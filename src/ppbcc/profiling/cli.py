"""Command-line interface for batch kernel profiling with Nsight Compute."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from ppbcc.benchmark.runner import find_files, print_found_files
from ppbcc.constants import BENCHMARK_PROBLEM, EXECUTABLE
from ppbcc.plot.roofline import plot_roofline
from ppbcc.plot.styles import save_figure
from ppbcc.profiling.metrics import MEMORY_LEVELS, PRECISIONS, ROOFLINE_METRICS
from ppbcc.profiling.reports import (
    aggregate_kernels,
    filter_kernels,
    load_reports,
)
from ppbcc.profiling.runner import REPORT_SUFFIX, find_ncu, run_profiles


def build_parser() -> argparse.ArgumentParser:
    """Build the profile command parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="profile",
        description=(
            "Profile benchmark executables one after another with Nvidia "
            "Nsight Compute (ncu), consolidate the per-kernel counters into a "
            "single CSV, and draw a roofline model from them. The executables "
            "are expected to be built with -DPPB_PROFILING=ON, which reduces "
            "them to a single input running a single iteration."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    discovery = parser.add_argument_group("discovery")
    discovery.add_argument(
        "-p",
        "--path",
        nargs="+",
        type=Path,
        default=[Path(".")],
        help="Directories to search recursively (relative to --build-dir).",
    )
    discovery.add_argument(
        "-r",
        "--regex",
        nargs="+",
        type=str,
        required=True,
        help="Regex pattern(s) matched against file paths to select "
        "executables (or reports with --skip-profile).",
    )
    discovery.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        type=str,
        default=[],
        help="Regex pattern(s) to exclude matched paths (e.g. '.*_cpp').",
    )
    discovery.add_argument(
        "-b",
        "--build-dir",
        type=Path,
        default=Path.cwd(),
        help="Build folder used as the working directory for the whole pipeline.",
    )
    discovery.add_argument(
        "-s",
        "--skip-profile",
        action="store_true",
        help="Skip profiling; only parse the reports already in --report-dir.",
    )
    discovery.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-profile even if a report already exists.",
    )
    discovery.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Only list the matched executables (or reports), then exit.",
    )

    profiler = parser.add_argument_group("profiler")
    profiler.add_argument(
        "--ncu",
        type=str,
        default=None,
        help="Path to the Nsight Compute CLI, defaulting to 'ncu' in PATH.",
    )
    profiler.add_argument(
        "-d",
        "--report-dir",
        type=Path,
        default=Path("profiling"),
        help="Directory the '<executable>.ncu-rep' reports are written to.",
    )
    profiler.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=None,
        help="Override the collected metrics (default: the roofline set).",
    )
    profiler.add_argument(
        "--ncu-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument passed through to ncu; repeatable.",
    )
    profiler.add_argument(
        "--target-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument passed through to the benchmark binary; repeatable.",
    )

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument(
        "-m",
        "--memory-level",
        choices=sorted(MEMORY_LEVELS),
        default="dram",
        help="Level of the memory hierarchy the arithmetic intensity refers to.",
    )
    analysis.add_argument(
        "--precision",
        choices=["auto", *sorted(PRECISIONS)],
        default="auto",
        help="Precision the FLOPs and the compute ceiling refer to; 'auto' "
        "follows the precision the binary was built with.",
    )
    analysis.add_argument(
        "-k",
        "--exclude-kernel",
        action="append",
        default=[],
        metavar="REGEX",
        help="Drop kernels matching this pattern from the plot (not from the "
        "CSV); repeatable. Useful for framework bootstrap kernels such as "
        "'init_lock_arrays' or 'query_cuda_kernel_arch'.",
    )
    analysis.add_argument(
        "-a",
        "--aggregate",
        choices=["sum", "dominant", "none"],
        default="sum",
        help="How the kernels of one executable become plot points: 'sum' "
        "adds all launches up, 'dominant' keeps the longest-running kernel, "
        "'none' plots every launch.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-H",
        "--hardware",
        type=str,
        default="",
        help="Hardware identifier stored in the 'Hardware' column and used in "
        "the plot title (e.g. 'NVIDIA RTX5080').",
    )
    default_output = datetime.now().strftime("%Y-%m-%d_%H-%M_Profiling_Result")
    output.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(default_output),
        help="Output base name for the consolidated CSV.",
    )
    output.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the consolidated CSV. Useful for a pure collection "
        "run whose reports are consolidated later with --skip-profile.",
    )
    output.add_argument(
        "--roofline",
        action="store_true",
        help="Also render a roofline chart.",
    )
    output.add_argument(
        "--roofline-output",
        type=Path,
        default=None,
        help="Output path for the roofline chart "
        "(default: '<problem>_roofline.pdf').",
    )
    output.add_argument(
        "--label-points",
        action="store_true",
        help="Annotate every roofline point with its paradigm. Off by default: "
        "the legend already carries that, and the points tend to cluster.",
    )
    output.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v: DEBUG, -vv: TRACE).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the profile command.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=["INFO", "DEBUG", "TRACE"][min(args.verbose, 2)])

    # The build folder is always the working directory: the benchmark binaries
    # resolve their input meshes relative to it.
    build_dir = args.build_dir.resolve()
    if not build_dir.is_dir():
        logger.error(f"Build directory {build_dir} does not exist.")
        return 1
    os.chdir(build_dir)
    logger.info(f"Working directory: {build_dir}")

    try:
        ncu = find_ncu(args.ncu)
    except FileNotFoundError as error:
        logger.error(str(error))
        return 1

    # 1. Discover --------------------------------------------------------------
    search_paths = [args.report_dir] if args.skip_profile else args.path
    patterns = [f".*{REPORT_SUFFIX}$"] if args.skip_profile else args.regex
    files = find_files(
        search_paths,
        patterns,
        require_executable=not args.skip_profile,
        exclude=args.exclude,
    )
    if args.skip_profile:
        # --regex still selects, but now among the existing reports.
        keep = find_files([args.report_dir], args.regex, exclude=args.exclude)
        files = [path for path in files if path in set(keep)]

    if args.dry_run:
        print_found_files(files, build_dir, are_reports=args.skip_profile)
        return 0 if files else 1
    if not files:
        logger.error("Nothing matched the given path/regex.")
        return 1

    # 2. Profile (unless skipped) ---------------------------------------------
    if args.skip_profile:
        reports = files
    else:
        reports = run_profiles(
            files,
            args.report_dir,
            ncu,
            force=args.force,
            metrics=args.metrics or ROOFLINE_METRICS,
            extra_ncu_args=args.ncu_arg,
            extra_target_args=args.target_arg,
            stream_output=args.verbose >= 2,
        )
    if not reports:
        logger.error("No profiler reports available to process.")
        return 1

    # 3. Consolidate -----------------------------------------------------------
    data = load_reports(
        reports,
        ncu=ncu,
        hardware=args.hardware,
        precision=args.precision,
        memory_level=args.memory_level,
    )
    if data.empty:
        return 1
    if not args.no_csv:
        csv_path = args.output.with_suffix(".csv")
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(csv_path, index=False)
        except OSError as error:
            logger.error(f"Could not write {csv_path}: {error}")
            return 1
        logger.success(f"Wrote unified CSV: {csv_path.resolve()}")

    # 4. Plot ------------------------------------------------------------------
    if not args.roofline:
        if args.roofline_output is not None:
            logger.warning("Ignoring --roofline-output without --roofline.")
        return 0
    points = aggregate_kernels(
        filter_kernels(data, args.exclude_kernel), args.aggregate
    )
    problems = sorted({str(value) for value in points[BENCHMARK_PROBLEM]})
    problem_title = ", ".join(problems)
    output = args.roofline_output
    if output is None:
        slug = "_".join(problems).lower() or "profile"
        output = Path(f"{slug}_roofline.pdf")
    if not output.suffix:
        output = output.with_suffix(".pdf")
    try:
        figure = plot_roofline(
            points,
            hardware=args.hardware,
            problem_title=problem_title,
            label_points=args.label_points,
            subtitle={
                "sum": "one point per implementation, summed over all its kernels",
                "dominant": "one point per implementation, its longest kernel only",
                "none": "one point per kernel launch",
            }[args.aggregate],
        )
    except ValueError as error:
        logger.error(str(error))
        return 1
    save_figure(figure, output)
    logger.info(
        "Roofline points: "
        + ", ".join(sorted(str(value) for value in points[EXECUTABLE]))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
