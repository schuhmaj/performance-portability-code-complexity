"""Command-line interface for batch kernel profiling.

Four backends are available, selected with ``--profiler``. They differ in what
they can see, not in what they report: each one writes the same columns, so
their tables concatenate into one roofline (``--from-csv``).

* ``ncu``    -- Nsight Compute. Counts instructions and bytes per kernel launch,
  but only inside a CUDA context.
* ``nsys``   -- Nsight Systems. Samples the GPU's performance monitors
  device-wide, which is what makes OpenCL and Vulkan visible.
* ``ngfx``   -- Nsight Graphics GPU Trace. Sums the same counters over a Vulkan
  queue submission; an independent cross-check for the Vulkan paradigms.
* ``likwid`` -- reads counters inside the regions marked in the benchmark source.

Backend-specific settings are passed as ``-O <name>=<value>`` rather than as one
flag each; ``--help`` lists the names each backend accepts.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from loguru import logger

from ppbcc.benchmark.runner import find_files, print_found_files
from ppbcc.constants import BENCHMARK_PROBLEM, EXECUTABLE, REGION
from ppbcc.plot.roofline import plot_roofline
from ppbcc.plot.styles import save_figure
from ppbcc.profiling.metrics import MEMORY_LEVELS, PRECISIONS, ROOFLINE_METRICS
from ppbcc.profiling.reports import (
    aggregate_kernels,
    filter_regions,
    load_csv,
    load_reports,
    select_regions,
)
from ppbcc.profiling.runner import REPORT_SUFFIX, find_ncu, run_profiles
from ppbcc.profiling import likwid as likwid_backend
from ppbcc.profiling import ngfx as ngfx_backend
from ppbcc.profiling import nsys as nsys_backend

#: Backend settings reachable through ``-O``: name -> (backend, converter,
#: default, help). Keeping them out of the flag namespace is what stops the
#: parser from growing a flag per backend per knob; a name that does not belong
#: to the selected backend is an error, so a typo cannot pass silently.
BACKEND_OPTIONS: dict[str, tuple[str, object, object, str]] = {
    "metrics": (
        "ncu",
        lambda value: value.split(","),
        None,
        "comma-separated metrics to collect instead of the roofline set",
    ),
    "ncu-arg": (
        "ncu",
        "append",
        [],
        "extra argument passed through to ncu; repeatable",
    ),
    "aslr": (
        "ncu",
        lambda value: value.lower() in ("on", "true", "yes", "1"),
        False,
        "'on' keeps address space randomization. Off by default, which stops "
        "Google Benchmark from re-executing the process under the profiler; "
        "Nsight Compute otherwise hangs on some binaries (matMul_kokkos, "
        "matMul_omp)",
    ),
    "metric-set": (
        "nsys",
        str,
        "gb20x",
        "GPU metric set alias; must match the architecture ('gb20x' for a "
        "Blackwell consumer part). 'nsys profile --gpu-metrics-set=help' lists "
        "the aliases",
    ),
    "frequency": ("nsys", int, 100_000, "sampling frequency in Hz"),
    "iterations": (
        "nsys",
        int,
        20,
        "iterations of the many-iteration run. The sampler cannot tell a kernel "
        "launch from the pipeline compilation and buffer uploads a lazily "
        "initialising backend does on its first call, so every executable is "
        "profiled twice, once with a single iteration and once with N; "
        "subtracting the first from the second cancels the setup both paid. "
        "1 disables the second run and leaves the setup cost in the result",
    ),
    "activity-ratio": (
        "nsys",
        float,
        0.7,
        "compute-active windows whose mean occupancy is below this fraction of "
        "the busiest window's in the same region are treated as data movement "
        "rather than as the kernel. Some runtimes move buffers with a compute "
        "shader instead of a copy engine, and those phases sit inside the same "
        "marked region as the kernel",
    ),
    "architecture": (
        "ngfx",
        str,
        "Blackwell GB20x",
        "architecture whose metric set is selected; 'ngfx --help-all' lists the "
        "names",
    ),
    "submit": (
        "ngfx",
        str,
        "auto",
        "queue submission to trace. Without a swapchain there are no frames, so "
        "the traced region is bounded by submit index instead. 'auto' traces "
        "the first 'probes' submissions and keeps the one that spent the most "
        "cycles in compute",
    ),
    "probes": ("ngfx", int, 4, "how many submissions 'submit=auto' tries"),
    "lib": (
        "likwid",
        Path,
        None,
        "lib directory of the LIKWID installation, prepended to LD_LIBRARY_PATH",
    ),
    "gpu": ("likwid", int, 0, "index of the GPU the counters are read from"),
}


def _option_help() -> str:
    """Render the ``-O`` names as a help epilog, grouped by backend.

    Returns:
        The epilog text.
    """
    lines = ["backend settings for -O/--option, by --profiler:"]
    for backend in ("ncu", "nsys", "ngfx", "likwid"):
        lines.append(f"  {backend}:")
        for name, (owner, _, default, text) in BACKEND_OPTIONS.items():
            if owner != backend:
                continue
            shown = "" if default is None or default == [] else f" [{default}]"
            lines += textwrap.wrap(
                f"{name}{shown}: {text}",
                width=78,
                initial_indent="    ",
                subsequent_indent="        ",
            )
    return "\n".join(lines)


def parse_options(pairs: list[str], profiler: str) -> dict[str, object]:
    """Turn the ``-O name=value`` arguments into the selected backend's settings.

    Args:
        pairs: The raw ``name=value`` strings, in order.
        profiler: The backend the settings have to belong to.

    Returns:
        Every option of that backend, defaults filled in.

    Raises:
        ValueError: On a malformed pair, an unknown name, a name belonging to a
            different backend, or a value the converter rejects.
    """
    settings = {
        name: default
        for name, (backend, _, default, _) in BACKEND_OPTIONS.items()
        if backend == profiler
    }
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator:
            raise ValueError(f"-O expects name=value, got {pair!r}")
        if name not in BACKEND_OPTIONS:
            raise ValueError(f"Unknown -O name {name!r}")
        backend, convert, _, _ = BACKEND_OPTIONS[name]
        if backend != profiler:
            raise ValueError(
                f"-O {name} belongs to --profiler {backend}, not {profiler}"
            )
        if convert == "append":
            settings[name] = [*settings[name], value]
            continue
        try:
            settings[name] = convert(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Bad value for -O {name}: {value!r} ({error})") from error
    return settings


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show defaults in the option help, but leave the epilog table unwrapped."""


def build_parser() -> argparse.ArgumentParser:
    """Build the profile command parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="profile",
        description=textwrap.fill(
            "Profile benchmark executables one after another, consolidate the "
            "per-kernel counters into a single CSV, and draw a roofline model "
            "from them. The executables are expected to be built with "
            "-DPPB_PROFILING=ON, which reduces them to a single input running a "
            "single iteration and names their kernels with NVTX ranges.",
            width=78,
        ),
        epilog=_option_help(),
        formatter_class=_HelpFormatter,
    )

    discovery = parser.add_argument_group("discovery")
    discovery.add_argument(
        "-p",
        "--path",
        nargs="+",
        type=Path,
        default=[Path(".")],
        help="Directories to search recursively, relative to --build-dir "
        "(default: the build directory itself).",
    )
    discovery.add_argument(
        "-r",
        "--regex",
        nargs="+",
        type=str,
        default=[],
        help="Regex pattern(s) matched against file paths to select "
        "executables (or reports with --skip-profile). With --from-csv they "
        "select rows by executable name instead. Required unless --from-csv "
        "is given.",
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
        help="Build folder used as the working directory for the whole pipeline "
        "(default: the current directory). Note that --output is relative to it.",
    )
    discovery.add_argument(
        "--from-csv",
        nargs="+",
        type=Path,
        default=[],
        metavar="CSV",
        help=(
            "Skip discovery and profiling entirely and read these consolidated "
            "profiling CSVs instead. No single profiler covers every paradigm "
            "-- Nsight Compute reads counters only inside a CUDA context -- so "
            "this is how the tables of several backends become one roofline."
        ),
    )
    discovery.add_argument(
        "-s",
        "--skip-profile",
        action="store_true",
        help="Skip profiling; only parse the artefacts already in --report-dir.",
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
        "--profiler",
        choices=("ncu", "nsys", "ngfx", "likwid"),
        default="ncu",
        help=(
            "Profiling backend. 'ncu' replays every kernel launch from the "
            "outside and needs no instrumentation; 'nsys' samples the GPU's "
            "performance monitors device-wide, which is the only one that also "
            "sees OpenCL and Vulkan; 'ngfx' runs Nsight Graphics' GPU Trace "
            "Profiler, which reads the same counters as sums rather than "
            "samples but only for Vulkan; 'likwid' reads the counters inside "
            "the regions marked in the benchmark source (build with "
            "-DPPB_ENABLE_LIKWID=ON) and reports one row per region."
        ),
    )
    profiler.add_argument(
        "--profiler-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the selected backend's CLI. Found automatically: 'ncu' "
        "and 'nsys' in PATH, 'ngfx' under /opt/nvidia/nsight-graphics-for-linux.",
    )
    profiler.add_argument(
        "-O",
        "--option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Backend setting; repeatable. The names the selected --profiler "
        "accepts are listed at the end of this help.",
    )
    profiler.add_argument(
        "--target-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument passed through to the benchmark binary; repeatable.",
    )
    profiler.add_argument(
        "-d",
        "--report-dir",
        type=Path,
        default=Path("profiling"),
        help="Directory the profiler artefacts are written to.",
    )
    profiler.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Wall-clock limit per executable. A run that hits it is skipped and "
            "the batch continues. Without a limit a profiler that stops making "
            "progress -- ncu's kernel replay on a large working set does -- "
            "blocks the whole batch indefinitely."
        ),
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
        "--analytic-flop",
        action="append",
        default=[],
        metavar="REGEX=FLOP",
        help=(
            "Analytic FLOP count for rows matching REGEX; repeatable, first "
            "match wins. REGEX is matched against '<executable>[<region>]', so "
            "'matMul_' addresses a whole binary and "
            "'polyhedral_.*\\[evaluate\\]' only its evaluation kernel. "
            "Required for --profiler nsys and ngfx: their metric sets carry "
            "pipe utilisations, not instruction counts. The value is not a "
            "guess for these benchmarks -- Nsight Compute reproduces 2*M*N*K "
            "exactly on every CUDA-backed matrix multiplication."
        ),
    )
    analysis.add_argument(
        "--peak-performance",
        type=float,
        default=None,
        metavar="FLOPS",
        help=(
            "Compute ceiling in FLOP/s. Only Nsight Compute measures the "
            "ceilings itself; the sampling backends report percentages of a "
            "peak they do not name, so it has to be supplied. It is a property "
            "of the hardware, and an ncu run on the same machine measures it."
        ),
    )
    analysis.add_argument(
        "--peak-bandwidth",
        type=float,
        default=None,
        metavar="BYTES_PER_S",
        help="Memory ceiling in bytes/s, supplied for the same reason *and* "
        "needed to turn the sampled traffic percentages into bytes.",
    )
    analysis.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="REGEX",
        help="Plot only the regions matching this pattern; repeatable. The "
        "benchmark names the work it is about with NVTX ranges ('matmul', "
        "'init', 'evaluate'), so '--region evaluate' plots the polyhedral "
        "kernel without its one-off setup. The CSV always keeps every region.",
    )
    analysis.add_argument(
        "--all-kernels",
        action="store_true",
        help="Keep the launches that happened outside every named region. Off "
        "by default: those are the runtime setting itself up (Kokkos' "
        "architecture query, desul's lock arrays) and say nothing about the "
        "algorithm.",
    )
    analysis.add_argument(
        "-a",
        "--aggregate",
        choices=["sum", "dominant", "none"],
        default="sum",
        help="How the kernel launches of one region become plot points: 'sum' "
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
        help="Output base name for the consolidated CSV, relative to "
        "--build-dir unless absolute.",
    )
    output.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the consolidated CSV. Useful for a pure collection "
        "run whose reports are consolidated later with --skip-profile.",
    )
    output.add_argument(
        "--roofline",
        nargs="?",
        type=str,
        const="",
        default=None,
        metavar="PATH",
        help="Also render a roofline chart, optionally to this path "
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


def _select_rows(data, patterns: list[str], excludes: list[str]):
    """Apply ``--regex``/``--exclude`` to a table read with ``--from-csv``.

    Args:
        data: The concatenated table.
        patterns: Patterns a row's executable has to match, or none to keep all.
        excludes: Patterns a row's executable must not match.

    Returns:
        The surviving rows.
    """
    names = data[EXECUTABLE].astype(str)
    if patterns:
        data = data[names.map(lambda name: any(re.search(p, name) for p in patterns))]
    if excludes:
        names = data[EXECUTABLE].astype(str)
        data = data[~names.map(lambda name: any(re.search(p, name) for p in excludes))]
    return data.reset_index(drop=True)


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

    try:
        options = parse_options(args.option, args.profiler)
    except ValueError as error:
        logger.error(str(error))
        return 1

    # The build folder is always the working directory: the benchmark binaries
    # resolve their input meshes relative to it.
    build_dir = args.build_dir.resolve()
    if not build_dir.is_dir():
        logger.error(f"Build directory {build_dir} does not exist.")
        return 1
    os.chdir(build_dir)
    logger.info(f"Working directory: {build_dir}")

    # 0. Plot straight from finished CSVs --------------------------------------
    if args.from_csv:
        data = load_csv([Path(path).resolve() for path in args.from_csv])
        if data.empty:
            return 1
        # Here --regex/--exclude select rows rather than files.
        data = _select_rows(data, args.regex, args.exclude)
        if data.empty:
            logger.error("No row survived --regex/--exclude.")
            return 1
        if args.dry_run:
            for name in sorted(set(data[EXECUTABLE].astype(str))):
                print(name)
            return 0
        return _write_and_plot(args, data)

    if not args.regex:
        logger.error("--regex is required unless --from-csv is given.")
        return 1

    # The backend's own CLI is only needed for a run that actually profiles.
    tool = ""
    if not args.skip_profile:
        finder = {
            "ncu": find_ncu,
            "nsys": nsys_backend.find_nsys,
            "ngfx": ngfx_backend.find_ngfx,
        }.get(args.profiler)
        if finder is not None:
            try:
                tool = finder(args.profiler_path)
            except FileNotFoundError as error:
                logger.error(str(error))
                return 1

    # 1. Discover --------------------------------------------------------------
    # Each backend re-reads a finished batch from its own artefacts: ncu from the
    # .ncu-rep reports, LIKWID from the marker files, nsys from the SQLite
    # exports, ngfx from the exported trace tables.
    reuse_reports = args.skip_profile and args.profiler == "ncu"
    if args.skip_profile and args.profiler in ("likwid", "nsys", "ngfx"):
        backend = {
            "likwid": likwid_backend,
            "nsys": nsys_backend,
            "ngfx": ngfx_backend,
        }[args.profiler]
        files = backend.find_profiled(args.report_dir)
        patterns = [re.compile(pattern) for pattern in args.regex]
        excludes = [re.compile(pattern) for pattern in args.exclude]
        files = [
            path
            for path in files
            if any(pattern.search(path.name) for pattern in patterns)
            and not any(pattern.search(path.name) for pattern in excludes)
        ]
    else:
        search_paths = [args.report_dir] if reuse_reports else args.path
        patterns = [f".*{REPORT_SUFFIX}$"] if reuse_reports else args.regex
        files = find_files(
            search_paths,
            patterns,
            require_executable=not args.skip_profile,
            exclude=args.exclude,
        )
        if reuse_reports:
            # --regex still selects, but now among the existing reports.
            keep = find_files([args.report_dir], args.regex, exclude=args.exclude)
            files = [path for path in files if path in set(keep)]

    if args.dry_run:
        print_found_files(files, build_dir, are_reports=reuse_reports)
        return 0 if files else 1
    if not files:
        logger.error("Nothing matched the given path/regex.")
        return 1

    # 2. Profile (unless skipped) and 3. consolidate ---------------------------
    if args.profiler == "ngfx":
        traces = (
            files
            if args.skip_profile
            else ngfx_backend.run_profiles(
                files,
                args.report_dir,
                tool,
                working_directory=build_dir,
                architecture=options["architecture"],
                submit=options["submit"],
                probes=options["probes"],
                force=args.force,
                stream_output=args.verbose >= 2,
                timeout=args.timeout,
            )
        )
        if not traces:
            logger.error("No traces available to process.")
            return 1
        data = ngfx_backend.load_reports(
            traces,
            args.report_dir,
            hardware=args.hardware,
            precision=args.precision,
            peak_performance=args.peak_performance,
            peak_bandwidth=args.peak_bandwidth,
            flop_rules=args.analytic_flop,
        )
    elif args.profiler == "nsys":
        databases = (
            files
            if args.skip_profile
            else nsys_backend.run_profiles(
                files,
                args.report_dir,
                tool,
                metric_set=options["metric-set"],
                frequency=options["frequency"],
                iterations=options["iterations"],
                force=args.force,
                extra_target_args=args.target_arg,
                stream_output=args.verbose >= 2,
                timeout=args.timeout,
            )
        )
        if not databases:
            logger.error("No profiler reports available to process.")
            return 1
        data = nsys_backend.load_reports(
            databases,
            args.report_dir,
            hardware=args.hardware,
            precision=args.precision,
            peak_performance=args.peak_performance,
            peak_bandwidth=args.peak_bandwidth,
            flop_rules=args.analytic_flop,
            iterations=options["iterations"],
            activity_ratio=options["activity-ratio"],
        )
    elif args.profiler == "likwid":
        # Unlike ncu, LIKWID has to know the precision up front: the counters are
        # programmed before the run, and FADD/FMUL/FFMA are different events from
        # their FP64 and FP16 counterparts. "auto" cannot be resolved from the
        # Google-Benchmark report yet, because that report is written by the run
        # being set up, so it falls back to the default build precision.
        likwid_precision = "fp32" if args.precision == "auto" else args.precision
        targets = (
            files
            if args.skip_profile
            else likwid_backend.run_profiles(
                files,
                args.report_dir,
                likwid_lib=options["lib"],
                gpu=options["gpu"],
                precision=likwid_precision,
                force=args.force,
                extra_target_args=args.target_arg,
                stream_output=args.verbose >= 2,
                timeout=args.timeout,
            )
        )
        if not targets:
            logger.error("No marker files available to process.")
            return 1
        data = likwid_backend.load_reports(
            targets,
            args.report_dir,
            hardware=args.hardware,
            precision=likwid_precision,
            peak_performance=args.peak_performance,
            peak_bandwidth=args.peak_bandwidth,
        )
    else:
        reports = (
            files
            if args.skip_profile
            else run_profiles(
                files,
                args.report_dir,
                tool,
                force=args.force,
                metrics=options["metrics"] or ROOFLINE_METRICS,
                extra_ncu_args=options["ncu-arg"],
                extra_target_args=args.target_arg,
                stream_output=args.verbose >= 2,
                timeout=args.timeout,
                disable_aslr=not options["aslr"],
            )
        )
        if not reports:
            logger.error("No profiler reports available to process.")
            return 1
        data = load_reports(
            reports,
            ncu=tool or find_ncu(args.profiler_path),
            hardware=args.hardware,
            precision=args.precision,
            memory_level=args.memory_level,
        )
    if data.empty:
        return 1
    return _write_and_plot(args, data)


def _write_and_plot(args: argparse.Namespace, data) -> int:
    """Write the consolidated CSV and render the roofline.

    Args:
        args: Parsed command-line arguments.
        data: One row per profiled kernel launch, sampled window or marked
            region.

    Returns:
        Process exit status.
    """
    # The unlabelled launches go before the CSV is written: the table is about
    # the benchmark's kernels, not about the runtime bootstrapping itself.
    data = filter_regions(data, keep_unlabelled=args.all_kernels)
    if data.empty:
        logger.error("No launch was left after dropping the unlabelled ones.")
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
    if args.roofline is None:
        return 0
    points = aggregate_kernels(select_regions(data, args.region), args.aggregate)
    if points.empty:
        logger.error("No row was left after --region.")
        return 1
    problems = sorted({str(value) for value in points[BENCHMARK_PROBLEM]})
    problem_title = ", ".join(problems)
    if args.roofline:
        output = Path(args.roofline)
    else:
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
            # A row means something different per backend: ncu reports kernel
            # launches, LIKWID the regions marked in the source.
            subtitle={
                "sum": "one point per implementation and region, summed over all its {plural}",
                "dominant": "one point per implementation and region, its longest {singular} only",
                "none": "one point per {singular}",
            }[args.aggregate].format(
                singular={
                    "likwid": "marked region",
                    "nsys": "sampled compute window",
                    "ngfx": "traced submission",
                }.get(args.profiler, "kernel launch"),
                plural={
                    "likwid": "regions",
                    "nsys": "windows",
                    "ngfx": "submissions",
                }.get(args.profiler, "kernels"),
            ),
        )
    except ValueError as error:
        logger.error(str(error))
        return 1
    save_figure(figure, output)
    logger.info(
        "Roofline points: "
        + ", ".join(
            sorted(
                f"{row[EXECUTABLE]} [{row[REGION]}]" if REGION in points else
                str(row[EXECUTABLE])
                for _, row in points.iterrows()
            )
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
