"""Command-line interface for benchmark execution and consolidation."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from ppbcc.benchmark.reports import load_reports
from ppbcc.benchmark.runner import find_files, print_found_files, run_benchmarks


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description=(
            "Run the benchmarking targets and consolidate their Google-Benchmark "
            "JSON reports into a single tidy CSV."
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
        "executables (or reports with --skip-benchmark).",
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
        "--skip-benchmark",
        action="store_true",
        help="Skip running benchmarks; use --path/--regex to locate reports directly.",
    )
    discovery.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-run benchmarks even if a report already exists.",
    )
    discovery.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Only list the executables (or reports with --skip-benchmark) "
        "matched by --path/--regex, then exit without doing anything else.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-H",
        "--hardware",
        type=str,
        default="",
        help="Hardware identifier stored in the 'Hardware' column (e.g. RTX5080).",
    )
    output.add_argument(
        "-P",
        "--json-prefix",
        type=str,
        default="",
        help="Prefix prepended to the intermediate JSON report file names "
        "(e.g. 'intel_' -> 'intel_nbody_kokkos.json').",
    )
    default_output = datetime.now().strftime("%Y-%m-%d_%H-%M_Benchmark_Result")
    output.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(default_output),
        help="Output base name for the consolidated CSV.",
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
    """Run the benchmark command.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=["INFO", "DEBUG", "TRACE"][min(args.verbose, 2)])

    # The build folder is always the working directory.
    build_dir = args.build_dir.resolve()
    if not build_dir.is_dir():
        logger.error(f"Build directory {build_dir} does not exist.")
        return 1
    os.chdir(build_dir)
    logger.info(f"Working directory: {build_dir}")

    # 1. Discover --------------------------------------------------------------
    files = find_files(
        args.path,
        args.regex,
        require_executable=not args.skip_benchmark,
        exclude=args.exclude,
    )

    if args.dry_run:
        print_found_files(files, build_dir, are_reports=args.skip_benchmark)
        return 0 if files else 1

    if not files:
        logger.error("Nothing matched the given path/regex.")
        return 1

    # 2. Benchmark (unless skipped) -------------------------------------------
    if args.skip_benchmark:
        reports = files
    else:
        reports = run_benchmarks(
            files,
            force=args.force,
            json_prefix=args.json_prefix,
            stream_output=args.verbose >= 2,
        )
    if not reports:
        logger.error("No report files available to process.")
        return 1

    # 3. Consolidate -----------------------------------------------------------
    df = load_reports(reports, hardware=args.hardware)
    if df.empty:
        return 1
    csv_path = args.output.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.success(f"Wrote unified CSV: {csv_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
