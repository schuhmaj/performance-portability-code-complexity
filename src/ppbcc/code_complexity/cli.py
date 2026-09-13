"""Command-line interface for code-complexity analysis.

Examples:
    Analyse a directory with automatic dialect detection and save a CSV::

        python -m ppbcc.code_complexity path/to/src -o report.csv

    Force the Kokkos dialect, restrict the metrics and show the difference
    to the plain-C++ baseline::

        python -m ppbcc.code_complexity src/kokkos -d kokkos \\
            -m halstead_effort halstead_volume loc --diff -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from . import __version__
from .config import load_dialects
from .evaluate import evaluate
from .report import format_table

#: Loguru levels selected by the number of ``--verbose`` flags.
_VERBOSITY_LEVELS: dict[int, str] = {0: "INFO", 1: "DEBUG", 2: "TRACE"}


def configure_logging(verbosity: int) -> None:
    """Configures the loguru sink according to the CLI verbosity.

    Args:
        verbosity: Number of ``-v`` flags: 0 = INFO, 1 = DEBUG, >= 2 = TRACE.
    """
    level = _VERBOSITY_LEVELS.get(min(verbosity, 2), "TRACE")
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="code-complexity",
        description=(
            "Halstead complexity and LOC/SLOC metrics for C++ and GPU-enriched C++ "
            "(OpenMP, OpenACC, Kokkos, RAJA, Alpaka, CUDA, HIP, SYCL, OpenCL, "
            "Vulkan, Boost.Compute, WebGPU/WGSL, GLSL, Slang, Metal)."
        ),
        epilog="Example: code-complexity src/ -d kokkos -m halstead loc --diff -o report.csv",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="source files and/or directories (directories are searched recursively)",
    )
    parser.add_argument(
        "-d",
        "--dialect",
        default="auto",
        metavar="DIALECT",
        help=(
            "language dialect: 'auto' (per-file detection, default), 'cpp' (plain "
            "C++), or one or more dialect names separated by commas, e.g. "
            "'kokkos,openmp' (see --list-dialects)"
        ),
    )
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        metavar="METRIC",
        help=(
            "metrics to report, e.g. 'halstead', 'loc', 'sloc', 'halstead_effort', "
            "'halstead_volume', 'dialect' (default: all)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="CSV",
        help="write the resulting table to this CSV file",
    )
    parser.add_argument(
        "--csv-separator",
        default=",",
        metavar="SEP",
        help="field separator for the CSV output (default: ',')",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help=(
            "additionally compute the plain-C++ baseline (all dialect tokens "
            "removed) and report baseline_*/delta_* columns"
        ),
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="append a TOTAL row aggregating all analysed files",
    )
    parser.add_argument(
        "--exclude-macro",
        nargs="+",
        default=None,
        metavar="REGEX",
        help=(
            "macros to disregard, as regular expressions matched against the whole name: "
            "their invocations are removed and conditionals on them are resolved as if "
            r"they were undefined, e.g. 'PPB_MARKER_\w+'"
        ),
    )
    parser.add_argument(
        "--exclude-header",
        nargs="+",
        default=None,
        metavar="GLOB",
        help=(
            "headers to disregard, as glob patterns matched against an #include and the "
            "tail of a file path: they are not analysed and their #include lines are "
            "removed, e.g. 'common/Marker.h'"
        ),
    )
    parser.add_argument(
        "--table-format",
        default="rounded_outline",
        metavar="FMT",
        help=(
            "tabulate format for the table printed to stdout, e.g. "
            "'rounded_outline' (default), 'github', 'psql', 'simple', 'tsv'"
        ),
    )
    parser.add_argument(
        "--keywords-config",
        type=Path,
        metavar="TOML",
        help="override the packaged cpp_keywords.toml",
    )
    parser.add_argument(
        "--dialects-config",
        type=Path,
        metavar="TOML",
        help="override the packaged dialects.toml",
    )
    parser.add_argument(
        "--list-dialects",
        action="store_true",
        help="list all known dialects and their aliases, then exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (default: INFO, -v: DEBUG, -vv: TRACE)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def list_dialects(dialects_config: Path | None) -> None:
    """Prints all known dialects with their aliases to stdout.

    Args:
        dialects_config: Optional override for the packaged ``dialects.toml``.
    """
    registry = load_dialects(dialects_config)
    for name, spec in registry.dialects.items():
        aliases = ", ".join(sorted(spec.aliases - {name}))
        print(f"{name:15s}{aliases}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 on error.
    """
    parser = build_parser()
    # --list-dialects works without sources, so peek before full parsing.
    raw_args = sys.argv[1:] if argv is None else argv
    if "--list-dialects" in raw_args:
        configure_logging(raw_args.count("-v") + raw_args.count("--verbose"))
        list_dialects(None)
        return 0

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        frame = evaluate(
            sources=args.sources,
            language_dialect=args.dialect,
            metrics=args.metrics,
            diff=args.diff,
            aggregate=args.aggregate,
            output=args.output,
            csv_separator=args.csv_separator,
            keywords_path=args.keywords_config,
            dialects_path=args.dialects_config,
            exclude_macros=args.exclude_macro,
            exclude_headers=args.exclude_header,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        message = error.args[0] if error.args else error
        logger.error("{}", message)
        return 1

    print(format_table(frame, table_format=args.table_format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
