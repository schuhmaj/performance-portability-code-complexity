"""Unified command-line interface for :mod:`ppbcc`."""

from __future__ import annotations

import argparse
import sys

from ppbcc import __version__


def _run_command(name: str, argv: list[str]) -> int:
    """Load and run one command handler lazily.

    Args:
        name: Selected command name.
        argv: Remaining command-line arguments.

    Returns:
        Process exit status from the selected command.
    """
    if name == "benchmark":
        from ppbcc.benchmark.cli import main as command
    elif name == "code-complexity":
        from ppbcc.code_complexity.cli import main as command
    elif name == "profile":
        from ppbcc.profiling.cli import main as command
    elif name == "p2analysis":
        from ppbcc.performance_portability.cli import p2analysis_main as command
    else:
        from ppbcc.performance_portability.cli import p3analysis_main as command
    return command(argv)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser.

    Returns:
        Configured parser for the grouped CLI.
    """
    parser = argparse.ArgumentParser(
        prog="ppbcc",
        description=(
            "Performance-portability benchmarking, analysis, plotting, and "
            "code-complexity tools."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "benchmark",
            "profile",
            "code-complexity",
            "p2analysis",
            "p3analysis",
        ),
        help="tool to run",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch the selected ppbcc subcommand.

    Args:
        argv: Arguments without the executable name.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 0
    return _run_command(args.command, args.arguments)


if __name__ == "__main__":
    sys.exit(main())
