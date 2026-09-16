"""Argument definitions for the P2- and P3-analysis CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ppbcc.constants import TIME_COLUMNS
from ppbcc.performance_portability.selection import (
    ALL_SIZE,
    AVERAGE_OVER_EFFICIENCY,
    AVERAGE_OVER_PP,
    parse_problem_size,
)

#: Charts that need benchmark results only.
P2_CHARTS = ("cascade", "heatmap", "boxplot", "time-barplot")
#: Charts that combine benchmark results with code complexity.
P3_CHARTS = ("navchart", "combined", "complexity-comparison")


def _new_parser(
    prog: str, description: str, charts: tuple[str, ...]
) -> argparse.ArgumentParser:
    """Create a parser whose first positional argument selects the chart.

    Args:
        prog: Program name shown in usage and help.
        description: Parser description.
        charts: Chart choices accepted by this command.

    Returns:
        The parser with the chart argument added.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "chart",
        choices=charts,
        metavar="PLOT",
        help=f"Chart to create: {', '.join(charts)}.",
    )
    return parser


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the benchmark inputs and the options both commands accept.

    Args:
        parser: Parser to extend. Its earlier positional arguments precede the
            benchmark CSVs.
    """
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        metavar="CSV",
        help="One or more CSV files produced by ppbcc benchmark.",
    )
    general = parser.add_argument_group("general script options")
    general.add_argument(
        "-n",
        "--name",
        help=(
            "Benchmark problem to plot. Exact case-insensitive matches are "
            "preferred; a unique substring match is accepted. May be omitted "
            "when the CSVs contain exactly one problem."
        ),
    )
    general.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v: DEBUG, -vv: TRACE).",
    )
    general.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output plot path. If no suffix is provided, .pdf is appended. "
            "Defaults to <problem>_<chart>.pdf."
        ),
    )
    general.add_argument(
        "-e",
        "--export-to-csv",
        action="store_true",
        help=(
            "Export application-efficiency and performance-portability data "
            "to <plot-prefix>_application_efficiency.csv and "
            "<plot-prefix>_performance_portability.csv. Exported metrics "
            "include separate and average rows for all sizes and precisions, "
            "and all available complexity metrics for p3analysis."
        ),
    )
    general.add_argument(
        "-l",
        "--legend",
        action="store_true",
        help=(
            "Omit legends from the plot and save them as a separate PDF with "
            "four columns by default."
        ),
    )
    general.add_argument(
        "--legend--vertical",
        dest="legend_vertical",
        action="store_true",
        help=(
            "Arrange entries in the separate legend in one column. Requires "
            "-l/--legend."
        ),
    )
    general.add_argument(
        "--remove-description",
        action="store_true",
        help=(
            "Remove bracketed descriptions such as [Naive] from plot labels "
            "and legends; efficiency charts combine variants by paradigm."
        ),
    )

    metrics = parser.add_argument_group(
        "performance portability / application efficiency metric options"
    )
    metrics.add_argument(
        "-i",
        "--include",
        dest="description_include",
        help=("Only keep rows whose Description matches this regular expression."),
    )
    metrics.add_argument(
        "-x",
        "--exclude",
        dest="description_exclude",
        help="Exclude rows whose Description matches this regular expression.",
    )
    metrics.add_argument(
        "-s",
        "--size",
        type=parse_problem_size,
        default=ALL_SIZE,
        help=(
            "Use all problem sizes (default), or this exact Problem Size, for "
            "application efficiency and the "
            "aggregate PP/complexity point; 'avg', 'average', or 'mean' takes "
            "the arithmetic mean; 'best' takes the maximum; and 'worst' "
            "takes the minimum of each metric over problem sizes. Scaling "
            "still uses all sizes. Boxplots accept only 'all' or a numeric size; "
            "time-barplot accepts only a numeric size and uses the largest "
            "size for 'all'."
        ),
    )
    metrics.add_argument(
        "--average-over",
        choices=(AVERAGE_OVER_PP, AVERAGE_OVER_EFFICIENCY),
        default=AVERAGE_OVER_PP,
        help=(
            "How --size avg reduces PP over problem sizes: 'pp' computes PP at "
            "every size and takes the arithmetic mean of those scores; "
            "'efficiency' averages each application efficiency over the sizes "
            "first and computes PP once from the averages. Application "
            "efficiency and the per-size heatmap are the same either way. "
            "Only valid with --size avg."
        ),
    )
    metrics.add_argument(
        "-p",
        "--precision",
        type=int,
        choices=[32, 64],
        help="Keep results with this floating-point precision.",
    )
    metrics.add_argument(
        "--non-zero-pp",
        action="store_true",
        help=(
            "Calculate PP over supported platforms only. Missing platforms "
            "remain zero in application-efficiency plots."
        ),
    )


def build_p2_parser() -> argparse.ArgumentParser:
    """Build the parser for charts that need benchmark results only.

    Returns:
        The configured argument parser.
    """
    parser = _new_parser(
        "p2analysis",
        "Create an application-efficiency or performance-portability plot "
        "from ppbcc benchmark CSV output.",
        P2_CHARTS,
    )
    _add_shared_arguments(parser)
    parser.add_argument(
        "-H",
        "--hardware",
        help=(
            "Only include results from this hardware in a boxplot or "
            "time-barplot."
        ),
    )
    runtime = parser.add_argument_group("time-barplot options")
    runtime.add_argument(
        "-t",
        "--time",
        choices=tuple(TIME_COLUMNS),
        help=(
            "Runtime column to plot (default: wall-clock). It is an error if "
            "the selected results do not contain it."
        ),
    )
    runtime.add_argument(
        "--normalize-time-to-peak",
        action="store_true",
        help=(
            "Multiply every runtime by the published peak performance of its "
            "platform and precision, i.e. plot the FLOPs the platform could "
            "have executed in that time."
        ),
    )
    return parser


def build_p3_parser() -> argparse.ArgumentParser:
    """Build the parser for charts that combine benchmarks and complexity.

    Returns:
        The configured argument parser.
    """
    parser = _new_parser(
        "p3analysis",
        "Create a performance-portability and code-complexity plot from ppbcc "
        "benchmark and code-complexity CSV output.",
        P3_CHARTS,
    )
    parser.add_argument(
        "complexity",
        type=Path,
        metavar="COMPLEXITY_CSV",
        help=(
            "Code-complexity CSV with one row per implementation; also "
            "included in CSV exports."
        ),
    )
    _add_shared_arguments(parser)

    complexity = parser.add_argument_group("code complexity options")
    complexity.add_argument(
        "-c",
        "--complexity-metric",
        default="halstead-difficulty",
        help=(
            "Complexity metric to plot: SLOC, Halstead vocabulary, Halstead "
            "program length, Halstead volume, Halstead difficulty, or Halstead "
            "effort (common short aliases are accepted). The y axis of "
            "complexity-comparison."
        ),
    )
    complexity.add_argument(
        "--complexity-metric-absolute",
        dest="complexity_absolute",
        action="store_true",
        help=(
            "Plot absolute complexity values instead of a percentage of the "
            "sequential CPP score. Not valid for complexity-comparison, whose "
            "identity line needs both metrics on the relative scale."
        ),
    )
    complexity.add_argument(
        "--compare-metric",
        default="sloc",
        help=(
            "Second complexity metric for complexity-comparison, plotted on "
            "the x axis against --complexity-metric on the y axis."
        ),
    )
    complexity.add_argument(
        "--log-complexity",
        action="store_true",
        help="Use logarithmic complexity axes.",
    )
    return parser
