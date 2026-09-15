"""Argument definitions for the P3-analysis CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from ppbcc.performance_portability.selection import (
    ALL_SIZE,
    AVERAGE_OVER_EFFICIENCY,
    AVERAGE_OVER_PP,
    parse_problem_size,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="p3analysis",
        description=("Create a P3 analysis plot from " "benchmark.py CSV output."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    general = parser.add_argument_group("general script options")
    general.add_argument(
        "name",
        metavar="NAME",
        help=(
            "Benchmark problem to plot. Exact case-insensitive matches are "
            "preferred; a unique substring match is accepted."
        ),
    )
    general.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        metavar="CSV",
        help="One or more CSV files produced by benchmark.py.",
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
            "and all available complexity metrics when --complexity is "
            "supplied."
        ),
    )
    general.add_argument(
        "-c",
        "--chart",
        choices=(
            "cascade",
            "navchart",
            "combined",
            "complexity-comparison",
            "heatmap",
            "boxplot",
        ),
        default="cascade",
        help=(
            "Chart to create. Navchart, combined and complexity-comparison "
            "require --complexity."
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

    complexity = parser.add_argument_group("code complexity options")
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
            "still uses all sizes. Boxplots accept only 'all' or a numeric size."
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
        "-H",
        "--hardware",
        help=(
            "Only include results from this hardware in a boxplot. This option "
            "is invalid for all other chart types."
        ),
    )
    metrics.add_argument(
        "--non-zero-pp",
        action="store_true",
        help=(
            "Calculate PP over supported platforms only. Missing platforms "
            "remain zero in application-efficiency plots."
        ),
    )
    metrics.add_argument(
        "--log-size",
        action="store_true",
        help=(
            "Deprecated and ignored. The combined scaling panel is a heatmap "
            "over the discrete benchmark sizes and has no continuous axis."
        ),
    )

    complexity.add_argument(
        "--complexity",
        type=Path,
        help=(
            "Code-complexity CSV used by navchart and combined charts, and "
            "included in CSV exports."
        ),
    )
    complexity.add_argument(
        "--complexity-metric",
        default="halstead-effort",
        help=(
            "Navchart metric: SLOC, Halstead vocabulary, Halstead program "
            "length, Halstead volume, Halstead difficulty, or Halstead effort "
            "(common short aliases are accepted)."
        ),
    )
    complexity_scaling = complexity.add_mutually_exclusive_group()
    complexity_scaling.add_argument(
        "--normalize",
        action="store_true",
        help="Divide complexity by the CPP score (CPP = 100%%).",
    )
    complexity_scaling.add_argument(
        "--additive",
        action="store_true",
        help="Subtract the CPP complexity score from every paradigm score.",
    )
    complexity.add_argument(
        "--compare-metric",
        default="sloc",
        help=(
            "Second complexity metric for --chart complexity-comparison, "
            "plotted on the x axis against --complexity-metric on the y axis. "
            "Both are expressed relative to the sequential CPP baseline."
        ),
    )
    complexity.add_argument(
        "--legend-complexity-comparison-coefficients",
        dest="legend_comparison_coefficients",
        action="store_true",
        help=(
            "Box Spearman's rho and Kendall's tau in the corner of the "
            "complexity-comparison chart. Off by default: the coefficients "
            "belong in the running text, where they can be discussed."
        ),
    )
    complexity.add_argument(
        "--log-complexity",
        action="store_true",
        help=(
            "Use logarithmic complexity axes in navchart, combined and "
            "complexity-comparison plots."
        ),
    )
    return parser
