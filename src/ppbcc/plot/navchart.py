"""Navchart plotting for complexity and performance portability."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import EngFormatter

from ppbcc.constants import APPLICATION, PERFORMANCE_PORTABILITY, PROBLEM
from ppbcc.plot.styles import (
    PP_SYMBOL,
    SCATTER_MARKER_AREA,
    _application_legend_handles,
    _framework_colors,
    _problem_legend_handles,
    _problem_markers,
)


def plot_navchart(
    data: pd.DataFrame,
    metric: str,
    problem_title: str,
    remove_description: bool = False,
    log_complexity: bool = False,
    show_legends: bool = True,
) -> plt.Figure:
    """Create a Navchart of complexity and performance portability.

    Args:
        data: Matched problem, application, PP, and complexity data.
        metric: Complexity metric column and x-axis label.
        problem_title: Problem name or names used in the title.
        remove_description: Whether to hide bracketed legend descriptions.
        log_complexity: Whether the complexity axis is logarithmic.
        show_legends: Whether legends are embedded in the plot.

    Returns:
        The Matplotlib figure.
    """
    applications = list(dict.fromkeys(data[APPLICATION]))
    problems = list(dict.fromkeys(data[PROBLEM]))
    colors = _framework_colors(applications, remove_description)
    markers = _problem_markers(problems)
    figure, axis = plt.subplots(figsize=(9.5, 7.2))
    for _, row in data.iterrows():
        application = str(row[APPLICATION])
        problem = str(row[PROBLEM])
        marker = markers[problem]
        marker_options = (
            {"linewidth": 1.2}
            if marker in {"x", "+", "1", "2", "3", "4", "|", "_"}
            else {"edgecolor": "black", "linewidth": 0.5}
        )
        axis.scatter(
            row[metric],
            row[PERFORMANCE_PORTABILITY],
            color=colors[application],
            marker=marker,
            s=SCATTER_MARKER_AREA,
            zorder=3,
            **marker_options,
        )

    axis.set_xlabel(metric)
    axis.set_ylabel(PP_SYMBOL)
    axis.set_ylim(0.0, 1.05)
    figure.suptitle(f"{problem_title} {PP_SYMBOL} - Code Complexity")
    if log_complexity:
        axis.set_xscale("log")
    else:
        axis.xaxis.set_major_formatter(EngFormatter())
    axis.grid(True, linestyle="--", alpha=0.45)
    if show_legends:
        application_legend = figure.legend(
            handles=_application_legend_handles(
                applications, colors, remove_description
            ),
            title="Paradigm",
            loc="center left",
            bbox_to_anchor=(0.72, 0.55),
            frameon=True,
        )
        figure.add_artist(application_legend)
        if len(problems) > 1:
            figure.legend(
                handles=_problem_legend_handles(problems, markers),
                title="Problem",
                loc="lower center",
                bbox_to_anchor=(0.34, 0.01),
                ncol=len(problems),
                frameon=True,
            )
        figure.subplots_adjust(right=0.68, bottom=0.28 if len(problems) > 1 else 0.11)
    else:
        figure.subplots_adjust(right=0.96, bottom=0.11)
    return figure
