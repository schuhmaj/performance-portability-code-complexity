"""Cascade and combined performance-portability plots."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import EngFormatter

from ppbcc.constants import (
    APPLICATION,
    APPLICATION_EFFICIENCY,
    HARDWARE,
    PERFORMANCE_PORTABILITY,
    PROBLEM,
    PROBLEM_SIZE,
)
from ppbcc.performance_portability.selection import (
    AVERAGE_SIZE,
    BEST_SIZE,
    WORST_SIZE,
)
from ppbcc.plot.styles import (
    PLOT_MARKER_SIZE,
    PP_SYMBOL,
    SCATTER_MARKER_AREA,
    _application_legend_handles,
    _framework_colors,
    _hardware_colors,
    _problem_legend_handles,
    _problem_markers,
)

PLATFORM_ROW_START_X = 0.35


def plot_cascade(
    efficiency: pd.DataFrame,
    portability: pd.DataFrame,
    problem_title: str,
    remove_description: bool = False,
    navchart_data: pd.DataFrame | None = None,
    complexity_metric: str | None = None,
    scaling_data: pd.DataFrame | None = None,
    log_complexity: bool = False,
    log_size: bool = False,
    selected_size: float | str | None = None,
    show_legends: bool = True,
) -> plt.Figure:
    """Create a Cascade Plot following the P3 Analysis Library layout.

    Args:
        efficiency: Application efficiency by problem, application, and hardware.
        portability: Performance portability by problem and application.
        problem_title: Problem name or names used in the title.
        remove_description: Whether to hide bracketed legend descriptions.
        navchart_data: Optional complexity/PP data for combined mode.
        complexity_metric: Complexity column used in combined mode.
        scaling_data: Optional per-size PP data for combined mode.
        log_complexity: Whether the complexity axis is logarithmic.
        log_size: Whether the problem-size scaling axis is logarithmic.
        selected_size: Optional benchmark size or size-summary mode for the two
            upper panels.
        show_legends: Whether legends are embedded in the plot.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If combined-mode inputs are incomplete.
    """
    combined = navchart_data is not None
    if combined and (complexity_metric is None or scaling_data is None):
        raise ValueError(
            "Combined plotting requires complexity and per-size scaling data."
        )

    series = portability.sort_values(
        PERFORMANCE_PORTABILITY, ascending=False
    ).reset_index(drop=True)
    applications = list(dict.fromkeys(series[APPLICATION]))
    problems = list(dict.fromkeys(series[PROBLEM]))
    platforms = sorted(efficiency[HARDWARE].unique())
    colors = _framework_colors(applications, remove_description)
    markers = _problem_markers(problems)
    platform_colors = _hardware_colors(platforms)
    platform_labels = {
        platform: chr(ord("A") + index) if index < 26 else str(index + 1)
        for index, platform in enumerate(platforms)
    }

    series_count = len(series)
    figure_width = max(11.0, 7.5 + 0.25 * series_count)
    figure_height = max(7.5, 5.6 + 0.27 * series_count)
    figure = plt.figure(figsize=(figure_width, figure_height))
    right_ratio = 4.0 if combined else max(1.7, 0.22 * series_count)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[4.5, max(1.5, 0.34 * series_count)],
        width_ratios=[5.5, right_ratio],
        hspace=0.3 if combined else 0.03,
        wspace=0.08,
    )
    efficiency_axis = figure.add_subplot(grid[0, 0])
    platform_axis = figure.add_subplot(grid[1, 0], sharex=efficiency_axis)
    portability_axis = figure.add_subplot(grid[0, 1], sharey=efficiency_axis)
    scaling_axis = (
        figure.add_subplot(grid[1, 1], sharey=efficiency_axis) if combined else None
    )

    for _, item in series.iterrows():
        problem = str(item[PROBLEM])
        application = str(item[APPLICATION])
        rows = efficiency.loc[
            (efficiency[PROBLEM] == problem)
            & (efficiency[APPLICATION] == application)
            & efficiency[APPLICATION_EFFICIENCY].gt(0.0)
        ].sort_values(APPLICATION_EFFICIENCY, ascending=False)
        ranks = np.arange(1, len(rows) + 1)
        efficiency_axis.plot(
            ranks,
            rows[APPLICATION_EFFICIENCY],
            color=colors[application],
            marker=markers[problem],
            linewidth=1.6,
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=1.5,
        )

    efficiency_axis.set_ylabel("Application Efficiency")
    efficiency_axis.set_ylim(0.0, 1.05)
    efficiency_axis.set_xlim(0.5, len(platforms) + 0.5)
    efficiency_axis.set_xticks(np.arange(1, len(platforms) + 1))
    efficiency_axis.tick_params(axis="x", labelbottom=False)
    efficiency_axis.grid(True, linestyle="--", alpha=0.45)
    plot_name = (
        f"{problem_title} {PP_SYMBOL} - Code Complexity Plot"
        if combined
        else f"{problem_title} {PP_SYMBOL} Cascade Plot"
    )
    figure.suptitle(plot_name, y=0.98)

    row_height = 1.0
    reversed_series = series.iloc[::-1].reset_index(drop=True)
    for row_index, item in reversed_series.iterrows():
        problem = str(item[PROBLEM])
        application = str(item[APPLICATION])
        rows = efficiency.loc[
            (efficiency[PROBLEM] == problem)
            & (efficiency[APPLICATION] == application)
            & efficiency[APPLICATION_EFFICIENCY].gt(0.0)
        ].sort_values(APPLICATION_EFFICIENCY, ascending=False)
        y = row_index * row_height
        platform_axis.plot(
            [0.5, len(platforms) + 0.5],
            [y + 0.5, y + 0.5],
            color=colors[application],
            marker=markers[problem],
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=1.5,
            linewidth=1.2,
            zorder=1,
        )
        # Draw the row's start just outside the table so its paradigm color is
        # visible even when platform rectangles cover the entire row. Keep the
        # existing row line unchanged so its right endpoint retains its current
        # appearance for applications that do not support every platform.
        platform_axis.plot(
            [PLATFORM_ROW_START_X, 0.5],
            [y + 0.5, y + 0.5],
            color=colors[application],
            marker=markers[problem],
            markevery=[0],
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=1.5,
            linewidth=1.2,
            clip_on=False,
            zorder=3,
        )
        for rank, platform in enumerate(rows[HARDWARE], start=1):
            platform_axis.add_patch(
                Rectangle(
                    (rank - 0.5, y),
                    1.0,
                    row_height,
                    facecolor=platform_colors[platform],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=2,
                )
            )
            platform_axis.text(
                rank,
                y + 0.5,
                platform_labels[platform],
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )
    platform_axis.set_xlabel("Platform rank (highest efficiency first)")
    platform_axis.set_ylim(0.0, max(series_count, 1))
    platform_axis.set_yticks([])
    platform_axis.grid(False)

    if combined:
        assert navchart_data is not None
        assert complexity_metric is not None
        for _, item in navchart_data.iterrows():
            problem = str(item[PROBLEM])
            application = str(item[APPLICATION])
            portability_axis.scatter(
                item[complexity_metric],
                item[PERFORMANCE_PORTABILITY],
                color=colors[application],
                marker=markers[problem],
                s=SCATTER_MARKER_AREA,
                linewidth=1.5,
                zorder=3,
            )
        portability_axis.set_xlabel(complexity_metric)
        if log_complexity:
            portability_axis.set_xscale("log")
        else:
            portability_axis.xaxis.set_major_formatter(EngFormatter())
        portability_axis.grid(True, linestyle="--", alpha=0.45)
    else:
        positions = np.arange(series_count)
        portability_axis.bar(
            positions,
            series[PERFORMANCE_PORTABILITY],
            color="white",
            edgecolor=[colors[name] for name in series[APPLICATION]],
            linewidth=1.5,
        )
        for position, (_, item) in zip(positions, series.iterrows()):
            portability_axis.scatter(
                position,
                item[PERFORMANCE_PORTABILITY],
                color=colors[str(item[APPLICATION])],
                marker=markers[str(item[PROBLEM])],
                s=SCATTER_MARKER_AREA,
                linewidth=1.5,
                zorder=3,
            )
        portability_axis.set_xticks([])
        portability_axis.grid(True, axis="y", linestyle="--", alpha=0.45)
    portability_axis.yaxis.tick_right()
    portability_axis.yaxis.set_label_position("right")
    portability_axis.set_ylabel(PP_SYMBOL)

    if combined:
        assert scaling_axis is not None
        assert scaling_data is not None
        for (problem, application), rows in scaling_data.groupby(
            [PROBLEM, APPLICATION], sort=False
        ):
            rows = rows.sort_values(PROBLEM_SIZE)
            scaling_axis.plot(
                rows[PROBLEM_SIZE],
                rows[PERFORMANCE_PORTABILITY],
                color=colors[str(application)],
                marker=markers[str(problem)],
                linewidth=1.5,
                markersize=PLOT_MARKER_SIZE,
                markeredgewidth=1.5,
            )
        scaling_axis.set_xlabel("Problem Size")
        if log_size:
            scaling_axis.set_xscale("log")
        scaling_axis.yaxis.tick_right()
        scaling_axis.yaxis.set_label_position("right")
        scaling_axis.set_ylabel(PP_SYMBOL)
        scaling_axis.grid(True, linestyle="--", alpha=0.45)

    if show_legends:
        application_handles = _application_legend_handles(
            applications, colors, remove_description
        )
        app_legend = figure.legend(
            handles=application_handles,
            title="Paradigm",
            loc="center left",
            bbox_to_anchor=(0.72, 0.55),
            frameon=True,
        )
        figure.add_artist(app_legend)

        if len(problems) > 1:
            problem_legend = figure.legend(
                handles=_problem_legend_handles(problems, markers),
                title="Problem",
                loc="lower left",
                bbox_to_anchor=(0.72, 0.08),
                frameon=True,
            )
            figure.add_artist(problem_legend)

    platform_handles = [
        Patch(
            facecolor=platform_colors[name],
            edgecolor="black",
            label=f"{platform_labels[name]}: {name}",
        )
        for name in platforms
    ]
    hardware_columns = min(2, max(len(platforms), 1))
    hardware_rows = (len(platforms) + hardware_columns - 1) // hardware_columns
    bottom_margin = 0.23 + 0.045 * max(hardware_rows - 1, 0)
    if show_legends:
        figure.legend(
            handles=platform_handles,
            title="Device",
            loc="lower center",
            bbox_to_anchor=(0.35, 0.01),
            ncol=hardware_columns,
            frameon=True,
        )
        figure.subplots_adjust(right=0.68, bottom=bottom_margin, top=0.9)
    else:
        figure.subplots_adjust(right=0.97, bottom=0.1, top=0.9)

    if selected_size is not None:
        upper_left = efficiency_axis.get_position()
        upper_right = portability_axis.get_position()
        if selected_size == AVERAGE_SIZE:
            size_note = f"{PP_SYMBOL} and $e_A$ averaged over benchmark sizes"
        elif selected_size in {BEST_SIZE, WORST_SIZE}:
            size_note = (
                f"{PP_SYMBOL} and $e_A$ use the {selected_size} value over "
                "benchmark sizes"
            )
        else:
            size_note = (
                f"{PP_SYMBOL} and $e_A$ plotted for benchmark size = "
                f"{float(selected_size):g}"
            )
        figure.text(
            (upper_left.x0 + upper_right.x1) / 2.0,
            max(upper_left.y1, upper_right.y1) + 0.012,
            size_note,
            ha="center",
            va="bottom",
            fontsize=matplotlib.rcParams["axes.titlesize"],
        )
    return figure
