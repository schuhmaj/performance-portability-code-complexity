"""Cascade and combined performance-portability plots."""

from __future__ import annotations

import math

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
    font_points,
    format_relative_log_axis,
    short_metric_label,
    use_inward_ticks,
)

PLATFORM_ROW_START_X = 0.35
HARDWARE_CELL_FONT_SIZE = 11
SCALING_COLORMAP = "viridis"
SCALING_CELL_TEXT_THRESHOLD = 0.55


def _format_problem_size(size: float) -> str:
    """Abbreviate a benchmark size for a categorical tick label.

    Exact multiples are abbreviated, so a decimal size such as 100000 becomes
    ``100k`` while a power of two such as 16384 stays itself rather than
    turning into a misleading ``16.4k``. Sizes that would otherwise print more
    than five digits are abbreviated anyway: the labels are set upright, so a
    long one grows the figure's bounding box rather than colliding with its
    neighbours.

    Args:
        size: Benchmark problem size.

    Returns:
        A short label such as ``100k``, ``16384`` or ``3.15M``.
    """
    value = float(size)
    exact = f"{value:.0f}" if value == int(value) else f"{value:g}"
    for divisor, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) < divisor:
            continue
        if value % divisor == 0.0:
            return f"{value / divisor:g}{suffix}"
        if len(exact) > 5:
            return f"{value / divisor:.3g}{suffix}"
        break
    return exact


def _exponents(sizes: list[float], base: int) -> list[int] | None:
    """Return the exponents of ``sizes`` in ``base``, or None if inexact.

    Args:
        sizes: Benchmark problem sizes.
        base: Base to test, 2 or 10.

    Returns:
        One exponent per size, or ``None`` unless every size is an exact power.
    """
    exponents: list[int] = []
    for size in sizes:
        if size <= 0.0:
            return None
        exponent = round(math.log(size, base))
        if float(base) ** exponent != float(size):
            return None
        exponents.append(exponent)
    return exponents


def _problem_size_labels(sizes: list[float]) -> tuple[list[str], bool]:
    """Label the benchmark sizes, using powers when they all are powers.

    Matrix multiplication sweeps powers of two and the other problems sweep
    powers of ten, so ``2^14`` and ``10^8`` say the same as ``16384`` and
    ``100M`` in a third of the width — which buys back the font size the
    heatmap needs at print scale. A sweep that is not a clean power sequence,
    such as the polyhedral meshes, keeps decimal labels.

    Args:
        sizes: Benchmark problem sizes, ascending.

    Returns:
        The labels and whether power notation was used.
    """
    # Exponents are compact because they make a *sweep* regular; for a lone
    # size they only obscure, so a single column keeps its plain value.
    if len(sizes) >= 2:
        for base in (10, 2):
            exponents = _exponents(sizes, base)
            if exponents is not None and len(set(exponents)) == len(exponents):
                return [f"${base}^{{{exponent}}}$" for exponent in exponents], True
    return [_format_problem_size(size) for size in sizes], False


def _draw_scaling_heatmap(
    figure: plt.Figure,
    axis: plt.Axes,
    scaling_data: pd.DataFrame,
    series: pd.DataFrame,
) -> None:
    """Draw performance portability over problem size as a heatmap.

    One row per plotted implementation and one column per benchmark size. The
    rows keep the descending-PP order of the platform-ranking panel to the
    left, so the two lower panels line up row for row and the paradigm labels
    need not be repeated. This replaces a line plot in which fourteen series
    overlapped to the point of being unreadable.

    Args:
        figure: Figure that owns the axis, used for the color bar.
        axis: Axis to draw into.
        scaling_data: Per-size performance portability.
        series: Plotted implementations, ordered by descending PP.
    """
    sizes = sorted(scaling_data[PROBLEM_SIZE].unique())
    size_index = {size: column for column, size in enumerate(sizes)}
    values = np.full((len(series), len(sizes)), np.nan)
    for row, (_, item) in enumerate(series.iterrows()):
        rows = scaling_data.loc[
            (scaling_data[PROBLEM] == item[PROBLEM])
            & (scaling_data[APPLICATION] == item[APPLICATION])
        ]
        for _, entry in rows.iterrows():
            values[row, size_index[entry[PROBLEM_SIZE]]] = entry[
                PERFORMANCE_PORTABILITY
            ]

    image = axis.imshow(
        values,
        aspect="auto",
        cmap=SCALING_COLORMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    # The combined figure is printed at well under half its rendered width, so
    # the cell values need to be set generously to survive the reduction. The
    # column pitch binds before the row height does: fourteen rows leave ~17 pt
    # each, while ten columns leave ~28 pt for a three-character number.
    cell_font_size = max(8.0, min(13.0, 95.0 / max(len(sizes), 1)))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                continue
            axis.text(
                column,
                row,
                f"{value * 100.0:.0f}",
                ha="center",
                va="center",
                fontsize=cell_font_size,
                color=(
                    "white" if value < SCALING_CELL_TEXT_THRESHOLD else "black"
                ),
            )

    # Upright labels never collide horizontally however long they are, so the
    # size that fits is set by the column pitch. Power notation is short enough
    # to carry the same size as the cell values; decimal labels are longer and
    # would grow the figure's bounding box, so they stay smaller.
    size_labels, uses_powers = _problem_size_labels(sizes)
    tick_font_size = (
        max(9.0, min(13.0, 95.0 / max(len(sizes), 1)))
        if uses_powers
        else max(7.0, min(10.0, 70.0 / max(len(sizes), 1)))
    )
    axis.set_xticks(np.arange(len(sizes)))
    axis.set_xticklabels(
        size_labels,
        fontsize=tick_font_size,
        rotation=90,
        ha="center",
    )
    axis.set_yticks([])
    axis.set_xlabel("Problem Size")
    axis.tick_params(length=0)
    axis.grid(False)

    # An inset color bar is expressed in axes coordinates, so it follows the
    # panel through the later subplots_adjust; a figure-level one would not.
    # Its PP scale reads as a continuation of the PP axis of the panel above,
    # so it carries that panel's tick and label sizes rather than the
    # heatmap's own.
    color_bar = figure.colorbar(image, cax=axis.inset_axes([1.03, 0.0, 0.04, 1.0]))
    color_bar.set_label(PP_SYMBOL, fontsize=font_points("axes.labelsize"))
    color_bar.outline.set_linewidth(0.8)
    color_bar.ax.tick_params(labelsize=font_points("ytick.labelsize"))
    # A color bar is a right-hand axis too, so it needs the same treatment as
    # the PP axis above it or its labels read as negative numbers.
    use_inward_ticks(color_bar.ax)


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
        log_size: Deprecated and ignored. The scaling panel is a categorical
            heatmap over benchmark sizes, which has no continuous axis to
            scale logarithmically.
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
    # The scaling panel is a categorical heatmap over benchmark sizes, so it
    # shares neither the efficiency axis' limits nor its ticks.
    scaling_axis = figure.add_subplot(grid[1, 1]) if combined else None

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
                fontsize=HARDWARE_CELL_FONT_SIZE,
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
        portability_axis.set_xlabel(short_metric_label(complexity_metric))
        if log_complexity:
            portability_axis.set_xscale("log")
            format_relative_log_axis(portability_axis.xaxis)
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
    use_inward_ticks(portability_axis)

    if combined:
        assert scaling_axis is not None
        assert scaling_data is not None
        _draw_scaling_heatmap(figure, scaling_axis, scaling_data, series)

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
        # The combined chart's heatmap carries an inset color bar past its
        # right edge, which needs a little more margin than a plain cascade.
        figure.subplots_adjust(
            right=0.92 if combined else 0.97, bottom=0.1, top=0.9
        )

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
