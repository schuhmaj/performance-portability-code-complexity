"""Application-efficiency heatmap and boxplot visualizations."""

from __future__ import annotations

import math
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon, Rectangle
from matplotlib.textpath import TextPath

from ppbcc.constants import (
    APPLICATION,
    APPLICATION_EFFICIENCY,
    HARDWARE,
)
from ppbcc.performance_portability.selection import ALL_SIZE
from ppbcc.plot.styles import _display_application, _framework_colors, font_points

#: Colormap shared by both efficiency heatmaps.
HEATMAP_COLORMAP = "viridis"
#: Fill of a cell whose paradigm was never benchmarked on the platform.
MISSING_COLOR = "black"
#: Annotation of a cell whose paradigm was never benchmarked on the platform.
MISSING_LABEL = "-"
#: Rotation of the double heatmap's paradigm labels, in degrees.
DOUBLE_HEATMAP_LABEL_ANGLE = 35
#: Inches of clear space between two neighbouring paradigm labels.
DOUBLE_HEATMAP_COLUMN_PADDING = 0.12
#: Line height of a paradigm label, as a multiple of its font size.
DOUBLE_HEATMAP_LINE_SPACING = 1.3
#: Narrowest double-heatmap column, for problems with short paradigm names.
DOUBLE_HEATMAP_MIN_COLUMN_WIDTH = 1.5
#: Inches of figure height per platform row of the double heatmap.
DOUBLE_HEATMAP_ROW_HEIGHT = 0.85
#: Inches of figure width taken by the axis labels and the color bar.
DOUBLE_HEATMAP_SIDE_CHROME = 3.6
#: Inches of figure height taken by the title and the axis label.
DOUBLE_HEATMAP_VERTICAL_CHROME = 2.2
#: Share of a column one cell value may occupy. It sets the column pitch, so
#: a smaller share is a wider figure: at one text width that is a shorter
#: chart whose values still print at about 5.5 pt.
DOUBLE_HEATMAP_ANNOTATION_SHARE = 0.40


def _text_width(text: str, size: float) -> float:
    """Measure a string's width in inches at a font size, without rendering.

    Args:
        text: String to measure.
        size: Font size in points.

    Returns:
        The width the string occupies, in inches.
    """
    path = TextPath((0.0, 0.0), text, size=size, prop=FontProperties(size=size))
    return float(path.get_extents().width) / 72.0


def _format_size(size: float) -> str:
    """Print a problem size without losing digits to scientific notation.

    Args:
        size: Numeric benchmark problem size.

    Returns:
        ``3145728`` rather than ``3.14573e+06`` for integral sizes.
    """
    value = float(size)
    return f"{value:.0f}" if value.is_integer() else f"{value:g}"


def _size_description(selected_size: float | str) -> str:
    """Return a concise subtitle for a chart's size selection.

    Args:
        selected_size: Numeric problem size or a size-selection literal.

    Returns:
        Human-readable size-selection text.
    """
    if selected_size == ALL_SIZE:
        return "All problem sizes"
    if isinstance(selected_size, str):
        return f"Problem-size summary: {selected_size}"
    return f"Problem size: {_format_size(selected_size)}"


def _missing_mask(matrix: pd.DataFrame) -> np.ndarray:
    """Mark the cells of paradigms that were never benchmarked on a platform.

    ``calculate_metrics`` fills an absent application/platform pair with an
    efficiency of exactly zero, while a measured result is the ratio of two
    positive runtimes and therefore always positive. Averaging variants or
    workloads keeps that property: the mean is zero only if nothing at all was
    measured. A zero or undefined cell is therefore a missing result, not a
    very slow one.

    Args:
        matrix: Application efficiency, one cell per platform and paradigm.

    Returns:
        Boolean array, ``True`` where no result exists.
    """
    values = matrix.to_numpy(dtype=float)
    return np.isnan(values) | (values <= 0.0)


def _prepare_efficiency(
    efficiency: pd.DataFrame, remove_description: bool
) -> pd.DataFrame:
    """Copy efficiency rows and merge implementation variants if requested.

    Args:
        efficiency: Rows containing application, hardware, and efficiency.
        remove_description: Whether to hide bracketed implementation details.

    Returns:
        The rows with display-ready application labels.
    """
    plot_data = efficiency.copy()
    if remove_description:
        plot_data[APPLICATION] = (
            plot_data[APPLICATION]
            .astype(str)
            .map(lambda application: _display_application(application, True))
        )
    return plot_data


def _application_order(
    plot_data: pd.DataFrame, sort_alphabetically: bool
) -> list[str]:
    """Order the paradigm columns of a heatmap.

    Args:
        plot_data: Rows containing application and efficiency.
        sort_alphabetically: Whether to sort by name (case-insensitive)
            instead of by descending mean efficiency.

    Returns:
        Application labels in column order.
    """
    if sort_alphabetically:
        present = set(plot_data[APPLICATION].astype(str))
        return sorted(present, key=lambda item: (item.casefold(), item))
    return (
        plot_data.groupby(APPLICATION)[APPLICATION_EFFICIENCY]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def _efficiency_matrix(
    plot_data: pd.DataFrame, platforms: list[str], applications: list[str]
) -> pd.DataFrame:
    """Pivot efficiency rows into a platform-by-paradigm matrix.

    Args:
        plot_data: Rows containing application, hardware, and efficiency.
        platforms: Row order.
        applications: Column order.

    Returns:
        Mean efficiency per cell, ``NaN`` where the rows hold no pair.
    """
    return plot_data.pivot_table(
        index=HARDWARE,
        columns=APPLICATION,
        values=APPLICATION_EFFICIENCY,
        aggfunc="mean",
    ).reindex(index=platforms, columns=applications)


def _text_color(color: tuple[float, float, float, float]) -> str:
    """Pick black or white annotation text for a cell color.

    Args:
        color: RGBA fill of the cell.

    Returns:
        ``black`` on light fills, ``white`` on dark ones.
    """
    red, green, blue, _ = color
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.408 else "white"


def plot_efficiency_heatmap(
    efficiency: pd.DataFrame,
    problem_title: str,
    remove_description: bool = False,
    selected_size: float | str = ALL_SIZE,
    sort_alphabetically: bool = False,
) -> plt.Figure:
    """Plot application efficiency by platform and paradigm as a heatmap.

    Duplicate application/platform observations are averaged. The color scale
    is fixed to the valid efficiency range so separate plots remain comparable.
    A paradigm that was never benchmarked on a platform is drawn as a black
    cell with a dash rather than as an efficiency of zero.

    Args:
        efficiency: Rows containing application, hardware, and efficiency.
        problem_title: Benchmark problem shown in the title.
        remove_description: Whether to hide bracketed implementation details.
        selected_size: Numeric problem size or size-selection literal.
        sort_alphabetically: Whether to order paradigms by name instead of by
            descending mean efficiency.

    Returns:
        Matplotlib figure containing the heatmap.

    Raises:
        ValueError: If no efficiency observations are available.
    """
    if efficiency.empty:
        raise ValueError("No application-efficiency data available for heatmap.")

    plot_data = _prepare_efficiency(efficiency, remove_description)
    application_order = _application_order(plot_data, sort_alphabetically)
    platforms = sorted(plot_data[HARDWARE].astype(str).unique())
    matrix = _efficiency_matrix(plot_data, platforms, application_order)
    missing = _missing_mask(matrix)

    width = max(8.0, 3.0 + 1.05 * len(application_order))
    height = max(4.8, 2.5 + 0.62 * len(platforms))
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        matrix,
        ax=axis,
        mask=missing,
        cmap=HEATMAP_COLORMAP,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Application Efficiency", "shrink": 0.9},
    )
    # Masked cells are left empty by seaborn; a black box with a dash keeps a
    # missing result from reading as an efficiency of zero.
    annotation_size = axis.texts[0].get_fontsize() if axis.texts else None
    for row, column in zip(*np.nonzero(missing)):
        axis.add_patch(
            Rectangle(
                (column, row),
                1.0,
                1.0,
                facecolor=MISSING_COLOR,
                edgecolor="white",
                linewidth=0.8,
            )
        )
        axis.text(
            column + 0.5,
            row + 0.5,
            MISSING_LABEL,
            ha="center",
            va="center",
            color="white",
            fontsize=annotation_size,
        )
    axis.set_xlabel("Paradigm")
    axis.set_ylabel("Platform")
    axis.set_title(
        f"{problem_title} Application Efficiency Heatmap\n"
        f"{_size_description(selected_size)}"
    )
    axis.set_xticklabels(
        [
            _display_application(str(application), remove_description)
            for application in application_order
        ],
        rotation=35,
        ha="right",
    )
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    figure.tight_layout()
    return figure


def plot_efficiency_double_heatmap(
    first_efficiency: pd.DataFrame,
    second_efficiency: pd.DataFrame,
    problem_title: str,
    first_size: float,
    second_size: float,
    remove_description: bool = False,
    sort_alphabetically: bool = False,
) -> plt.Figure:
    """Compare application efficiency at two problem sizes in one heatmap.

    Every platform/paradigm cell is split along its diagonal: the upper-left
    triangle shows the first size, the lower-right triangle the second. Both
    halves share one color scale fixed to ``[0, 1]``. A half whose paradigm was
    never benchmarked on the platform at that size is black with a dash.

    Args:
        first_efficiency: Efficiency rows at the first size.
        second_efficiency: Efficiency rows at the second size.
        problem_title: Benchmark problem shown in the title.
        first_size: Problem size of the upper-left triangles.
        second_size: Problem size of the lower-right triangles.
        remove_description: Whether to hide bracketed implementation details.
        sort_alphabetically: Whether to order paradigms by name instead of by
            descending mean efficiency over both sizes.

    Returns:
        Matplotlib figure containing the split heatmap.

    Raises:
        ValueError: If either size has no efficiency observations.
    """
    if first_efficiency.empty or second_efficiency.empty:
        raise ValueError(
            "No application-efficiency data available for double-heatmap."
        )

    first_data = _prepare_efficiency(first_efficiency, remove_description)
    second_data = _prepare_efficiency(second_efficiency, remove_description)
    combined = pd.concat([first_data, second_data], ignore_index=True)
    application_order = _application_order(combined, sort_alphabetically)
    platforms = sorted(combined[HARDWARE].astype(str).unique())
    halves = [
        _efficiency_matrix(data, platforms, application_order)
        for data in (first_data, second_data)
    ]

    paradigm_labels = [
        _display_application(str(application), remove_description)
        for application in application_order
    ]

    # A page gives this chart width but no height, so every size is derived
    # from the width one column needs and the height is kept to what the rows
    # and the label band actually use.
    tick_size = font_points("xtick.labelsize", 1.55)
    label_size = font_points("axes.labelsize", 1.65)
    title_size = font_points("axes.titlesize", 1.60)

    # The cell values are what the chart is read for, so the column pitch is
    # sized for them: a value takes a fixed share of its half-cell, wide enough
    # to read and narrow enough to stay off the cell border and the diagonal.
    font_size = font_points("font.size", 1.4)
    angle = math.radians(DOUBLE_HEATMAP_LABEL_ANGLE)
    # Rotated labels are parallel lines one column pitch apart, so what keeps
    # them legible is the perpendicular gap, pitch * sin(angle) — a shallow
    # angle needs a wider pitch, but only in proportion to the line height and
    # never to the label length, which merely runs alongside its neighbour.
    line_height = DOUBLE_HEATMAP_LINE_SPACING * tick_size / 72.0
    column_width = max(
        DOUBLE_HEATMAP_MIN_COLUMN_WIDTH,
        _text_width("0.00", font_size) / DOUBLE_HEATMAP_ANNOTATION_SHARE,
        line_height / math.sin(angle) + DOUBLE_HEATMAP_COLUMN_PADDING,
    )
    # The band the labels occupy is as deep as the longest one reaching down
    # at the label angle.
    longest_label = max(_text_width(label, tick_size) for label in paradigm_labels)
    width = DOUBLE_HEATMAP_SIDE_CHROME + column_width * len(application_order)
    height = (
        DOUBLE_HEATMAP_VERTICAL_CHROME
        + longest_label * math.sin(angle)
        + DOUBLE_HEATMAP_ROW_HEIGHT * len(platforms)
    )
    figure, axis = plt.subplots(figsize=(width, height))
    colormap = plt.get_cmap(HEATMAP_COLORMAP)
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Triangle corners in cell units, with rows growing downwards, and where
    # each half's annotation sits: pulled from the centroid towards the corner
    # so the two numbers stay clear of the diagonal.
    triangles = (
        (((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), (0.3, 0.27)),
        (((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), (0.7, 0.73)),
    )
    for half, (corners, (text_x, text_y)) in zip(halves, triangles):
        values = half.to_numpy(dtype=float)
        missing = _missing_mask(half)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                if missing[row, column]:
                    fill = to_rgba(MISSING_COLOR)
                    label = MISSING_LABEL
                else:
                    fill = colormap(norm(values[row, column]))
                    label = f"{values[row, column]:.2f}"
                axis.add_patch(
                    Polygon(
                        [(column + x, row + y) for x, y in corners],
                        closed=True,
                        facecolor=fill,
                        edgecolor=fill,
                        linewidth=0.3,
                    )
                )
                axis.text(
                    column + text_x,
                    row + text_y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color=_text_color(fill),
                )

    # Cell borders and diagonals are drawn over the fills in one pass each.
    rows, columns = len(platforms), len(application_order)
    for row in range(rows):
        for column in range(columns):
            axis.plot(
                [column, column + 1.0],
                [row + 1.0, row],
                color="white",
                linewidth=0.6,
            )
    for row in range(rows + 1):
        axis.axhline(row, color="white", linewidth=2.0)
    for column in range(columns + 1):
        axis.axvline(column, color="white", linewidth=2.0)

    axis.set_xlim(0.0, columns)
    axis.set_ylim(rows, 0.0)
    axis.set_aspect("auto")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    axis.set_xticks(np.arange(columns) + 0.5)
    # Anchored rotation keeps each label's end under its own column rather
    # than letting it drift left as the labels grow.
    axis.set_xticklabels(
        [
            _display_application(str(application), remove_description)
            for application in application_order
        ],
        rotation=DOUBLE_HEATMAP_LABEL_ANGLE,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_size,
    )
    axis.set_yticks(np.arange(rows) + 0.5)
    axis.set_yticklabels(platforms, rotation=0, fontsize=tick_size)
    axis.set_xlabel("Paradigm", fontsize=label_size)
    axis.set_ylabel("Platform", fontsize=label_size)
    axis.set_title(
        f"{problem_title} Application Efficiency Heatmap\n"
        f"\u25e4 Problem size: {_format_size(first_size)}    "
        f"\u25e2 Problem size: {_format_size(second_size)}",
        fontsize=title_size,
    )
    color_bar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=colormap), ax=axis, shrink=0.9
    )
    color_bar.set_label("Application Efficiency", fontsize=label_size)
    color_bar.ax.tick_params(labelsize=tick_size)
    color_bar.outline.set_visible(False)
    figure.tight_layout()
    return figure


def plot_efficiency_boxplot(
    efficiency: pd.DataFrame,
    problem_title: str,
    remove_description: bool = False,
    selected_size: float | str = ALL_SIZE,
    application_order: Iterable[str] | None = None,
) -> plt.Figure:
    """Plot efficiency distributions across platforms for each paradigm.

    Args:
        efficiency: Per-size, per-platform application-efficiency observations.
        problem_title: Benchmark problem shown in the title.
        remove_description: Whether to hide bracketed implementation details.
        selected_size: ``all`` or an exact numeric problem size.
        application_order: Retained for API compatibility. Boxplots always
            sort paradigms alphabetically.

    Returns:
        Matplotlib figure containing paradigm-colored boxplots and observations.

    Raises:
        ValueError: If no efficiency observations are available.
    """
    if efficiency.empty:
        raise ValueError("No application-efficiency data available for boxplot.")

    del application_order
    plot_data = _prepare_efficiency(efficiency, remove_description)
    present = set(plot_data[APPLICATION].astype(str))
    order = sorted(present, key=lambda item: (item.casefold(), item))
    colors = _framework_colors(order)

    width = max(9.0, 3.0 + 1.15 * len(order))
    figure, axis = plt.subplots(figsize=(width, 6.5))
    sns.boxplot(
        data=plot_data,
        x=APPLICATION,
        y=APPLICATION_EFFICIENCY,
        hue=APPLICATION,
        order=order,
        hue_order=order,
        palette=colors,
        dodge=False,
        legend=False,
        saturation=1.0,
        width=0.62,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        ax=axis,
    )
    sns.stripplot(
        data=plot_data,
        x=APPLICATION,
        y=APPLICATION_EFFICIENCY,
        hue=APPLICATION,
        order=order,
        hue_order=order,
        palette=colors,
        dodge=False,
        legend=False,
        alpha=0.65,
        size=5,
        edgecolor="white",
        linewidth=0.5,
        ax=axis,
    )
    axis.set_xlabel("Paradigm")
    axis.set_ylabel("Application Efficiency")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(
        f"{problem_title} Application Efficiency by Paradigm\n"
        f"{_size_description(selected_size)}"
    )
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(
        [
            _display_application(application, remove_description)
            for application in order
        ],
        rotation=35,
        ha="right",
    )
    axis.grid(True, axis="y", linestyle="--", alpha=0.4)
    figure.tight_layout()
    return figure
