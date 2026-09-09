"""Scatter plot comparing two code-complexity metrics against each other."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ppbcc.constants import APPLICATION
from ppbcc.plot.styles import (
    SCATTER_MARKER_AREA,
    _application_legend_handles,
    _display_application,
    _framework_colors,
    font_points,
    format_relative_log_axis,
    short_metric_label,
    short_metric_name,
)

#: Region shading: above the identity line the y metric charges a paradigm
#: more than the x metric does, below it the x metric charges more.
ABOVE_LINE_COLOR = "#d62728"
BELOW_LINE_COLOR = "#1f77b4"
#: What each half-plane means, in the reader's terms rather than the metrics'.
ABOVE_LINE_TEXT = "Dense\nLines"
BELOW_LINE_TEXT = "Verbose\nCode"
LABEL_FONT_SIZE = 8.5
#: The chart is printed small beside a shared legend, so the axis labels, the
#: region captions and the markers are all set above the inherited defaults.
AXIS_LABEL_FONT_SCALE = 1.15
REGION_FONT_SCALE = 1.25
MARKER_AREA_SCALE = 1.45
#: Where the baseline key sits, in axes coordinates. Its top edge starts at
#: half height so the box grows downwards into the empty lower-right corner,
#: clearing both the markers and the "Verbose Code" caption below it.
BASELINE_BOX_POSITION = (0.975, 0.5)
BASELINE_FONT_SCALE = 1.0
#: Candidate label offsets in points, tried in order until one is free.
LABEL_OFFSETS = (
    (12, 0, "left"),
    (-12, 0, "right"),
    (12, 13, "left"),
    (-12, 13, "right"),
    (12, -13, "left"),
    (-12, -13, "right"),
    (0, 17, "center"),
    (0, -17, "center"),
    (12, 26, "left"),
    (-12, -26, "right"),
)


def _place_labels(
    axis: plt.Axes,
    points: list[tuple[float, float, str]],
) -> None:
    """Annotate every point, avoiding other labels and the markers.

    Labels are placed in descending y order, each taking the first candidate
    offset whose bounding box overlaps neither an already placed label nor any
    marker. Placement happens in display space, so the axis must already be
    drawn and scaled.

    Args:
        axis: Axis holding the scatter.
        points: Data coordinates and label of every marker.
    """
    positions = {
        label: axis.transData.transform((x, y)) for x, y, label in points
    }
    marker_radius = 10.5
    half_height = 7.0
    boxes: list[tuple[float, float, float, float]] = []
    for x, y, label in sorted(points, key=lambda point: -point[1]):
        point_x, point_y = positions[label]
        width = 6.4 * len(label)
        chosen = LABEL_OFFSETS[0]
        for offset_x, offset_y, alignment in LABEL_OFFSETS:
            shift = width / 2.0
            if alignment == "right":
                shift = -shift
            elif alignment == "center":
                shift = 0.0
            center_x = point_x + offset_x + shift
            box = (
                center_x - width / 2.0,
                point_y + offset_y - half_height,
                center_x + width / 2.0,
                point_y + offset_y + half_height,
            )
            overlaps_label = any(
                not (
                    box[2] < other[0]
                    or box[0] > other[2]
                    or box[3] < other[1]
                    or box[1] > other[3]
                )
                for other in boxes
            )
            overlaps_marker = any(
                box[0] - marker_radius < marker_x < box[2] + marker_radius
                and box[1] - marker_radius < marker_y < box[3] + marker_radius
                for marker_x, marker_y in positions.values()
            )
            if not overlaps_label and not overlaps_marker:
                chosen = (offset_x, offset_y, alignment)
                boxes.append(box)
                break
        else:
            offset_x, offset_y, alignment = chosen
            shift = width / 2.0 if alignment == "left" else -width / 2.0
            center_x = point_x + offset_x + shift
            boxes.append(
                (
                    center_x - width / 2.0,
                    point_y + offset_y - half_height,
                    center_x + width / 2.0,
                    point_y + offset_y + half_height,
                )
            )
        offset_x, offset_y, alignment = chosen
        axis.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=LABEL_FONT_SIZE,
            color="0.15",
            ha=alignment,
            va="center",
            zorder=6,
        )


def _key_symbol(metric: str) -> str:
    """Name a metric as compactly as the baseline key allows.

    The axis titles already carry the full name, so the key drops the
    ``Halstead`` qualifier and keeps the symbol alone. That matters: the key
    sits over the lower-right half-plane, and every character of width brings
    it closer to the markers on the identity line.

    Args:
        metric: Column name from the complexity loader.

    Returns:
        The bare symbol, for example ``"$D$"`` or ``"SLOC"``.
    """
    return short_metric_name(metric).removeprefix("Halstead ")


def _format_baseline(value: float) -> str:
    """Format an absolute complexity value for the baseline key.

    Args:
        value: Unscaled metric value.

    Returns:
        A thousands-separated integer for counts such as SLOC, two decimals for
        derived metrics such as the Halstead difficulty.
    """
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def plot_complexity_comparison(
    data: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    problem_title: str,
    remove_description: bool = False,
    log_axes: bool = True,
    show_legends: bool = True,
    show_coefficients: bool = False,
    baselines: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot two complexity metrics against each other with an identity line.

    Both metrics are relative to the sequential C++ baseline, so they share one
    dimensionless scale and the identity line is meaningful: a paradigm above it
    is charged more by ``y_metric`` than by ``x_metric``, and one below it is
    charged more by ``x_metric``. The two half-planes are captioned with what
    they mean for the source rather than for the metrics.

    Args:
        data: Application labels with both relative metric columns.
        x_metric: Column plotted on the x axis.
        y_metric: Column plotted on the y axis.
        problem_title: Problem name or names used in the title.
        remove_description: Whether to hide bracketed descriptions.
        log_axes: Whether both axes are logarithmic.
        show_legends: Whether the legend and the per-point labels are drawn.
            With an external legend the paradigm colors are keyed there, so the
            labels are dropped: they do not survive the reduction to a narrow
            column anyway.
        show_coefficients: Whether Spearman's rho and Kendall's tau are added to
            the baseline key. Off by default; the numbers belong in the text,
            where they can be given to three decimals and discussed.
        baselines: Absolute sequential C++ values of ``x_metric`` and
            ``y_metric``. Both axes are percentages of these, so the key states
            what 100 % stands for. Omitted when the values are unknown or when
            several problems with different baselines share one chart.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If fewer than two paradigms can be plotted.
    """
    plotted = data.dropna(subset=[x_metric, y_metric])
    if len(plotted) < 2:
        raise ValueError(
            "Comparing complexity metrics needs at least two paradigms with "
            f"values for both {x_metric!r} and {y_metric!r}."
        )
    applications = list(dict.fromkeys(plotted[APPLICATION].astype(str)))
    colors = _framework_colors(applications, remove_description)

    figure, axis = plt.subplots(figsize=(6.6, 6.2))
    low = min(plotted[x_metric].min(), plotted[y_metric].min()) * 0.92
    high = max(plotted[x_metric].max(), plotted[y_metric].max()) * 1.12

    axis.plot(
        [low, high], [low, high], color="0.35", linestyle="--", linewidth=1.3, zorder=1
    )
    axis.fill_between(
        [low, high], [low, high], [high, high], color=ABOVE_LINE_COLOR, alpha=0.05,
        zorder=0,
    )
    axis.fill_between(
        [low, high], [low, low], [low, high], color=BELOW_LINE_COLOR, alpha=0.05,
        zorder=0,
    )
    for _, row in plotted.iterrows():
        application = str(row[APPLICATION])
        axis.scatter(
            row[x_metric],
            row[y_metric],
            color=colors[application],
            marker="o",
            s=SCATTER_MARKER_AREA * MARKER_AREA_SCALE,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    if log_axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        format_relative_log_axis(axis.xaxis)
        format_relative_log_axis(axis.yaxis)
    axis_font_size = font_points("axes.labelsize", AXIS_LABEL_FONT_SCALE)
    axis.set_xlabel(short_metric_label(x_metric), fontsize=axis_font_size)
    axis.set_ylabel(short_metric_label(y_metric), fontsize=axis_font_size)
    axis.grid(True, linestyle="--", alpha=0.45)

    region_font_size = font_points("axes.labelsize", REGION_FONT_SCALE)
    axis.text(
        0.04,
        0.96,
        ABOVE_LINE_TEXT,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=region_font_size,
        color=ABOVE_LINE_COLOR,
        style="italic",
        linespacing=1.1,
        zorder=2,
    )
    axis.text(
        0.96,
        0.04,
        BELOW_LINE_TEXT,
        transform=axis.transAxes,
        va="bottom",
        ha="right",
        fontsize=region_font_size,
        color=BELOW_LINE_COLOR,
        style="italic",
        linespacing=1.1,
        zorder=2,
    )
    key_lines: list[str] = []
    if baselines is not None:
        key_lines = [
            r"$\mathbf{100\,\%}$ means",
            f"{_key_symbol(x_metric)} = {_format_baseline(baselines[0])}",
            f"{_key_symbol(y_metric)} = {_format_baseline(baselines[1])}",
        ]
    if show_coefficients:
        rho = spearmanr(plotted[x_metric], plotted[y_metric]).statistic
        tau = kendalltau(plotted[x_metric], plotted[y_metric]).statistic
        key_lines += [rf"$\rho={rho:.3f}$", rf"$\tau={tau:.3f}$"]
    if key_lines:
        axis.text(
            *BASELINE_BOX_POSITION,
            "\n".join(key_lines),
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=font_points("axes.labelsize", BASELINE_FONT_SCALE),
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.85,
            },
            zorder=2,
        )
    # The panel is narrow in print, so the title is set a little below the
    # inherited title size to keep it inside the figure width.
    figure.suptitle(
        f"{problem_title} Code Complexity",
        fontsize=font_points("axes.titlesize"),
    )

    if show_legends:
        figure.legend(
            handles=_application_legend_handles(
                applications, colors, remove_description
            ),
            title="Paradigm",
            loc="center left",
            bbox_to_anchor=(0.69, 0.5),
            frameon=True,
        )
        figure.subplots_adjust(left=0.13, right=0.67, top=0.92, bottom=0.10)
        figure.canvas.draw()
        _place_labels(
            axis,
            [
                (
                    float(row[x_metric]),
                    float(row[y_metric]),
                    _display_application(str(row[APPLICATION]), remove_description),
                )
                for _, row in plotted.iterrows()
            ],
        )
    else:
        figure.subplots_adjust(left=0.15, right=0.96, top=0.92, bottom=0.11)
    return figure
