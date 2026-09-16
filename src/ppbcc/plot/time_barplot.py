"""Bar chart of benchmark runtimes per paradigm and platform."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter

from ppbcc.constants import APPLICATION, HARDWARE
from ppbcc.plot.styles import (
    _application_legend_handles,
    _display_application,
    _framework_colors,
    format_relative_log_axis,
)

#: Column of the plotted value: nanoseconds, or FLOP when normalized.
RUNTIME_NS = "Runtime (ns)"
PEAK_FLOP = "Runtime x Peak Performance (FLOP)"
#: Display units for runtimes, largest first, with their size in nanoseconds.
TIME_UNITS = (("s", 1e9), ("ms", 1e6), (r"$\mu$s", 1e3), ("ns", 1.0))
#: Share of one platform group's width filled with bars.
GROUP_FILL = 0.84
#: Below this many decades a runtime axis gets plain decimal ticks, since
#: Matplotlib would otherwise label it with cluttered minor ticks.
DECIMAL_TICK_DECADES = 1.5


def _time_unit(values_ns: pd.Series) -> tuple[str, float]:
    """Pick the largest unit in which the median runtime is at least one.

    Args:
        values_ns: Plotted runtimes in nanoseconds.

    Returns:
        The unit label and its size in nanoseconds.
    """
    median = float(values_ns.median())
    for unit, factor in TIME_UNITS:
        if median >= factor:
            return unit, factor
    return TIME_UNITS[-1]


def plot_time_barplot(
    runtimes: pd.DataFrame,
    problem_title: str,
    time_label: str,
    selected_size: float,
    precision: int,
    normalized: bool = False,
    remove_description: bool = False,
    show_legends: bool = True,
) -> plt.Figure:
    """Plot one runtime bar per paradigm, grouped by platform.

    With a single platform the bars stand side by side and are labeled with
    their paradigm. With several platforms every platform is one labeled group
    holding a slot per paradigm, in the same order in every group, so a
    paradigm keeps its position; a slot stays empty where the paradigm has no
    result on that platform. The runtime axis is logarithmic, because runtimes
    of one problem differ by orders of magnitude between paradigms.

    Args:
        runtimes: One row per application and hardware with ``Runtime (ns)``,
            and ``Runtime x Peak Performance (FLOP)`` when ``normalized``.
        problem_title: Benchmark problem shown in the title.
        time_label: Name of the plotted runtime, e.g. ``Kernel Time``.
        selected_size: Problem size the runtimes were measured at.
        precision: Floating-point precision in bits.
        normalized: Whether the runtime is multiplied by the platform's peak
            performance, i.e. plotted as the FLOPs the platform could have
            executed in that time.
        remove_description: Whether bracketed descriptions are hidden.
        show_legends: Whether the paradigm legend is drawn into the figure.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If there is nothing to plot.
    """
    if runtimes.empty:
        raise ValueError("No runtimes available for the time bar plot.")

    applications = sorted(
        runtimes[APPLICATION].astype(str).unique(),
        key=lambda application: (application.casefold(), application),
    )
    platforms = sorted(runtimes[HARDWARE].astype(str).unique())
    colors = _framework_colors(applications, remove_description)
    lookup = runtimes.set_index([APPLICATION, HARDWARE])

    if normalized:
        column, factor = PEAK_FLOP, 1.0
        y_label = f"{time_label} × Peak Performance [FLOP]"
    else:
        unit, factor = _time_unit(runtimes[RUNTIME_NS])
        column = RUNTIME_NS
        y_label = f"{time_label} [{unit}]"

    single = len(platforms) == 1
    if single:
        width = max(8.0, 2.5 + 0.62 * len(applications))
    else:
        width = max(9.0, 2.5 + len(platforms) * (0.16 * len(applications) + 0.35))
    figure, axis = plt.subplots(figsize=(width, 6.0))

    bar_width = 0.7 if single else GROUP_FILL / len(applications)
    plotted: list[float] = []
    for group, platform in enumerate(platforms):
        for slot, application in enumerate(applications):
            if (application, platform) not in lookup.index:
                continue
            value = float(lookup.loc[(application, platform), column]) / factor
            plotted.append(value)
            if single:
                position = slot
            else:
                position = group + (slot - (len(applications) - 1) / 2) * bar_width
            axis.bar(
                position,
                value,
                width=bar_width,
                color=colors[application],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

    # Bars on a log axis have no natural base, so the axis starts a little
    # below the shortest bar to keep every bar visible as a bar.
    low, high = min(plotted), max(plotted)
    axis.set_yscale("log")
    axis.set_ylim(low / 1.6, high * 1.6)
    if not normalized and math.log10(high / low) < DECIMAL_TICK_DECADES:
        format_relative_log_axis(axis.yaxis)
    else:
        axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_ylabel(y_label)
    axis.grid(True, axis="y", which="major", linestyle="--", alpha=0.45, zorder=0)
    if single:
        axis.set_xticks(range(len(applications)))
        axis.set_xticklabels(
            [
                _display_application(application, remove_description)
                for application in applications
            ],
            rotation=35,
            ha="right",
        )
        axis.set_xlim(-0.6, len(applications) - 0.4)
        axis.set_xlabel("Paradigm")
    else:
        axis.set_xticks(range(len(platforms)))
        axis.set_xticklabels(platforms)
        axis.set_xlim(-0.5, len(platforms) - 0.5)
        for boundary in range(1, len(platforms)):
            axis.axvline(boundary - 0.5, color="0.75", linewidth=0.8, zorder=1)
        axis.set_xlabel("Platform")
    axis.grid(False, axis="x")

    size_text = (
        f"{selected_size:,.0f}"
        if float(selected_size).is_integer()
        else f"{selected_size:g}"
    )
    subtitle = f"Problem size: {size_text}, FP{precision}"
    if single:
        subtitle += f", {platforms[0]}"
    axis.set_title(f"{problem_title} {time_label}\n{subtitle}")

    if show_legends:
        axis.legend(
            handles=_application_legend_handles(
                applications, colors, remove_description
            ),
            title="Paradigm",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
        )
    figure.tight_layout()
    return figure
