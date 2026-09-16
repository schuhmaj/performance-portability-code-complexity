"""Roofline plotting for profiled GPU kernels."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import EngFormatter

from ppbcc.constants import (
    ARITHMETIC_INTENSITY,
    MEMORY_LEVEL,
    PARADIGM,
    PEAK_BANDWIDTH,
    PEAK_PERFORMANCE,
    PERFORMANCE,
    PRECISION,
)
from ppbcc.plot.styles import (
    SCATTER_MARKER_AREA,
    _application_legend_handles,
    _framework_colors,
    font_points,
)

#: Factor applied to every font of the roofline chart; Matplotlib's defaults read
#: too small once the figure is scaled down to a column in a paper.
FONT_SCALE = 1.2


def _ceilings(data: pd.DataFrame) -> tuple[float, float]:
    """Derive the two roofs from the measured per-kernel peaks.

    Args:
        data: Rows to plot, carrying ``Peak Performance`` and
            ``Peak Bandwidth``.

    Returns:
        Compute ceiling in FLOP/s and memory ceiling in byte/s.

    Raises:
        ValueError: If neither ceiling could be measured.
    """
    peak_flops = float(pd.to_numeric(data[PEAK_PERFORMANCE], errors="coerce").max())
    peak_bandwidth = float(pd.to_numeric(data[PEAK_BANDWIDTH], errors="coerce").max())
    if not np.isfinite(peak_flops) or not np.isfinite(peak_bandwidth):
        raise ValueError(
            "No ceilings available, so no roofline can be drawn. Nsight Compute "
            "measures them from its peak_sustained metrics; LIKWID has none, so "
            "pass --peak-performance and --peak-bandwidth there."
        )
    return peak_flops, peak_bandwidth


def plot_roofline(
    data: pd.DataFrame,
    hardware: str = "",
    problem_title: str = "",
    show_legend: bool = True,
    subtitle: str = "",
) -> plt.Figure:
    """Create a roofline chart from profiled kernels.

    The roof is measured rather than taken from a datasheet: every
    ``.peak_sustained`` metric Nsight Compute reports is scaled with the clock
    the corresponding unit ran at, and the highest value observed across all
    profiled kernels becomes the ceiling.

    Args:
        data: One row per plotted point, as produced by
            :func:`~ppbcc.profiling.reports.aggregate_kernels`.
        hardware: Hardware label used in the title.
        problem_title: Benchmark problem name used in the title.
        show_legend: Whether a paradigm legend is embedded in the plot.
        subtitle: Optional second title line, e.g. how kernels were aggregated.

    Returns:
        The Matplotlib figure.
    """
    data = data.copy()
    data[ARITHMETIC_INTENSITY] = pd.to_numeric(
        data[ARITHMETIC_INTENSITY], errors="coerce"
    )
    data[PERFORMANCE] = pd.to_numeric(data[PERFORMANCE], errors="coerce")
    finite = data[
        np.isfinite(data[ARITHMETIC_INTENSITY]) & (data[ARITHMETIC_INTENSITY] > 0)
    ]
    finite = finite[np.isfinite(finite[PERFORMANCE]) & (finite[PERFORMANCE] > 0)]
    if finite.empty:
        raise ValueError("No kernel with a positive intensity and performance.")

    peak_flops, peak_bandwidth = _ceilings(finite)
    ridge = peak_flops / peak_bandwidth

    paradigms = list(dict.fromkeys(finite[PARADIGM].astype(str)))
    colors = _framework_colors(paradigms)
    small = font_points("legend.fontsize", FONT_SCALE)

    figure, axis = plt.subplots(figsize=(10.0, 7.0))

    # The roof spans the measured intensities, but always shows the ridge point.
    low = min(finite[ARITHMETIC_INTENSITY].min(), ridge) / 4.0
    high = max(finite[ARITHMETIC_INTENSITY].max(), ridge) * 4.0
    intensity = np.logspace(np.log10(low), np.log10(high), 512)
    roof = np.minimum(peak_flops, peak_bandwidth * intensity)
    axis.plot(intensity, roof, color="black", linewidth=2.0, zorder=2)
    axis.fill_between(intensity, roof, low, color="black", alpha=0.03, zorder=0)

    level = str(finite[MEMORY_LEVEL].iloc[0])
    precision = str(finite[PRECISION].iloc[0])
    axis.axhline(peak_flops, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    axis.annotate(
        f"{peak_flops / 1e12:.1f} TFLOP/s ({precision})",
        xy=(high, peak_flops),
        xytext=(-6, 8),
        textcoords="offset points",
        ha="right",
        fontsize=small,
    )
    axis.annotate(
        f"{peak_bandwidth / 1e9:.0f} GB/s ({level})",
        xy=(low * 1.15, peak_bandwidth * low * 1.15),
        xytext=(6, -14),
        textcoords="offset points",
        rotation=38,
        rotation_mode="anchor",
        fontsize=small,
    )

    for _, row in finite.iterrows():
        axis.scatter(
            row[ARITHMETIC_INTENSITY],
            row[PERFORMANCE],
            color=colors[str(row[PARADIGM])],
            marker="o",
            s=SCATTER_MARKER_AREA,
            edgecolor="black",
            linewidth=0.5,
            # Implementations of the same algorithm land close together, so
            # overlapping points have to stay visible.
            alpha=0.85,
            zorder=3,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    label_size = font_points("axes.labelsize", FONT_SCALE)
    axis.set_xlabel(f"Arithmetic Intensity [FLOP / {level} byte]", fontsize=label_size)
    axis.set_ylabel("Performance [FLOP/s]", fontsize=label_size)
    # The unit is in the axis label already; without it the vertical tick labels
    # stay short enough not to run into the titles.
    axis.yaxis.set_major_formatter(EngFormatter(places=0))
    axis.tick_params(axis="x", labelsize=font_points("xtick.labelsize", FONT_SCALE))
    # Horizontal y tick labels are the widest element of the chart, so they run
    # along the axis instead.
    axis.tick_params(
        axis="y",
        labelsize=font_points("ytick.labelsize", FONT_SCALE),
        labelrotation=90,
    )
    axis.set_xlim(low, high)
    axis.set_ylim(finite[PERFORMANCE].min() / 4.0, peak_flops * 2.0)
    # The tick labels exist only once the limits are final.
    for label in axis.get_yticklabels():
        label.set_verticalalignment("center")
    axis.grid(True, which="both", linestyle="--", alpha=0.35)

    # The ridge point separates the memory- from the compute-bound regime.
    axis.axvline(ridge, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    axis.annotate(
        f"ridge point {ridge:.0f} FLOP/byte",
        xy=(ridge, finite[PERFORMANCE].min() / 4.0),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        fontsize=small,
        alpha=0.7,
    )

    title = " ".join(filter(None, [problem_title, "Roofline"]))
    title_artist = figure.suptitle(
        f"{title} - {hardware}" if hardware else title,
        fontsize=font_points("figure.titlesize", FONT_SCALE),
    )
    if subtitle:
        axis.set_title(subtitle, fontsize=small, color="0.35")
    if show_legend:
        axis.legend(
            handles=_application_legend_handles(paradigms, colors, False),
            title="Paradigm",
            loc="lower right",
            frameon=True,
            fontsize=small,
            title_fontsize=small,
        )
    figure.tight_layout()
    # The subtitle is centred on the axes, the figure title on the whole figure,
    # which the y-axis label pushes off centre; align the title with the axes.
    position = axis.get_position()
    title_artist.set_x((position.x0 + position.x1) / 2.0)
    return figure
