"""Roofline plotting for profiled GPU kernels."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import EngFormatter

from ppbcc.constants import (
    AGGREGATED_KERNEL,
    ARITHMETIC_INTENSITY,
    KERNEL,
    MEMORY_LEVEL,
    PARADIGM,
    PEAK_BANDWIDTH,
    PEAK_PERFORMANCE,
    PERFORMANCE,
    PRECISION,
    REGION,
)
from ppbcc.plot.styles import (
    SCATTER_MARKER_AREA,
    _application_legend_handles,
    _framework_colors,
)


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
    label_points: bool = False,
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
        label_points: Whether every point is annotated with its paradigm. The
            legend already carries that information, so this is only useful when
            the points are far enough apart.
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
        fontsize="small",
    )
    axis.annotate(
        f"{peak_bandwidth / 1e9:.0f} GB/s ({level})",
        xy=(low * 2.0, peak_bandwidth * low * 2.0),
        xytext=(6, -14),
        textcoords="offset points",
        rotation=38,
        rotation_mode="anchor",
        fontsize="small",
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
        if label_points:
            # The paradigm alone is ambiguous whenever one implementation
            # contributes more than one point: the polyhedral binaries have an
            # "init" and an "evaluate" region, and the CUDA matrix
            # multiplication links cuBLAS next to its own kernel.
            label = str(row[PARADIGM])
            region = str(row.get(REGION, "") or "")
            if region:
                label = f"{label}: {region}"
            elif row.get(KERNEL, AGGREGATED_KERNEL) != AGGREGATED_KERNEL:
                label = f"{label}: {row[KERNEL]}"
            axis.annotate(
                label,
                xy=(row[ARITHMETIC_INTENSITY], row[PERFORMANCE]),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize="x-small",
            )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(f"Arithmetic Intensity [FLOP / {level} byte]")
    axis.set_ylabel("Performance [FLOP/s]")
    axis.yaxis.set_major_formatter(EngFormatter(unit="FLOP/s", places=0))
    axis.set_xlim(low, high)
    axis.set_ylim(finite[PERFORMANCE].min() / 4.0, peak_flops * 2.0)
    axis.grid(True, which="both", linestyle="--", alpha=0.35)

    # The ridge point separates the memory- from the compute-bound regime.
    axis.axvline(ridge, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    axis.annotate(
        f"ridge point {ridge:.0f} FLOP/byte",
        xy=(ridge, finite[PERFORMANCE].min() / 4.0),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        fontsize="small",
        alpha=0.7,
    )

    title = " ".join(filter(None, [problem_title, "Roofline"]))
    figure.suptitle(f"{title} - {hardware}" if hardware else title)
    if subtitle:
        axis.set_title(subtitle, fontsize="small", color="0.35")
    if show_legend:
        axis.legend(
            handles=_application_legend_handles(paradigms, colors, False),
            title="Paradigm",
            loc="lower right",
            frameon=True,
            fontsize="small",
        )
    figure.tight_layout()
    return figure
