"""Application-efficiency heatmap and boxplot visualizations."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ppbcc.constants import (
    APPLICATION,
    APPLICATION_EFFICIENCY,
    HARDWARE,
)
from ppbcc.performance_portability.selection import ALL_SIZE
from ppbcc.plot.styles import _display_application, _framework_colors


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
    return f"Problem size: {selected_size:g}"


def plot_efficiency_heatmap(
    efficiency: pd.DataFrame,
    problem_title: str,
    remove_description: bool = False,
    selected_size: float | str = ALL_SIZE,
) -> plt.Figure:
    """Plot application efficiency by platform and paradigm as a heatmap.

    Duplicate application/platform observations are averaged. The color scale
    is fixed to the valid efficiency range so separate plots remain comparable.

    Args:
        efficiency: Rows containing application, hardware, and efficiency.
        problem_title: Benchmark problem shown in the title.
        remove_description: Whether to hide bracketed implementation details.
        selected_size: Numeric problem size or size-selection literal.

    Returns:
        Matplotlib figure containing the heatmap.

    Raises:
        ValueError: If no efficiency observations are available.
    """
    if efficiency.empty:
        raise ValueError("No application-efficiency data available for heatmap.")

    plot_data = efficiency.copy()
    if remove_description:
        plot_data[APPLICATION] = (
            plot_data[APPLICATION]
            .astype(str)
            .map(lambda application: _display_application(application, True))
        )
    application_order = (
        plot_data.groupby(APPLICATION)[APPLICATION_EFFICIENCY]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    platforms = sorted(plot_data[HARDWARE].astype(str).unique())
    matrix = plot_data.pivot_table(
        index=HARDWARE,
        columns=APPLICATION,
        values=APPLICATION_EFFICIENCY,
        aggfunc="mean",
    ).reindex(index=platforms, columns=application_order)

    width = max(8.0, 3.0 + 1.05 * len(application_order))
    height = max(4.8, 2.5 + 0.62 * len(platforms))
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Application Efficiency", "shrink": 0.9},
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
        application_order: Optional paradigm order, normally descending PP.

    Returns:
        Matplotlib figure containing paradigm-colored boxplots and observations.

    Raises:
        ValueError: If no efficiency observations are available.
    """
    if efficiency.empty:
        raise ValueError("No application-efficiency data available for boxplot.")

    plot_data = efficiency.copy()
    if remove_description:
        plot_data[APPLICATION] = (
            plot_data[APPLICATION]
            .astype(str)
            .map(lambda application: _display_application(application, True))
        )
    present = set(plot_data[APPLICATION].astype(str))
    requested_order = () if application_order is None else application_order
    requested_labels = [
        _display_application(str(item), remove_description) for item in requested_order
    ]
    order = list(dict.fromkeys(item for item in requested_labels if item in present))
    order.extend(sorted(present - set(order), key=str.casefold))
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
