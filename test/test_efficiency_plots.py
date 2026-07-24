"""Tests for P3 application-efficiency plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba

from ppbcc.constants import APPLICATION, APPLICATION_EFFICIENCY, HARDWARE
from ppbcc.plot.heatmap import (
    plot_efficiency_boxplot,
    plot_efficiency_heatmap,
)
from ppbcc.plot.styles import _framework_colors


def _efficiency_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Cuda[Naive]", "AMD MI250", 1.0],
            ["Cuda[Naive]", "NVIDIA H100", 0.9],
            ["Kokkos", "AMD MI250", 0.7],
            ["Kokkos", "NVIDIA H100", 0.8],
        ],
        columns=[APPLICATION, HARDWARE, APPLICATION_EFFICIENCY],
    )


def test_heatmap_uses_paradigms_as_columns_and_platforms_as_rows():
    figure = plot_efficiency_heatmap(
        _efficiency_frame(), "NBody", remove_description=True
    )
    axis = figure.axes[0]

    assert axis.get_xlabel() == "Paradigm"
    assert axis.get_ylabel() == "Platform"
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        "Cuda",
        "Kokkos",
    ]
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "AMD MI250",
        "NVIDIA H100",
    ]
    assert len(axis.texts) == 4
    assert figure.axes[1].get_ylabel() == "Application Efficiency"
    plt.close(figure)


def test_boxplot_sorts_paradigms_and_uses_framework_colors():
    order = ["Kokkos", "Cuda[Naive]"]
    figure = plot_efficiency_boxplot(
        _efficiency_frame(),
        "NBody",
        remove_description=True,
        application_order=order,
    )
    axis = figure.axes[0]
    expected_order = ["Cuda", "Kokkos"]
    expected = _framework_colors(expected_order)

    assert axis.get_xlabel() == "Paradigm"
    assert axis.get_ylabel() == "Application Efficiency"
    assert [label.get_text() for label in axis.get_xticklabels()] == expected_order
    assert to_rgba(axis.patches[0].get_facecolor()) == to_rgba(
        expected[expected_order[0]]
    )
    assert to_rgba(axis.patches[1].get_facecolor()) == to_rgba(
        expected[expected_order[1]]
    )
    plt.close(figure)
