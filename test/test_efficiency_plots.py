"""Tests for P3 application-efficiency plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba

from ppbcc.constants import APPLICATION, APPLICATION_EFFICIENCY, HARDWARE
from ppbcc.plot.heatmap import (
    MISSING_LABEL,
    plot_efficiency_boxplot,
    plot_efficiency_double_heatmap,
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


def test_heatmap_marks_never_benchmarked_cells_instead_of_zero():
    efficiency = _efficiency_frame()
    efficiency.loc[3, APPLICATION_EFFICIENCY] = 0.0
    figure = plot_efficiency_heatmap(efficiency, "NBody")
    axis = figure.axes[0]
    labels = [text.get_text() for text in axis.texts]

    assert labels.count(MISSING_LABEL) == 1
    assert "0.00" not in labels
    assert len(labels) == 4
    assert any(
        to_rgba(patch.get_facecolor()) == to_rgba("black") for patch in axis.patches
    )
    plt.close(figure)


def test_double_heatmap_splits_every_cell_into_two_sizes():
    first = _efficiency_frame()
    second = _efficiency_frame()
    second.loc[0, APPLICATION_EFFICIENCY] = 0.0
    figure = plot_efficiency_double_heatmap(
        first, second, "NBody", 100.0, 3145728.0, remove_description=True
    )
    axis = figure.axes[0]
    labels = [text.get_text() for text in axis.texts]

    assert len(axis.patches) == 8
    assert len(labels) == 8
    assert labels.count(MISSING_LABEL) == 1
    # The missing Cuda result pulls its mean over both sizes below Kokkos'.
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        "Kokkos",
        "Cuda",
    ]
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "AMD MI250",
        "NVIDIA H100",
    ]
    assert "100" in axis.get_title() and "3145728" in axis.get_title()
    plt.close(figure)


def test_heatmaps_sort_paradigms_alphabetically_on_request():
    efficiency = pd.DataFrame(
        [
            ["raja", "AMD MI250", 1.0],
            ["Kokkos", "AMD MI250", 0.2],
            ["Cuda", "AMD MI250", 0.5],
        ],
        columns=[APPLICATION, HARDWARE, APPLICATION_EFFICIENCY],
    )
    expected = ["Cuda", "Kokkos", "raja"]

    single = plot_efficiency_heatmap(efficiency, "NBody", sort_alphabetically=True)
    double = plot_efficiency_double_heatmap(
        efficiency, efficiency, "NBody", 100.0, 200.0, sort_alphabetically=True
    )
    default = plot_efficiency_heatmap(efficiency, "NBody")

    for figure in (single, double):
        assert [
            label.get_text() for label in figure.axes[0].get_xticklabels()
        ] == expected
    assert [label.get_text() for label in default.axes[0].get_xticklabels()] == [
        "raja",
        "Cuda",
        "Kokkos",
    ]
    for figure in (single, double, default):
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
