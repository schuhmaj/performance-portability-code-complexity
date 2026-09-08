"""Tests for Cascade and combined performance-portability plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ppbcc.constants import (
    APPLICATION,
    APPLICATION_EFFICIENCY,
    HARDWARE,
    PERFORMANCE_PORTABILITY,
    PROBLEM,
)
from ppbcc.constants import PROBLEM_SIZE
from ppbcc.plot.cascade import (
    HARDWARE_CELL_FONT_SIZE,
    PLATFORM_ROW_START_X,
    _format_problem_size,
    _problem_size_labels,
    plot_cascade,
)


def test_platform_rows_have_visible_start_marker_without_changing_endpoint():
    efficiency = pd.DataFrame(
        [
            ["NBody", "Kokkos", "AMD MI250", 1.0],
            ["NBody", "Kokkos", "NVIDIA H100", 0.8],
        ],
        columns=[PROBLEM, APPLICATION, HARDWARE, APPLICATION_EFFICIENCY],
    )
    portability = pd.DataFrame(
        [["NBody", "Kokkos", 0.9]],
        columns=[PROBLEM, APPLICATION, PERFORMANCE_PORTABILITY],
    )

    figure = plot_cascade(
        efficiency,
        portability,
        "NBody",
        show_legends=False,
    )
    platform_axis = figure.axes[1]
    row_line, start_line = platform_axis.lines

    assert list(row_line.get_xdata()) == [0.5, 2.5]
    assert row_line.get_marker() == "o"
    assert row_line.get_markevery() is None
    assert list(start_line.get_xdata()) == [PLATFORM_ROW_START_X, 0.5]
    assert start_line.get_marker() == "o"
    assert start_line.get_markevery() == [0]
    assert start_line.get_clip_on() is False
    assert {text.get_fontsize() for text in platform_axis.texts} == {
        HARDWARE_CELL_FONT_SIZE
    }
    assert HARDWARE_CELL_FONT_SIZE > 8
    plt.close(figure)


def test_problem_size_labels_stay_short_and_do_not_fake_precision():
    """A power of two must not be rounded into a decimal-looking label."""
    assert _format_problem_size(16384) == "16384"
    assert _format_problem_size(100000) == "100k"
    assert _format_problem_size(1e6) == "1M"
    # Too long to set upright without growing the figure, so abbreviated.
    assert _format_problem_size(3145728) == "3.15M"
    assert _format_problem_size(2780) == "2780"


def test_combined_scaling_panel_is_a_heatmap_over_the_benchmark_sizes():
    efficiency = pd.DataFrame(
        [
            ["NBody", "Kokkos", "AMD MI250", 1.0],
            ["NBody", "RAJA", "AMD MI250", 0.5],
        ],
        columns=[PROBLEM, APPLICATION, HARDWARE, APPLICATION_EFFICIENCY],
    )
    portability = pd.DataFrame(
        [["NBody", "Kokkos", 0.9], ["NBody", "RAJA", 0.4]],
        columns=[PROBLEM, APPLICATION, PERFORMANCE_PORTABILITY],
    )
    navchart = pd.DataFrame(
        [["NBody", "Kokkos", 0.9, 120.0], ["NBody", "RAJA", 0.4, 150.0]],
        columns=[PROBLEM, APPLICATION, PERFORMANCE_PORTABILITY, "Metric"],
    )
    scaling = pd.DataFrame(
        [
            ["NBody", "Kokkos", 100.0, 0.8],
            ["NBody", "Kokkos", 1000.0, 1.0],
            ["NBody", "RAJA", 100.0, 0.3],
            ["NBody", "RAJA", 1000.0, 0.5],
        ],
        columns=[PROBLEM, APPLICATION, PROBLEM_SIZE, PERFORMANCE_PORTABILITY],
    )

    figure = plot_cascade(
        efficiency,
        portability,
        "NBody",
        navchart_data=navchart,
        complexity_metric="Metric",
        scaling_data=scaling,
        show_legends=False,
    )
    scaling_axis = figure.axes[3]
    assert len(scaling_axis.images) == 1
    values = scaling_axis.images[0].get_array()
    # Rows follow the descending-PP order of the platform-ranking panel.
    assert values.tolist() == [[0.8, 1.0], [0.3, 0.5]]
    assert [t.get_text() for t in scaling_axis.get_xticklabels()] == [
        "$10^{2}$",
        "$10^{3}$",
    ]
    plt.close(figure)


def test_power_sweeps_get_exponent_labels_and_others_keep_decimals():
    """Exponents buy back the width the heatmap needs at print scale."""
    matmul = [float(2**exponent) for exponent in range(5, 15)]
    labels, powers = _problem_size_labels(matmul)
    assert powers
    assert labels[0] == "$2^{5}$" and labels[-1] == "$2^{14}$"

    decimal_sweep = [10.0**exponent for exponent in range(1, 6)]
    labels, powers = _problem_size_labels(decimal_sweep)
    assert powers
    assert labels == [f"$10^{{{exponent}}}$" for exponent in range(1, 6)]

    # The polyhedral meshes are neither, and must not be forced into a power.
    meshes = [2780.0, 12796.0, 14744.0, 32040.0, 196608.0, 255932.0, 3145728.0]
    labels, powers = _problem_size_labels(meshes)
    assert not powers
    assert labels[0] == "2780" and labels[-1] == "3.15M"


def test_a_single_size_is_not_dressed_up_as_a_power():
    """One column carries no sweep to read, so plain notation is honest."""
    labels, powers = _problem_size_labels([1024.0])
    assert not powers
    assert labels == ["1024"]
