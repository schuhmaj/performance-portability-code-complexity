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
from ppbcc.plot.cascade import (
    HARDWARE_CELL_FONT_SIZE,
    PLATFORM_ROW_START_X,
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
