"""Tests for shared plot styles and standalone legends."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ppbcc.plot.styles import create_separate_legend


def test_separate_vertical_legend_combines_sections_in_one_column():
    figure = create_separate_legend(
        ["Cuda", "Kokkos", "RAJA", "OpenMP"],
        ["NBody", "BabelStream"],
        ["AMD MI250", "NVIDIA H100", "Intel PVC"],
        vertical=True,
    )

    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert legend._ncols == 1
    assert [text.get_text() for text in legend.get_texts()] == [
        "Paradigm",
        "Cuda",
        "Kokkos",
        "OpenMP",
        "RAJA",
        "Problem",
        "NBody",
        "BabelStream",
        "Device",
        "A: AMD MI250",
        "B: NVIDIA H100",
        "C: Intel PVC",
    ]
    heading_indexes = {0, 5, 8}
    assert all(
        text.get_fontweight() == ("bold" if index in heading_indexes else "normal")
        for index, text in enumerate(legend.get_texts())
    )
    assert figure.get_figwidth() == 6.5

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer).transformed(
        figure.transFigure.inverted()
    )
    assert legend_box.y0 >= 0.0
    assert legend_box.y1 <= 1.0
    plt.close(figure)


def test_classic_separate_legend_keeps_independent_four_column_sections():
    figure = create_separate_legend(
        ["Cuda", "Kokkos"],
        ["NBody", "BabelStream"],
        ["AMD MI250", "NVIDIA H100"],
    )

    assert [legend.get_title().get_text() for legend in figure.legends] == [
        "Paradigm",
        "Problem",
        "Device",
    ]
    assert all(legend._ncols == 4 for legend in figure.legends)
    plt.close(figure)
