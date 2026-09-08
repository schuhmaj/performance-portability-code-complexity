"""Tests for the two-metric code-complexity comparison plot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

from ppbcc.constants import APPLICATION, PROBLEM
from ppbcc.plot.complexity_comparison import plot_complexity_comparison
from ppbcc.plot.styles import format_relative_log_axis, short_metric_label

SLOC = "Source Lines of Code [normalized]"
DIFFICULTY = "Halstead Difficulty [normalized]"


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["VecAdd", "Stdpar", 102.0, 101.0],
            ["VecAdd", "Kokkos", 108.0, 115.0],
            ["VecAdd", "OpenCL", 175.0, 219.0],
            ["VecAdd", "Vulkan[Naive]", 224.0, 204.0],
        ],
        columns=[PROBLEM, APPLICATION, SLOC, DIFFICULTY],
    )


def test_every_paradigm_becomes_one_marker():
    figure = plot_complexity_comparison(
        _data(), SLOC, DIFFICULTY, "VecAdd", show_legends=False
    )
    axis = figure.axes[0]
    # The region shading is a PolyCollection; only the scatter markers count.
    points = [
        tuple(collection.get_offsets()[0])
        for collection in axis.collections
        if isinstance(collection, PathCollection)
    ]
    assert sorted(points) == [(102.0, 101.0), (108.0, 115.0), (175.0, 219.0),
                              (224.0, 204.0)]
    plt.close(figure)


def test_remove_description_strips_bracketed_variant_from_labels():
    figure = plot_complexity_comparison(
        _data(), SLOC, DIFFICULTY, "VecAdd", remove_description=True
    )
    labels = {text.get_text() for text in figure.axes[0].texts}
    assert "Vulkan" in labels
    assert "Vulkan[Naive]" not in labels
    plt.close(figure)


def test_external_legend_drops_the_point_labels():
    """With -l the paradigm colors are keyed by the shared legend instead."""
    figure = plot_complexity_comparison(
        _data(), SLOC, DIFFICULTY, "VecAdd", show_legends=False
    )
    labels = {text.get_text() for text in figure.axes[0].texts}
    assert not labels & {"Stdpar", "Kokkos", "OpenCL", "Vulkan[Naive]"}
    plt.close(figure)


def test_a_single_paradigm_is_rejected():
    with pytest.raises(ValueError, match="at least two paradigms"):
        plot_complexity_comparison(
            _data().head(1), SLOC, DIFFICULTY, "VecAdd", show_legends=False
        )


@pytest.mark.parametrize(
    ("low", "high"),
    [(100.5, 615.0), (110.5, 190.2), (101.2, 218.8), (100.9, 173.4)],
)
def test_narrow_log_ranges_keep_plain_decimal_tick_labels(low, high):
    """A sub-decade range must not be left without any major tick label."""
    figure, axis = plt.subplots()
    axis.set_xscale("log")
    axis.set_xlim(low, high)
    format_relative_log_axis(axis.xaxis)
    figure.canvas.draw()
    labels = [text.get_text() for text in axis.get_xticklabels() if text.get_text()]
    assert len(labels) >= 2
    assert all(label.replace(".", "").isdigit() for label in labels)
    plt.close(figure)


def test_axis_titles_are_short_and_name_the_baseline():
    figure = plot_complexity_comparison(
        _data(), SLOC, DIFFICULTY, "VecAdd", show_legends=False
    )
    axis = figure.axes[0]
    assert axis.get_xlabel() == "SLOC [% of sequential C++]"
    assert axis.get_ylabel() == "Halstead $D$ [% of sequential C++]"
    plt.close(figure)


def test_half_planes_are_captioned_for_the_reader():
    figure = plot_complexity_comparison(
        _data(), SLOC, DIFFICULTY, "VecAdd", show_legends=False
    )
    texts = {text.get_text() for text in figure.axes[0].texts}
    assert "Dense\nVocabulary" in texts
    assert "Verbose\nCode" in texts
    plt.close(figure)


def test_correlation_coefficients_are_opt_in():
    hidden = plot_complexity_comparison(
        _data(), SLOC, DIFFICULTY, "VecAdd", show_legends=False
    )
    assert not [t for t in hidden.axes[0].texts if r"\rho" in t.get_text()]
    plt.close(hidden)

    shown = plot_complexity_comparison(
        _data(), SLOC, DIFFICULTY, "VecAdd", show_legends=False,
        show_coefficients=True,
    )
    assert [t for t in shown.axes[0].texts if r"\rho" in t.get_text()]
    plt.close(shown)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("Halstead Difficulty [normalized]", "Halstead $D$ [% of sequential C++]"),
        ("Source Lines of Code [normalized]", "SLOC [% of sequential C++]"),
        ("Halstead Difficulty [additive]", "Halstead $D$ [added over sequential C++]"),
        # An unknown metric or scaling mode must survive unchanged.
        ("Cyclomatic Complexity [normalized]", "Cyclomatic Complexity [normalized]"),
        ("Halstead Volume [rescaled]", "Halstead Volume [rescaled]"),
    ],
)
def test_short_metric_label_keeps_unknown_metrics_intact(metric, expected):
    assert short_metric_label(metric) == expected
