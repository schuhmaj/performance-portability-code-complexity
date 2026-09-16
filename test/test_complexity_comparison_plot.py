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
    assert "Dense\nLines" in texts
    assert "Verbose\nCode" in texts
    plt.close(figure)


def _key(figure: plt.Figure) -> str | None:
    """Return the baseline key of a comparison figure, if it has one."""
    keys = [
        text.get_text()
        for text in figure.axes[0].texts
        if "means" in text.get_text()
    ]
    return keys[0] if keys else None


def test_baseline_key_states_what_hundred_percent_is():
    figure = plot_complexity_comparison(
        _data(),
        SLOC,
        DIFFICULTY,
        "VecAdd",
        show_legends=False,
        baselines=(173.0, 102.88),
    )
    key = _key(figure)
    assert key is not None
    # SLOC is a whole count, the difficulty is not; both are spelled out with
    # the qualifier dropped so the box stays narrow.
    assert "SLOC = 173" in key
    assert "$D$ = 102.88" in key
    assert "Halstead" not in key
    plt.close(figure)


def test_baseline_key_is_absent_without_baselines():
    figure = plot_complexity_comparison(
        _data(), SLOC, DIFFICULTY, "VecAdd", show_legends=False
    )
    assert _key(figure) is None
    plt.close(figure)


def test_baseline_key_clears_the_lower_right_caption():
    figure = plot_complexity_comparison(
        _data(),
        SLOC,
        DIFFICULTY,
        "VecAdd",
        show_legends=False,
        baselines=(173.0, 102.88),
    )
    axis = figure.axes[0]
    figure.canvas.draw()
    key = next(text for text in axis.texts if "means" in text.get_text())
    caption = next(text for text in axis.texts if text.get_text() == "Verbose\nCode")
    key_box = key.get_window_extent(figure.canvas.get_renderer())
    caption_box = caption.get_window_extent(figure.canvas.get_renderer())
    assert key_box.y0 > caption_box.y1
    plt.close(figure)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("Halstead Difficulty [normalized]", "Halstead $D$ [% of sequential C++]"),
        ("Source Lines of Code [normalized]", "SLOC [% of sequential C++]"),
        ("Halstead Difficulty [absolute]", "Halstead $D$ [absolute]"),
        # An unknown metric or scaling mode must survive unchanged.
        ("Cyclomatic Complexity [normalized]", "Cyclomatic Complexity [normalized]"),
        ("Halstead Volume [rescaled]", "Halstead Volume [rescaled]"),
    ],
)
def test_short_metric_label_keeps_unknown_metrics_intact(metric, expected):
    assert short_metric_label(metric) == expected
