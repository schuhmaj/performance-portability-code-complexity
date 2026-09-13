"""Tests for application-efficiency and portability analysis."""

import pandas as pd
import pytest

from ppbcc.constants import (
    APPLICATION,
    APPLICATION_EFFICIENCY,
    BENCHMARK_PROBLEM,
    DESCRIPTION,
    HARDWARE,
    PARADIGM,
    PERFORMANCE_PORTABILITY,
    PRECISION,
    PROBLEM_SIZE,
    TIME_UNIT,
    WALL_CLOCK_TIME,
)
from ppbcc.performance_portability.complexity import (
    load_complexity_baseline,
    load_complexity_data,
)
from ppbcc.performance_portability.metrics import calculate_metrics
from ppbcc.performance_portability.options import build_parser
from ppbcc.performance_portability.selection import (
    ALL_SIZE,
    parse_problem_size,
    select_problem_rows,
)


def _benchmark_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["NBody", "Cuda", "", 64, "NVIDIA", 100, 10.0, "ms"],
            ["NBody", "Kokkos", "", 64, "NVIDIA", 100, 20.0, "ms"],
            ["NBody", "Cuda", "", 64, "AMD", 100, 20.0, "ms"],
        ],
        columns=[
            BENCHMARK_PROBLEM,
            PARADIGM,
            DESCRIPTION,
            PRECISION,
            HARDWARE,
            PROBLEM_SIZE,
            WALL_CLOCK_TIME,
            TIME_UNIT,
        ],
    )


def test_missing_hardware_gives_zero_default_portability():
    efficiency, portability = calculate_metrics(_benchmark_frame(), False)

    kokkos_efficiency = efficiency.loc[efficiency[APPLICATION].eq("Kokkos")].set_index(
        HARDWARE
    )[APPLICATION_EFFICIENCY]
    assert kokkos_efficiency["NVIDIA"] == pytest.approx(0.5)
    assert kokkos_efficiency["AMD"] == 0.0
    kokkos_pp = portability.loc[
        portability[APPLICATION].eq("Kokkos"), PERFORMANCE_PORTABILITY
    ].item()
    assert kokkos_pp == 0.0


def test_non_zero_portability_omits_unsupported_hardware():
    _, portability = calculate_metrics(_benchmark_frame(), False, non_zero_pp=True)

    kokkos_pp = portability.loc[
        portability[APPLICATION].eq("Kokkos"), PERFORMANCE_PORTABILITY
    ].item()
    assert kokkos_pp == pytest.approx(0.5)


def test_problem_selection_supports_unique_partial_name():
    selected, name, description_is_workload = select_problem_rows(
        _benchmark_frame(), "body", None, None, None, 64
    )

    assert name == "NBody"
    assert len(selected) == 3
    assert not description_is_workload


def test_size_all_is_supported_and_is_the_default():
    assert parse_problem_size("ALL") == ALL_SIZE
    args = build_parser().parse_args(["NBody", "results.csv"])
    assert args.size == ALL_SIZE


@pytest.mark.parametrize("flag", ["-H", "--hardware"])
def test_hardware_option_supports_short_and_long_flags(flag):
    args = build_parser().parse_args(
        ["NBody", "results.csv", "--chart", "boxplot", flag, "AMD MI250"]
    )

    assert args.hardware == "AMD MI250"


def test_vertical_legend_option_is_parsed():
    args = build_parser().parse_args(
        ["NBody", "results.csv", "-l", "--legend--vertical"]
    )

    assert args.legend is True
    assert args.legend_vertical is True


@pytest.mark.parametrize("literal", ["mean", "average", "best", "worst"])
def test_boxplot_rejects_summary_size_literals(literal, tmp_path):
    from ppbcc.performance_portability.cli import main

    assert (
        main(
            [
                "NBody",
                str(tmp_path / "unused.csv"),
                "--chart",
                "boxplot",
                "--size",
                literal,
            ]
        )
        == 1
    )


def _complexity_csv(tmp_path):
    """Write a minimal complexity CSV with a CPP reference row."""
    path = tmp_path / "code-complexity.csv"
    pd.DataFrame(
        [
            ["VecAdd", "CPP", 173, 102.88],
            ["VecAdd", "Kokkos", 187, 118.5],
            ["VecAdd", "OpenCL", 303, 225.4],
        ],
        columns=["Name", "Framework", "SLOC", "Halstead Difficulty"],
    ).to_csv(path, index=False)
    return path


def test_baseline_reports_the_unscaled_cpp_value(tmp_path):
    path = _complexity_csv(tmp_path)
    assert load_complexity_baseline(path, "VecAdd", "sloc") == 173.0
    assert load_complexity_baseline(
        path, "VecAdd", "halstead-difficulty"
    ) == pytest.approx(102.88)


def test_baseline_matches_the_hundred_percent_of_normalized_data(tmp_path):
    path = _complexity_csv(tmp_path)
    normalized, label = load_complexity_data(
        path, "VecAdd", "sloc", normalize=True, additive=False
    )
    baseline = load_complexity_baseline(path, "VecAdd", "sloc")
    cpp = normalized.loc[normalized["Framework"] == "CPP", label]
    assert float(cpp.iloc[0]) == pytest.approx(100.0)
    # The key on the plot claims this equivalence, so assert it directly.
    other = normalized.loc[normalized["Framework"] == "OpenCL", label]
    assert float(other.iloc[0]) == pytest.approx(303.0 / baseline * 100.0)


def test_baseline_requires_a_cpp_row(tmp_path):
    path = tmp_path / "no-cpp.csv"
    pd.DataFrame(
        [["VecAdd", "Kokkos", 187, 118.5]],
        columns=["Name", "Framework", "SLOC", "Halstead Difficulty"],
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly one CPP"):
        load_complexity_baseline(path, "VecAdd", "sloc")
