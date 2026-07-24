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
