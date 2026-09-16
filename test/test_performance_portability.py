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
from ppbcc.performance_portability.metrics import (
    calculate_average_size_metrics,
    calculate_export_metrics,
    calculate_metrics,
)
from ppbcc.performance_portability.options import (
    build_p2_parser,
    build_p3_parser,
)
from ppbcc.performance_portability.selection import (
    ALL_SIZE,
    AVERAGE_OVER_EFFICIENCY,
    AVERAGE_OVER_PP,
    AVERAGE_SIZE,
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
    args = build_p2_parser().parse_args(["cascade", "results.csv"])
    assert args.size == ALL_SIZE
    assert args.name is None


@pytest.mark.parametrize("flag", ["-H", "--hardware"])
def test_hardware_option_supports_short_and_long_flags(flag):
    args = build_p2_parser().parse_args(
        ["boxplot", "results.csv", flag, "AMD MI250"]
    )

    assert args.hardware == "AMD MI250"


def test_vertical_legend_option_is_parsed():
    args = build_p2_parser().parse_args(
        ["cascade", "results.csv", "-l", "--legend--vertical"]
    )

    assert args.legend is True
    assert args.legend_vertical is True


def test_p3_parser_takes_complexity_csv_before_benchmark_csvs():
    args = build_p3_parser().parse_args(
        ["combined", "complexity.csv", "a.csv", "b.csv", "-n", "NBody"]
    )

    assert args.chart == "combined"
    assert args.complexity.name == "complexity.csv"
    assert [path.name for path in args.csv_files] == ["a.csv", "b.csv"]
    assert args.name == "NBody"
    assert args.complexity_metric == "halstead-difficulty"
    assert args.complexity_absolute is False


@pytest.mark.parametrize("literal", ["mean", "average", "best", "worst"])
def test_boxplot_rejects_summary_size_literals(literal, tmp_path):
    from ppbcc.performance_portability.cli import p2analysis_main

    assert (
        p2analysis_main(
            [
                "boxplot",
                str(tmp_path / "unused.csv"),
                "--size",
                literal,
            ]
        )
        == 1
    )


def _size_swap_frame() -> pd.DataFrame:
    """Kokkos is best on NVIDIA at the small size and on AMD at the large one.

    Kokkos' efficiency is (1, 0.25) at size 10 and (0.25, 1) at size 100, so
    its PP is 0.4 at either size while its size-averaged efficiency is 0.625
    on both platforms.
    """
    rows = [
        (10, "NVIDIA", 10.0, 10.0),
        (10, "AMD", 10.0, 40.0),
        (100, "NVIDIA", 10.0, 40.0),
        (100, "AMD", 10.0, 10.0),
    ]
    return pd.DataFrame(
        [
            ["NBody", paradigm, "", 64, hardware, size, runtime, "ms"]
            for size, hardware, cuda, kokkos in rows
            for paradigm, runtime in (("Cuda", cuda), ("Kokkos", kokkos))
        ],
        columns=_benchmark_frame().columns,
    )


def _kokkos_pp(portability: pd.DataFrame) -> float:
    return portability.loc[
        portability[APPLICATION].eq("Kokkos"), PERFORMANCE_PORTABILITY
    ].item()


def test_average_over_pp_averages_the_per_size_scores():
    _, portability = calculate_average_size_metrics(
        _size_swap_frame(), False, average_over=AVERAGE_OVER_PP
    )
    assert _kokkos_pp(portability) == pytest.approx(0.4)


def test_average_over_efficiency_computes_pp_from_the_averages():
    efficiency, portability = calculate_average_size_metrics(
        _size_swap_frame(), False, average_over=AVERAGE_OVER_EFFICIENCY
    )
    kokkos = efficiency.loc[efficiency[APPLICATION].eq("Kokkos")]
    assert kokkos[APPLICATION_EFFICIENCY].tolist() == pytest.approx([0.625, 0.625])
    assert _kokkos_pp(portability) == pytest.approx(0.625)
    # Pooling every size into one workload set is the same black-box reduction.
    _, pooled = calculate_metrics(_size_swap_frame(), False)
    assert _kokkos_pp(pooled) == pytest.approx(0.625)


@pytest.mark.parametrize(
    ("average_over", "expected"),
    [(AVERAGE_OVER_PP, 0.4), (AVERAGE_OVER_EFFICIENCY, 0.625)],
)
def test_export_average_row_follows_average_over(average_over, expected):
    _, portability = calculate_export_metrics(
        _size_swap_frame(), False, average_over=average_over
    )
    average = portability.loc[portability[PROBLEM_SIZE].eq(AVERAGE_SIZE)]
    assert _kokkos_pp(average) == pytest.approx(expected)


def test_average_over_efficiency_requires_size_average(tmp_path):
    from ppbcc.performance_portability.cli import p2analysis_main

    arguments = ["cascade", str(tmp_path / "unused.csv"), "--average-over"]
    assert p2analysis_main([*arguments, AVERAGE_OVER_EFFICIENCY]) == 1


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
        path, "VecAdd", "sloc", normalize=True
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
