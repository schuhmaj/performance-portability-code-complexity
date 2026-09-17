"""Tests for the cross-problem rank-correlation table."""

import pandas as pd
import pytest

from ppbcc.constants import (
    BENCHMARK_PROBLEM,
    DESCRIPTION,
    HARDWARE,
    PARADIGM,
    PERFORMANCE_PORTABILITY,
    PRECISION,
    PROBLEM,
    PROBLEM_SIZE,
    TIME_UNIT,
    WALL_CLOCK_TIME,
)
from ppbcc.performance_portability import cli
from ppbcc.performance_portability.correlation import (
    CORRELATION_PP,
    RANK,
    base_paradigm,
    build_value_table,
    rank_correlation_matrix,
    resolve_correlation_variable,
    shared_paradigm_counts,
)

#: Runtimes that order VecAdd Cuda > Kokkos > RAJA > OpenMP and reverse it for
#: NBody, so the two problems rank the paradigms exactly opposite each other.
_RUNTIMES = {
    "VecAdd": [
        ("Cuda", "Naive", 20.0),
        ("Cuda", "Cublas", 10.0),
        ("Kokkos", "", 20.0),
        ("RAJA", "", 40.0),
        ("OpenMP", "", 80.0),
    ],
    "NBody": [
        ("Cuda", "", 80.0),
        ("Kokkos", "", 40.0),
        ("RAJA", "", 20.0),
        ("OpenMP", "", 10.0),
    ],
}

#: SLOC and Halstead counts that order both problems the same way, with rows
#: for the C++ baseline and for frameworks that were never benchmarked.
_COMPLEXITY = [
    ("VecAdd", "CPP", 100, 10, 20, 100, 50),
    ("VecAdd", "Cuda[Naive]", 200, 20, 40, 200, 100),
    ("VecAdd", "Cuda[Cublas]", 220, 24, 44, 220, 110),
    ("VecAdd", "Kokkos", 150, 15, 30, 150, 75),
    ("VecAdd", "RAJA", 300, 30, 60, 300, 150),
    ("VecAdd", "OpenMP", 400, 40, 80, 400, 200),
    ("VecAdd", "Metal", 999, 99, 198, 999, 500),
    ("NBody", "CPP", 100, 10, 20, 100, 50),
    ("NBody", "Cuda", 210, 22, 42, 210, 105),
    ("NBody", "Kokkos", 150, 15, 30, 150, 75),
    ("NBody", "RAJA", 300, 30, 60, 300, 150),
    ("NBody", "OpenMP", 400, 40, 80, 400, 200),
    ("NBody", "Thrust", 50, 5, 10, 50, 25),
]


def _write_benchmarks(path):
    rows = []
    for problem, implementations in _RUNTIMES.items():
        for paradigm, description, runtime in implementations:
            for hardware in ("AMD MI250", "NVIDIA H100"):
                rows.append(
                    [
                        problem,
                        paradigm,
                        description,
                        64,
                        hardware,
                        100,
                        runtime,
                        "ms",
                    ]
                )
    pd.DataFrame(
        rows,
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
    ).to_csv(path, index=False)


def _write_complexity(path):
    pd.DataFrame(
        _COMPLEXITY,
        columns=["Name", "Framework", "SLOC", "n1", "n2", "N1", "N2"],
    ).to_csv(path, index=False)


@pytest.fixture
def inputs(tmp_path):
    benchmarks = tmp_path / "benchmarks.csv"
    complexity = tmp_path / "complexity.csv"
    _write_benchmarks(benchmarks)
    _write_complexity(complexity)
    return complexity, benchmarks


def _run(inputs, tmp_path, *extra):
    complexity, benchmarks = inputs
    output = tmp_path / "correlation.csv"
    result = cli.p3analysis_main(
        [
            "rank-correlation",
            str(complexity),
            str(benchmarks),
            "-o",
            str(output),
            *extra,
        ]
    )
    return result, output


@pytest.mark.parametrize(
    ("request_value", "expected"),
    [
        ("pp", CORRELATION_PP),
        ("PP", CORRELATION_PP),
        ("performance_portability", CORRELATION_PP),
        ("sloc", "SLOC"),
        ("halstead-difficulty", "Halstead Difficulty"),
        ("Halstead Effort", "Halstead Effort"),
    ],
)
def test_resolve_correlation_variable_accepts_aliases(request_value, expected):
    assert resolve_correlation_variable(request_value) == expected


def test_resolve_correlation_variable_rejects_unknown_variable():
    with pytest.raises(ValueError, match="Unknown correlation variable"):
        resolve_correlation_variable("runtime")


def test_base_paradigm_strips_implementation_variants():
    assert base_paradigm("Cuda[SharedMemory]") == "Cuda"
    assert base_paradigm("Kokkos") == "Kokkos"


def test_pp_correlation_ranks_paradigms_and_writes_matrix(inputs, tmp_path):
    result, output = _run(inputs, tmp_path, "--correlation", "pp")

    assert result == 0
    matrix = pd.read_csv(output, index_col=0)
    assert list(matrix.columns) == ["NBody", "VecAdd"]
    assert matrix.loc["NBody", "NBody"] == pytest.approx(1.0)
    assert matrix.loc["VecAdd", "VecAdd"] == pytest.approx(1.0)
    # The two problems order the paradigms exactly opposite each other.
    assert matrix.loc["NBody", "VecAdd"] == pytest.approx(-1.0)
    assert matrix.loc["VecAdd", "NBody"] == pytest.approx(-1.0)


def test_pp_correlation_lets_the_best_variant_stand_for_its_paradigm(
    inputs, tmp_path
):
    result, output = _run(inputs, tmp_path, "--correlation", "pp", "-e")

    assert result == 0
    ranks = pd.read_csv(output.with_name("correlation_ranks.csv"))
    assert list(ranks.columns) == [
        PROBLEM,
        PARADIGM,
        PERFORMANCE_PORTABILITY,
        RANK,
    ]
    vecadd = ranks.loc[ranks[PROBLEM].eq("VecAdd")].set_index(PARADIGM)
    assert set(vecadd.index) == {"Cuda", "Kokkos", "RAJA", "OpenMP"}
    # Cuda[Cublas] is twice as fast as Cuda[Naive], so Cuda is perfectly
    # portable and ranks first; the slower variant does not drag it down.
    assert vecadd.loc["Cuda", PERFORMANCE_PORTABILITY] == pytest.approx(1.0)
    assert vecadd.loc["Cuda", RANK] == 1
    assert vecadd.loc["OpenMP", RANK] == 4


def test_complexity_correlation_uses_benchmarked_paradigms_only(inputs, tmp_path):
    result, output = _run(inputs, tmp_path, "--correlation", "sloc", "-e")

    assert result == 0
    matrix = pd.read_csv(output, index_col=0)
    # Both problems order the paradigms by SLOC the same way.
    assert matrix.loc["NBody", "VecAdd"] == pytest.approx(1.0)

    ranks = pd.read_csv(output.with_name("correlation_ranks.csv"))
    assert list(ranks.columns) == [PROBLEM, PARADIGM, "SLOC", RANK]
    for problem in ("VecAdd", "NBody"):
        paradigms = set(ranks.loc[ranks[PROBLEM].eq(problem), PARADIGM])
        assert paradigms == {"Cuda", "Kokkos", "RAJA", "OpenMP"}
    vecadd = ranks.loc[ranks[PROBLEM].eq("VecAdd")].set_index(PARADIGM)
    # Cuda[Naive] and Cuda[Cublas] are reduced to their median.
    assert vecadd.loc["Cuda", "SLOC"] == pytest.approx(210.0)
    # Fewest lines ranks first for a complexity metric.
    assert vecadd.loc["Kokkos", RANK] == 1
    assert vecadd.loc["OpenMP", RANK] == 4


def test_halstead_correlation_is_supported(inputs, tmp_path):
    result, output = _run(inputs, tmp_path, "--correlation", "halstead-difficulty")

    assert result == 0
    matrix = pd.read_csv(output, index_col=0)
    assert set(matrix.columns) == {"NBody", "VecAdd"}
    assert matrix.loc["NBody", "VecAdd"] == pytest.approx(1.0)


def test_name_selects_a_subset_of_problems(inputs, tmp_path):
    result, output = _run(inputs, tmp_path, "-n", "VecAdd,NBody")

    assert result == 0
    matrix = pd.read_csv(output, index_col=0)
    assert set(matrix.columns) == {"NBody", "VecAdd"}


def test_a_single_problem_is_rejected(inputs, tmp_path):
    result, output = _run(inputs, tmp_path, "-n", "VecAdd")

    assert result == 1
    assert not output.exists()


def test_a_repeated_problem_is_rejected(inputs, tmp_path):
    result, _ = _run(inputs, tmp_path, "-n", "VecAdd,VecAdd")

    assert result == 1


def test_options_without_a_figure_are_rejected(inputs, tmp_path):
    for option in ("-l", "--remove-description", "--log-complexity"):
        result, _ = _run(inputs, tmp_path, option)
        assert result == 1, option


def test_correlation_is_rejected_for_other_charts(inputs, tmp_path):
    complexity, benchmarks = inputs
    result = cli.p3analysis_main(
        [
            "navchart",
            str(complexity),
            str(benchmarks),
            "-n",
            "VecAdd",
            "--correlation",
            "sloc",
        ]
    )

    assert result == 1


def test_build_value_table_needs_two_problems():
    with pytest.raises(ValueError, match="at least two"):
        build_value_table({"VecAdd": pd.Series({"Cuda": 1.0})})


def test_pairs_are_correlated_over_their_shared_paradigms():
    table = build_value_table(
        {
            "VecAdd": pd.Series({"Cuda": 1.0, "Kokkos": 2.0, "RAJA": 3.0}),
            "NBody": pd.Series({"Cuda": 3.0, "Kokkos": 2.0, "RAJA": 1.0}),
            "MatMul": pd.Series({"Cuda": 1.0, "Kokkos": 2.0, "OpenMP": 9.0}),
        }
    )

    counts = shared_paradigm_counts(table)
    assert counts.loc["VecAdd", "NBody"] == 3
    # MatMul shares only Cuda and Kokkos with the others.
    assert counts.loc["VecAdd", "MatMul"] == 2

    matrix = rank_correlation_matrix(table)
    assert matrix.loc["VecAdd", "NBody"] == pytest.approx(-1.0)
    # Two shared paradigms are below the minimum, leaving the pair undefined.
    assert pd.isna(matrix.loc["VecAdd", "MatMul"])


def test_rank_correlation_needs_one_usable_pair():
    table = build_value_table(
        {
            "VecAdd": pd.Series({"Cuda": 1.0, "Kokkos": 2.0}),
            "NBody": pd.Series({"Cuda": 2.0, "Kokkos": 1.0}),
        }
    )

    with pytest.raises(ValueError, match="rank correlation would be meaningless"):
        rank_correlation_matrix(table)
