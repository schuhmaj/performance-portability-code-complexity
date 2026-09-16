"""Integration tests for the P2- and P3-analysis CLIs."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ppbcc.constants import (
    APPLICATION,
    APPLICATION_EFFICIENCY,
    BENCHMARK_PROBLEM,
    DESCRIPTION,
    HARDWARE,
    PARADIGM,
    PRECISION,
    PROBLEM_SIZE,
    TIME_UNIT,
    WALL_CLOCK_TIME,
)
from ppbcc.performance_portability import cli


def _write_benchmarks(path):
    rows = []
    for size, kokkos_runtime in ((100, 20.0), (200, 40.0)):
        for hardware in ("AMD MI250", "NVIDIA H100"):
            rows.extend(
                [
                    ["NBody", "Cuda", "Naive", 64, hardware, size, 10.0, "ms"],
                    [
                        "NBody",
                        "Cuda",
                        "Cublas",
                        64,
                        hardware,
                        size,
                        8.0,
                        "ms",
                    ],
                    [
                        "NBody",
                        "Kokkos",
                        "Portable",
                        64,
                        hardware,
                        size,
                        kokkos_runtime,
                        "ms",
                    ],
                    ["NBody", "RAJA", "", 64, hardware, size, 30.0, "ms"],
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


def test_heatmap_honors_size_regex_and_remove_description(tmp_path, monkeypatch):
    csv_path = tmp_path / "benchmarks.csv"
    _write_benchmarks(csv_path)
    captured = {}

    def fake_heatmap(efficiency, problem_title, **kwargs):
        captured["efficiency"] = efficiency.copy()
        captured.update(kwargs)
        return plt.figure()

    monkeypatch.setattr(cli, "plot_efficiency_heatmap", fake_heatmap)

    def fake_save(figure, output):
        captured["output"] = output
        plt.close(figure)

    monkeypatch.setattr(cli, "save_figure", fake_save)

    result = cli.p2analysis_main(
        [
            "heatmap",
            str(csv_path),
            "-n",
            "NBody",
            "--size",
            "100",
            "--include",
            "Naive|Portable",
            "--exclude",
            "Cublas",
            "--remove-description",
        ]
    )

    assert result == 0
    assert captured["selected_size"] == 100.0
    assert captured["remove_description"] is True
    assert captured["output"].name == "nbody_heatmap.pdf"
    assert set(captured["efficiency"][APPLICATION]) == {
        "Cuda[Naive]",
        "Kokkos[Portable]",
    }
    kokkos = captured["efficiency"].loc[
        captured["efficiency"][APPLICATION].eq("Kokkos[Portable]"),
        APPLICATION_EFFICIENCY,
    ]
    assert set(kokkos) == {0.5}


def test_boxplot_all_retains_every_selected_size(tmp_path, monkeypatch):
    csv_path = tmp_path / "benchmarks.csv"
    _write_benchmarks(csv_path)
    captured = {}

    def fake_boxplot(efficiency, problem_title, **kwargs):
        captured["efficiency"] = efficiency.copy()
        captured.update(kwargs)
        return plt.figure()

    monkeypatch.setattr(cli, "plot_efficiency_boxplot", fake_boxplot)
    monkeypatch.setattr(cli, "save_figure", lambda figure, output: plt.close(figure))

    result = cli.p2analysis_main(
        [
            "boxplot",
            str(csv_path),
            "--name",
            "NBody",
            "--size",
            "all",
            "--include",
            "Portable",
            "--exclude",
            "Cublas|Naive",
        ]
    )

    assert result == 0
    assert captured["selected_size"] == "all"
    assert set(captured["efficiency"][PROBLEM_SIZE]) == {100.0, 200.0}
    assert set(captured["efficiency"][APPLICATION]) == {
        "Kokkos[Portable]",
    }


def test_boxplot_hardware_filter_retains_only_requested_hardware(
    tmp_path, monkeypatch
):
    csv_path = tmp_path / "benchmarks.csv"
    _write_benchmarks(csv_path)
    captured = {}

    def fake_boxplot(efficiency, problem_title, **kwargs):
        captured["efficiency"] = efficiency.copy()
        return plt.figure()

    monkeypatch.setattr(cli, "plot_efficiency_boxplot", fake_boxplot)
    monkeypatch.setattr(cli, "save_figure", lambda figure, output: plt.close(figure))

    result = cli.p2analysis_main(
        [
            "boxplot",
            str(csv_path),
            "--name",
            "NBody",
            "-H",
            "AMD MI250",
        ]
    )

    assert result == 0
    assert set(captured["efficiency"][HARDWARE]) == {"AMD MI250"}


@pytest.mark.parametrize("chart", ["cascade", "heatmap"])
def test_hardware_filter_is_rejected_for_non_boxplot_charts(chart, tmp_path):
    result = cli.p2analysis_main(
        [chart, str(tmp_path / "unused.csv"), "--hardware", "AMD MI250"]
    )

    assert result == 1


def test_vertical_legend_requires_separate_legend(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "load_benchmark_csvs",
        lambda paths: pytest.fail("input should not be loaded"),
    )

    result = cli.p2analysis_main(
        ["cascade", str(tmp_path / "unused.csv"), "--legend--vertical"]
    )

    assert result == 1


def test_name_may_be_omitted_for_a_single_problem(tmp_path, monkeypatch):
    csv_path = tmp_path / "benchmarks.csv"
    _write_benchmarks(csv_path)
    captured = {}
    monkeypatch.setattr(
        cli,
        "plot_efficiency_heatmap",
        lambda efficiency, problem_title, **kwargs: plt.figure(),
    )

    def fake_save(figure, output):
        captured["output"] = output
        plt.close(figure)

    monkeypatch.setattr(cli, "save_figure", fake_save)

    assert cli.p2analysis_main(["heatmap", str(csv_path)]) == 0
    assert captured["output"].name == "nbody_heatmap.pdf"


def test_name_is_required_for_several_problems(tmp_path, monkeypatch):
    csv_path = tmp_path / "benchmarks.csv"
    _write_benchmarks(csv_path)
    frame = pd.read_csv(csv_path)
    frame[BENCHMARK_PROBLEM] = "VecAdd"
    pd.concat([pd.read_csv(csv_path), frame]).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        cli,
        "plot_efficiency_heatmap",
        lambda *args, **kwargs: pytest.fail("nothing should be plotted"),
    )

    assert cli.p2analysis_main(["heatmap", str(csv_path)]) == 1


@pytest.mark.parametrize("chart", ["navchart", "combined", "complexity-comparison"])
def test_p2analysis_rejects_complexity_charts(chart, tmp_path):
    with pytest.raises(SystemExit):
        cli.p2analysis_main([chart, str(tmp_path / "unused.csv")])


def _write_complexity(path):
    pd.DataFrame(
        [
            ["NBody", "CPP", 100, 10.0],
            ["NBody", "Cuda", 150, 30.0],
            ["NBody", "Kokkos", 200, 15.0],
            ["NBody", "RAJA", 400, 20.0],
        ],
        columns=["Name", "Framework", "SLOC", "Halstead Difficulty"],
    ).to_csv(path, index=False)


@pytest.mark.parametrize(
    ("extra", "expected_metric", "expected_kokkos"),
    [
        ([], "Halstead Difficulty [normalized]", 150.0),
        (["--complexity-metric-absolute"], "Halstead Difficulty [absolute]", 15.0),
        (["-c", "sloc"], "Source Lines of Code [normalized]", 200.0),
    ],
)
def test_navchart_normalizes_complexity_by_default(
    extra, expected_metric, expected_kokkos, tmp_path, monkeypatch
):
    csv_path = tmp_path / "benchmarks.csv"
    complexity_path = tmp_path / "complexity.csv"
    _write_benchmarks(csv_path)
    _write_complexity(complexity_path)
    captured = {}

    def fake_navchart(data, metric, problem_title, **kwargs):
        captured["data"] = data.copy()
        captured["metric"] = metric
        return plt.figure()

    monkeypatch.setattr(cli, "plot_navchart", fake_navchart)
    monkeypatch.setattr(cli, "save_figure", lambda figure, output: plt.close(figure))

    result = cli.p3analysis_main(
        [
            "navchart",
            str(complexity_path),
            str(csv_path),
            "-n",
            "NBody",
            "--remove-description",
            *extra,
        ]
    )

    assert result == 0
    assert captured["metric"] == expected_metric
    kokkos = captured["data"].loc[
        captured["data"][APPLICATION].str.startswith("Kokkos"), expected_metric
    ]
    assert set(kokkos) == {expected_kokkos}


def test_complexity_comparison_rejects_absolute_values(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "load_benchmark_csvs",
        lambda paths: pytest.fail("input should not be loaded"),
    )

    result = cli.p3analysis_main(
        [
            "complexity-comparison",
            str(tmp_path / "complexity.csv"),
            str(tmp_path / "unused.csv"),
            "--complexity-metric-absolute",
        ]
    )

    assert result == 1


@pytest.mark.parametrize("chart", ["cascade", "heatmap", "boxplot"])
def test_p3analysis_rejects_benchmark_only_charts(chart, tmp_path):
    with pytest.raises(SystemExit):
        cli.p3analysis_main(
            [chart, str(tmp_path / "complexity.csv"), str(tmp_path / "unused.csv")]
        )
