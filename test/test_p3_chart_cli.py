"""Integration tests for P3 efficiency-chart CLI filtering."""

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

    result = cli.main(
        [
            "NBody",
            str(csv_path),
            "--chart",
            "heatmap",
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

    result = cli.main(
        [
            "NBody",
            str(csv_path),
            "--chart",
            "boxplot",
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

    result = cli.main(
        [
            "NBody",
            str(csv_path),
            "--chart",
            "boxplot",
            "-H",
            "AMD MI250",
        ]
    )

    assert result == 0
    assert set(captured["efficiency"][HARDWARE]) == {"AMD MI250"}


@pytest.mark.parametrize("chart", ["cascade", "navchart", "combined", "heatmap"])
def test_hardware_filter_is_rejected_for_non_boxplot_charts(chart, tmp_path):
    result = cli.main(
        [
            "NBody",
            str(tmp_path / "unused.csv"),
            "--chart",
            chart,
            "--hardware",
            "AMD MI250",
        ]
    )

    assert result == 1
