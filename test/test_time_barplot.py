"""Tests for the runtime bar chart, its CLI and the peak-performance table."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ppbcc.constants import (
    APPLICATION,
    BENCHMARK_PROBLEM,
    DESCRIPTION,
    HARDWARE,
    KERNEL_TIME,
    PARADIGM,
    PRECISION,
    PROBLEM_SIZE,
    TIME_UNIT,
    WALL_CLOCK_TIME,
)
from ppbcc.hardware import PEAK_PERFORMANCE, peak_performance
from ppbcc.performance_portability import cli
from ppbcc.performance_portability.metrics import calculate_runtimes
from ppbcc.plot.time_barplot import PEAK_FLOP, RUNTIME_NS, plot_time_barplot

HARDWARE_LABELS = ("NVIDIA RTX5080", "AMD MI210")


def _write_benchmarks(path, kernel_time=True):
    rows = []
    for size in (100, 200):
        for hardware in HARDWARE_LABELS:
            for paradigm, description, runtime in (
                ("Cuda", "Naive", 4.0),
                ("Cuda", "Cublas", 2.0),
                ("Kokkos", "", 8.0),
            ):
                rows.append(
                    [
                        "MatMul",
                        paradigm,
                        description,
                        32,
                        hardware,
                        size,
                        runtime * size,
                        runtime * size / 2 if kernel_time else None,
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
            KERNEL_TIME,
            TIME_UNIT,
        ],
    ).to_csv(path, index=False)


@pytest.fixture
def captured(monkeypatch):
    result = {}

    def fake_plot(runtimes, problem_title, time_label, size, precision, **kwargs):
        result.update(
            runtimes=runtimes.copy(),
            time_label=time_label,
            size=size,
            precision=precision,
            **kwargs,
        )
        return plt.figure()

    def fake_save(figure, output):
        result.setdefault("outputs", []).append(output)
        plt.close(figure)

    monkeypatch.setattr(cli, "plot_time_barplot", fake_plot)
    monkeypatch.setattr(cli, "save_figure", fake_save)
    return result


def test_peak_performance_is_tabulated_for_every_platform():
    assert set(PEAK_PERFORMANCE) == {
        "NVIDIA RTX3080",
        "NVIDIA RTX4060",
        "NVIDIA RTX5080",
        "NVIDIA GH200",
        "AMD MI210",
        "Intel Max 1550",
    }
    assert peak_performance("nvidia rtx-5080", 32) == pytest.approx(56.3e12)


def test_peak_performance_rejects_unknown_platforms():
    with pytest.raises(ValueError, match="Known platforms"):
        peak_performance("NVIDIA A100", 32)


def test_runtimes_combine_variants_by_their_fastest():
    frame = pd.DataFrame(
        [
            ["MatMul", "Cuda", "Naive", 32, "X", 100, 4.0, "ms"],
            ["MatMul", "Cuda", "Naive", 32, "X", 100, 6.0, "ms"],
            ["MatMul", "Cuda", "Cublas", 32, "X", 100, 2.0, "ms"],
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
    separate = calculate_runtimes(frame, False, WALL_CLOCK_TIME)
    assert dict(zip(separate[APPLICATION], separate[RUNTIME_NS])) == {
        "Cuda[Cublas]": 2e6,
        "Cuda[Naive]": 5e6,
    }
    combined = calculate_runtimes(
        frame, False, WALL_CLOCK_TIME, remove_description=True
    )
    assert combined[RUNTIME_NS].tolist() == [2e6]


def test_time_barplot_uses_the_largest_size_and_the_selected_time(
    tmp_path, captured
):
    csv_path = tmp_path / "benchmarks.csv"
    _write_benchmarks(csv_path)

    assert cli.p2analysis_main(["time-barplot", str(csv_path), "-t", "kernel"]) == 0

    assert captured["size"] == 200.0
    assert captured["time_label"] == KERNEL_TIME
    assert captured["outputs"][0].name == "matmul_time_barplot_kernel.pdf"
    kokkos = captured["runtimes"].loc[
        captured["runtimes"][APPLICATION].eq("Kokkos"), RUNTIME_NS
    ]
    assert set(kokkos) == {800.0 * 1e6}


def test_time_barplot_normalizes_to_peak_and_splits_the_legend(
    tmp_path, captured
):
    csv_path = tmp_path / "benchmarks.csv"
    _write_benchmarks(csv_path)

    result = cli.p2analysis_main(
        [
            "time-barplot",
            str(csv_path),
            "-H",
            "NVIDIA RTX5080",
            "--normalize-time-to-peak",
            "-l",
        ]
    )

    assert result == 0
    assert captured["normalized"] is True
    assert captured["show_legends"] is False
    assert set(captured["runtimes"][HARDWARE]) == {"NVIDIA RTX5080"}
    kokkos = captured["runtimes"].loc[
        captured["runtimes"][APPLICATION].eq("Kokkos")
    ].iloc[0]
    assert kokkos[PEAK_FLOP] == pytest.approx(1.6 * 56.3e12)
    assert [output.name for output in captured["outputs"]] == [
        "matmul_time_barplot_wall_clock.pdf",
        "matmul_time_barplot_wall_clock_legend.pdf",
    ]


def test_time_barplot_rejects_a_missing_time_column(tmp_path, captured):
    csv_path = tmp_path / "benchmarks.csv"
    _write_benchmarks(csv_path, kernel_time=False)

    assert cli.p2analysis_main(["time-barplot", str(csv_path), "-t", "kernel"]) == 1
    assert "outputs" not in captured


@pytest.mark.parametrize("size", ["avg", "best", "worst"])
def test_time_barplot_rejects_size_summaries(size, tmp_path, captured):
    csv_path = tmp_path / "benchmarks.csv"
    _write_benchmarks(csv_path)

    assert cli.p2analysis_main(["time-barplot", str(csv_path), "-s", size]) == 1


def test_time_options_are_rejected_for_other_charts(tmp_path):
    assert cli.p2analysis_main(["cascade", str(tmp_path / "x.csv"), "-t", "kernel"]) == 1


@pytest.mark.parametrize("platforms", [HARDWARE_LABELS, HARDWARE_LABELS[:1]])
def test_plot_draws_one_bar_per_result(platforms):
    runtimes = pd.DataFrame(
        [
            [application, platform, 32, value]
            for platform in platforms
            for application, value in (("Cuda", 2e9), ("Kokkos", 4e9))
        ],
        columns=[APPLICATION, HARDWARE, PRECISION, RUNTIME_NS],
    )
    figure = plot_time_barplot(runtimes, "MatMul", KERNEL_TIME, 16384.0, 32)
    axis = figure.axes[0]
    assert len(axis.patches) == len(runtimes)
    assert axis.get_yscale() == "log"
    assert axis.get_ylabel() == "Kernel Time [s]"
    assert "16,384" in axis.get_title()
    plt.close(figure)
