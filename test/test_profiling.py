"""Tests for Nsight Compute profiling, parsing, and roofline plotting."""

import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from ppbcc.constants import (  # noqa: E402
    ARITHMETIC_INTENSITY,
    BENCHMARK_PROBLEM,
    BLOCK_SIZE,
    DURATION,
    EXECUTABLE,
    FLOP,
    GRID_SIZE,
    KERNEL,
    KERNEL_SIGNATURE,
    MEMORY_TRAFFIC,
    PARADIGM,
    PEAK_BANDWIDTH,
    PEAK_PERFORMANCE,
    PERFORMANCE,
    PRECISION,
)
from ppbcc.plot.roofline import plot_roofline  # noqa: E402
from ppbcc.profiling.cli import build_parser  # noqa: E402
from ppbcc.profiling.metrics import (  # noqa: E402
    ROOFLINE_METRICS,
    flop_columns,
    peak_flop_column,
)
from ppbcc.profiling.reports import (  # noqa: E402
    _short_kernel_name,
    aggregate_kernels,
    filter_kernels,
    load_reports,
)
from ppbcc.profiling.runner import profile_command  # noqa: E402


# --------------------------------------------------------------------------- #
# Metric set
# --------------------------------------------------------------------------- #


def test_flop_and_peak_columns_are_part_of_the_collected_metrics():
    for precision in ("fp32", "fp64", "fp16"):
        for column in flop_columns(precision):
            assert column in ROOFLINE_METRICS
        assert peak_flop_column(precision) in ROOFLINE_METRICS


# --------------------------------------------------------------------------- #
# Kernel names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "signature, expected",
    [
        ("run_eval(const float3 *, int)", "run_eval"),
        ("void cub::DeviceReduceKernel<A<B>, C>(T2, T5 *)", "cub::DeviceReduceKernel"),
        ("__acpp_sscp_kernel", "__acpp_sscp_kernel"),
    ],
)
def test_short_kernel_name_strips_templates_and_parameters(signature, expected):
    assert _short_kernel_name(signature) == expected


# --------------------------------------------------------------------------- #
# Profiler invocation
# --------------------------------------------------------------------------- #


def test_profile_command_exports_next_to_the_executable_name(tmp_path):
    command = profile_command(
        "ncu",
        tmp_path / "polyhedral_cuda",
        tmp_path / "out" / "polyhedral_cuda.ncu-rep",
        metrics=["dram__bytes.sum"],
        benchmark_report=tmp_path / "out" / "polyhedral_cuda.json",
    )

    assert command[0] == "ncu"
    assert "--metrics" in command and "dram__bytes.sum" in command
    # The profiler appends its own suffix, so it must not be passed in.
    assert str(tmp_path / "out" / "polyhedral_cuda") in command
    assert command.index(str(tmp_path / "polyhedral_cuda")) > command.index("--export")
    assert f"--benchmark_out={tmp_path / 'out' / 'polyhedral_cuda.json'}" in command


# --------------------------------------------------------------------------- #
# Report parsing
# --------------------------------------------------------------------------- #


def _write_report(tmp_path, monkeypatch, rows, float_type="32"):
    """Write a fake report pair and stub the profiler import out."""
    report = tmp_path / "polyhedral_cuda.ncu-rep"
    report.write_bytes(b"")
    (tmp_path / "polyhedral_cuda.json").write_text(
        json.dumps(
            {
                "context": {"paradigm": "Cuda", "float_type": float_type},
                "benchmarks": [{"name": "Polyhedral-Eros"}],
            }
        )
    )

    header = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context",'
        '"Stream","Block Size","Grid Size","Device","CC","Section Name",'
        '"Metric Name","Metric Unit","Metric Value"'
    )
    lines = [header]
    for kernel_id, kernel, metric, value in rows:
        lines.append(
            f'"{kernel_id}","1","polyhedral_cuda","host","{kernel}","1","7",'
            f'"(256, 1, 1)","(58, 1, 1)","0","12.0","Command line profiler '
            f'metrics","{metric}","unit","{value}"'
        )
    csv = "\n".join(lines) + "\n"

    monkeypatch.setattr(
        "ppbcc.profiling.reports._import_csv",
        lambda path, ncu: pd.read_csv(__import__("io").StringIO(csv), dtype=str),
    )
    return report


def test_load_reports_derives_roofline_quantities(tmp_path, monkeypatch):
    # 1000 add + 500 fma = 2000 FLOP over 100 bytes in 1 us -> AI 20, 2 GFLOP/s.
    rows = [
        (0, "run_eval", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "1,000"),
        (0, "run_eval", "sm__sass_thread_inst_executed_op_fmul_pred_on.sum", "0"),
        (0, "run_eval", "sm__sass_thread_inst_executed_op_ffma_pred_on.sum", "500"),
        (0, "run_eval", "dram__bytes.sum", "100"),
        (0, "run_eval", "gpu__time_duration.sum", "1,000"),
        (0, "run_eval", "sm__cycles_elapsed.avg.per_second", "2"),
        (
            0,
            "run_eval",
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
            "3",
        ),
        (0, "run_eval", "dram__bytes.sum.peak_sustained", "4"),
        (0, "run_eval", "dram__cycles_elapsed.avg.per_second", "5"),
    ]
    report = _write_report(tmp_path, monkeypatch, rows)

    frame = load_reports([report], hardware="NVIDIA RTX5080")

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row[EXECUTABLE] == "polyhedral_cuda"
    assert row[PARADIGM] == "Cuda"
    assert row[BENCHMARK_PROBLEM] == "Polyhedral"
    assert row[PRECISION] == "FP32"
    assert row[FLOP] == pytest.approx(2000.0)
    assert row[MEMORY_TRAFFIC] == pytest.approx(100.0)
    assert row[DURATION] == pytest.approx(1e-6)
    assert row[ARITHMETIC_INTENSITY] == pytest.approx(20.0)
    assert row[PERFORMANCE] == pytest.approx(2e9)
    # An FMA is two FLOP, and peak_sustained is per cycle.
    assert row[PEAK_PERFORMANCE] == pytest.approx(2 * 3 * 2)
    assert row[PEAK_BANDWIDTH] == pytest.approx(4 * 5)
    # The launch configuration explains a point that sits far below the roof.
    assert row[GRID_SIZE] == "(58, 1, 1)"
    assert row[BLOCK_SIZE] == "(256, 1, 1)"


def test_load_reports_follows_the_build_precision_for_setup_kernels(
    tmp_path, monkeypatch
):
    # A kernel whose only floating-point instructions are half precision still
    # belongs to an FP32 build and must not switch the whole row to FP16.
    rows = [
        (0, "setup", "sm__sass_thread_inst_executed_op_hfma_pred_on.sum", "8"),
        (0, "setup", "dram__bytes.sum", "10"),
        (0, "setup", "gpu__time_duration.sum", "1,000"),
    ]
    report = _write_report(tmp_path, monkeypatch, rows, float_type="32")

    frame = load_reports([report])

    assert frame.iloc[0][PRECISION] == "FP32"
    assert frame.iloc[0][FLOP] == pytest.approx(0.0)
    assert frame.iloc[0][f"{FLOP} FP16"] == pytest.approx(16.0)


# --------------------------------------------------------------------------- #
# Aggregation and filtering
# --------------------------------------------------------------------------- #


def _points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_PROBLEM: ["Polyhedral"] * 3,
            PARADIGM: ["Kokkos"] * 3,
            PRECISION: ["FP32"] * 3,
            "Hardware": ["RTX5080"] * 3,
            "Memory Level": ["DRAM"] * 3,
            EXECUTABLE: ["polyhedral_kokkos"] * 3,
            KERNEL: ["init_lock_arrays_cuda_kernel", "main", "reduce"],
            KERNEL_SIGNATURE: [
                "desul::init_lock_arrays_cuda_kernel()",
                "main()",
                "reduce()",
            ],
            DURATION: [1e-5, 2e-5, 1e-5],
            FLOP: [0.0, 800.0, 200.0],
            MEMORY_TRAFFIC: [100.0, 100.0, 50.0],
            PEAK_PERFORMANCE: [1e13, 2e13, 1e13],
            PEAK_BANDWIDTH: [1e11, 9e11, 1e11],
            ARITHMETIC_INTENSITY: [0.0, 8.0, 4.0],
            PERFORMANCE: [0.0, 4e7, 2e7],
        }
    )


def test_filter_kernels_drops_framework_bootstrap_kernels():
    filtered = filter_kernels(_points(), ["init_lock_arrays"])
    assert list(filtered[KERNEL]) == ["main", "reduce"]


def test_aggregate_kernels_sums_work_time_and_traffic():
    aggregated = aggregate_kernels(filter_kernels(_points(), ["init_lock"]), "sum")

    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row[FLOP] == pytest.approx(1000.0)
    assert row[MEMORY_TRAFFIC] == pytest.approx(150.0)
    assert row[ARITHMETIC_INTENSITY] == pytest.approx(1000.0 / 150.0)
    assert row[PERFORMANCE] == pytest.approx(1000.0 / 3e-5)
    # The ceilings are properties of the device, not something to add up.
    assert row[PEAK_PERFORMANCE] == pytest.approx(2e13)


def test_aggregate_kernels_dominant_keeps_the_longest_kernel():
    aggregated = aggregate_kernels(_points(), "dominant")
    assert list(aggregated[KERNEL]) == ["main"]


def test_aggregate_kernels_none_keeps_every_launch():
    assert len(aggregate_kernels(_points(), "none")) == 3


# --------------------------------------------------------------------------- #
# Plot and CLI
# --------------------------------------------------------------------------- #


def test_plot_roofline_draws_a_point_per_row():
    figure = plot_roofline(
        aggregate_kernels(_points(), "none").iloc[1:],
        hardware="NVIDIA RTX5080",
        problem_title="Polyhedral",
    )
    axis = figure.axes[0]
    # One PathCollection per scatter point, next to the shaded area under the roof.
    scatters = [
        collection
        for collection in axis.collections
        if isinstance(collection, matplotlib.collections.PathCollection)
    ]
    assert len(scatters) == 2
    assert axis.get_xscale() == "log" and axis.get_yscale() == "log"


def test_plot_roofline_rejects_data_without_a_usable_point():
    empty = _points().iloc[:1]
    with pytest.raises(ValueError):
        plot_roofline(empty)


def test_profile_cli_requires_a_regex():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    args = build_parser().parse_args(["-r", "polyhedral_.*", "--roofline"])
    assert args.regex == ["polyhedral_.*"]
    assert args.roofline is True
    assert args.aggregate == "sum"
    assert args.memory_level == "dram"
