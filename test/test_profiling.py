"""Tests for kernel profiling, report parsing, and roofline plotting."""

import json
from pathlib import Path

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
    PROFILE_COLUMN_LIST,
    REGION,
)
from ppbcc.plot.roofline import plot_roofline  # noqa: E402
from ppbcc.profiling import likwid, ngfx, nsys  # noqa: E402
from ppbcc.profiling.cli import build_parser, parse_options  # noqa: E402
from ppbcc.profiling.metrics import (  # noqa: E402
    ROOFLINE_METRICS,
    column_unit,
    flop_columns,
    peak_flop_column,
)
from ppbcc.profiling.reports import (  # noqa: E402
    _nvtx_region,
    _short_kernel_name,
    aggregate_kernels,
    filter_regions,
    load_reports,
    select_regions,
    with_units,
    without_units,
)
from ppbcc.profiling.runner import aslr_prefix, profile_command  # noqa: E402


# --------------------------------------------------------------------------- #
# Metric set
# --------------------------------------------------------------------------- #


def test_flop_and_peak_columns_are_part_of_the_collected_metrics():
    for precision in ("fp32", "fp64", "fp16"):
        for column in flop_columns(precision):
            assert column in ROOFLINE_METRICS
        assert peak_flop_column(precision) in ROOFLINE_METRICS


@pytest.mark.parametrize(
    "column, unit",
    [
        ("Arithmetic Intensity", "FLOP/Byte"),
        ("Peak Bandwidth", "Byte/s"),
        ("gpu__time_duration.sum", "ns"),
        ("dram__bytes.sum", "Byte"),
        ("dram__bytes.sum.peak_sustained", "Byte/cycle"),
        ("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained", "inst/cycle"),
        ("sm__cycles_elapsed.avg", "cycle"),
        ("lts__cycles_elapsed.avg.per_second", "cycle/s"),
        ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "%"),
        ("Executable", None),
    ],
)
def test_column_unit_follows_the_metric_name(column, unit):
    assert column_unit(column) == unit


def test_every_collected_metric_has_a_unit():
    assert all(column_unit(metric) for metric in ROOFLINE_METRICS)


def test_units_round_trip_through_the_csv_header():
    frame = pd.DataFrame(columns=["Executable", "FLOP", "FLOP FP32", *ROOFLINE_METRICS])
    annotated = with_units(frame)
    assert "Arithmetic Intensity [FLOP/Byte]" in with_units(
        pd.DataFrame(columns=["Arithmetic Intensity"])
    )
    assert "Executable" in annotated
    assert list(without_units(annotated).columns) == list(frame.columns)


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

    assert "ncu" in command
    assert "--metrics" in command and "dram__bytes.sum" in command
    # The profiler appends its own suffix, so it must not be passed in.
    assert str(tmp_path / "out" / "polyhedral_cuda") in command
    assert command.index(str(tmp_path / "polyhedral_cuda")) > command.index("--export")
    assert f"--benchmark_out={tmp_path / 'out' / 'polyhedral_cuda.json'}" in command


def test_profile_command_disables_aslr_by_default(tmp_path, monkeypatch):
    # Google Benchmark re-executes the process to drop ASLR, and Nsight Compute
    # hangs on some binaries when it follows that exec. setarch only exists on
    # Linux, so stub the lookup to keep the test independent of the host.
    monkeypatch.setattr(
        "ppbcc.profiling.runner.shutil.which",
        lambda name: "/usr/bin/setarch" if name == "setarch" else None,
    )
    command = profile_command("ncu", tmp_path / "matMul_kokkos", tmp_path / "r.ncu-rep")
    assert command[:1] == aslr_prefix()[:1] == ["/usr/bin/setarch"]
    assert "-R" in command[: command.index("ncu")]

    plain = profile_command(
        "ncu", tmp_path / "matMul_kokkos", tmp_path / "r.ncu-rep", disable_aslr=False
    )
    assert plain[0] == "ncu"


def test_profile_command_without_setarch_runs_ncu_directly(tmp_path, monkeypatch):
    monkeypatch.setattr("ppbcc.profiling.runner.shutil.which", lambda name: None)
    assert aslr_prefix() == []
    command = profile_command("ncu", tmp_path / "matMul_kokkos", tmp_path / "r.ncu-rep")
    assert command[0] == "ncu"


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
            REGION: ["", "evaluate", "evaluate"],
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


@pytest.mark.parametrize(
    "stack, expected",
    [
        ("", ""),
        ('123  "<default domain>:evaluate:none:none:none:none:none:none" ', "evaluate"),
        # A library's own domain sits inside the benchmark's range and must not
        # win: CCCL brackets every thrust::reduce with one of these.
        (
            '123  "<default domain>:evaluate:none:none:none:none:none:none" '
            ' "CCCL:cub::DeviceReduce::Reduce:none:none:none:none:REGISTERED:x" ',
            "evaluate",
        ),
        ('123  "CCCL:thrust::reduce:none:none:none:none:REGISTERED:x" ', ""),
    ],
)
def test_nvtx_region_reads_the_benchmarks_own_domain(stack, expected):
    assert _nvtx_region(stack) == expected


def test_filter_regions_drops_the_launches_outside_every_region():
    filtered = filter_regions(_points())
    assert list(filtered[KERNEL]) == ["main", "reduce"]


def test_filter_regions_keeps_an_executable_without_any_region():
    frame = _points()
    frame[REGION] = ""
    assert len(filter_regions(frame)) == 3


def test_filter_regions_keeps_everything_when_asked():
    assert len(filter_regions(_points(), keep_unlabelled=True)) == 3


def test_select_regions_keeps_only_the_matching_ones():
    frame = _points()
    frame[REGION] = ["init", "evaluate", "evaluate"]
    assert list(select_regions(frame, ["evaluate"])[KERNEL]) == ["main", "reduce"]
    assert len(select_regions(frame, [])) == 3


def test_aggregate_kernels_groups_by_region():
    frame = filter_regions(_points())
    frame.loc[frame[KERNEL] == "reduce", REGION] = "init"
    aggregated = aggregate_kernels(frame, "sum")
    assert sorted(aggregated[REGION]) == ["evaluate", "init"]


def test_aggregate_kernels_sums_work_time_and_traffic():
    aggregated = aggregate_kernels(filter_regions(_points()), "sum")

    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row[FLOP] == pytest.approx(1000.0)
    assert row[MEMORY_TRAFFIC] == pytest.approx(150.0)
    assert row[ARITHMETIC_INTENSITY] == pytest.approx(1000.0 / 150.0)
    assert row[PERFORMANCE] == pytest.approx(1000.0 / 3e-5)
    # The ceilings are properties of the device, not something to add up.
    assert row[PEAK_PERFORMANCE] == pytest.approx(2e13)


def test_aggregate_kernels_dominant_keeps_the_longest_kernel():
    aggregated = aggregate_kernels(filter_regions(_points()), "dominant")
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


def test_plot_roofline_leaves_points_unlabelled_and_centres_the_titles():
    figure = plot_roofline(
        aggregate_kernels(_points(), "none").iloc[1:],
        hardware="NVIDIA RTX5080",
        problem_title="Polyhedral",
        subtitle="one point per kernel launch",
        show_legend=False,
    )
    axis = figure.axes[0]
    assert axis.get_legend() is None
    # Only the two ceiling annotations and the ridge point remain, no per-point label.
    assert len(axis.texts) == 3
    position = axis.get_position()
    assert figure._suptitle.get_position()[0] == pytest.approx((position.x0 + position.x1) / 2.0)
    assert all(label.get_rotation() == 90 for label in axis.get_yticklabels())


def test_plot_roofline_rejects_data_without_a_usable_point():
    empty = _points().iloc[:1]
    with pytest.raises(ValueError):
        plot_roofline(empty)


def test_profile_cli_parses_the_analysis_defaults():
    args = build_parser().parse_args(["-r", "polyhedral_.*", "--roofline"])
    assert args.regex == ["polyhedral_.*"]
    # --roofline takes an optional path, so its presence is "not None".
    assert args.roofline == ""
    assert args.aggregate == "sum"
    assert args.memory_level == "dram"
    # --regex is only required without --from-csv, which selects rows instead.
    assert build_parser().parse_args(["--from-csv", "a.csv"]).regex == []


def test_profile_cli_no_legend_flag():
    assert build_parser().parse_args(["-r", "x", "-l"]).no_legend
    assert not build_parser().parse_args(["-r", "x"]).no_legend


def test_profile_cli_roofline_takes_an_optional_path():
    args = build_parser().parse_args(["-r", "x", "--roofline", "out.pdf"])
    assert args.roofline == "out.pdf"
    assert build_parser().parse_args(["-r", "x"]).roofline is None


def test_parse_options_fills_in_the_backends_defaults():
    options = parse_options([], "nsys")
    assert options == {
        "metric-set": "gb20x",
        "frequency": 100_000,
        "iterations": 20,
        "activity-ratio": 0.7,
    }


def test_parse_options_converts_and_appends():
    options = parse_options(["iterations=5"], "nsys")
    assert options["iterations"] == 5
    options = parse_options(["ncu-arg=--a", "ncu-arg=--b"], "ncu")
    assert options["ncu-arg"] == ["--a", "--b"]
    assert parse_options(["metrics=a,b"], "ncu")["metrics"] == ["a", "b"]


@pytest.mark.parametrize(
    "pairs, profiler",
    [
        (["iterations"], "nsys"),          # no '='
        (["nonsense=1"], "nsys"),          # unknown name
        (["iterations=1"], "ncu"),         # belongs to another backend
        (["iterations=many"], "nsys"),     # unconvertible value
    ],
)
def test_parse_options_rejects_bad_input(pairs, profiler):
    with pytest.raises(ValueError):
        parse_options(pairs, profiler)


# --------------------------------------------------------------------------- #
# LIKWID backend
# --------------------------------------------------------------------------- #

#: One region, one group, three counters, as an instrumented binary writes it.
_FLOP_MARKER = "1 1 1\n0:matmul-0\n0 0 0 1 1.820479e+01 3 1.0e+02 2.0e+02 4.0e+02\n"
_DRAM_MARKER = "1 1 1\n0:matmul-0\n0 0 0 1 1.820479e+01 1 8.0e+02\n"


def test_parse_marker_file_reads_tags_time_and_counters(tmp_path):
    path = tmp_path / "matMul_cuda.flops.likwid-marker"
    path.write_text(_FLOP_MARKER)
    regions = likwid.parse_marker_file(path)
    assert set(regions) == {"matmul"}
    assert regions["matmul"]["time"] == pytest.approx(18.20479)
    assert regions["matmul"]["values"] == [100.0, 200.0, 400.0]


def test_parse_marker_file_handles_an_empty_file(tmp_path):
    path = tmp_path / "empty.likwid-marker"
    path.write_text("")
    assert likwid.parse_marker_file(path) == {}


def test_event_string_follows_the_requested_precision():
    assert "OP_FFMA" in likwid.event_string("flops", "fp32")
    assert "OP_DFMA" in likwid.event_string("flops", "fp64")
    # The memory group has no precision variants.
    assert likwid.event_string("memory", "fp64") == "DRAM_BYTES_SUM:GPU0"


def test_event_string_numbers_the_counters_in_order():
    events = likwid.event_string("flops", "fp32").split(",")
    assert [event.split(":")[1] for event in events] == ["GPU0", "GPU1", "GPU2"]


def _write_likwid_run(directory, name="matMul_cuda"):
    (directory / f"{name}.flops.likwid-marker").write_text(_FLOP_MARKER)
    (directory / f"{name}.memory.likwid-marker").write_text(_DRAM_MARKER)
    (directory / f"{name}.json").write_text(
        json.dumps(
            {
                "context": {"paradigm": "CUDA", "float_type": "32"},
                "benchmarks": [{"name": "MatrixMultiplication/16384"}],
            }
        )
    )


def test_load_reports_derives_the_roofline_quantities(tmp_path):
    _write_likwid_run(tmp_path)
    frame = likwid.load_reports(
        [Path("matMul_cuda")], tmp_path, hardware="RTX5080",
        peak_performance=1.0e13, peak_bandwidth=1.0e12,
    )
    assert len(frame) == 1
    row = frame.iloc[0]
    scale = likwid.COUNTER_SCALE
    # add + mul + 2 * fma, every counter corrected by COUNTER_SCALE.
    expected_flop = scale * (100.0 + 200.0 + 2.0 * 400.0)
    assert row[FLOP] == pytest.approx(expected_flop)
    assert row[MEMORY_TRAFFIC] == pytest.approx(scale * 800.0)
    assert row[ARITHMETIC_INTENSITY] == pytest.approx(expected_flop / (scale * 800.0))
    assert row[PERFORMANCE] == pytest.approx(expected_flop / 18.20479)
    assert row[PEAK_PERFORMANCE] == 1.0e13
    assert row[KERNEL] == "matmul"
    assert row[PARADIGM] == "CUDA"


def test_load_reports_leaves_intensity_untouched_by_the_scale(tmp_path):
    _write_likwid_run(tmp_path)
    kwargs = dict(hardware="", peak_performance=None, peak_bandwidth=None)
    corrected = likwid.load_reports([Path("matMul_cuda")], tmp_path, scale=2.0, **kwargs)
    raw = likwid.load_reports([Path("matMul_cuda")], tmp_path, scale=1.0, **kwargs)
    # The correction cancels in a ratio of two equally scaled counters.
    assert corrected.iloc[0][ARITHMETIC_INTENSITY] == pytest.approx(
        raw.iloc[0][ARITHMETIC_INTENSITY]
    )
    # It does not cancel in an absolute rate.
    assert corrected.iloc[0][PERFORMANCE] == pytest.approx(2.0 * raw.iloc[0][PERFORMANCE])


def test_load_reports_skips_an_executable_without_marker_files(tmp_path):
    frame = likwid.load_reports([Path("matMul_ocl")], tmp_path)
    assert frame.empty
    assert list(frame.columns) == PROFILE_COLUMN_LIST


def test_find_profiled_lists_executables_from_marker_files(tmp_path):
    _write_likwid_run(tmp_path, "matMul_cuda")
    _write_likwid_run(tmp_path, "polyhedral_kokkos")
    assert [path.name for path in likwid.find_profiled(tmp_path)] == [
        "matMul_cuda",
        "polyhedral_kokkos",
    ]


def test_find_profiled_is_empty_without_marker_files(tmp_path):
    assert likwid.find_profiled(tmp_path) == []


def test_load_reports_uses_the_least_perturbed_region_time(tmp_path):
    # Collecting three SMSP counters perturbs the kernel far more than one DRAM
    # counter, so the groups disagree; the smaller time is the honest one.
    (tmp_path / "matMul_cuda.flops.likwid-marker").write_text(
        "1 1 1\n0:matmul-0\n0 0 0 1 3.6e+01 3 1.0e+02 2.0e+02 4.0e+02\n"
    )
    (tmp_path / "matMul_cuda.memory.likwid-marker").write_text(
        "1 1 1\n0:matmul-0\n0 0 0 1 3.6e+00 1 8.0e+02\n"
    )
    (tmp_path / "matMul_cuda.json").write_text(
        json.dumps({"context": {"paradigm": "CUDA", "float_type": "32"},
                    "benchmarks": [{"name": "MatrixMultiplication/16384"}]})
    )
    frame = likwid.load_reports([Path("matMul_cuda")], tmp_path)
    assert frame.iloc[0][DURATION] == pytest.approx(3.6)


#: Two regions in one file: the header counts GPUs first and regions second, and
#: the region index leads each data line. Both are ambiguous with a single region.
_TWO_REGION_MARKER = (
    "1 2 1\n"
    "0:matmul-cublas-0\n"
    "1:matmul-naive-0\n"
    "0 0 0 1 2.663338e+00 3 0.0e+00 1.0e+02 2.0e+02\n"
    "1 0 0 1 1.783196e+01 3 0.0e+00 3.0e+02 4.0e+02\n"
)


def test_parse_marker_file_separates_two_regions(tmp_path):
    path = tmp_path / "matMul_cuda.flops.likwid-marker"
    path.write_text(_TWO_REGION_MARKER)
    regions = likwid.parse_marker_file(path)
    assert set(regions) == {"matmul-cublas", "matmul-naive"}
    assert regions["matmul-cublas"]["time"] == pytest.approx(2.663338)
    assert regions["matmul-cublas"]["values"] == [0.0, 100.0, 200.0]
    assert regions["matmul-naive"]["time"] == pytest.approx(17.83196)
    assert regions["matmul-naive"]["values"] == [0.0, 300.0, 400.0]


# --------------------------------------------------------------------------- #
# Nsight Systems backend
# --------------------------------------------------------------------------- #


def _sampled_report(path, samples, metric_names=None, regions=None):
    """Write a minimal stand-in for an ``nsys export --type sqlite`` database.

    Args:
        path: Destination database.
        samples: (timestamp, metric name, value) triples.
        metric_names: Metric names to declare, defaulting to the three the
            backend needs.
        regions: (name, start, end) triples for NVTX_EVENTS, or ``None`` to
            leave the table out entirely, as a run without --trace=nvtx does.
    """
    import sqlite3

    names = metric_names or [
        nsys.COMPUTE_ACTIVITY_METRIC,
        nsys.DRAM_READ_METRIC,
        nsys.DRAM_WRITE_METRIC,
    ]
    identifiers = {name: index for index, name in enumerate(names)}
    connection = sqlite3.connect(path)
    connection.execute(
        "create table TARGET_INFO_GPU_METRICS (metricId int, metricName text)"
    )
    connection.executemany(
        "insert into TARGET_INFO_GPU_METRICS values (?, ?)",
        [(index, name) for name, index in identifiers.items()],
    )
    connection.execute(
        "create table GPU_METRICS (timestamp int, metricId int, value real)"
    )
    connection.executemany(
        "insert into GPU_METRICS values (?, ?, ?)",
        [
            (timestamp, identifiers[name], value)
            for timestamp, name, value in samples
            if name in identifiers
        ],
    )
    if regions is not None:
        connection.execute(
            "create table NVTX_EVENTS "
            "(text text, start int, end int, domainId int)"
        )
        connection.executemany(
            "insert into NVTX_EVENTS values (?, ?, ?, 0)", regions
        )
    connection.commit()
    connection.close()


def _busy(start, count, step=10_000, activity=90.0, dram=50.0):
    """Generate samples for one busy stretch.

    Args:
        start: Timestamp of the first sample, in nanoseconds.
        count: Number of samples.
        step: Sampling interval in nanoseconds.
        activity: Compute occupancy in percent.
        dram: DRAM read throughput in percent.

    Returns:
        Triples for :func:`_sampled_report`.
    """
    return [
        (start + index * step, name, value)
        for index in range(count)
        for name, value in (
            (nsys.COMPUTE_ACTIVITY_METRIC, activity),
            (nsys.DRAM_READ_METRIC, dram),
            (nsys.DRAM_WRITE_METRIC, 0.0),
        )
    ]


def test_nsys_measure_finds_the_busy_stretch(tmp_path):
    database = tmp_path / "matMul_ocl.sqlite"
    # 1 ms idle, 1 ms busy, 1 ms idle: only the middle stretch counts.
    _sampled_report(
        database,
        _busy(0, 100, activity=0.0, dram=0.0)
        + _busy(1_000_000, 100)
        + _busy(2_000_000, 100, activity=0.0, dram=0.0),
    )
    # Without NVTX_EVENTS the whole run is one unnamed region.
    duration, traffic, windows, _ = nsys._measure(database, peak_bandwidth=1.0e12)[""]
    assert windows == 1
    assert duration == pytest.approx(990e-6, rel=0.02)
    # 50 % of 1 TB/s over ~1 ms.
    assert traffic == pytest.approx(0.5e12 * 990e-6, rel=0.05)


def test_nsys_load_reports_reports_a_one_time_region_as_measured(tmp_path):
    """A region entered once in both runs is a one-time phase, not per-iteration."""
    for suffix, busy in ((".sqlite", 4), (".ref.sqlite", 4)):
        _sampled_report(
            tmp_path / f"polyhedral_ocl{suffix}",
            _busy(0, 100),
            regions=[("init", 0, 1_500_000)],
        )
    (tmp_path / "polyhedral_ocl.json").write_text("{}")
    frame = nsys.load_reports(
        [tmp_path / "polyhedral_ocl.sqlite"],
        tmp_path,
        peak_bandwidth=1.0e12,
        iterations=20,
    )
    # Subtracting would leave zero and drop the row; dividing by 20 would be a
    # twentieth of the truth. Neither happens.
    assert list(frame[REGION]) == ["init"]
    assert frame.iloc[0][DURATION] == pytest.approx(990e-6, rel=0.02)


def test_nsys_measure_splits_the_run_by_nvtx_region(tmp_path):
    database = tmp_path / "polyhedral_ocl.sqlite"
    # Two busy stretches, each inside its own range; a third outside both.
    _sampled_report(
        database,
        _busy(0, 100) + _busy(2_000_000, 100) + _busy(4_000_000, 100),
        regions=[("init", 0, 1_500_000), ("evaluate", 1_900_000, 3_500_000)],
    )
    measured = nsys._measure(database, peak_bandwidth=1.0e12)
    assert sorted(measured) == ["evaluate", "init"]
    for duration, _, windows, entries in measured.values():
        assert windows == 1
        assert entries == 1
        assert duration == pytest.approx(990e-6, rel=0.02)


def test_nsys_measure_drops_low_occupancy_windows(tmp_path):
    database = tmp_path / "matMul_vulkan.sqlite"
    # A transfer phase at half the dispatch's occupancy must not be counted:
    # some runtimes move buffers with a compute shader.
    _sampled_report(
        database,
        _busy(0, 50, activity=45.0)
        + _busy(1_000_000, 100, activity=95.0)
        + _busy(3_000_000, 50, activity=45.0),
    )
    duration, _, windows, _ = nsys._measure(database, peak_bandwidth=1.0e12)[""]
    assert windows == 1
    assert duration == pytest.approx(990e-6, rel=0.02)
    # Without the filter all three stretches would be summed.
    everything, _, count, _ = nsys._measure(
        database, peak_bandwidth=1.0e12, activity_ratio=0.0
    )[""]
    assert count == 3
    assert everything > duration


def test_nsys_measure_reports_nothing_without_the_needed_metrics(tmp_path):
    database = tmp_path / "matMul_ocl.sqlite"
    _sampled_report(database, [], metric_names=["GPC Clock Frequency [MHz]"])
    assert nsys._measure(database, peak_bandwidth=1.0e12) is None


def test_nsys_analytic_flop_takes_the_first_matching_rule():
    rules = ["polyhedral_.*=4.1e9", "matMul_.*=1.37e11", "matMul_ocl=1.0"]
    assert nsys.analytic_flop("matMul_ocl", rules) == pytest.approx(1.37e11)
    assert nsys.analytic_flop("polyhedral_vulkan", rules) == pytest.approx(4.1e9)
    assert nsys.analytic_flop("vec_ocl", rules) is None
    assert nsys.analytic_flop("matMul_ocl", ["broken"]) is None


def test_nsys_analytic_flop_can_address_one_region():
    # The polyhedral init kernel does different work from the evaluation, so a
    # rule has to be able to name the region rather than only the binary.
    rules = [r"polyhedral_.*\[evaluate\]=4.1e9"]
    assert nsys.analytic_flop("polyhedral_ocl[evaluate]", rules) == pytest.approx(4.1e9)
    assert nsys.analytic_flop("polyhedral_ocl[init]", rules) is None
    # A rule naming only the executable still matches every one of its regions.
    assert nsys.analytic_flop(
        "polyhedral_ocl[init]", ["polyhedral_=1.0"]
    ) == pytest.approx(1.0)


def test_nsys_load_reports_subtracts_the_reference_run(tmp_path):
    # 20 iterations of a 1 ms kernel after 5 ms of one-time setup, against a
    # reference run of the same setup plus one kernel.
    _sampled_report(tmp_path / "polyhedral_vulkan.sqlite", _busy(0, 2500))
    _sampled_report(tmp_path / "polyhedral_vulkan.ref.sqlite", _busy(0, 600))
    (tmp_path / "polyhedral_vulkan.json").write_text(
        json.dumps(
            {
                "context": {"paradigm": "Vulkan", "float_type": "32"},
                "benchmarks": [{"name": "Polyhedral-SHAPE_SFM_3M"}],
            }
        )
    )
    frame = nsys.load_reports(
        [tmp_path / "polyhedral_vulkan.sqlite"],
        tmp_path,
        peak_bandwidth=1.0e12,
        flop_rules=["polyhedral_.*=1.0e9"],
        iterations=20,
    )
    row = frame.iloc[0]
    assert row[PARADIGM] == "Vulkan"
    # (24.99 ms - 5.99 ms) / 19 iterations.
    assert row[DURATION] == pytest.approx(1.0e-3, rel=0.02)
    assert row[PERFORMANCE] == pytest.approx(1.0e12, rel=0.02)


def test_nsys_find_profiled_skips_the_reference_runs(tmp_path):
    for name in ("a.sqlite", "a.ref.sqlite", "b.sqlite"):
        (tmp_path / name).touch()
    assert [path.name for path in nsys.find_profiled(tmp_path)] == [
        "a.sqlite",
        "b.sqlite",
    ]


# --------------------------------------------------------------------------- #
# Nsight Graphics backend
# --------------------------------------------------------------------------- #


def _trace(directory, frame_time_ms, dram_bytes, compute_cycles=1.0e8):
    """Write a minimal stand-in for an ``ngfx --auto-export`` result.

    Args:
        directory: The ``--output-dir`` of the run.
        frame_time_ms: Value of the exported GPU frame time.
        dram_bytes: Value of the exported ``dram__sectors.sum``.
        compute_cycles: Value of the compute-cycle counter.
    """
    (directory / "BASE").mkdir(parents=True, exist_ok=True)
    (directory / ngfx.FRAME_TIME_FILE).write_text(f"GPU frame time\t{frame_time_ms}\n")
    (directory / ngfx.FRAME_METRICS_FILE).write_text(
        f"{ngfx.DRAM_METRIC}\t{dram_bytes}\n"
        f"{ngfx.COMPUTE_CYCLES_METRIC}\t{compute_cycles}\n"
        "not a number\tn/a\n"
    )


def test_ngfx_load_reports_reads_the_exported_tables(tmp_path):
    _trace(tmp_path / "matMul_vulkan", 40.0, 8.0e9)
    (tmp_path / "matMul_vulkan.json").write_text(
        json.dumps(
            {
                "context": {"paradigm": "Vulkan", "float_type": "32"},
                "benchmarks": [{"name": "MatrixMultiplication/4096"}],
            }
        )
    )
    frame = ngfx.load_reports(
        [tmp_path / "matMul_vulkan"],
        tmp_path,
        peak_bandwidth=9.592e11,
        flop_rules=["matMul_.*=1.6e11"],
    )
    row = frame.iloc[0]
    assert row[PARADIGM] == "Vulkan"
    assert row[DURATION] == pytest.approx(0.04)
    assert row[MEMORY_TRAFFIC] == pytest.approx(8.0e9)
    assert row[ARITHMETIC_INTENSITY] == pytest.approx(20.0)
    assert row[PERFORMANCE] == pytest.approx(4.0e12)


def test_ngfx_find_profiled_skips_the_probe_directories(tmp_path):
    for name in ("matMul_vulkan", "matMul_vulkan.submit0", "matMul_vulkan.submit1"):
        _trace(tmp_path / name, 1.0, 1.0)
    assert [path.name for path in ngfx.find_profiled(tmp_path)] == ["matMul_vulkan"]


def test_ngfx_profile_command_bounds_the_trace_by_submit_index(tmp_path):
    command = ngfx.profile_command(
        "ngfx", tmp_path / "matMul_vulkan", tmp_path, tmp_path / "out",
        "Blackwell GB20x", submit=1,
    )
    assert command[command.index("--start-after-submits") + 1] == "1"
    assert command[command.index("--limit-to-submits") + 1] == "1"
    assert int(command[command.index("--max-duration-ms") + 1]) <= 10_000
