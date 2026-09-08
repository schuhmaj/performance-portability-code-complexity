"""Nsight Graphics GPU Trace backend: hardware counters for the Vulkan paradigms.

Nsight Systems (:mod:`ppbcc.profiling.nsys`) sees Vulkan work, but only through
device-wide *sampling*: it reports what fraction of peak each unit was running
at, averaged over a sampling interval. Nsight Graphics' GPU Trace Profiler reads
the same performance monitors the way Nsight Compute does -- as counter sums
over a bounded region -- and it is the one Nvidia tool that does so for a Vulkan
workload. That makes it the independent check on the sampled numbers, which is
the role it plays here.

Its unit of attribution is the *trace*, not the dispatch: with no swapchain
there are no frames, and the per-regime table it exports stays empty. The region
is bounded by submit index instead: ``--start-after-submits k --limit-to-submits
n`` traces exactly the k-th submission. A ``PPB_PROFILING`` binary issues a
handful of submits -- upload, dispatch, download -- so ``--ngfx-submit auto``
simply traces each of the first few and keeps the one that spent the most cycles
in compute.

As with Nsight Systems, the FLOP count is not measured: the Blackwell metric set
carries pipe utilisations rather than instruction counts, so it comes from the
analytic work model (``--analytic-flop``).
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from ppbcc.constants import (
    ARITHMETIC_INTENSITY,
    BENCHMARK_PROBLEM,
    DURATION,
    EXECUTABLE,
    FLOP,
    HARDWARE,
    KERNEL,
    KERNEL_ID,
    KERNEL_SIGNATURE,
    MEMORY_LEVEL,
    MEMORY_TRAFFIC,
    PARADIGM,
    PEAK_BANDWIDTH,
    PEAK_PERFORMANCE,
    PERFORMANCE,
    PRECISION,
    PROFILE_COLUMN_LIST,
    REGION,
)
from ppbcc.profiling.metrics import PRECISIONS
from ppbcc.profiling.nsys import analytic_flop

#: Where the tool writes the exported tables, relative to ``--output-dir``.
FRAME_TIME_FILE = Path("BASE") / "FRAME.xls"
#: Trace-wide counter sums, one ``name<TAB>value`` line each.
FRAME_METRICS_FILE = Path("BASE") / "GPUTRACE_FRAME.xls"

#: Total DRAM traffic of the trace. Despite the name the exported value is in
#: bytes: dividing it by the frame time reproduces the accompanying
#: ``dram__sectors.sum.per_second`` in GB/s and the matching
#: ``pct_of_peak_sustained_elapsed`` against a 960 GB/s peak.
DRAM_METRIC = "dram__sectors.sum"
#: Cycles the graphics/compute engine spent on compute work, used by
#: ``--ngfx-submit auto`` to recognise the dispatch among the submissions.
COMPUTE_CYCLES_METRIC = "gr__compute_cycles_active_queue_sync.sum"

#: Nsight Graphics refuses longer traces.
MAXIMUM_DURATION_MS = 9_000


def find_ngfx(explicit: str | None = None) -> str:
    """Locate the ``ngfx`` executable.

    Args:
        explicit: User-supplied path, or ``None`` to search ``PATH`` and the
            default installation directory.

    Returns:
        Path to the Nsight Graphics CLI.

    Raises:
        FileNotFoundError: If it cannot be found.
    """
    if explicit:
        if Path(explicit).is_file():
            return explicit
        raise FileNotFoundError(f"No Nsight Graphics CLI at {explicit!r}")
    resolved = shutil.which("ngfx")
    if resolved is not None:
        return resolved
    # The installer does not put ngfx on PATH; only the UI launcher.
    candidates = sorted(
        Path("/opt/nvidia/nsight-graphics-for-linux").glob(
            "*/host/linux-desktop-nomad-x64/ngfx"
        )
    )
    if candidates:
        logger.debug(f"Using profiler: {candidates[-1]}")
        return str(candidates[-1])
    raise FileNotFoundError(
        "Could not find the Nsight Graphics CLI ('ngfx'). Install Nsight "
        "Graphics or pass --ngfx with an explicit path."
    )


def profile_command(
    ngfx: str,
    target: Path,
    working_directory: Path,
    output_dir: Path,
    architecture: str,
    submit: int,
    submits: int = 1,
    metric_set: int = 0,
    benchmark_report: Path | None = None,
) -> list[str]:
    """Build the ``ngfx`` command line for one benchmark executable.

    Args:
        ngfx: Nsight Graphics CLI.
        target: Benchmark binary to profile.
        working_directory: Directory the binary is launched in; the benchmarks
            resolve their input meshes relative to it.
        output_dir: Directory the trace and the exported tables land in.
        architecture: Architecture whose metric set is selected, e.g.
            ``"Blackwell GB20x"``. ``ngfx --help-all`` lists the names.
        submit: Index of the first traced queue submission.
        submits: How many submissions to trace.
        metric_set: Index of the metric set for ``architecture``.
        benchmark_report: Where the target writes its Google-Benchmark JSON
            report, which carries the paradigm and precision.

    Returns:
        The argument vector to execute.
    """
    command = [
        ngfx,
        "--activity",
        "GPU Trace Profiler",
        "--exe",
        str(target.resolve()),
        "--dir",
        str(working_directory),
        "--output-dir",
        str(output_dir),
        "--auto-export",
        "--start-after-submits",
        str(submit),
        "--limit-to-submits",
        str(submits),
        "--max-duration-ms",
        str(MAXIMUM_DURATION_MS),
        "--architecture",
        architecture,
        "--metric-set-id",
        str(metric_set),
        "--collect-screenshot",
        "0",
    ]
    if benchmark_report is not None:
        command += [
            "--args",
            f"--benchmark_out={benchmark_report} --benchmark_out_format=json",
        ]
    return command


def _read_metrics(directory: Path) -> dict[str, float]:
    """Read the exported trace-wide counters of one run.

    Args:
        directory: The ``--output-dir`` of that run.

    Returns:
        Metric name -> value, plus ``"frame_time"`` in seconds. Empty if the
        run exported nothing.
    """
    metrics_file = directory / FRAME_METRICS_FILE
    if not metrics_file.is_file():
        return {}
    metrics: dict[str, float] = {}
    with open(metrics_file, newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 2:
                continue
            try:
                metrics[row[0]] = float(row[1])
            except ValueError:
                continue
    frame_file = directory / FRAME_TIME_FILE
    if frame_file.is_file():
        with open(frame_file, newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) >= 2 and row[0].startswith("GPU frame time"):
                    try:
                        metrics["frame_time"] = float(row[1]) * 1e-3
                    except ValueError:
                        pass
    return metrics


def run_profiles(
    executables: list[Path],
    output_dir: Path,
    ngfx: str,
    working_directory: Path,
    architecture: str,
    submit: str = "auto",
    probes: int = 4,
    metric_set: int = 0,
    force: bool = False,
    stream_output: bool = False,
    timeout: float | None = None,
) -> list[Path]:
    """Trace every executable and keep the submission that did the compute work.

    Args:
        executables: Benchmark binaries to profile.
        output_dir: Directory the per-executable trace directories are created in.
        ngfx: Nsight Graphics CLI.
        working_directory: Directory the binaries are launched in.
        architecture: Architecture whose metric set is selected.
        submit: ``auto`` to probe the first ``probes`` submissions and keep the
            one with the most compute cycles, or an explicit index.
        probes: How many submissions ``auto`` tries.
        metric_set: Index of the metric set for ``architecture``.
        force: Re-trace even when a result already exists.
        stream_output: Forward the tool's output to the logger at TRACE level.
        timeout: Wall-clock limit per trace in seconds, or ``None``.

    Returns:
        The per-executable directories holding the kept trace, in execution order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    kept: list[Path] = []
    environment = dict(os.environ)
    # The CLI still initialises Qt, which has no display in a batch run.
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")

    for index, target in enumerate(executables, start=1):
        destination = output_dir / target.name
        if (destination / FRAME_METRICS_FILE).is_file() and not force:
            logger.warning(
                f"{destination.name} already traced, reusing it (use --force to re-trace)"
            )
            kept.append(destination)
            continue
        indices = range(probes) if submit == "auto" else [int(submit)]
        best: tuple[float, Path] | None = None
        for candidate in indices:
            probe = output_dir / f"{target.name}.submit{candidate}"
            shutil.rmtree(probe, ignore_errors=True)
            probe.mkdir(parents=True, exist_ok=True)
            command = profile_command(
                ngfx,
                target,
                working_directory,
                probe,
                architecture,
                candidate,
                metric_set=metric_set,
                benchmark_report=output_dir / f"{target.name}.json",
            )
            logger.info(
                f"[{index}/{len(executables)}] Tracing {target.name}, submit {candidate} ..."
            )
            logger.debug(f"Command: {' '.join(command)}")
            try:
                completed = subprocess.run(
                    command,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if stream_output:
                    for line in (completed.stdout or "").splitlines():
                        logger.trace(f"[{target.name}] {line}")
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Timed out after {timeout} s tracing {target.name} "
                    f"(submit {candidate})"
                )
                continue
            metrics = _read_metrics(probe)
            if not metrics:
                logger.debug(f"{target.name}: submit {candidate} exported nothing")
                continue
            cycles = metrics.get(COMPUTE_CYCLES_METRIC, 0.0)
            logger.debug(
                f"{target.name}: submit {candidate} -> "
                f"{metrics.get('frame_time', float('nan')) * 1e3:.3f} ms, "
                f"{cycles:.4g} compute cycles"
            )
            if best is None or cycles > best[0]:
                best = (cycles, probe)
        if best is None or best[0] <= 0.0:
            logger.warning(
                f"{target.name}: no traced submission did compute work. The "
                "dispatch may be beyond the probed range; raise --ngfx-probes "
                "or pass --ngfx-submit explicitly."
            )
            continue
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(best[1], destination)
        logger.success(f"Finished {target.name} -> {destination}")
        kept.append(destination)
    return kept


def find_profiled(report_dir: Path) -> list[Path]:
    """List the traces a previous batch left behind.

    Probe directories carry a ``.submitN`` suffix and are skipped; only the
    kept trace of each executable is returned.

    Args:
        report_dir: Directory holding the per-executable trace directories.

    Returns:
        The directories, sorted by name.
    """
    return sorted(
        path
        for path in report_dir.glob("*")
        if path.is_dir()
        and ".submit" not in path.name
        and (path / FRAME_METRICS_FILE).is_file()
    )


def load_reports(
    traces: list[Path],
    report_dir: Path,
    hardware: str = "",
    precision: str = "auto",
    peak_performance: float | None = None,
    peak_bandwidth: float | None = None,
    flop_rules: list[str] | None = None,
) -> pd.DataFrame:
    """Turn the traces of a batch into the tidy roofline table.

    Args:
        traces: Per-executable trace directories.
        report_dir: Directory holding the Google-Benchmark reports; they supply
            paradigm, precision and problem.
        hardware: Label for the hardware column.
        precision: ``auto`` or an explicit key of
            :data:`~ppbcc.profiling.metrics.PRECISIONS`.
        peak_performance: Compute ceiling in FLOP/s; Nsight Graphics reports
            percentages of peak but not the peak itself.
        peak_bandwidth: Memory ceiling in bytes/s, for the same reason.
        flop_rules: ``<regex>=<value>`` rules supplying the analytic FLOP count.

    Returns:
        One row per traced submission, ordered as
        :data:`~ppbcc.constants.PROFILE_COLUMN_LIST`.
    """
    from ppbcc.profiling.reports import _load_context

    rows: list[dict[str, object]] = []
    for trace in traces:
        name = trace.name
        metrics = _read_metrics(trace)
        if not metrics:
            logger.warning(f"{name}: no exported metrics, skipping")
            continue
        paradigm, built_precision, problem = _load_context(report_dir / f"{name}.json")
        key = precision
        if key == "auto":
            key = {"32": "fp32", "64": "fp64", "16": "fp16"}.get(
                str(built_precision), "fp32"
            )
        duration = metrics.get("frame_time", float("nan"))
        traffic = metrics.get(DRAM_METRIC, float("nan"))
        # A traced submission carries no region name, so only a rule naming the
        # executable can match here.
        flop = analytic_flop(f"{name}[]", flop_rules or [])
        if flop is None:
            logger.warning(
                f"{name}: no --analytic-flop rule matched; the row carries no work"
            )
        rows.append(
            {
                HARDWARE: hardware,
                BENCHMARK_PROBLEM: problem,
                PARADIGM: paradigm,
                PRECISION: PRECISIONS[key][1],
                EXECUTABLE: name,
                KERNEL_ID: 0,
                # A GPU Trace capture is bounded by queue submission index, and
                # a Vulkan submission carries no NVTX range, so this backend
                # cannot name the region it traced. The row stays unlabelled,
                # which is what keeps it in the table.
                REGION: "",
                KERNEL: "traced submission",
                KERNEL_SIGNATURE: f"GPU Trace of one queue submission ({duration * 1e3:.3f} ms)",
                DURATION: duration,
                MEMORY_LEVEL: "DRAM",
                FLOP: flop if flop is not None else float("nan"),
                MEMORY_TRAFFIC: traffic,
                ARITHMETIC_INTENSITY: (
                    flop / traffic if flop is not None and traffic else float("nan")
                ),
                PERFORMANCE: (
                    flop / duration if flop is not None and duration else float("nan")
                ),
                PEAK_PERFORMANCE: peak_performance or float("nan"),
                PEAK_BANDWIDTH: peak_bandwidth or float("nan"),
            }
        )
    if not rows:
        logger.error("No profiling data could be loaded.")
        return pd.DataFrame(columns=PROFILE_COLUMN_LIST)
    frame = pd.DataFrame(rows)
    logger.success(
        f"Loaded {len(frame)} trace(s) across {frame[EXECUTABLE].nunique()} executable(s)"
    )
    # Grid and block size are launch properties no sampling profiler sees.
    return frame[[column for column in PROFILE_COLUMN_LIST if column in frame]]
