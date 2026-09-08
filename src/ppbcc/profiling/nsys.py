"""Nsight Systems backend: roofline quantities for paradigms without a CUDA context.

Nsight Compute and LIKWID both read their counters through CUPTI, which needs a
*current CUDA context*. OpenCL and Vulkan build their own, so neither tool sees
them -- which leaves four of this benchmark's paradigms without a roofline point.

Nsight Systems closes that gap from the other side. ``--gpu-metrics-devices``
programs the GPU's performance monitors in **time-based sampling** mode: the
counters are read device-wide at a fixed frequency, with no context filter and no
kernel boundaries, so the samples cover whatever the GPU is doing regardless of
which API submitted it. What the sampler gives up is attribution -- a sample
knows *when*, not *which kernel* -- and this module buys it back from the
benchmark itself. ``--trace=nvtx`` records the NVTX ranges of
``src/common/Marker.h`` with the same clock as the samples, so every sample
falls inside a named region (``matmul``, ``init``, ``evaluate``) or outside all
of them. Within a region, the stretches where compute warps are in flight are
the kernel.

Two quantities come out of such a window:

* its length, which is the kernel duration, and
* the integral of the DRAM read and write throughputs over it, which is the
  memory traffic.

The third roofline quantity, the FLOP count, is the one the sampler cannot
supply: the metric sets available on this hardware carry pipe *utilisations*,
not instruction counts. It comes from the analytic work model instead
(``--analytic-flop``), which for these benchmarks is not a guess -- Nsight
Compute reproduces it exactly on every CUDA-backed implementation of the same
problem.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
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

#: Suffix of the profiler's own report.
REPORT_SUFFIX = ".nsys-rep"
#: Suffix of the SQLite export the report is read from.
DATABASE_SUFFIX = ".sqlite"

#: Metric names the window detection and the traffic integral need. Nsight
#: Systems numbers its metrics per metric set, so they are resolved by name from
#: ``TARGET_INFO_GPU_METRICS`` rather than hard-coded.
COMPUTE_ACTIVITY_METRIC = "Compute Warps in Flight [Throughput %]"
DRAM_READ_METRIC = "DRAM Read Bandwidth [Throughput %]"
DRAM_WRITE_METRIC = "DRAM Write Bandwidth [Throughput %]"

#: Compute activity above this percentage counts as "a kernel is running".
ACTIVITY_THRESHOLD = 1.0

#: Samples separated by less than this are treated as one window. The sampler
#: reports a per-interval average, so a kernel whose occupancy dips between two
#: waves produces a short run of empty samples inside a single launch. 0.5 ms
#: was calibrated against binaries whose kernel time is known independently:
#: it recovers 41.43 ms for ``matMul_ocl`` (the application's own
#: ``clGetEventProfilingInfo`` delta is 41.45 ms), 38.09 ms for
#: ``matMul_vulkan`` (38.91 ms) and 44.17 ms for ``matMul_cuda`` (Nsight
#: Compute: 43.11 ms), while anything below 0.2 ms splits every one of them.
WINDOW_GAP_SECONDS = 5.0e-4

#: Shortest window that can be a kernel rather than a bootstrap launch.
MINIMUM_WINDOW_SECONDS = 1.0e-4

#: A window whose mean compute occupancy is below this fraction of the busiest
#: window's is treated as data movement rather than as the kernel. Some runtimes
#: move their buffers with a compute shader instead of a copy engine -- Kompute
#: brackets every matrix-multiplication dispatch with two such phases -- and
#: those show up as compute-active windows that the roofline must not count. The
#: separation is wide: the dispatch runs at ~96 % occupancy, the transfers at
#: ~49 %.
ACTIVITY_RATIO = 0.7

#: Infix of the reference run of the two-point measurement, see
#: :func:`run_profiles`.
REFERENCE_INFIX = ".ref"


def find_nsys(explicit: str | None = None) -> str:
    """Locate the ``nsys`` executable.

    Args:
        explicit: User-supplied path, or ``None`` to search ``PATH``.

    Returns:
        Path to the profiler executable.

    Raises:
        FileNotFoundError: If no ``nsys`` executable can be found.
    """
    import shutil

    candidate = explicit or "nsys"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find the Nsight Systems CLI ({candidate!r}). Install the "
            "CUDA toolkit or pass --nsys with an explicit path."
        )
    logger.debug(f"Using profiler: {resolved}")
    return resolved


def profile_command(
    nsys: str,
    target: Path,
    report: Path,
    metric_set: str,
    frequency: int,
    benchmark_report: Path | None = None,
    extra_target_args: list[str] | None = None,
) -> list[str]:
    """Build the ``nsys profile`` command line for one benchmark executable.

    Args:
        nsys: Profiler executable.
        target: Benchmark binary to profile.
        report: Destination report; the profiler appends ``.nsys-rep``.
        metric_set: GPU metric set alias, e.g. ``gb20x`` for a Blackwell
            consumer part. ``nsys profile --gpu-metrics-set=help`` lists them.
        frequency: Sampling frequency in Hz.
        benchmark_report: Where the target writes its Google-Benchmark JSON
            report, which carries the paradigm and precision.
        extra_target_args: Additional arguments for the benchmark binary.

    Returns:
        The argument vector to execute.
    """
    command = [
        nsys,
        "profile",
        # The NVTX ranges are what names the sampled windows; nothing else in
        # the default trace set is read, and each API traced costs run time.
        "--trace=nvtx",
        "--gpu-metrics-devices=all",
        f"--gpu-metrics-set={metric_set}",
        f"--gpu-metrics-frequency={frequency}",
        "--force-overwrite",
        "true",
        "-o",
        str(report.with_suffix("")),
        str(target),
    ]
    if benchmark_report is not None:
        command += [
            f"--benchmark_out={benchmark_report}",
            "--benchmark_out_format=json",
        ]
    command += extra_target_args or []
    return command


def _export_sqlite(nsys: str, report: Path, database: Path) -> bool:
    """Export a report to SQLite, which is the only readable form of the samples.

    Args:
        nsys: Profiler executable.
        report: The ``.nsys-rep`` to export.
        database: Destination ``.sqlite`` file.

    Returns:
        Whether the export produced a database.
    """
    command = [
        nsys,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite",
        "true",
        "-o",
        str(database),
        str(report),
    ]
    logger.debug(f"Command: {' '.join(command)}")
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as error:
        logger.error(f"Could not export {report.name}: {error}")
        return False
    return database.is_file()


def _profile_once(
    nsys: str,
    target: Path,
    output_dir: Path,
    stem: str,
    metric_set: str,
    frequency: int,
    iterations: int,
    write_benchmark_report: bool,
    extra_target_args: list[str] | None,
    stream_output: bool,
    timeout: float | None,
) -> Path | None:
    """Profile one executable once and export the result to SQLite.

    Args:
        nsys: Profiler executable.
        target: Benchmark binary.
        output_dir: Directory the artefacts are written to.
        stem: File name stem of the artefacts.
        metric_set: GPU metric set alias.
        frequency: Sampling frequency in Hz.
        iterations: Google-Benchmark iterations to run.
        write_benchmark_report: Whether this run writes the JSON report that
            supplies paradigm and precision.
        extra_target_args: Additional arguments for the benchmark binary.
        stream_output: Forward the profiler output to the logger.
        timeout: Wall-clock limit in seconds, or ``None``.

    Returns:
        The SQLite export, or ``None`` if the run or the export failed.
    """
    report = output_dir / f"{stem}{REPORT_SUFFIX}"
    database = output_dir / f"{stem}{DATABASE_SUFFIX}"
    # Profiling.h prepends "--benchmark_min_time=1x"; Google Benchmark keeps the
    # last occurrence of a flag, so appending here wins.
    arguments = [f"--benchmark_min_time={iterations}x", *(extra_target_args or [])]
    command = profile_command(
        nsys,
        target,
        report,
        metric_set,
        frequency,
        benchmark_report=(
            output_dir / f"{target.name}.json" if write_benchmark_report else None
        ),
        extra_target_args=arguments,
    )
    logger.debug(f"Command: {' '.join(command)}")
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=timeout
        )
        if stream_output:
            for line in (completed.stdout or "").splitlines():
                logger.trace(f"[{stem}] {line}")
    except subprocess.TimeoutExpired:
        logger.error(f"Timed out after {timeout} s profiling {stem}")
        return None
    except subprocess.CalledProcessError as error:
        logger.error(f"Failed to profile {stem}: {error}")
        return None
    database.unlink(missing_ok=True)
    if not _export_sqlite(nsys, report, database):
        return None
    return database


def run_profiles(
    executables: list[Path],
    output_dir: Path,
    nsys: str,
    metric_set: str,
    frequency: int = 100_000,
    iterations: int = 1,
    force: bool = False,
    extra_target_args: list[str] | None = None,
    stream_output: bool = False,
    timeout: float | None = None,
) -> list[Path]:
    """Profile every executable, one after another, and export each report.

    With ``iterations`` above one every executable is profiled **twice**, once
    with a single iteration and once with ``iterations``. The sampler cannot
    tell a kernel launch from the pipeline compilation and buffer uploads that
    a lazily initialising backend does on its first call -- both are compute
    work on the same GPU, and they end up inside the same window. Two runs
    separate them: the constant setup cost appears in both, so subtracting the
    single-iteration measurement from the many-iteration one and dividing by
    the difference in iteration count leaves the per-iteration kernel alone.

    Args:
        executables: Benchmark binaries to profile.
        output_dir: Directory the reports and SQLite exports are written to.
        nsys: Profiler executable.
        metric_set: GPU metric set alias.
        frequency: Sampling frequency in Hz.
        iterations: Iterations of the many-iteration run; 1 disables the
            two-point measurement.
        force: Re-profile even when a database already exists.
        extra_target_args: Additional arguments for the benchmark binaries.
        stream_output: Forward the profiler output to the logger at TRACE level.
        timeout: Wall-clock limit per run in seconds, or ``None``.

    Returns:
        The SQLite databases of the main run, in execution order. The reference
        run of each is named ``<executable>.ref.sqlite`` next to it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    databases: list[Path] = []
    for index, target in enumerate(executables, start=1):
        database = output_dir / f"{target.name}{DATABASE_SUFFIX}"
        reference = output_dir / f"{target.name}{REFERENCE_INFIX}{DATABASE_SUFFIX}"
        complete = database.is_file() and (iterations == 1 or reference.is_file())
        if complete and not force:
            logger.warning(
                f"{database.name} already exists, reusing it (use --force to re-profile)"
            )
            databases.append(database)
            continue
        logger.info(f"[{index}/{len(executables)}] Profiling {target.name} ...")
        produced = _profile_once(
            nsys,
            target,
            output_dir,
            target.name,
            metric_set,
            frequency,
            iterations,
            True,
            extra_target_args,
            stream_output,
            timeout,
        )
        if produced is None:
            continue
        if iterations > 1:
            logger.info(
                f"[{index}/{len(executables)}] Profiling {target.name} "
                "again with one iteration (reference for the setup cost) ..."
            )
            if (
                _profile_once(
                    nsys,
                    target,
                    output_dir,
                    f"{target.name}{REFERENCE_INFIX}",
                    metric_set,
                    frequency,
                    1,
                    False,
                    extra_target_args,
                    stream_output,
                    timeout,
                )
                is None
            ):
                continue
        logger.success(f"Finished {target.name} -> {produced}")
        databases.append(produced)
    return databases


def find_profiled(report_dir: Path) -> list[Path]:
    """List the SQLite exports a previous batch left behind.

    Args:
        report_dir: Directory holding the exports.

    Returns:
        The databases, sorted by name.
    """
    return sorted(
        path
        for path in report_dir.glob(f"*{DATABASE_SUFFIX}")
        if not path.name.endswith(f"{REFERENCE_INFIX}{DATABASE_SUFFIX}")
    )


def _metric_ids(connection: sqlite3.Connection) -> dict[str, int]:
    """Map metric names to the ids the samples are stored under.

    Args:
        connection: Open connection to an exported report.

    Returns:
        Metric name -> metric id.
    """
    return {
        str(name): int(identifier)
        for identifier, name in connection.execute(
            "select metricId, metricName from TARGET_INFO_GPU_METRICS"
        )
    }


def _samples(
    connection: sqlite3.Connection, ids: dict[str, int]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Read the three metrics the analysis needs, aligned on a common time axis.

    A twenty-second run at 100 kHz holds a few million samples, so the rows are
    pulled straight into arrays rather than Python objects.

    Args:
        connection: Open connection to an exported report.
        ids: Metric name -> metric id, from :func:`_metric_ids`.

    Returns:
        The sorted sample timestamps in nanoseconds, and one value array per
        metric id, aligned with them. A metric that has no sample at a given
        timestamp reads zero there.
    """
    wanted = [
        ids[name]
        for name in (COMPUTE_ACTIVITY_METRIC, DRAM_READ_METRIC, DRAM_WRITE_METRIC)
        if name in ids
    ]
    placeholders = ",".join("?" * len(wanted))
    frame = pd.read_sql_query(
        f"select timestamp, metricId, value from GPU_METRICS "
        f"where metricId in ({placeholders})",
        connection,
        params=wanted,
    )
    timestamps = np.unique(frame["timestamp"].to_numpy())
    position = np.searchsorted(timestamps, frame["timestamp"].to_numpy())
    series: dict[int, np.ndarray] = {}
    for metric in wanted:
        values = np.zeros(len(timestamps))
        mask = (frame["metricId"] == metric).to_numpy()
        values[position[mask]] = frame["value"].to_numpy()[mask]
        series[metric] = values
    return timestamps, series


def _regions(connection: sqlite3.Connection) -> dict[str, list[tuple[int, int]]]:
    """Read the benchmark's NVTX ranges, grouped by region name.

    Only the default domain is read. Libraries annotate ranges of their own --
    CCCL brackets every ``thrust::reduce`` -- and they register their names
    rather than passing them inline, so they arrive with ``text`` unset and a
    domain of their own.

    Args:
        connection: Open connection to an exported report.

    Returns:
        Region name -> its occurrences as (start, end) timestamps in
        nanoseconds, in chronological order. Empty if the run recorded no NVTX
        range at all.
    """
    try:
        rows = connection.execute(
            "select text, start, end from NVTX_EVENTS "
            "where text is not null and end is not null "
            "and (domainId is null or domainId = 0) "
            "order by start"
        ).fetchall()
    except sqlite3.Error:
        # No NVTX_EVENTS table: the run was collected without --trace=nvtx.
        return {}
    occurrences: dict[str, list[tuple[int, int]]] = {}
    for text, start, end in rows:
        occurrences.setdefault(str(text), []).append((int(start), int(end)))
    return occurrences


def _windows(
    timestamps: np.ndarray,
    activity: np.ndarray,
    bounds: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Group the compute-active samples into kernel windows.

    Args:
        timestamps: Sorted sample timestamps in nanoseconds.
        activity: The compute-activity signal, aligned with ``timestamps``.
        bounds: Only consider samples inside these (start, end) timestamps,
            or ``None`` for the whole run.

    Returns:
        The windows as (start, end) timestamps, longest first.
    """
    inside = activity > ACTIVITY_THRESHOLD
    if bounds is not None:
        inside &= (timestamps >= bounds[0]) & (timestamps <= bounds[1])
    active = timestamps[inside]
    if active.size == 0:
        return []
    gap = WINDOW_GAP_SECONDS * 1e9
    # A gap larger than the tolerance starts a new window.
    breaks = np.flatnonzero(np.diff(active) > gap)
    starts = np.concatenate(([active[0]], active[breaks + 1]))
    ends = np.concatenate((active[breaks], [active[-1]]))
    windows = [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start > MINIMUM_WINDOW_SECONDS * 1e9
    ]
    return sorted(windows, key=lambda window: window[1] - window[0], reverse=True)


def _traffic(
    timestamps: np.ndarray,
    read: np.ndarray,
    write: np.ndarray,
    window: tuple[int, int],
    peak_bandwidth: float,
) -> float:
    """Integrate the DRAM throughputs over one window.

    Each sample reports the average throughput of the interval that *ends* at
    its timestamp, as a percentage of the peak. The integral therefore weights
    every sample with how much of its interval falls inside the window, so a
    window boundary that lands mid-interval is not counted twice.

    Args:
        timestamps: Sorted sample timestamps in nanoseconds.
        read: DRAM read throughput in percent, aligned with ``timestamps``.
        write: DRAM write throughput in percent, aligned with ``timestamps``.
        window: The (start, end) timestamps to integrate over.
        peak_bandwidth: Bandwidth the percentages refer to, in bytes/s.

    Returns:
        Bytes moved between the GPU and DRAM inside the window.
    """
    start, end = window
    lower = np.maximum(timestamps[:-1], start)
    upper = np.minimum(timestamps[1:], end)
    overlap = np.clip(upper - lower, 0, None) * 1e-9
    fraction = (read[1:] + write[1:]) / 100.0
    return float(np.sum(fraction * peak_bandwidth * overlap))


def analytic_flop(name: str, rules: list[str]) -> float | None:
    """Resolve the analytic FLOP count of one row.

    Args:
        name: What the rules are matched against. The callers pass
            ``<executable>[<region>]``, e.g. ``polyhedral_ocl[evaluate]``, so a
            rule can address one phase of a binary -- the polyhedral ``init``
            kernel computes normals and segment vectors, not the gravity model,
            and the work model of ``evaluate`` does not describe it. A rule
            naming only the executable still matches every one of its regions.
        rules: ``<regex>=<value>`` rules, in order; the first match wins.

    Returns:
        The FLOP count, or ``None`` if no rule matches.
    """
    for rule in rules:
        pattern, _, value = rule.partition("=")
        if not _:
            logger.error(f"Ignoring malformed --analytic-flop rule {rule!r}")
            continue
        if re.search(pattern, name):
            try:
                return float(value)
            except ValueError:
                logger.error(f"Ignoring --analytic-flop rule with a bad value: {rule!r}")
                return None
    return None


def _measure(
    database: Path, peak_bandwidth: float, activity_ratio: float = ACTIVITY_RATIO
) -> dict[str, tuple[float, float, int, int]] | None:
    """Reduce one report to the compute time and DRAM traffic of each region.

    Every compute-active window inside a region counts: with more than one
    iteration the run holds several, and whether the sampler resolves them
    separately or merges them into one long stretch depends on how much idle
    time the host leaves between launches. Summing makes the result independent
    of that.

    A run without NVTX ranges falls back to measuring the whole run as one
    unnamed region, which is what the ``""`` key means to the caller.

    Args:
        database: SQLite export to read.
        peak_bandwidth: Bandwidth the sampled percentages refer to, in bytes/s.
        activity_ratio: Windows whose mean compute occupancy falls below this
            fraction of the busiest window's *in the same region* are dropped,
            see :data:`ACTIVITY_RATIO`.

    Returns:
        Region name -> (seconds of compute activity, bytes moved, number of
        windows, number of times the region was entered); ``None`` if the
        report could not be read or holds no compute at all. The entry count is
        zero for the synthetic whole-run region, which was never entered
        because it was never marked.
    """
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error as error:
        logger.error(f"Could not open {database}: {error}")
        return None
    with connection:
        ids = _metric_ids(connection)
        missing = [
            metric
            for metric in (
                COMPUTE_ACTIVITY_METRIC,
                DRAM_READ_METRIC,
                DRAM_WRITE_METRIC,
            )
            if metric not in ids
        ]
        if missing:
            logger.error(
                f"{database.name} does not carry {missing}; the run most likely "
                "used a metric set without them "
                "(nsys profile --gpu-metrics-set=help lists the sets)"
            )
            return None
        timestamps, series = _samples(connection, ids)
        activity = series[ids[COMPUTE_ACTIVITY_METRIC]]
        regions = _regions(connection)
        if not regions:
            logger.warning(
                f"{database.name} carries no NVTX range, so the whole run is "
                "measured as one region. Build the benchmark with "
                "-DPPB_ENABLE_NVTX=ON to have its kernels named."
            )
            regions = (
                {"": [(int(timestamps[0]), int(timestamps[-1]))]}
                if timestamps.size
                else {}
            )
            unmarked = True
        else:
            unmarked = False

        measured: dict[str, tuple[float, float, int, int]] = {}
        for name, occurrences in regions.items():
            windows = [
                window
                for bounds in occurrences
                for window in _windows(timestamps, activity, bounds)
            ]
            if not windows:
                continue
            # Some runtimes move their buffers with a compute shader rather than
            # a copy engine -- Kompute brackets every matrix-multiplication
            # dispatch with two such phases -- and those land inside the same
            # region as the kernel. They run at roughly half the occupancy of
            # the dispatch, which is what separates them.
            occupancy = {
                window: float(
                    np.mean(
                        activity[(timestamps >= window[0]) & (timestamps <= window[1])]
                    )
                )
                for window in windows
            }
            limit = activity_ratio * max(occupancy.values())
            windows = [window for window in windows if occupancy[window] >= limit]
            if not windows:
                continue
            duration = sum(end - start for start, end in windows) * 1e-9
            traffic = sum(
                _traffic(
                    timestamps,
                    series[ids[DRAM_READ_METRIC]],
                    series[ids[DRAM_WRITE_METRIC]],
                    window,
                    peak_bandwidth,
                )
                for window in windows
            )
            measured[name] = (
                duration,
                traffic,
                len(windows),
                0 if unmarked else len(occurrences),
            )
    return measured or None


def load_reports(
    databases: list[Path],
    report_dir: Path,
    hardware: str = "",
    precision: str = "auto",
    peak_performance: float | None = None,
    peak_bandwidth: float | None = None,
    flop_rules: list[str] | None = None,
    iterations: int = 1,
    activity_ratio: float = ACTIVITY_RATIO,
) -> pd.DataFrame:
    """Turn the sampled reports of a batch into the tidy roofline table.

    The columns are the ones the Nsight Compute backend writes, so both feed the
    same CSV writer and the same plot. A row is one named region of one
    implementation -- ``matmul``, or ``init`` and ``evaluate``.

    Args:
        databases: SQLite exports of the main run.
        report_dir: Directory holding the Google-Benchmark reports written
            alongside them; they supply paradigm, precision and problem.
        hardware: Label for the hardware column.
        precision: ``auto`` to follow the precision the binary was built with,
            or an explicit key of
            :data:`~ppbcc.profiling.metrics.PRECISIONS`.
        peak_performance: Compute ceiling in FLOP/s. Nsight Systems reports
            percentages of peak but not the peak itself, so it has to be
            supplied; it is a property of the hardware, and an ``ncu`` run on
            the same machine measures it.
        peak_bandwidth: Memory ceiling in bytes/s, supplied for the same reason
            *and* needed to turn the sampled percentages into bytes.
        flop_rules: ``<regex>=<value>`` rules supplying the analytic FLOP count
            per executable, see :func:`analytic_flop`.
        iterations: Iteration count the main run used. Above one, the ``.ref``
            report next to each database is subtracted per region and the
            remainder divided by how many more times that region was entered,
            which cancels the one-time setup both runs paid; see
            :func:`run_profiles`.
        activity_ratio: Occupancy threshold that separates the kernel from
            data-movement phases, see :data:`ACTIVITY_RATIO`.

    Returns:
        One row per (executable, NVTX region), ordered as
        :data:`~ppbcc.constants.PROFILE_COLUMN_LIST`.
    """
    from ppbcc.profiling.reports import _load_context

    if not peak_bandwidth:
        logger.error(
            "The sampler reports DRAM traffic as a percentage of peak, so "
            "--peak-bandwidth is required to turn it into bytes."
        )
        return pd.DataFrame(columns=PROFILE_COLUMN_LIST)

    rows: list[dict[str, object]] = []
    for database in databases:
        name = database.name[: -len(DATABASE_SUFFIX)]
        measured = _measure(database, peak_bandwidth, activity_ratio)
        if measured is None:
            logger.warning(
                f"{name}: no compute-active window found — the GPU stayed idle "
                "for the whole run"
            )
            continue
        reference: dict[str, tuple[float, float, int, int]] = {}
        if iterations > 1:
            reference_path = database.with_name(
                f"{name}{REFERENCE_INFIX}{DATABASE_SUFFIX}"
            )
            reference = (
                _measure(reference_path, peak_bandwidth, activity_ratio) or {}
                if reference_path.is_file()
                else {}
            )
        paradigm, built_precision, problem = _load_context(report_dir / f"{name}.json")
        key = precision
        if key == "auto":
            key = {"32": "fp32", "64": "fp64", "16": "fp16"}.get(
                str(built_precision), "fp32"
            )
        for identifier, (region, (duration, traffic, count, entries)) in enumerate(
            sorted(measured.items())
        ):
            flop = analytic_flop(f"{name}[{region}]", flop_rules or [])
            if flop is None:
                logger.warning(
                    f"{name} [{region or 'unnamed'}]: no --analytic-flop rule "
                    "matched. The sampler reports pipe utilisations, not "
                    "instruction counts, so this row carries no work and cannot "
                    "be placed on a roofline."
                )
            note = f"{count} compute window(s)"
            if iterations > 1:
                if region not in reference:
                    logger.warning(
                        f"{name} [{region or 'unnamed'}]: no usable reference "
                        f"run; dividing by {iterations} instead of subtracting, "
                        "so any one-time setup cost stays in the result"
                    )
                    duration /= iterations
                    traffic /= iterations
                    note += f", divided by {iterations} iterations"
                else:
                    base = reference[region]
                    # How often the region was actually entered is what the two
                    # runs differ by, and it is not always the iteration count:
                    # a lazily initialised backend enters "init" exactly once no
                    # matter how many iterations follow. Such a region is a
                    # one-time phase and is reported as measured.
                    # A synthetic whole-run region reports no entry count, and
                    # the whole run *is* proportional to the iterations.
                    calls = (
                        entries - base[3]
                        if entries and base[3]
                        else iterations - 1
                    )
                    if calls <= 0:
                        note += f", entered {entries} time(s) in both runs"
                    else:
                        duration = (duration - base[0]) / calls
                        traffic = (traffic - base[1]) / calls
                        note += (
                            f", minus a {base[0] * 1e3:.3f} ms reference run, "
                            f"over {calls} call(s)"
                        )
                        if duration <= 0.0:
                            logger.warning(
                                f"{name} [{region or 'unnamed'}]: the reference "
                                f"run was not shorter than the "
                                f"{iterations}-iteration one; the two-point "
                                "measurement cannot be applied and the row is "
                                "dropped"
                            )
                            continue
            logger.debug(
                f"{name} [{region or 'unnamed'}]: {duration * 1e3:.3f} ms, "
                f"{traffic / 1e9:.3f} GB ({note})"
            )
            rows.append(
                {
                    HARDWARE: hardware,
                    BENCHMARK_PROBLEM: problem,
                    PARADIGM: paradigm,
                    PRECISION: PRECISIONS[key][1],
                    EXECUTABLE: name,
                    REGION: region,
                    KERNEL_ID: identifier,
                    KERNEL: "compute window",
                    KERNEL_SIGNATURE: f"sampled compute activity: {note}",
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
                    PEAK_BANDWIDTH: peak_bandwidth,
                }
            )
    if not rows:
        logger.error("No profiling data could be loaded.")
        return pd.DataFrame(columns=PROFILE_COLUMN_LIST)
    frame = pd.DataFrame(rows)
    logger.success(
        f"Loaded {len(frame)} row(s) across {frame[EXECUTABLE].nunique()} executable(s)"
    )
    # Grid and block size are launch properties no sampling profiler sees.
    return frame[[column for column in PROFILE_COLUMN_LIST if column in frame]]
