"""LIKWID NvMarker backend: GPU counters read from regions marked in the source.

Where Nsight Compute attaches to a process from the outside and replays every
kernel launch it sees, LIKWID reads the counters from inside the application:
``src/common/Marker.h`` in the benchmark repository opens a region around the
kernel each implementation already times, and CUPTI reads the counters when that
region closes. The two tools therefore answer slightly different questions --
ncu reports one row per *kernel launch*, LIKWID one row per *marked region* --
and the region is the more useful unit whenever an implementation reaches the
GPU through a runtime that launches several kernels per call.

Two properties of LIKWID 5.5.1 shape this module.

*The wrapper is bypassed.* ``likwid-perfctr`` programs the counters from a
separate daemon process, which fails on this hardware
(``cuptiProfilerGetCounterAvailability`` -> ``CUPTI_ERROR_INVALID_PARAMETER``,
independent of the CUDA version). The NvMarker API is therefore driven directly:
the environment variables below make the instrumented binary program its own
counters, and no wrapper is involved.

*The counters are read in two passes.* The SMSP domain (floating-point
instructions) and the DRAM domain cannot be programmed at the same time --
``createConfigImage`` rejects the combination -- so every executable is run once
per :data:`EVENT_GROUPS` entry and the results are merged afterwards.
"""

from __future__ import annotations

import os
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

#: File name suffix of a parsed marker file.
MARKER_SUFFIX = ".likwid-marker"

#: Event groups, collected in separate runs. The keys name the run, the values
#: are the LIKWID event names in the order their counts appear in the marker
#: file.
EVENT_GROUPS: dict[str, list[str]] = {
    "flops": [
        "SMSP_SASS_THREAD_INST_EXECUTED_OP_FADD_PRED_ON_SUM",
        "SMSP_SASS_THREAD_INST_EXECUTED_OP_FMUL_PRED_ON_SUM",
        "SMSP_SASS_THREAD_INST_EXECUTED_OP_FFMA_PRED_ON_SUM",
    ],
    "memory": ["DRAM_BYTES_SUM"],
}

#: Double precision uses the same counters with a ``D`` infix; half precision an
#: ``H``. Which set is read follows the precision the binary was built with.
_PRECISION_INFIX = {"fp32": "F", "fp64": "D", "fp16": "H"}

#: LIKWID 5.5.1 reports every Nvidia counter at exactly half its hardware value
#: on this machine. The factor was established against kernels with an
#: analytically known instruction count and cross-checked with Nsight Compute:
#: for FADD, FMUL, FFMA and DRAM_BYTES_SUM alike, ncu reproduces the analytic
#: value exactly and LIKWID reports half of it. The factor is independent of the
#: counter domain and of the number of launches inside a region, so it is a
#: constant scale rather than an averaging error, and correcting for it here
#: restores agreement with ncu. Note that arithmetic intensity is a ratio of two
#: equally scaled counters and is therefore unaffected either way.
COUNTER_SCALE = 2.0


def event_string(group: str, precision: str) -> str:
    """Build the ``LIKWID_NVMON_EVENTS`` value for one event group.

    Args:
        group: Key of :data:`EVENT_GROUPS`.
        precision: Key of :data:`~ppbcc.profiling.metrics.PRECISIONS`.

    Returns:
        Comma-separated ``EVENT:GPUn`` pairs, in counter order.
    """
    infix = _PRECISION_INFIX[precision]
    events = [
        event.replace("_OP_F", f"_OP_{infix}") if group == "flops" else event
        for event in EVENT_GROUPS[group]
    ]
    return ",".join(f"{event}:GPU{index}" for index, event in enumerate(events))


def parse_marker_file(path: Path) -> dict[str, dict[str, float]]:
    """Parse a LIKWID NvMarker output file.

    The format is three blocks of whitespace-separated fields, as written by
    ``nvmon_markerClose`` in LIKWID's ``src/libnvctr.c``::

        <gpus> <regions> <groups>
        <region index>:<tag>-<group id>            (one line per region)
        <region index> <group id> <gpu id> <calls> <time> <events> <value>...

    Note that the header counts GPUs first and regions second, and that the
    region index leads the data lines while the group id follows it. Both are
    easy to mistake for one another while every file holds a single region and
    a single group, where the two are indistinguishable.

    Args:
        path: Marker file written by the instrumented binary.

    Returns:
        Region tag -> ``{"time": seconds, "calls": n, "values": [...]}``.
    """
    tokens = path.read_text().split("\n")
    lines = [line for line in tokens if line.strip()]
    if not lines:
        return {}
    region_count = int(lines[0].split()[1])
    # "0:matmul-0" -> region index 0 carries the tag "matmul"
    tags = {
        int(line.split(":", 1)[0]): line.split(":", 1)[1].rsplit("-", 1)[0]
        for line in lines[1 : 1 + region_count]
    }
    regions: dict[str, dict[str, float]] = {}
    for line in lines[1 + region_count :]:
        fields = line.split()
        if len(fields) < 6:
            continue
        region, calls = int(fields[0]), float(fields[3])
        time, event_count = float(fields[4]), int(fields[5])
        values = [float(value) for value in fields[6 : 6 + event_count]]
        tag = tags.get(region, str(region))
        entry = regions.setdefault(tag, {"time": time, "calls": calls, "values": []})
        # A second group's run appends its counters to the same region.
        entry["values"] = list(entry["values"]) + values  # type: ignore[arg-type]
        entry["time"] = max(float(entry["time"]), time)
    return regions


def run_profiles(
    executables: list[Path],
    output_dir: Path,
    likwid_lib: Path | None = None,
    gpu: int = 0,
    precision: str = "fp32",
    force: bool = False,
    extra_target_args: list[str] | None = None,
    stream_output: bool = False,
    timeout: float | None = None,
) -> list[Path]:
    """Run every executable once per event group and collect its marker files.

    Args:
        executables: Instrumented benchmark binaries.
        output_dir: Directory the marker files are written to.
        likwid_lib: ``lib`` directory of the LIKWID installation, prepended to
            ``LD_LIBRARY_PATH`` when given.
        gpu: Index of the GPU to read counters from.
        precision: Key of :data:`~ppbcc.profiling.metrics.PRECISIONS`.
        force: Re-profile even when marker files already exist.
        extra_target_args: Additional arguments for the benchmark binaries.
        stream_output: Forward the binary's output to the logger.
        timeout: Wall-clock limit per run in seconds, or ``None`` for no limit.

    Returns:
        The executables that produced at least one marker file, in execution
        order. The files themselves are named
        ``<executable>.<group>.likwid-marker`` and stay separate per group --
        each carries its own header block, so they cannot be concatenated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for index, target in enumerate(executables, start=1):
        existing = [
            output_dir / f"{target.name}.{group}{MARKER_SUFFIX}" for group in EVENT_GROUPS
        ]
        if all(path.is_file() for path in existing) and not force:
            logger.warning(
                f"Marker files for {target.name} already exist, reusing them "
                "(use --force to re-profile)"
            )
            produced.append(target)
            continue
        parts: list[Path] = []
        for group in EVENT_GROUPS:
            part = output_dir / f"{target.name}.{group}{MARKER_SUFFIX}"
            environment = dict(os.environ)
            environment["LIKWID_NVMON_GPUS"] = str(gpu)
            environment["LIKWID_NVMON_EVENTS"] = event_string(group, precision)
            environment["LIKWID_NVMON_FILEPATH"] = str(part.resolve())
            if likwid_lib is not None:
                environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                    [str(likwid_lib), environment.get("LD_LIBRARY_PATH", "")]
                ).rstrip(os.pathsep)
            command = [str(target)]
            # The Google-Benchmark report supplies paradigm and precision, which
            # the counters themselves do not carry.
            if group == next(iter(EVENT_GROUPS)):
                command += [
                    f"--benchmark_out={output_dir / f'{target.name}.json'}",
                    "--benchmark_out_format=json",
                ]
            command += extra_target_args or []
            logger.info(
                f"[{index}/{len(executables)}] Profiling {target.name} ({group}) ..."
            )
            logger.debug(f"Events: {environment['LIKWID_NVMON_EVENTS']}")
            part.unlink(missing_ok=True)
            try:
                completed = subprocess.run(
                    command,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                )
                if stream_output:
                    for line in (completed.stdout or "").splitlines():
                        logger.trace(f"[{target.name}] {line}")
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Timed out after {timeout} s profiling {target.name} ({group}); "
                    "skipping it."
                )
                continue
            except subprocess.CalledProcessError as error:
                logger.error(f"Failed to profile {target.name} ({group}): {error}")
                continue
            if not part.is_file():
                # No marked region ran, or CUPTI saw no CUDA context: OpenCL and
                # Vulkan build their own and stay invisible to NvMarker.
                logger.warning(
                    f"{target.name} ({group}) wrote no marker file — it most "
                    "likely reaches the GPU without a CUDA context (OpenCL, "
                    "Vulkan) or ran no marked region"
                )
                continue
            parts.append(part)
        if not parts:
            continue
        logger.success(f"Finished {target.name} -> {', '.join(p.name for p in parts)}")
        produced.append(target)
    return produced


def find_profiled(report_dir: Path) -> list[Path]:
    """List the executables a previous run left marker files for.

    Used by ``--skip-profile``, which re-reads an existing batch instead of
    running one. Only the file *name* matters downstream, so the returned paths
    are name stubs, not runnable binaries.

    Args:
        report_dir: Directory holding the marker files.

    Returns:
        One path per executable, sorted by name.
    """
    group = next(iter(EVENT_GROUPS))
    suffix = f".{group}{MARKER_SUFFIX}"
    names = sorted(
        path.name[: -len(suffix)]
        for path in report_dir.glob(f"*{suffix}")
    )
    return [Path(name) for name in names]


def load_reports(
    executables: list[Path],
    report_dir: Path,
    hardware: str = "",
    precision: str = "auto",
    peak_performance: float | None = None,
    peak_bandwidth: float | None = None,
    scale: float = COUNTER_SCALE,
) -> pd.DataFrame:
    """Turn the marker files of a batch into the tidy roofline table.

    The table has the same columns as the Nsight Compute one, so both backends
    feed the same CSV writer and the same plot. The one structural difference is
    the row granularity: a row is a marked *region*, not a kernel launch, and its
    kernel column carries the region tag.

    Args:
        executables: Benchmark binaries whose marker files should be read.
        report_dir: Directory holding the marker files and the Google-Benchmark
            reports written alongside them.
        hardware: Label for the hardware column.
        precision: The precision whose counters were collected -- pass the same
            value that :func:`run_profiles` ran with. ``auto`` falls back to the
            precision the binary was built with, which is only correct when the
            two happen to agree.
        peak_performance: Compute ceiling in FLOP/s. LIKWID exposes no
            ``peak_sustained`` counters, so the ceiling has to be supplied; it
            is a property of the hardware, not of the measuring tool.
        peak_bandwidth: Memory ceiling in bytes/s, supplied for the same reason.
        scale: Correction applied to every counter, see :data:`COUNTER_SCALE`.

    Returns:
        One row per marked region, ordered as
        :data:`~ppbcc.constants.PROFILE_COLUMN_LIST`.
    """
    # Imported here to keep the two backends independent at module level.
    from ppbcc.profiling.reports import _load_context

    rows: list[dict[str, object]] = []
    for target in executables:
        paradigm, built_precision, problem = _load_context(
            report_dir / f"{target.name}.json"
        )
        key = precision
        if key == "auto":
            key = {"32": "fp32", "64": "fp64", "16": "fp16"}.get(
                str(built_precision), "fp32"
            )
        groups = {
            group: parse_marker_file(path)
            for group in EVENT_GROUPS
            if (path := report_dir / f"{target.name}.{group}{MARKER_SUFFIX}").is_file()
        }
        if not groups:
            logger.warning(f"No marker files for {target.name}, skipping")
            continue
        tags = sorted({tag for regions in groups.values() for tag in regions})
        for identifier, tag in enumerate(tags):
            flops = groups.get("flops", {}).get(tag, {})
            memory = groups.get("memory", {}).get(tag, {})
            counters = [float(value) * scale for value in flops.get("values", [])]
            # EVENT_GROUPS["flops"] is ordered add, mul, fma; an FMA is 2 FLOP.
            add, mul, fma = (counters + [0.0, 0.0, 0.0])[:3]
            flop = add + mul + 2.0 * fma
            traffic = next(
                (float(value) * scale for value in memory.get("values", [])), float("nan")
            )
            # Each event group times the same region, but not at the same cost:
            # programming three SMSP counters perturbs a kernel far more than a
            # single DRAM counter, so the groups disagree by up to an order of
            # magnitude. The smallest time is the least perturbed measurement of
            # the kernel, and it is the one a roofline wants -- the counters are
            # unaffected by the overhead, the clock is not.
            times = [
                float(region["time"])
                for region in (flops, memory)
                if region.get("time") is not None
            ]
            duration = min(times) if times else float("nan")
            if len(times) > 1 and min(times) > 0 and max(times) / min(times) > 1.5:
                logger.debug(
                    f"{target.name}/{tag}: region time differs by "
                    f"{max(times) / min(times):.1f}x between event groups "
                    f"({', '.join(f'{value:.3f}s' for value in times)}); "
                    "using the smallest as the kernel time"
                )
            calls = float(flops.get("calls", memory.get("calls", 1)))
            if flop == 0.0:
                # The region ran and was timed, but CUPTI saw no CUDA context to
                # read counters from. OpenCL and Vulkan reach the GPU through
                # their own contexts and look exactly like this.
                logger.warning(
                    f"{target.name}/{tag}: region timed but all floating-point "
                    "counters are zero — the paradigm most likely reaches the "
                    "GPU without a CUDA context (OpenCL, Vulkan), so CUPTI "
                    "cannot see it. The row is kept, but it carries no work."
                )
            rows.append(
                {
                    HARDWARE: hardware,
                    BENCHMARK_PROBLEM: problem,
                    PARADIGM: paradigm,
                    PRECISION: PRECISIONS[key][1],
                    EXECUTABLE: target.name,
                    KERNEL_ID: identifier,
                    # This backend measures regions directly, so the region
                    # name is both its label and its kernel column.
                    REGION: tag,
                    KERNEL: tag,
                    KERNEL_SIGNATURE: f"{tag} ({calls:.0f} call(s))",
                    DURATION: duration,
                    MEMORY_LEVEL: "DRAM",
                    FLOP: flop,
                    MEMORY_TRAFFIC: traffic,
                    ARITHMETIC_INTENSITY: flop / traffic if traffic else float("nan"),
                    PERFORMANCE: flop / duration if duration else float("nan"),
                    PEAK_PERFORMANCE: peak_performance,
                    PEAK_BANDWIDTH: peak_bandwidth,
                }
            )
    if not rows:
        return pd.DataFrame(columns=PROFILE_COLUMN_LIST)
    frame = pd.DataFrame(rows)
    return frame[[column for column in PROFILE_COLUMN_LIST if column in frame]]
