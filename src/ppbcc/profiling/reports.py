"""Read ``.ncu-rep`` reports and turn them into tidy roofline tables."""

from __future__ import annotations

import io
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from ppbcc.constants import (
    ARITHMETIC_INTENSITY,
    BENCHMARK_PROBLEM,
    BLOCK_SIZE,
    DURATION,
    EXECUTABLE,
    FLOP,
    GRID_SIZE,
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
)
from ppbcc.profiling.metrics import (
    MEMORY_LEVEL_CLOCK,
    MEMORY_LEVELS,
    PRECISIONS,
    flop_columns,
    peak_flop_column,
)

#: Clock rate the compute ceiling is scaled with.
SM_CLOCK = "sm__cycles_elapsed.avg.per_second"
#: Kernel duration reported by ncu, in nanoseconds.
DURATION_METRIC = "gpu__time_duration.sum"

#: Columns identifying a single kernel launch inside one report. "Block Size"
#: and "Grid Size" already carry the names the tidy table uses.
_LAUNCH_KEYS = ["ID", "Kernel Name", BLOCK_SIZE, GRID_SIZE]


def _short_kernel_name(signature: str) -> str:
    """Reduce a kernel signature to a legend-sized name.

    Args:
        signature: Kernel signature as reported by the profiler, e.g.
            ``void cub::DeviceReduceKernel<...>(T2, T5 *)``.

    Returns:
        The bare (possibly namespace-qualified) kernel name.
    """
    name = signature.strip()
    # Drop the parameter list: the first "(" outside any template argument list.
    depth = 0
    for index, character in enumerate(name):
        if character == "(" and depth == 0:
            name = name[:index]
            break
        depth += {"<": 1, ">": -1}.get(character, 0)
    # Collapse template argument lists from the inside out.
    while True:
        collapsed = re.sub(r"<[^<>]*>", "", name)
        if collapsed == name:
            break
        name = collapsed
    # Whatever is left in front (a return type such as "void") is not the name.
    words = name.split("<")[0].split()
    return words[-1] if words else signature


def _import_csv(report: Path, ncu: str) -> pd.DataFrame:
    """Re-import one profiler report as a long metric table.

    Args:
        report: Path to a ``.ncu-rep`` file.
        ncu: Profiler executable used for the import.

    Returns:
        The profiler's own CSV export, one row per (kernel launch, metric).

    Raises:
        subprocess.CalledProcessError: If the import fails.
    """
    command = [ncu, "--import", str(report), "--csv", "--print-units", "base"]
    logger.debug(f"Command: {' '.join(command)}")
    output = subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout
    # The profiler may prepend warnings; the table starts at the header row.
    start = output.find('"ID"')
    if start < 0:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(output[start:]), dtype=str)


def _pivot(long_table: pd.DataFrame) -> pd.DataFrame:
    """Pivot the profiler's long metric table into one row per kernel launch.

    Args:
        long_table: Output of :func:`_import_csv`.

    Returns:
        A wide table indexed by launch, with one column per metric.
    """
    long_table = long_table.copy()
    long_table["value"] = pd.to_numeric(
        long_table["Metric Value"].str.replace(",", "", regex=False),
        errors="coerce",
    )
    wide = long_table.pivot_table(
        index=_LAUNCH_KEYS,
        columns="Metric Name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def _load_context(benchmark_report: Path) -> tuple[str, str, str]:
    """Read paradigm, precision, and problem from a Google-Benchmark report.

    The profiler knows nothing about the paradigm a kernel was written in, so
    the JSON report the binary writes during the same run supplies it.

    Args:
        benchmark_report: Path to the ``.json`` report written next to the
            profiler report.

    Returns:
        Paradigm, precision, and benchmark problem; ``"Unknown"`` where the
        report is missing or does not carry the information.
    """
    if not benchmark_report.is_file():
        logger.warning(
            f"No Google-Benchmark report next to {benchmark_report.stem}; "
            "paradigm and precision stay unknown"
        )
        return "Unknown", "Unknown", "Unknown"
    try:
        with open(benchmark_report, "r") as handle:
            data = json.load(handle)
    except (ValueError, OSError) as error:
        logger.error(f"Could not read {benchmark_report}: {error}")
        return "Unknown", "Unknown", "Unknown"
    context = data.get("context", {})
    benchmarks = data.get("benchmarks", [])
    # "Polyhedral-Eros" / "VecAdd-Naive/100000000" -> "Polyhedral" / "VecAdd"
    name = str(benchmarks[0]["name"]) if benchmarks else ""
    problem = name.split("/")[0].split("-")[0] or "Unknown"
    return (
        str(context.get("paradigm", "Unknown")),
        str(context.get("float_type", "Unknown")),
        problem,
    )


def _resolve_precision(row: pd.Series, requested: str) -> str:
    """Pick the precision a kernel's roofline is drawn in.

    Args:
        row: One kernel launch with its floating-point instruction counts.
        requested: ``auto`` or an explicit key of
            :data:`~ppbcc.profiling.metrics.PRECISIONS`.

    Returns:
        A key of :data:`~ppbcc.profiling.metrics.PRECISIONS`.
    """
    if requested != "auto":
        return requested
    # The floating-point type the binary was built with decides. It is a
    # property of the whole executable, so setup kernels that happen to contain
    # a stray conversion do not end up in a different precision than the kernels
    # doing the actual work.
    preferred = {"32": "fp32", "64": "fp64", "16": "fp16"}.get(str(row[PRECISION]))
    if preferred is not None:
        return preferred
    # Without that information, the precision that did the most work wins.
    counts = {
        key: sum(row.get(column, 0.0) or 0.0 for column in flop_columns(key))
        for key in PRECISIONS
    }
    return max(counts, key=lambda key: counts[key]) if max(counts.values()) > 0 else "fp32"


def _derive(frame: pd.DataFrame, precision: str, memory_level: str) -> pd.DataFrame:
    """Add the roofline quantities to a table of kernel launches.

    Args:
        frame: Kernel launches with their raw metric columns.
        precision: ``auto`` or an explicit precision key.
        memory_level: Key of :data:`~ppbcc.profiling.metrics.MEMORY_LEVELS`.

    Returns:
        The same table with duration, FLOP, traffic, intensity, performance,
        and both ceilings filled in.
    """
    bytes_column, level_label = MEMORY_LEVELS[memory_level]
    clock_column = MEMORY_LEVEL_CLOCK[memory_level]

    resolved = frame.apply(lambda row: _resolve_precision(row, precision), axis=1)
    frame[PRECISION] = [PRECISIONS[key][1] for key in resolved]
    frame[MEMORY_LEVEL] = level_label

    def flop_of(row: pd.Series, key: str) -> float:
        add, mul, fma = (row.get(column, 0.0) or 0.0 for column in flop_columns(key))
        return add + mul + 2.0 * fma

    # Every precision is reported, so a mixed-precision kernel stays inspectable.
    for key, (_, label) in PRECISIONS.items():
        frame[f"{FLOP} {label}"] = frame.apply(lambda row: flop_of(row, key), axis=1)

    frame[FLOP] = [flop_of(row, key) for (_, row), key in zip(frame.iterrows(), resolved)]
    frame[DURATION] = frame.get(DURATION_METRIC, pd.NA) * 1e-9
    frame[MEMORY_TRAFFIC] = frame.get(f"{bytes_column}.sum", pd.NA)
    # A ".peak_sustained" metric is a per-cycle rate; the clock the unit actually
    # ran at during the kernel turns it into an absolute rate. An FMA is 2 FLOP.
    frame[PEAK_PERFORMANCE] = [
        2.0 * (row.get(peak_flop_column(key), 0.0) or 0.0) * (row.get(SM_CLOCK, 0.0) or 0.0)
        for (_, row), key in zip(frame.iterrows(), resolved)
    ]
    frame[PEAK_BANDWIDTH] = frame.get(
        f"{bytes_column}.sum.peak_sustained", pd.NA
    ) * frame.get(clock_column, pd.NA)

    frame[PERFORMANCE] = frame[FLOP] / frame[DURATION]
    frame[ARITHMETIC_INTENSITY] = frame[FLOP] / frame[MEMORY_TRAFFIC]
    return frame


def load_reports(
    reports: list[Path],
    ncu: str = "ncu",
    hardware: str = "",
    precision: str = "auto",
    memory_level: str = "dram",
) -> pd.DataFrame:
    """Load profiler reports into one tidy table, one row per kernel launch.

    Args:
        reports: Paths to ``.ncu-rep`` files.
        ncu: Profiler executable used to re-import the reports.
        hardware: Free-form hardware identifier stored in the ``Hardware``
            column (e.g. ``"NVIDIA RTX5080"``).
        precision: ``auto`` to follow the precision the binary was built with,
            or an explicit key of
            :data:`~ppbcc.profiling.metrics.PRECISIONS`.
        memory_level: Level of the memory hierarchy the arithmetic intensity is
            measured against; a key of
            :data:`~ppbcc.profiling.metrics.MEMORY_LEVELS`.

    Returns:
        One row per profiled kernel launch, with the raw metrics and the derived
        roofline quantities. Empty if nothing could be loaded.
    """
    frames: list[pd.DataFrame] = []
    logger.info(f"Loading {len(reports)} profiler report(s)")
    for report in reports:
        try:
            long_table = _import_csv(report, ncu)
        except subprocess.CalledProcessError as error:
            logger.error(f"Could not import {report}: {error}")
            continue
        if long_table.empty:
            logger.warning(
                f"{report.name} contains no profiled kernels — the paradigm most "
                "likely does not run on CUDA (Nsight Compute sees no OpenCL, "
                "Vulkan or host-side work)"
            )
            continue

        frame = _pivot(long_table)
        # ncu appends its own suffix, so the stem is the executable's name.
        executable = report.name[: -len(".ncu-rep")]
        paradigm, float_type, problem = _load_context(
            report.with_name(f"{executable}.json")
        )
        frame[EXECUTABLE] = executable
        frame[PARADIGM] = paradigm
        frame[PRECISION] = float_type
        frame[BENCHMARK_PROBLEM] = problem
        frame[HARDWARE] = hardware
        frame[KERNEL_ID] = pd.to_numeric(frame["ID"], errors="coerce")
        frame[KERNEL_SIGNATURE] = frame["Kernel Name"]
        frame[KERNEL] = frame["Kernel Name"].map(_short_kernel_name)
        # "Block Size"/"Grid Size" already carry their final names.
        frame = frame.drop(columns=["ID", "Kernel Name"])

        frame = _derive(frame, precision, memory_level)
        logger.debug(f"{executable}: {len(frame)} kernel launch(es)")
        frames.append(frame)

    if not frames:
        logger.error("No profiling data could be loaded.")
        return pd.DataFrame(columns=PROFILE_COLUMN_LIST)

    combined = pd.concat(frames, ignore_index=True)
    # Lead with the columns a reader cares about, keep every raw metric behind.
    ordered = [column for column in PROFILE_COLUMN_LIST if column in combined]
    combined = combined[ordered + [c for c in combined.columns if c not in ordered]]
    combined = combined.sort_values([EXECUTABLE, KERNEL_ID]).reset_index(drop=True)
    logger.success(
        f"Loaded {len(combined)} kernel launch(es) across "
        f"{combined[EXECUTABLE].nunique()} executable(s)"
    )
    return combined


def filter_kernels(frame: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    """Drop kernel launches whose name matches one of the given patterns.

    Runtime bootstrap kernels (Kokkos' architecture query and desul's lock-array
    initialization, for instance) are launched once by the framework and have
    nothing to do with the algorithm under study, so they would distort an
    aggregate over all launches.

    Args:
        frame: Table returned by :func:`load_reports`.
        patterns: Regular expressions matched (``re.search``) against both the
            short kernel name and the full signature.

    Returns:
        The table without the matching launches.
    """
    if not patterns or frame.empty:
        return frame
    compiled = [re.compile(pattern) for pattern in patterns]
    matched = frame.apply(
        lambda row: any(
            expression.search(str(row[KERNEL]))
            or expression.search(str(row[KERNEL_SIGNATURE]))
            for expression in compiled
        ),
        axis=1,
    )
    for name in sorted(set(frame.loc[matched, KERNEL].astype(str))):
        logger.info(f"Excluding kernel from the plot: {name}")
    return frame[~matched].reset_index(drop=True)


def aggregate_kernels(frame: pd.DataFrame, mode: str = "sum") -> pd.DataFrame:
    """Collapse the kernel launches of each executable into plot points.

    Args:
        frame: Table returned by :func:`load_reports`.
        mode: ``sum`` adds up the work and the time of every kernel launch,
            which places one point per implementation; ``dominant`` keeps only
            the longest-running kernel; ``none`` keeps every launch.

    Returns:
        The rows to plot, with intensity and performance recomputed for ``sum``.
    """
    if mode == "none" or frame.empty:
        return frame.copy()
    if mode == "dominant":
        index = frame.groupby(EXECUTABLE)[DURATION].idxmax()
        return frame.loc[index].reset_index(drop=True)

    grouped = frame.groupby(EXECUTABLE, as_index=False).agg(
        {
            BENCHMARK_PROBLEM: "first",
            PARADIGM: "first",
            PRECISION: "first",
            HARDWARE: "first",
            MEMORY_LEVEL: "first",
            DURATION: "sum",
            FLOP: "sum",
            MEMORY_TRAFFIC: "sum",
            PEAK_PERFORMANCE: "max",
            PEAK_BANDWIDTH: "max",
        }
    )
    grouped[KERNEL] = "all kernels"
    grouped[PERFORMANCE] = grouped[FLOP] / grouped[DURATION]
    grouped[ARITHMETIC_INTENSITY] = grouped[FLOP] / grouped[MEMORY_TRAFFIC]
    return grouped
