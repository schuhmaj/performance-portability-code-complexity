"""Read ``.ncu-rep`` reports and turn them into tidy roofline tables."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from ppbcc.constants import (
    AGGREGATED_KERNEL,
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
    REGION,
)
from ppbcc.profiling.metrics import (
    MEMORY_LEVEL_CLOCK,
    MEMORY_LEVELS,
    PRECISIONS,
    ROOFLINE_METRICS,
    column_unit,
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

#: Substring identifying the column holding the NVTX push/pop stack a launch
#: happened under. The full header spells out every payload field, so it is
#: matched rather than written out.
_NVTX_COLUMN_MARKER = "Push/Pop_Range"

#: NVTX domain the benchmark's own regions live in. Third-party libraries open
#: domains of their own -- CCCL brackets every ``thrust::reduce`` with a
#: ``CCCL:cub::DeviceReduce::Reduce`` range -- and those sit *inside* the
#: benchmark's range, so the innermost entry is the wrong one to read.
_NVTX_DOMAIN = "<default domain>"

#: One entry of the stack, as the profiler's CSV quotes it.
_NVTX_ENTRY = re.compile(r'"([^"]*)"')


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


def _nvtx_region(stack: object) -> str:
    """Read the benchmark's region name out of an NVTX push/pop stack.

    Args:
        stack: One cell of the profiler's NVTX column, e.g.
            ``12345  "<default domain>:evaluate:none:..."``. Every field after
            the range name is an NVTX payload the benchmark does not use.

    Returns:
        The innermost range of :data:`_NVTX_DOMAIN`, or ``""`` for a launch
        that happened outside every region -- a framework bootstrap kernel, or
        a binary built without ``PPB_ENABLE_NVTX``.
    """
    if not isinstance(stack, str) or not stack.strip():
        return ""
    names = [
        fields[1]
        for entry in _NVTX_ENTRY.findall(stack)
        for fields in [entry.split(":")]
        if len(fields) > 1 and fields[0] == _NVTX_DOMAIN and fields[1]
    ]
    return names[-1] if names else ""


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


#: Places the ``ncu_report`` Python module ships in, tried in order when it is not
#: importable already. On macOS the host application carries it; on Linux the
#: ``extras/python`` folder of every installed Nsight Compute version does.
_NCU_REPORT_LOCATIONS = [
    Path("/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python"),
    *sorted(Path("/opt/nvidia/nsight-compute").glob("*/extras/python"), reverse=True),
]


def _ncu_report_module():
    """Import Nsight Compute's ``ncu_report`` module.

    Returns:
        The imported module.

    Raises:
        ImportError: If neither the Python path nor a known installation
            provides it.
    """
    try:
        import ncu_report  # type: ignore[import-not-found]

        return ncu_report
    except ImportError:
        pass
    for location in _NCU_REPORT_LOCATIONS:
        if (location / "ncu_report.py").is_file():
            sys.path.append(str(location))
            import ncu_report  # type: ignore[import-not-found]

            logger.debug(f"Using ncu_report from {location}")
            return ncu_report
    raise ImportError(
        "Neither the Nsight Compute CLI nor its 'ncu_report' Python module is "
        "available. Install Nsight Compute or add its python folder to PYTHONPATH."
    )


def _import_api(report: Path, metrics: list[str]) -> pd.DataFrame:
    """Read one profiler report through Nsight Compute's Python API.

    This is the route on a host without the ``ncu`` CLI, e.g. the macOS
    application, which ships only the Linux CLI but the report reader for the
    host itself. The values are the ones the CLI prints, at full precision
    rather than rounded to two decimals. Kernel names are the demangled ones;
    the CLI additionally shortens some namespaces (``Kokkos::Impl::`` becomes
    ``Kokkos::``), which the API does not expose.

    Args:
        report: Path to a ``.ncu-rep`` file.
        metrics: Metric names to read; missing ones are skipped.

    Returns:
        One row per kernel launch with the columns :func:`_pivot` produces,
        except that the region is already resolved into :data:`REGION`.
    """
    ncu_report = _ncu_report_module()
    context = ncu_report.load_report(report)
    rows: list[dict[str, object]] = []
    for range_index in range(context.num_ranges()):
        for launch_id, action in enumerate(context.range_by_idx(range_index)):
            available = set(action.metric_names())

            def dims(prefix: str) -> str:
                values = [
                    action.metric_by_name(f"{prefix}_dim_{axis}").value()
                    if f"{prefix}_dim_{axis}" in available
                    else 1
                    for axis in "xyz"
                ]
                return f"({values[0]}, {values[1]}, {values[2]})"

            state = action.nvtx_state()
            region = ""
            for domain in state.domains():
                info = state.domain_by_id(domain)
                if info.name() != _NVTX_DOMAIN:
                    continue
                names = [
                    info.push_pop_range(index).name()
                    for index in range(len(info.push_pop_ranges()))
                ]
                region = next((name for name in reversed(names) if name), "")
            row: dict[str, object] = {
                "ID": launch_id,
                "Kernel Name": action.name(action.NameBase_DEMANGLED),
                BLOCK_SIZE: dims("launch__block"),
                GRID_SIZE: dims("launch__grid"),
                REGION: region,
            }
            for metric in metrics:
                if metric in available:
                    value = action.metric_by_name(metric).value()
                    row[metric] = float(value) if value is not None else float("nan")
            rows.append(row)
    frame = pd.DataFrame(rows)
    # Same column order as the pivoted CLI export: keys first, metrics sorted.
    keys = ["ID", "Kernel Name", BLOCK_SIZE, GRID_SIZE, REGION]
    return frame[[*keys, *sorted(c for c in frame if c not in keys)]] if rows else frame


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
        index=_LAUNCH_KEYS
        + [column for column in long_table if _NVTX_COLUMN_MARKER in column],
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
    ncu: str | None = "ncu",
    hardware: str = "",
    precision: str = "auto",
    memory_level: str = "dram",
) -> pd.DataFrame:
    """Load profiler reports into one tidy table, one row per kernel launch.

    Args:
        reports: Paths to ``.ncu-rep`` files.
        ncu: Profiler executable used to re-import the reports, or ``None`` to
            read them through Nsight Compute's ``ncu_report`` Python module
            instead (a host without the CLI, such as macOS).
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
            if ncu is None:
                frame = _import_api(report, ROOFLINE_METRICS)
            else:
                long_table = _import_csv(report, ncu)
                frame = pd.DataFrame() if long_table.empty else _pivot(long_table)
        except subprocess.CalledProcessError as error:
            logger.error(f"Could not import {report}: {error}")
            continue
        if frame.empty:
            logger.warning(
                f"{report.name} contains no profiled kernels — the paradigm most "
                "likely does not run on CUDA (Nsight Compute sees no OpenCL, "
                "Vulkan or host-side work)"
            )
            continue

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
        # The NVTX column is only there when the report was collected with
        # --nvtx against a binary built with PPB_ENABLE_NVTX.
        nvtx = [column for column in frame if _NVTX_COLUMN_MARKER in column]
        if REGION not in frame:
            frame[REGION] = frame[nvtx[0]].map(_nvtx_region) if nvtx else ""
        # "Block Size"/"Grid Size" already carry their final names.
        frame = frame.drop(columns=["ID", "Kernel Name", *nvtx])

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


#: A unit appended to a CSV header, e.g. ``Arithmetic Intensity [FLOP/Byte]``.
_UNIT_SUFFIX = re.compile(r"\s+\[[^\[\]]*\]$")


def with_units(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the unit to every column that has one, for writing the CSV.

    Args:
        frame: A profiling table with bare column names.

    Returns:
        A copy whose headers read ``<name> [<unit>]``.
    """
    return frame.rename(
        columns={
            column: f"{column} [{unit}]"
            for column in frame
            if (unit := column_unit(str(column))) is not None
        }
    )


def without_units(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip the units :func:`with_units` appended, for reading a CSV back.

    Args:
        frame: A profiling table read from a CSV, with or without units.

    Returns:
        The table with bare column names.
    """
    return frame.rename(columns=lambda column: _UNIT_SUFFIX.sub("", str(column)))


def load_csv(paths: list[Path]) -> pd.DataFrame:
    """Concatenate consolidated profiling CSVs into one table.

    No single profiler covers every paradigm in this benchmark: Nsight Compute
    reads counters only inside a CUDA context, so Vulkan and OpenCL have to be
    measured with other tools. Each tool writes the same columns, and this is
    how their tables become one roofline.

    Args:
        paths: Paths to CSVs written by ``ppbcc profile``.

    Returns:
        The concatenated table, deduplicated on (executable, kernel id).
        Empty if nothing could be read.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = without_units(pd.read_csv(path))
        except (OSError, ValueError) as error:
            logger.error(f"Could not read {path}: {error}")
            continue
        missing = [column for column in (EXECUTABLE, PERFORMANCE) if column not in frame]
        if missing:
            logger.error(f"{path} is not a ppbcc profiling CSV (missing {missing})")
            continue
        logger.info(f"{path.name}: {len(frame)} row(s)")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=PROFILE_COLUMN_LIST)
    combined = pd.concat(frames, ignore_index=True)
    keys = [
        column
        for column in (EXECUTABLE, REGION, KERNEL_ID, KERNEL)
        if column in combined
    ]
    before = len(combined)
    combined = combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    if len(combined) < before:
        logger.warning(
            f"Dropped {before - len(combined)} duplicate row(s); a later CSV wins"
        )
    logger.success(
        f"Loaded {len(combined)} row(s) across "
        f"{combined[EXECUTABLE].nunique()} executable(s)"
    )
    return combined


def filter_regions(frame: pd.DataFrame, keep_unlabelled: bool = False) -> pd.DataFrame:
    """Keep only the launches that happened inside a named region.

    The benchmark brackets the work it is about -- ``matmul``, ``init``,
    ``evaluate`` -- with the NVTX ranges of ``src/common/Marker.h``. Everything
    outside them is the runtime setting itself up: Kokkos' architecture query,
    desul's lock-array initialization, a framework's buffer staging. Those
    launches say nothing about the algorithm and would distort any aggregate
    over an executable, so they do not belong in the table.

    An executable that carries no region at all is kept as it is, with a
    warning: it was built without ``PPB_ENABLE_NVTX``, profiled without
    ``--nvtx``, or measured by a backend that cannot see NVTX ranges -- Nsight
    Graphics traces a queue submission, which carries none. Dropping every one
    of its rows would hide that.

    Args:
        frame: Table returned by one of the backends' ``load_reports``.
        keep_unlabelled: Keep the launches outside every region as well.

    Returns:
        The table without the unlabelled launches.
    """
    if frame.empty or REGION not in frame or keep_unlabelled:
        return frame
    region = frame[REGION].fillna("").astype(str)
    labelled = region.str.len() > 0
    # Per executable, because "no region anywhere" means something different
    # (unannotated binary) than "no region on this launch" (bootstrap kernel).
    annotated = frame[EXECUTABLE].isin(set(frame.loc[labelled, EXECUTABLE]))
    for name in sorted(set(frame.loc[~annotated, EXECUTABLE].astype(str))):
        logger.warning(
            f"{name}: no region on any launch, so every row is kept. Either "
            "the binary was built without -DPPB_ENABLE_NVTX=ON, or the backend "
            "cannot see NVTX ranges at all (Nsight Graphics traces a queue "
            "submission, which carries none)."
        )
    dropped = annotated & ~labelled
    for name in sorted(set(frame.loc[dropped, KERNEL].astype(str))):
        logger.info(f"Dropping kernel outside every region: {name}")
    return frame[~dropped].reset_index(drop=True)


def select_regions(frame: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    """Keep only the regions whose name matches one of the given patterns.

    Args:
        frame: Table returned by :func:`filter_regions`.
        patterns: Regular expressions matched (``re.search``) against the
            region name; an empty list keeps every region.

    Returns:
        The table restricted to the matching regions.
    """
    if not patterns or frame.empty or REGION not in frame:
        return frame
    compiled = [re.compile(pattern) for pattern in patterns]
    region = frame[REGION].fillna("").astype(str)
    keep = region.map(lambda name: any(p.search(name) for p in compiled))
    for name in sorted(set(region[~keep])):
        logger.info(f"Excluding region from the plot: {name or '(unnamed)'}")
    return frame[keep].reset_index(drop=True)


def aggregate_kernels(frame: pd.DataFrame, mode: str = "sum") -> pd.DataFrame:
    """Collapse the kernel launches of each executable into plot points.

    A region -- ``matmul``, ``init``, ``evaluate`` -- is the unit of work the
    benchmark named, so it is also the unit the points are grouped by. An
    implementation with a separate initialization kernel therefore contributes
    two points, one per phase, rather than one point mixing them.

    Args:
        frame: Table returned by :func:`load_reports`.
        mode: ``sum`` adds up the work and the time of every kernel launch of a
            region, which places one point per implementation and region;
            ``dominant`` keeps only the longest-running kernel of each;
            ``none`` keeps every launch.

    Returns:
        The rows to plot, with intensity and performance recomputed for ``sum``.
    """
    if mode == "none" or frame.empty:
        return frame.copy()
    keys = [EXECUTABLE] + ([REGION] if REGION in frame else [])
    if mode == "dominant":
        index = frame.groupby(keys)[DURATION].idxmax()
        return frame.loc[index].reset_index(drop=True)

    grouped = frame.groupby(keys, as_index=False).agg(
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
    grouped[KERNEL] = AGGREGATED_KERNEL
    grouped[PERFORMANCE] = grouped[FLOP] / grouped[DURATION]
    grouped[ARITHMETIC_INTENSITY] = grouped[FLOP] / grouped[MEMORY_TRAFFIC]
    return grouped
