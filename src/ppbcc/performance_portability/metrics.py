"""Calculate application efficiency and performance portability."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from ppbcc.constants import (
    APPLICATION,
    APPLICATION_EFFICIENCY,
    DESCRIPTION,
    HARDWARE,
    PARADIGM,
    PERFORMANCE_PORTABILITY,
    PRECISION,
    PROBLEM_SIZE,
)
from ppbcc.performance_portability.selection import (
    AVERAGE_SIZE,
    BEST_SIZE,
    WORST_SIZE,
    _application_label,
    _convert_runtime_to_ns,
)


def calculate_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
    non_zero_pp: bool = False,
    hardware_universe: Iterable[str] | None = None,
    application_universe: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate application efficiency and performance portability.

    Duplicate measurements are reduced to their median runtime. Application
    efficiencies are computed per hardware/workload and then arithmetically
    averaged over workloads. PP is the harmonic mean across all selected
    hardware. By default, an absent result contributes zero and makes PP zero.
    In non-zero PP mode, platforms with zero efficiency are omitted from the
    harmonic mean; the application-efficiency output still retains those zeros.

    Args:
        df: Selected benchmark rows.
        description_is_workload: Whether Description belongs to the workload
            key rather than the application label.
        non_zero_pp: Whether to exclude unsupported platforms from PP.
        hardware_universe: Optional complete platform set. This is used for
            per-size scaling so a platform with no row at one size contributes
            zero rather than disappearing from that size's PP calculation.
        application_universe: Optional complete application set. This is used
            for per-size calculations so an implementation with no row at one
            size receives zero rather than disappearing from that size.

    Returns:
        A pair ``(efficiency, portability)``. The first DataFrame has one row
        per application/hardware; the second has one row per application.

    Raises:
        ValueError: If there are no positive numeric wall-clock runtimes.
    """
    data = df.copy()
    data[APPLICATION] = [
        _application_label(paradigm, description, description_is_workload)
        for paradigm, description in zip(data[PARADIGM], data[DESCRIPTION])
    ]
    data["Runtime (ns)"] = _convert_runtime_to_ns(data)
    invalid = data["Runtime (ns)"].isna() | data["Runtime (ns)"].le(0)
    if invalid.any():
        logger.warning(f"Dropping {int(invalid.sum())} rows with invalid runtimes")
        data = data.loc[~invalid].copy()
    if data.empty:
        raise ValueError("No positive numeric wall-clock runtimes remain.")

    workload_columns = [PROBLEM_SIZE]
    if PRECISION in data.columns and data[PRECISION].notna().any():
        workload_columns.append(PRECISION)
    if description_is_workload:
        workload_columns.append(DESCRIPTION)

    measurement_keys = [HARDWARE, APPLICATION, *workload_columns]
    runtimes = (
        data.groupby(measurement_keys, dropna=False, as_index=False)["Runtime (ns)"]
        .median()
        .sort_values(measurement_keys)
    )
    best_keys = [HARDWARE, *workload_columns]
    runtimes["Best Runtime (ns)"] = runtimes.groupby(best_keys, dropna=False)[
        "Runtime (ns)"
    ].transform("min")
    runtimes[APPLICATION_EFFICIENCY] = (
        runtimes["Best Runtime (ns)"] / runtimes["Runtime (ns)"]
    )

    applications = sorted(
        set(application_universe)
        if application_universe is not None
        else set(runtimes[APPLICATION].unique())
    )
    hardware = sorted(
        set(hardware_universe)
        if hardware_universe is not None
        else set(runtimes[HARDWARE].unique())
    )
    workloads = runtimes[workload_columns].drop_duplicates()
    full_index = pd.MultiIndex.from_frame(
        pd.DataFrame(
            [
                (application, platform, *workload)
                for application in applications
                for platform in hardware
                for workload in workloads.itertuples(index=False, name=None)
            ],
            columns=[APPLICATION, HARDWARE, *workload_columns],
        )
    )
    indexed = runtimes.set_index([APPLICATION, HARDWARE, *workload_columns])
    complete = indexed.reindex(full_index)
    complete.index.names = [APPLICATION, HARDWARE, *workload_columns]
    complete[APPLICATION_EFFICIENCY] = complete[APPLICATION_EFFICIENCY].fillna(0.0)
    complete = complete.reset_index()

    efficiency = (
        complete.groupby([APPLICATION, HARDWARE], as_index=False)[
            APPLICATION_EFFICIENCY
        ]
        .mean()
        .sort_values([APPLICATION, HARDWARE])
    )

    def harmonic_mean_or_zero(values: pd.Series) -> float:
        """Calculate PP, optionally omitting unsupported platforms."""
        numeric = values.to_numpy(dtype=float)
        if non_zero_pp:
            numeric = numeric[numeric > 0.0]
            if len(numeric) == 0:
                return 0.0
            return float(len(numeric) / np.reciprocal(numeric).sum())
        if len(numeric) != len(hardware) or np.any(numeric <= 0.0):
            return 0.0
        return float(len(numeric) / np.reciprocal(numeric).sum())

    portability = (
        efficiency.groupby(APPLICATION, as_index=False)[APPLICATION_EFFICIENCY]
        .agg(harmonic_mean_or_zero)
        .rename(columns={APPLICATION_EFFICIENCY: PERFORMANCE_PORTABILITY})
        .sort_values(PERFORMANCE_PORTABILITY, ascending=False)
    )
    logger.debug(f"Performance portability:\n{portability.to_string(index=False)}")
    return efficiency, portability


def calculate_metrics_by_size(
    df: pd.DataFrame,
    description_is_workload: bool,
    non_zero_pp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate both metrics independently for each problem size.

    Args:
        df: Selected rows for one benchmark problem.
        description_is_workload: Whether Description belongs to the workload key.
        non_zero_pp: Whether unsupported platforms are excluded from PP.

    Returns:
        Efficiency and portability DataFrames that retain problem size.
    """
    numeric_sizes = pd.to_numeric(df[PROBLEM_SIZE], errors="coerce")
    sizes = sorted(numeric_sizes.dropna().unique())
    if not sizes:
        raise ValueError("No numeric problem sizes remain for the scaling plot.")

    platforms = sorted(df[HARDWARE].unique())
    applications = sorted(
        {
            _application_label(paradigm, description, description_is_workload)
            for paradigm, description in zip(df[PARADIGM], df[DESCRIPTION])
        }
    )
    efficiency_frames: list[pd.DataFrame] = []
    portability_frames: list[pd.DataFrame] = []
    for size in sizes:
        size_rows = df.loc[
            np.isclose(numeric_sizes.to_numpy(dtype=float), size, equal_nan=False)
        ]
        efficiency, portability = calculate_metrics(
            size_rows,
            description_is_workload,
            non_zero_pp=non_zero_pp,
            hardware_universe=platforms,
            application_universe=applications,
        )
        efficiency.insert(1, PROBLEM_SIZE, float(size))
        portability.insert(1, PROBLEM_SIZE, float(size))
        efficiency_frames.append(efficiency)
        portability_frames.append(portability)
    return (
        pd.concat(efficiency_frames, ignore_index=True),
        pd.concat(portability_frames, ignore_index=True),
    )


def calculate_scaling_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
    non_zero_pp: bool = False,
) -> pd.DataFrame:
    """Calculate performance portability independently for each problem size."""
    _, portability = calculate_metrics_by_size(
        df,
        description_is_workload,
        non_zero_pp=non_zero_pp,
    )
    return portability


def calculate_average_size_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
    non_zero_pp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average application efficiency and PP independently over sizes.

    Both metrics are first calculated independently at every size, then
    arithmetically averaged. This gives every problem size equal weight and
    avoids taking the harmonic mean of already size-averaged efficiencies when
    ``--size average`` was explicitly requested.
    """
    per_size_efficiency, per_size_portability = calculate_metrics_by_size(
        df,
        description_is_workload,
        non_zero_pp=non_zero_pp,
    )
    efficiency = (
        per_size_efficiency.groupby([APPLICATION, HARDWARE], as_index=False)[
            APPLICATION_EFFICIENCY
        ]
        .mean()
        .sort_values([APPLICATION, HARDWARE])
    )
    portability = (
        per_size_portability.groupby(APPLICATION, as_index=False)[
            PERFORMANCE_PORTABILITY
        ]
        .mean()
        .sort_values(PERFORMANCE_PORTABILITY, ascending=False)
    )
    return efficiency, portability


def calculate_extreme_size_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
    mode: str,
    non_zero_pp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select each application's best or worst metrics over problem sizes.

    Application efficiency is reduced independently for every application and
    hardware pair. Performance portability is reduced independently for every
    application, so both Cascade panels represent the requested size-axis
    extreme of the metric they display.

    Args:
        df: Selected rows for one benchmark problem.
        description_is_workload: Whether Description belongs to the workload key.
        mode: ``best`` for maxima or ``worst`` for minima.
        non_zero_pp: Whether unsupported platforms are excluded from PP.

    Returns:
        Application-efficiency and performance-portability extrema.

    Raises:
        ValueError: If mode is not ``best`` or ``worst``.
    """
    if mode not in {BEST_SIZE, WORST_SIZE}:
        raise ValueError(f"Unsupported size-extreme mode: {mode!r}")

    per_size_efficiency, per_size_portability = calculate_metrics_by_size(
        df,
        description_is_workload,
        non_zero_pp=non_zero_pp,
    )
    reduction = "max" if mode == BEST_SIZE else "min"
    efficiency = (
        per_size_efficiency.groupby([APPLICATION, HARDWARE], as_index=False)[
            APPLICATION_EFFICIENCY
        ]
        .agg(reduction)
        .sort_values([APPLICATION, HARDWARE])
    )
    portability = (
        per_size_portability.groupby(APPLICATION, as_index=False)[
            PERFORMANCE_PORTABILITY
        ]
        .agg(reduction)
        .sort_values(PERFORMANCE_PORTABILITY, ascending=False)
    )
    return efficiency, portability


def calculate_export_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
    non_zero_pp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate export metrics without combining sizes or precisions.

    Returns one efficiency row per application, size, precision, and hardware,
    and one portability row per application, size, and precision. Additional
    rows with Problem Size ``average`` contain per-precision arithmetic means
    over the independently calculated size metrics.
    """
    dimensions = [PROBLEM_SIZE, PRECISION]
    platforms = sorted(df[HARDWARE].unique())
    applications = sorted(
        {
            _application_label(paradigm, description, description_is_workload)
            for paradigm, description in zip(df[PARADIGM], df[DESCRIPTION])
        }
    )
    efficiency_frames: list[pd.DataFrame] = []
    portability_frames: list[pd.DataFrame] = []
    for values, rows in df.groupby(dimensions, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        efficiency, portability = calculate_metrics(
            rows,
            description_is_workload,
            non_zero_pp=non_zero_pp,
            hardware_universe=platforms,
            application_universe=applications,
        )
        for position, (column, value) in enumerate(zip(dimensions, values), start=1):
            efficiency.insert(position, column, value)
            portability.insert(position, column, value)
        efficiency_frames.append(efficiency)
        portability_frames.append(portability)
    if not efficiency_frames:
        raise ValueError("No problem-size/precision groups remain for CSV export.")
    efficiency = pd.concat(efficiency_frames, ignore_index=True)
    portability = pd.concat(portability_frames, ignore_index=True)
    average_efficiency = (
        efficiency.groupby(
            [APPLICATION, PRECISION, HARDWARE],
            as_index=False,
            dropna=False,
        )[APPLICATION_EFFICIENCY]
        .mean()
        .sort_values([APPLICATION, PRECISION, HARDWARE])
    )
    average_efficiency.insert(1, PROBLEM_SIZE, AVERAGE_SIZE)
    average_portability = (
        portability.groupby(
            [APPLICATION, PRECISION],
            as_index=False,
            dropna=False,
        )[PERFORMANCE_PORTABILITY]
        .mean()
        .sort_values([APPLICATION, PRECISION])
    )
    average_portability.insert(1, PROBLEM_SIZE, AVERAGE_SIZE)
    return (
        pd.concat([efficiency, average_efficiency], ignore_index=True),
        pd.concat([portability, average_portability], ignore_index=True),
    )
