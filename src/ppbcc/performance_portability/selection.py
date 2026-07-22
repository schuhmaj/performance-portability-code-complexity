"""Load and select benchmark CSV data for portability analysis."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from ppbcc.constants import (
    BENCHMARK_PROBLEM,
    DESCRIPTION,
    HARDWARE,
    PARADIGM,
    PRECISION,
    PROBLEM_SIZE,
    TIME_UNIT,
    TIME_UNIT_TO_NS,
    WALL_CLOCK_TIME,
)

ALL_SIZE = "all"
AVERAGE_SIZE = "average"
BEST_SIZE = "best"
WORST_SIZE = "worst"


def parse_problem_size(value: str) -> float | str:
    """Parse a numeric problem size or a size-summary sentinel.

    Args:
        value: Numeric size or ``all``, ``average``, ``best``, or ``worst``.

    Returns:
        Numeric size or a normalized summary sentinel.

    Raises:
        argparse.ArgumentTypeError: If the value is unsupported.
    """
    normalized = value.casefold()
    if normalized == ALL_SIZE:
        return ALL_SIZE
    if normalized in {"avg", AVERAGE_SIZE, "mean"}:
        return AVERAGE_SIZE
    if normalized in {BEST_SIZE, WORST_SIZE}:
        return normalized
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "problem size must be numeric, 'avg', 'average', 'mean', "
            "'best', 'worst', or 'all'"
        ) from error


def configure_logging(verbosity: int) -> None:
    """Configure Loguru consistently with ``benchmark.py``.

    Args:
        verbosity: Number of ``-v`` flags supplied by the user.
    """
    logger.remove()
    logger.add(sys.stdout, level=["INFO", "DEBUG", "TRACE"][min(verbosity, 2)])


def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    """Validate that a DataFrame contains a set of columns.

    Args:
        df: DataFrame to validate.
        columns: Required column names.
        source: Input file used in an error message.

    Raises:
        ValueError: If one or more columns are absent.
    """
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_benchmark_csvs(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate benchmark result CSV files.

    Args:
        paths: CSV files produced by ``benchmark.py``.

    Returns:
        Concatenated benchmark data with a source-file column.

    Raises:
        FileNotFoundError: If an input path does not exist.
        ValueError: If a file does not have the benchmark output schema.
    """
    required = {
        BENCHMARK_PROBLEM,
        PARADIGM,
        DESCRIPTION,
        PRECISION,
        HARDWARE,
        PROBLEM_SIZE,
        WALL_CLOCK_TIME,
        TIME_UNIT,
    }
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark CSV does not exist: {path}")
        frame = pd.read_csv(path)
        _require_columns(frame, required, path)
        frame["Source File"] = str(path)
        frames.append(frame)
        logger.debug(f"Read {len(frame)} rows from {path}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(combined)} benchmark rows from {len(paths)} file(s)")
    return combined


def _resolve_unique_value(values: pd.Series, query: str, label: str) -> str:
    """Resolve a user query to one exact or uniquely partial-matching value.

    Args:
        values: Candidate values.
        query: User-supplied query.
        label: Human-readable value type for errors.

    Returns:
        The resolved value as it appears in the input.

    Raises:
        ValueError: If no value or more than one value matches.
    """
    candidates = sorted({str(value) for value in values.dropna().unique()})
    exact = [value for value in candidates if value.casefold() == query.casefold()]
    if len(exact) == 1:
        return exact[0]

    partial = [value for value in candidates if query.casefold() in value.casefold()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(
            f"No {label} matches {query!r}. Available values: {candidates}"
        )
    raise ValueError(f"{label.capitalize()} {query!r} is ambiguous; matches: {partial}")


def _compile_description_filter(
    pattern: str | None, option: str
) -> re.Pattern[str] | None:
    """Compile an optional description regular expression.

    Args:
        pattern: User-supplied regular expression, or ``None``.
        option: Command-line option name used in errors.

    Returns:
        A compiled expression, or ``None``.

    Raises:
        ValueError: If the expression is invalid.
    """
    if pattern is None:
        return None
    try:
        return re.compile(pattern)
    except re.error as error:
        raise ValueError(
            f"Invalid {option} description regex {pattern!r}: {error}"
        ) from error


def _description_is_workload(df: pd.DataFrame) -> bool:
    """Infer whether Description identifies workloads rather than variants.

    A descriptor is treated as a workload dimension when each selected,
    non-empty description is implemented by at least half of the paradigms.
    This identifies the polyhedral model names while retaining implementation
    variants such as ``Cuda[Naive]`` and ``Cuda[Cublas]`` as applications.

    Args:
        df: Filtered benchmark rows.

    Returns:
        Whether Description should be part of the workload identity.
    """
    described = df.loc[df[DESCRIPTION].ne("")]
    if described.empty:
        return False
    paradigm_count = max(df[PARADIGM].nunique(), 1)
    coverage = described.groupby(DESCRIPTION)[PARADIGM].nunique() / paradigm_count
    logger.debug(f"Description coverage ratios: {coverage.to_dict()}")
    return bool((coverage >= 0.5).all())


def _filter_problem_size(
    df: pd.DataFrame, problem_size: float, problem_name: str
) -> pd.DataFrame:
    """Select one exact numeric problem size from already filtered rows."""
    numeric_sizes = pd.to_numeric(df[PROBLEM_SIZE], errors="coerce")
    size_mask = np.isclose(
        numeric_sizes.to_numpy(dtype=float), problem_size, equal_nan=False
    )
    selected = df.loc[size_mask].copy()
    if selected.empty:
        available = sorted(numeric_sizes.dropna().unique())
        raise ValueError(
            f"Problem size {problem_size:g} has no rows for {problem_name}. "
            f"Available sizes: {available}"
        )
    return selected


def select_problem_rows(
    df: pd.DataFrame,
    name: str,
    description_include: str | None,
    description_exclude: str | None,
    problem_size: float | None,
    precision: int | None,
) -> tuple[pd.DataFrame, str, bool]:
    """Select a problem and optionally filter its descriptions.

    Args:
        df: Combined benchmark results.
        name: Requested benchmark problem.
        description_include: Optional regular expression descriptions must match.
        description_exclude: Optional regular expression descriptions must not match.
        problem_size: Optional exact problem size.
        precision: Optional floating-point precision in bits.

    Returns:
        A tuple of filtered rows, resolved problem name, and a flag indicating
        whether Description is a workload dimension.

    Raises:
        ValueError: If selection leaves no rows or required values are invalid.
    """
    resolved_name = _resolve_unique_value(df[BENCHMARK_PROBLEM], name, "problem")
    selected = df.loc[df[BENCHMARK_PROBLEM] == resolved_name].copy()
    selected[PARADIGM] = selected[PARADIGM].fillna("").astype(str).str.strip()
    selected[DESCRIPTION] = selected[DESCRIPTION].fillna("").astype(str).str.strip()
    selected[HARDWARE] = selected[HARDWARE].fillna("").astype(str).str.strip()

    if selected[PARADIGM].eq("").any():
        raise ValueError("Selected benchmark rows contain an empty Paradigm value.")
    if selected[HARDWARE].eq("").any():
        raise ValueError("Selected benchmark rows contain an empty Hardware value.")

    # Classify Description before filtering so a single selected variant does
    # not look like a workload implemented by every remaining paradigm.
    description_is_workload = _description_is_workload(selected)

    include_expression = _compile_description_filter(description_include, "--include")
    if include_expression is not None:
        mask = selected[DESCRIPTION].map(
            lambda value: include_expression.search(value) is not None
        )
        selected = selected.loc[mask].copy()
        if selected.empty:
            raise ValueError(
                f"Description include regex {description_include!r} matched no "
                f"rows for {resolved_name}."
            )

    exclude_expression = _compile_description_filter(description_exclude, "--exclude")
    if exclude_expression is not None:
        mask = selected[DESCRIPTION].map(
            lambda value: exclude_expression.search(value) is None
        )
        selected = selected.loc[mask].copy()
        if selected.empty:
            raise ValueError(
                f"Description exclude regex {description_exclude!r} excluded "
                f"all rows for {resolved_name}."
            )

    if problem_size is not None:
        selected = _filter_problem_size(selected, problem_size, resolved_name)

    if precision is not None:
        numeric_precision = pd.to_numeric(selected[PRECISION], errors="coerce")
        selected = selected.loc[numeric_precision.eq(precision)].copy()
        if selected.empty:
            available = sorted(
                pd.to_numeric(
                    df.loc[df[BENCHMARK_PROBLEM] == resolved_name, PRECISION],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )
            raise ValueError(
                f"Precision {precision} has no rows for {resolved_name}. "
                f"Available precisions: {available}"
            )

    logger.info(
        f"Selected {len(selected)} rows for {resolved_name} on "
        f"{selected[HARDWARE].nunique()} hardware platform(s)"
    )
    logger.debug(
        "Treating Description as a "
        + ("workload dimension" if description_is_workload else "variant label")
    )
    return selected, resolved_name, description_is_workload


def _application_label(paradigm: str, description: str, is_workload: bool) -> str:
    """Construct an application label from a paradigm and description.

    Args:
        paradigm: Programming paradigm or framework.
        description: Optional implementation description.
        is_workload: Whether descriptions identify workloads.

    Returns:
        A display/application identity.
    """
    if description and not is_workload:
        return f"{paradigm}[{description}]"
    return paradigm


def _convert_runtime_to_ns(df: pd.DataFrame) -> pd.Series:
    """Convert wall-clock runtimes to nanoseconds.

    Args:
        df: Selected benchmark rows.

    Returns:
        Numeric runtimes expressed in nanoseconds.

    Raises:
        ValueError: If a time unit is unsupported.
    """
    units = df[TIME_UNIT].fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(units) - set(TIME_UNIT_TO_NS))
    if unknown:
        raise ValueError(
            f"Unsupported values in {TIME_UNIT!r}: {unknown}. "
            f"Supported units: {sorted(TIME_UNIT_TO_NS)}"
        )
    runtimes = pd.to_numeric(df[WALL_CLOCK_TIME], errors="coerce")
    factors = units.map(TIME_UNIT_TO_NS).astype(float)
    return runtimes * factors
