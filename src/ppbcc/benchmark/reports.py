"""Normalize Google Benchmark JSON reports into tidy tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from ppbcc.constants import (
    BENCHMARK_PROBLEM,
    COLUMN_LIST,
    DESCRIPTION,
    HARDWARE,
    ITERATIONS,
    PARADIGM,
    PRECISION,
    PROBLEM_SIZE,
    RAW_KEY_TO_COLUMN,
    RUNTIME_COLUMNS,
)


def _normalize_report(json_data: dict, hardware: str) -> pd.DataFrame:
    """Turn a parsed Google Benchmark report into a tidy table.

    Args:
        json_data: Parsed report object.
        hardware: Hardware label to attach to every row.

    Returns:
        Normalized benchmark rows.
    """
    context = json_data["context"]
    paradigm = context.get("paradigm", "Unknown")
    float_type = str(context.get("float_type", "Unknown"))

    df = pd.DataFrame(json_data["benchmarks"])
    if df.empty:
        return df

    # Drop aggregate rows (BigO / RMS / mean / stddev ...); we only want raw runs.
    if "run_type" in df.columns:
        df = df[df["run_type"] == "iteration"].copy()
    if df.empty:
        return df

    # --- Parse the encoded name -------------------------------------------- #
    # Convention: "<BenchmarkProblem>[-<Description>][/<ProblemSize>]"
    name_size = df["name"].str.split("/", expand=True)
    base = name_size[0]
    has_size_in_name = name_size.shape[1] > 1

    details = base.str.split("-", n=1, expand=True)
    df[BENCHMARK_PROBLEM] = details[0].astype(str)
    df[DESCRIPTION] = details[1].fillna("").astype(str) if details.shape[1] > 1 else ""

    # --- Problem size ------------------------------------------------------- #
    if has_size_in_name:
        df[PROBLEM_SIZE] = pd.to_numeric(name_size[1], errors="coerce")
    elif "NumFaces" in df.columns:
        # The polyhedral benchmark has no size in the name; it reports the number
        # of faces of the processed mesh as a user counter instead.
        df[PROBLEM_SIZE] = pd.to_numeric(df["NumFaces"], errors="coerce")
    else:
        df[PROBLEM_SIZE] = pd.NA

    # --- Context columns ---------------------------------------------------- #
    df[PARADIGM] = paradigm
    df[PRECISION] = float_type
    df[HARDWARE] = hardware

    # --- Rename the runtime/counter columns we know about ------------------- #
    df = df.rename(columns={k: v for k, v in RAW_KEY_TO_COLUMN.items() if k in df})

    # Ensure every output column exists, then select & order them.
    for col in COLUMN_LIST:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COLUMN_LIST]

    # Coerce numeric columns.
    for col in [PROBLEM_SIZE, ITERATIONS, *RUNTIME_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_reports(report_files: list[Path], hardware: str = "") -> pd.DataFrame:
    """Load and normalize a list of Google-Benchmark JSON reports.

    Args:
        report_files: Paths to JSON report files.
        hardware: Free-form hardware identifier stored in the ``Hardware`` column
            (e.g. ``"RTX5080"``).

    Returns:
        A single tidy DataFrame (one row per benchmark run).
    """
    frames: list[pd.DataFrame] = []
    logger.info(f"Loading {len(report_files)} report file(s)")
    for file_path in report_files:
        logger.debug(f"Loading {file_path}")
        try:
            with open(file_path, "r") as fh:
                json_data = json.load(fh)
        except (ValueError, OSError) as e:
            logger.error(f"Could not read {file_path}: {e}")
            continue
        df = _normalize_report(json_data, hardware)
        if df.empty:
            logger.warning(f"No usable benchmark rows in {file_path}")
            continue
        frames.append(df)

    if not frames:
        logger.error("No benchmark data could be loaded.")
        return pd.DataFrame(columns=COLUMN_LIST)

    combined = pd.concat(frames, ignore_index=True)
    logger.success(
        f"Loaded {len(combined)} rows across "
        f"{combined[BENCHMARK_PROBLEM].nunique()} problem(s) and "
        f"{combined[PARADIGM].nunique()} paradigm(s)"
    )
    return combined


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
