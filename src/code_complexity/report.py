"""Result assembly: metric selection, DataFrame construction, CSV export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from tabulate import tabulate

#: Identifier columns always present in the report.
ID_COLUMNS: tuple[str, ...] = ("file", "dialect")

#: Line-metric columns (see :mod:`code_complexity.loc`).
LOC_COLUMNS: tuple[str, ...] = ("loc", "sloc", "comment_lines", "blank_lines")

#: Halstead columns (see :mod:`code_complexity.halstead`), in report order.
HALSTEAD_COLUMNS: tuple[str, ...] = (
    "distinct_operators",
    "distinct_operands",
    "total_operators",
    "total_operands",
    "vocabulary",
    "length",
    "calculated_length",
    "volume",
    "difficulty",
    "effort",
    "time_seconds",
    "delivered_bugs",
    "program_level",
    "language_level",
)

#: Columns counting the tokens contributed by the GPU dialect(s).
DIALECT_COLUMNS: tuple[str, ...] = (
    "dialect_distinct_operators",
    "dialect_total_operators",
    "dialect_distinct_operands",
    "dialect_total_operands",
)

#: Columns that additionally get ``baseline_``/``delta_`` variants in diff
#: mode (baseline = metrics of the code with all dialect tokens removed).
DIFF_COLUMNS: tuple[str, ...] = (
    "distinct_operators",
    "distinct_operands",
    "total_operators",
    "total_operands",
    "vocabulary",
    "length",
    "volume",
    "difficulty",
    "effort",
)

#: Metric-group names accepted by ``evaluate(metrics=...)`` and the CLI.
METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "all": LOC_COLUMNS + HALSTEAD_COLUMNS + DIALECT_COLUMNS,
    "loc": LOC_COLUMNS,
    "lines_of_code": LOC_COLUMNS,
    "sloc": ("sloc",),
    "source_lines_of_code": ("sloc",),
    "halstead": HALSTEAD_COLUMNS,
    "dialect": DIALECT_COLUMNS,
    "dialect_operators": DIALECT_COLUMNS,
}


def _normalize_metric(name: str) -> str:
    """Normalises a user-provided metric name.

    Args:
        name: Metric name, e.g. ``"Halstead_Effort"`` or ``"Lines of Code"``.

    Returns:
        Lower-case, underscore-separated form, e.g. ``"halstead_effort"``.
    """
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_metric_columns(metrics: list[str] | None) -> list[str]:
    """Resolves metric names/groups to the report columns they select.

    Accepted names are the group names in :data:`METRIC_GROUPS`, any single
    column name (e.g. ``"effort"``, ``"sloc"``) and Halstead columns with a
    ``halstead_`` prefix (e.g. ``"halstead_effort"``). Matching is
    case-insensitive; spaces and dashes are treated as underscores.

    Args:
        metrics: Requested metric names, or ``None``/empty for all metrics.

    Returns:
        The selected column names in canonical report order (without the
        identifier columns).

    Raises:
        ValueError: If a metric name is unknown.
    """
    if not metrics:
        return list(METRIC_GROUPS["all"])

    all_columns = METRIC_GROUPS["all"]
    selected: set[str] = set()
    for metric in metrics:
        name = _normalize_metric(metric)
        if name in METRIC_GROUPS:
            selected.update(METRIC_GROUPS[name])
            continue
        if name.removeprefix("halstead_") in HALSTEAD_COLUMNS:
            selected.add(name.removeprefix("halstead_"))
            continue
        if name in all_columns:
            selected.add(name)
            continue
        known = sorted(set(METRIC_GROUPS) | set(all_columns))
        raise ValueError(f"Unknown metric {metric!r}. Known metrics: {', '.join(known)}")
    return [column for column in all_columns if column in selected]


def report_columns(metrics: list[str] | None, diff: bool) -> list[str]:
    """Builds the full column list of the report.

    Args:
        metrics: Requested metric names (see :func:`resolve_metric_columns`).
        diff: If True, every selected column in :data:`DIFF_COLUMNS` is
            followed by its ``baseline_`` and ``delta_`` variant.

    Returns:
        The ordered column names, starting with :data:`ID_COLUMNS`.
    """
    columns = list(ID_COLUMNS)
    for column in resolve_metric_columns(metrics):
        columns.append(column)
        if diff and column in DIFF_COLUMNS:
            columns.append(f"baseline_{column}")
            columns.append(f"delta_{column}")
    return columns


def to_dataframe(rows: list[dict], metrics: list[str] | None, diff: bool) -> pd.DataFrame:
    """Builds the report DataFrame from per-file result rows.

    Args:
        rows: One dictionary per analysed file containing all metric values.
        metrics: Requested metric names (see :func:`resolve_metric_columns`).
        diff: Whether to include ``baseline_``/``delta_`` columns.

    Returns:
        DataFrame with one row per file and the selected metric columns.
    """
    columns = report_columns(metrics, diff)
    frame = pd.DataFrame(rows, columns=columns)
    return frame


def format_table(frame: pd.DataFrame, table_format: str = "rounded_outline") -> str:
    """Pretty-prints the report DataFrame as a text table.

    Args:
        frame: The report DataFrame.
        table_format: Any table format supported by ``tabulate`` (e.g.
            ``"rounded_outline"``, ``"github"``, ``"psql"``, ``"simple"``).

    Returns:
        The rendered table; floats are shown with six significant digits.
    """
    return tabulate(
        frame,
        headers="keys",
        tablefmt=table_format,
        showindex=False,
        floatfmt=".6g",
    )


def save_csv(frame: pd.DataFrame, path: Path, separator: str = ",") -> None:
    """Writes the report DataFrame to a CSV file.

    Args:
        frame: The report DataFrame.
        path: Destination file; parent directories are created as needed.
        separator: CSV field separator.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=separator, index=False)
    logger.success("Report with {} rows written to {}", len(frame), path)
