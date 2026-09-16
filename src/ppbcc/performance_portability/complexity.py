"""Load and join code-complexity data with portability results."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ppbcc.constants import APPLICATION, PERFORMANCE_PORTABILITY, PROBLEM
from ppbcc.performance_portability.selection import (
    _require_columns,
    _resolve_unique_value,
)

COMPLEXITY_NAME = "Name"
COMPLEXITY_FRAMEWORK = "Framework"
METRIC_ALIASES = {
    "sloc": "SLOC",
    "halstead-vocabulary": "Halstead Vocabulary",
    "vocabulary": "Halstead Vocabulary",
    "eta": "Halstead Vocabulary",
    "halstead-program-length": "Halstead Program Length",
    "halstead-length": "Halstead Program Length",
    "program-length": "Halstead Program Length",
    "length": "Halstead Program Length",
    "n": "Halstead Program Length",
    "halstead-volume": "Halstead Volume",
    "volume": "Halstead Volume",
    "v": "Halstead Volume",
    "halstead-difficulty": "Halstead Difficulty",
    "difficulty": "Halstead Difficulty",
    "d": "Halstead Difficulty",
    "halstead-effort": "Halstead Effort",
    "effort": "Halstead Effort",
    "e": "Halstead Effort",
}

#: Framework labels that name the sequential C++ reference implementation.
CPP_TOKENS = frozenset({"cpp", "cplusplus", "cpu"})


def _normalize_token(value: str) -> str:
    """Normalize a framework label for case-insensitive matching.

    Args:
        value: Framework or application label.

    Returns:
        Lowercase alphanumeric token.
    """
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _calculate_halstead_columns(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Add derived Halstead metrics to a complexity DataFrame.

    Args:
        df: Complexity data containing n1, n2, N1, and N2.
        source: Input file used in error messages.

    Returns:
        Copy of the data with derived metric columns.

    Raises:
        ValueError: If the primitive Halstead columns are absent or invalid.
    """
    required = ["n1", "n2", "N1", "N2"]
    _require_columns(df, required, source)
    result = df.copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[required].isna().any(axis=None):
        raise ValueError(f"{source} contains non-numeric Halstead counts.")

    vocabulary = result["n1"] + result["n2"]
    length = result["N1"] + result["N2"]
    result["Halstead Vocabulary"] = vocabulary
    result["Halstead Program Length"] = length
    result["Halstead Volume"] = np.where(
        vocabulary > 0.0, length * np.log2(vocabulary), np.nan
    )
    result["Halstead Difficulty"] = np.where(
        result["n2"] > 0.0,
        (result["n1"] / 2.0) * (result["N2"] / result["n2"]),
        np.nan,
    )
    result["Halstead Effort"] = (
        result["Halstead Volume"] * result["Halstead Difficulty"]
    )
    return result


def resolve_complexity_metric(
    df: pd.DataFrame,
    requested: str,
    source: Path,
) -> tuple[pd.DataFrame, str]:
    """Resolve or calculate a requested code-complexity metric.

    Args:
        df: Raw complexity data.
        requested: Metric name or alias supplied on the command line.
        source: Complexity CSV path.

    Returns:
        A pair containing enriched complexity data and the canonical metric
        column name.

    Raises:
        ValueError: If the metric is unknown or cannot be calculated.
    """
    alias = requested.casefold().replace("_", "-").replace(" ", "-")
    canonical = METRIC_ALIASES.get(alias)
    if canonical is None:
        direct = [
            column for column in df.columns if column.casefold() == requested.casefold()
        ]
        if len(direct) != 1:
            raise ValueError(
                f"Unknown complexity metric {requested!r}. Use one of: "
                "SLOC, Halstead vocabulary, Halstead program length, Halstead "
                "volume, Halstead difficulty, Halstead effort."
            )
        canonical = direct[0]

    enriched = df.copy()
    if canonical.startswith("Halstead") and canonical not in enriched.columns:
        enriched = _calculate_halstead_columns(enriched, source)
    if canonical not in enriched.columns:
        raise ValueError(f"{source} does not contain metric column {canonical!r}.")
    enriched[canonical] = pd.to_numeric(enriched[canonical], errors="coerce")
    if enriched[canonical].isna().all():
        raise ValueError(f"Complexity metric {canonical!r} has no numeric values.")
    return enriched, canonical


def _select_complexity(
    path: Path,
    problem_query: str,
    metric_request: str,
) -> tuple[pd.DataFrame, str, str]:
    """Load one problem's complexity rows and reduce them to one metric.

    Args:
        path: Complexity CSV path.
        problem_query: User-supplied benchmark problem query.
        metric_request: Requested metric or alias.

    Returns:
        The per-framework median of the resolved metric, that metric's column
        name, and the resolved problem name.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the input is invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Complexity CSV does not exist: {path}")
    raw = pd.read_csv(path)
    _require_columns(raw, [COMPLEXITY_NAME, COMPLEXITY_FRAMEWORK], path)
    problem = _resolve_unique_value(
        raw[COMPLEXITY_NAME], problem_query, "complexity problem"
    )
    selected = raw.loc[raw[COMPLEXITY_NAME] == problem].copy()
    selected[COMPLEXITY_FRAMEWORK] = (
        selected[COMPLEXITY_FRAMEWORK].fillna("").astype(str).str.strip()
    )
    selected, metric = resolve_complexity_metric(selected, metric_request, path)
    selected = (
        selected.groupby(COMPLEXITY_FRAMEWORK, as_index=False)[metric]
        .median()
        .dropna(subset=[metric])
    )
    return selected, metric, problem


def _cpp_baseline(selected: pd.DataFrame, metric: str, problem: str) -> float:
    """Return the sequential C++ value of ``metric``.

    Args:
        selected: Per-framework complexity values.
        metric: Metric column to read.
        problem: Problem name, used in error messages.

    Returns:
        The CPP reference value.

    Raises:
        ValueError: If there is not exactly one CPP row.
    """
    tokens = selected[COMPLEXITY_FRAMEWORK].map(_normalize_token)
    baseline_rows = selected.loc[tokens.isin(CPP_TOKENS), metric]
    if len(baseline_rows) != 1:
        raise ValueError(
            "Scaling against the CPP baseline requires exactly one CPP "
            f"complexity row for {problem}; found {len(baseline_rows)}."
        )
    return float(baseline_rows.iloc[0])


def load_complexity_baseline(
    path: Path,
    problem_query: str,
    metric_request: str,
) -> float:
    """Load the absolute sequential C++ value of one complexity metric.

    Normalized data says only that CPP is 100 %; a plot that wants to state
    what 100 % stands for has to read the unscaled value back out.

    Args:
        path: Complexity CSV path.
        problem_query: User-supplied benchmark problem query.
        metric_request: Requested metric or alias.

    Returns:
        The CPP value of the resolved metric, unscaled.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the input is invalid or no CPP baseline is available.
    """
    selected, metric, problem = _select_complexity(
        path, problem_query, metric_request
    )
    return _cpp_baseline(selected, metric, problem)


def load_complexity_data(
    path: Path,
    problem_query: str,
    metric_request: str,
    normalize: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load, select, calculate, and optionally normalize complexity data.

    Args:
        path: Complexity CSV path.
        problem_query: User-supplied benchmark problem query.
        metric_request: Requested metric or alias.
        normalize: Whether to express values as a percentage of CPP.

    Returns:
        A pair of selected complexity rows and the plotted metric label.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the input is invalid or no CPP baseline is available.
    """
    selected, metric, problem = _select_complexity(
        path, problem_query, metric_request
    )

    display_metric = "Source Lines of Code" if metric == "SLOC" else metric
    label = f"{display_metric} [absolute]"
    if normalize:
        baseline = _cpp_baseline(selected, metric, problem)
        if baseline <= 0.0:
            raise ValueError(
                "Normalizing requires a positive CPP complexity score; use "
                "--complexity-metric-absolute to plot absolute values."
            )
        selected[metric] = selected[metric] / baseline * 100.0
        label = f"{display_metric} [normalized]"
        logger.debug(f"Normalized {metric} to CPP baseline {baseline:g}")
    selected = selected.rename(columns={metric: label})
    logger.info(f"Loaded {len(selected)} complexity rows for {problem}")
    return selected, label


def load_complexity_export_data(
    path: Path,
    problem_query: str,
) -> pd.DataFrame:
    """Load every numeric complexity metric and derive Halstead metrics.

    Source metric columns such as SLOC and the primitive Halstead counts are
    retained. When all four primitive counts are available, all supported
    derived Halstead metrics are calculated even if they were not requested
    for the plot.

    Args:
        path: Complexity CSV path.
        problem_query: User-supplied benchmark problem query.

    Returns:
        Median numeric complexity metrics grouped by Framework.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the input schema or Halstead counts are invalid, or no
            numeric metric columns are available.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Complexity CSV does not exist: {path}")
    raw = pd.read_csv(path)
    _require_columns(raw, [COMPLEXITY_NAME, COMPLEXITY_FRAMEWORK], path)
    problem = _resolve_unique_value(
        raw[COMPLEXITY_NAME], problem_query, "complexity problem"
    )
    selected = raw.loc[raw[COMPLEXITY_NAME] == problem].copy()
    selected[COMPLEXITY_FRAMEWORK] = (
        selected[COMPLEXITY_FRAMEWORK].fillna("").astype(str).str.strip()
    )

    halstead_counts = {"n1", "n2", "N1", "N2"}
    if halstead_counts.issubset(selected.columns):
        selected = _calculate_halstead_columns(selected, path)

    metric_columns: list[str] = []
    identity_columns = {COMPLEXITY_NAME, COMPLEXITY_FRAMEWORK}
    for column in selected.columns:
        if column in identity_columns:
            continue
        numeric = pd.to_numeric(selected[column], errors="coerce")
        if numeric.notna().any():
            selected[column] = numeric
            metric_columns.append(column)
    if not metric_columns:
        raise ValueError(f"{path} contains no numeric complexity metrics.")

    result = selected.groupby(COMPLEXITY_FRAMEWORK, as_index=False)[
        metric_columns
    ].median()
    logger.info(f"Loaded {len(metric_columns)} export complexity metrics for {problem}")
    return result


def _complexity_candidates(application: str) -> list[str]:
    """Generate framework labels that may match an application.

    Args:
        application: Application label, possibly ``Paradigm[Description]``.

    Returns:
        Candidate labels ordered from most to least specific.
    """
    candidates = [application]
    match = re.fullmatch(r"(.+?)\[(.+)]", application)
    if match:
        paradigm, description = match.groups()
        candidates.extend([description, paradigm])
    return list(dict.fromkeys(candidates))


def merge_portability_complexity(
    portability: pd.DataFrame,
    complexity: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Match application PP values with code-complexity rows.

    Exact application labels are preferred. For labels containing a
    description, the description alone and then the base paradigm are tried as
    fallbacks; this maps ``Cuda[Cublas]`` to ``Cublas`` and polyhedral model
    labels such as ``Cuda[Eros]`` to ``Cuda``.

    Args:
        portability: Performance portability per application.
        complexity: Complexity per framework.
        metric: Complexity metric column.

    Returns:
        Matched application, PP, and complexity values.

    Raises:
        ValueError: If none of the applications can be matched.
    """
    lookup = {
        _normalize_token(row[COMPLEXITY_FRAMEWORK]): float(row[metric])
        for _, row in complexity.iterrows()
    }
    rows: list[dict[str, object]] = []
    unmatched: list[str] = []
    for _, row in portability.iterrows():
        application = str(row[APPLICATION])
        value = None
        for candidate in _complexity_candidates(application):
            token = _normalize_token(candidate)
            if token in lookup:
                value = lookup[token]
                break
        if value is None:
            unmatched.append(application)
            continue
        rows.append(
            {
                APPLICATION: application,
                PERFORMANCE_PORTABILITY: float(row[PERFORMANCE_PORTABILITY]),
                metric: value,
            }
        )
    if unmatched:
        logger.warning(
            "No complexity value for application(s), omitting them: "
            + ", ".join(unmatched)
        )
    if not rows:
        raise ValueError("No application labels match the complexity Framework values.")
    return pd.DataFrame(rows)


def add_complexity_to_export(
    data: pd.DataFrame,
    complexity: pd.DataFrame,
) -> pd.DataFrame:
    """Add all matching complexity values to an export DataFrame.

    Export rows are preserved when no complexity framework matches an
    application; the metrics are left empty for those rows.

    Args:
        data: Application-efficiency or performance-portability data.
        complexity: Complexity values keyed by Framework.

    Returns:
        A copy of ``data`` with every complexity metric after Application.
    """
    metric_columns = [
        column for column in complexity.columns if column != COMPLEXITY_FRAMEWORK
    ]
    overlapping = set(metric_columns) & set(data.columns)
    if overlapping:
        raise ValueError(
            "Complexity metric columns duplicate export columns: "
            f"{sorted(overlapping)}"
        )
    lookup = {
        _normalize_token(row[COMPLEXITY_FRAMEWORK]): {
            column: float(row[column]) for column in metric_columns
        }
        for _, row in complexity.iterrows()
    }
    matched_values: list[dict[str, float] | None] = []
    unmatched: set[str] = set()
    for application_value in data[APPLICATION]:
        application = str(application_value)
        value = next(
            (
                lookup[token]
                for candidate in _complexity_candidates(application)
                if (token := _normalize_token(candidate)) in lookup
            ),
            None,
        )
        matched_values.append(value)
        if value is None:
            unmatched.add(application)

    if unmatched:
        logger.warning(
            "No complexity value for exported application(s): "
            + ", ".join(sorted(unmatched))
        )
    result = data.copy()
    insert_position = result.columns.get_loc(APPLICATION) + 1
    for offset, column in enumerate(metric_columns):
        result.insert(
            insert_position + offset,
            column,
            [np.nan if values is None else values[column] for values in matched_values],
        )
    return result


def append_cpp_complexity_row(
    data: pd.DataFrame,
    complexity: pd.DataFrame,
    problem: str,
) -> pd.DataFrame:
    """Append the complexity-only CPP baseline to an export DataFrame.

    CPP is the complexity reference implementation but has no benchmark
    measurements. Its size, precision, hardware, and performance metric cells
    therefore remain empty in both exported datasets.
    """
    cpp_rows = complexity.loc[
        complexity[COMPLEXITY_FRAMEWORK]
        .astype(str)
        .map(_normalize_token)
        .isin(CPP_TOKENS)
    ]
    if cpp_rows.empty:
        logger.warning(f"No CPP complexity row available for {problem}")
        return data
    if len(cpp_rows) > 1:
        raise ValueError(
            f"Expected at most one CPP complexity row for {problem}; "
            f"found {len(cpp_rows)}."
        )

    cpp = cpp_rows.iloc[0]
    cpp_application = str(cpp[COMPLEXITY_FRAMEWORK])
    if (
        data[APPLICATION]
        .astype(str)
        .map(_normalize_token)
        .eq(_normalize_token(cpp_application))
        .any()
    ):
        return data

    row: dict[str, object] = {column: np.nan for column in data.columns}
    row[PROBLEM] = problem
    row[APPLICATION] = cpp_application
    for column in complexity.columns:
        if column != COMPLEXITY_FRAMEWORK and column in row:
            row[column] = cpp[column]
    return pd.concat([data, pd.DataFrame([row])], ignore_index=True)


def export_metrics_to_csv(
    efficiency: pd.DataFrame,
    portability: pd.DataFrame,
    output: Path,
) -> tuple[Path, Path]:
    """Export metric DataFrames next to the plot using its filename prefix.

    Args:
        efficiency: Application-efficiency data to export.
        portability: Performance-portability data to export.
        output: Resolved plot path whose stem supplies the filename prefix.

    Returns:
        The application-efficiency and performance-portability CSV paths.
    """
    efficiency_output = output.with_name(f"{output.stem}_application_efficiency.csv")
    portability_output = output.with_name(f"{output.stem}_performance_portability.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    efficiency.to_csv(efficiency_output, index=False)
    portability.to_csv(portability_output, index=False)
    logger.success(f"Wrote application efficiency: {efficiency_output.resolve()}")
    logger.success(f"Wrote performance portability: {portability_output.resolve()}")
    return efficiency_output, portability_output
