"""Rank-correlate paradigm orderings between benchmark problems.

A single benchmark problem orders the paradigms it was run with. Whether that
order carries over to another problem is a question about ranks, not values:
Spearman's :math:`\\rho` between two problems is high when both put the same
paradigms on top, and near zero when one problem says nothing about the other.

Every variable is reduced to one number per *paradigm*, because a paradigm is
what the problems have in common — implementation variants such as
``Cuda[Naive]`` and ``Cuda[SharedMemory]`` exist in one problem and not the
next. The sequential C++ reference is left out: it is the complexity baseline,
not one of the compared paradigms, and it has no benchmark results to rank.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
from loguru import logger

from ppbcc.constants import (
    APPLICATION,
    PARADIGM,
    PERFORMANCE_PORTABILITY,
    PROBLEM,
)
from ppbcc.performance_portability.complexity import (
    COMPLEXITY_FRAMEWORK,
    CPP_TOKENS,
    METRIC_ALIASES,
    _normalize_token,
    _select_complexity,
)

#: ``--correlation`` value that ranks paradigms by performance portability.
CORRELATION_PP = "pp"
#: Accepted spellings of :data:`CORRELATION_PP`.
PP_ALIASES = frozenset(
    {"pp", "p3", "phi", "portability", "performance-portability"}
)
#: Smallest number of shared paradigms for which a rho is worth reporting.
MINIMUM_PAIR_SIZE = 3

#: Column of the exported per-paradigm table holding the rank of a value.
RANK = "Rank (best first)"
#: Correlation method; the table is about orderings, not about values.
CORRELATION_METHOD = "spearman"


def resolve_correlation_variable(request: str) -> str:
    """Resolve a ``--correlation`` request to a rankable variable.

    Args:
        request: User-supplied variable name or alias.

    Returns:
        :data:`CORRELATION_PP`, or the canonical complexity metric column.

    Raises:
        ValueError: If the request names neither PP nor a complexity metric.
    """
    alias = request.casefold().replace("_", "-").replace(" ", "-")
    if alias in PP_ALIASES:
        return CORRELATION_PP
    canonical = METRIC_ALIASES.get(alias)
    if canonical is None:
        raise ValueError(
            f"Unknown correlation variable {request!r}. Use 'pp' for "
            "performance portability, or a complexity metric: SLOC, Halstead "
            "vocabulary, Halstead program length, Halstead volume, Halstead "
            "difficulty, Halstead effort."
        )
    return canonical


def base_paradigm(label: str) -> str:
    """Strip a bracketed implementation variant from a label.

    Args:
        label: Application or Framework label, e.g. ``Cuda[SharedMemory]``.

    Returns:
        The bare paradigm name, e.g. ``Cuda``.
    """
    return re.sub(r"\[[^]]*]", "", str(label)).strip()


def portability_by_paradigm(
    portability: pd.DataFrame, paradigms: Iterable[str]
) -> pd.Series:
    """Reduce performance portability to one value per paradigm.

    Where a paradigm was benchmarked in several variants, its best-performing
    variant stands for it, the same convention the time bar plot uses for
    runtimes.

    Args:
        portability: One PP value per application, as produced by
            :mod:`ppbcc.performance_portability.metrics`.
        paradigms: Paradigm universe of this problem; labels outside it are
            dropped.

    Returns:
        PP indexed by paradigm.
    """
    values = portability.copy()
    values[PARADIGM] = values[APPLICATION].map(base_paradigm)
    wanted = {_normalize_token(paradigm) for paradigm in paradigms}
    values = values.loc[values[PARADIGM].map(_normalize_token).isin(wanted)]
    return (
        values.groupby(PARADIGM)[PERFORMANCE_PORTABILITY]
        .max()
        .rename(PERFORMANCE_PORTABILITY)
    )


def complexity_by_paradigm(
    path: Path,
    problem_query: str,
    metric_request: str,
    paradigms: Iterable[str],
) -> tuple[pd.Series, str]:
    """Reduce one complexity metric to one value per benchmarked paradigm.

    Variants of a paradigm are reduced to their median, which is how
    :func:`ppbcc.performance_portability.complexity.load_complexity_data`
    already reduces several rows carrying one Framework label. Values are read
    unscaled: dividing every paradigm of a problem by that problem's C++
    baseline is a positive rescaling and cannot change the problem's order.

    Args:
        path: Complexity CSV path.
        problem_query: User-supplied benchmark problem query.
        metric_request: Requested metric or alias.
        paradigms: Paradigm universe of this problem, from the benchmark CSVs.
            Frameworks outside it — the C++ baseline, or a paradigm that was
            measured for complexity but never benchmarked — are dropped, so
            every variable ranks the same paradigms.

    Returns:
        The metric indexed by paradigm, and the resolved metric column name.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no benchmarked paradigm has a complexity value.
    """
    selected, metric, problem = _select_complexity(path, problem_query, metric_request)
    values = selected.copy()
    values[PARADIGM] = values[COMPLEXITY_FRAMEWORK].map(base_paradigm)
    tokens = values[PARADIGM].map(_normalize_token)
    wanted = {_normalize_token(paradigm) for paradigm in paradigms} - set(CPP_TOKENS)
    dropped = sorted(set(values.loc[~tokens.isin(wanted), PARADIGM]))
    if dropped:
        logger.debug(
            f"{problem}: ignoring complexity framework(s) without benchmark "
            "results: " + ", ".join(dropped)
        )
    values = values.loc[tokens.isin(wanted)]
    if values.empty:
        raise ValueError(
            f"No benchmarked paradigm of {problem} has a {metric!r} value."
        )
    return values.groupby(PARADIGM)[metric].median().rename(metric), metric


def build_value_table(values: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Align the per-problem paradigm values into one wide table.

    Args:
        values: One value series per problem, indexed by paradigm.

    Returns:
        Paradigms as rows and problems as columns, with a missing value where a
        paradigm has no value for a problem.

    Raises:
        ValueError: If fewer than two problems are given.
    """
    if len(values) < 2:
        raise ValueError(
            "rank-correlation compares the paradigm orderings of at least two "
            f"problems; got {len(values)}."
        )
    table = pd.DataFrame(dict(values))
    table.index.name = PARADIGM
    return table.sort_index()


def rank_correlation_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman's rho between every pair of problems.

    Each pair is correlated over the paradigms both problems have a value for,
    so a paradigm missing from one problem costs only the pairs that problem
    takes part in.

    Args:
        table: Wide value table from :func:`build_value_table`.

    Returns:
        A square matrix of rank correlations, problems on both axes.

    Raises:
        ValueError: If no pair of problems shares enough paradigms to rank.
    """
    shared = shared_paradigm_counts(table)
    problems = list(table.columns)
    off_diagonal = [
        int(shared.loc[first, second])
        for index, first in enumerate(problems)
        for second in problems[index + 1 :]
    ]
    if not any(count >= MINIMUM_PAIR_SIZE for count in off_diagonal):
        raise ValueError(
            "No two problems share at least "
            f"{MINIMUM_PAIR_SIZE} paradigms with a value; a rank correlation "
            "would be meaningless."
        )
    for index, first in enumerate(problems):
        for second in problems[index + 1 :]:
            count = int(shared.loc[first, second])
            if count < MINIMUM_PAIR_SIZE:
                logger.warning(
                    f"{first} and {second} share only {count} paradigm(s); "
                    "their rank correlation is not meaningful."
                )
    matrix = table.corr(method=CORRELATION_METHOD, min_periods=MINIMUM_PAIR_SIZE)
    matrix.index.name = PROBLEM
    return matrix


def shared_paradigm_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Count the paradigms each pair of problems has a value for.

    Args:
        table: Wide value table from :func:`build_value_table`.

    Returns:
        A square matrix of counts, problems on both axes.
    """
    present = table.notna().astype(int)
    counts = present.T.dot(present)
    counts.index.name = PROBLEM
    return counts


def rank_table(
    table: pd.DataFrame, variable: str, value_column: str
) -> pd.DataFrame:
    """Rank the paradigms of every problem, best first.

    "Best" is the largest value for performance portability and the smallest
    one for a complexity metric. The convention only labels the export: the
    correlations are calculated from the values themselves, and reversing every
    problem's direction at once leaves them unchanged.

    Args:
        table: Wide value table from :func:`build_value_table`.
        variable: :data:`CORRELATION_PP` or a complexity metric column, which
            decides which end of the scale rank one sits at.
        value_column: Name the values are written under.

    Returns:
        One row per problem and paradigm with the value and its rank.
    """
    ascending = variable != CORRELATION_PP
    long = (
        table.rename_axis(PARADIGM)
        .melt(ignore_index=False, var_name=PROBLEM, value_name=value_column)
        .reset_index()
        .dropna(subset=[value_column])
    )
    long[RANK] = long.groupby(PROBLEM)[value_column].rank(
        method="min", ascending=ascending
    )
    return long[[PROBLEM, PARADIGM, value_column, RANK]].sort_values(
        [PROBLEM, RANK, PARADIGM], ignore_index=True
    )


def write_rank_correlation_csv(matrix: pd.DataFrame, output: Path) -> Path:
    """Write the rank-correlation matrix next to the other analysis outputs.

    Args:
        matrix: Square matrix from :func:`rank_correlation_matrix`.
        output: Resolved output path.

    Returns:
        The written path.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output, index=True)
    logger.success(f"Wrote rank correlations: {output.resolve()}")
    return output
