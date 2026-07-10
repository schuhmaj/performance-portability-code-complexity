"""Top-level analysis pipeline: from source paths to the metric DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from .classification import TokenClassifier, TokenCounts, extract_namespace_aliases
from .config import (
    AUTO_DIALECT_NAME,
    SOURCE_EXTENSIONS,
    CppKeywords,
    DialectRegistry,
    DialectSpec,
    load_cpp_keywords,
    load_dialects,
)
from .detection import detect_dialects
from .halstead import HalsteadMetrics
from .loc import LineMetrics, count_lines
from .report import DIFF_COLUMNS, save_csv, to_dataframe
from .tokenizer import tokenize

#: Name used in the ``file`` column of the aggregation row.
AGGREGATE_ROW_NAME: str = "TOTAL"


def collect_source_files(sources: list[Path]) -> list[Path]:
    """Expands the given paths into a flat list of source files.

    Args:
        sources: Files and/or directories. Directories are searched
            recursively for files with an extension in
            :data:`code_complexity.config.SOURCE_EXTENSIONS`; explicitly
            listed files are taken as-is.

    Returns:
        Sorted list of unique source files.

    Raises:
        FileNotFoundError: If one of the paths does not exist.
    """
    files: set[Path] = set()
    for source in sources:
        source = Path(source)
        if source.is_dir():
            found = sorted(
                candidate
                for candidate in source.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in SOURCE_EXTENSIONS
            )
            logger.debug("Collected {} source files below {}", len(found), source)
            files.update(found)
        elif source.is_file():
            files.add(source)
        else:
            raise FileNotFoundError(f"Source path does not exist: {source}")
    return sorted(files)


def _build_row(
    name: str,
    dialect_names: list[str],
    counts: TokenCounts,
    lines: LineMetrics,
    diff: bool,
) -> dict:
    """Assembles one report row from the collected counts.

    Args:
        name: Value of the ``file`` column.
        dialect_names: Names of the active dialects (empty for plain C++).
        counts: Classified token multisets of the unit.
        lines: Line metrics of the unit.
        diff: If True, ``baseline_``/``delta_`` values (metrics without the
            dialect tokens and the difference to the full metrics) are added.

    Returns:
        Mapping from column name to value.
    """
    full = HalsteadMetrics.from_counts(counts.full_operators, counts.full_operands)
    row: dict = {
        "file": name,
        "dialect": "+".join(dialect_names) if dialect_names else "cpp",
        **lines.as_dict(),
        **full.as_dict(),
        "dialect_distinct_operators": len(counts.dialect_operators),
        "dialect_total_operators": sum(counts.dialect_operators.values()),
        "dialect_distinct_operands": len(counts.dialect_operands),
        "dialect_total_operands": sum(counts.dialect_operands.values()),
    }
    if diff:
        baseline = HalsteadMetrics.from_counts(counts.operators, counts.operands)
        full_values = full.as_dict()
        for column, value in baseline.as_dict().items():
            if column in DIFF_COLUMNS:
                row[f"baseline_{column}"] = value
                row[f"delta_{column}"] = full_values[column] - value
    return row


def analyze_source(
    code: str,
    path: Path | None,
    keywords: CppKeywords,
    registry: DialectRegistry,
    dialects: list[DialectSpec] | None,
) -> tuple[TokenCounts, LineMetrics, list[str]]:
    """Analyses one source text.

    Args:
        code: Raw source text.
        path: Path of the file (used for logging and auto-detection); may be
            ``None`` for in-memory analysis.
        keywords: Baseline C++ keyword sets.
        registry: The dialect registry.
        dialects: Active dialects, or ``None`` to auto-detect them per file.

    Returns:
        Tuple of the classified token counts, the line metrics and the names
        of the active dialects.
    """
    if dialects is None:
        dialects = detect_dialects(code, path, registry)
    tokens = tokenize(code)
    classifier = TokenClassifier(keywords, dialects, extract_namespace_aliases(code))
    counts = classifier.classify(tokens)
    lines = count_lines(code, tokens)
    logger.debug(
        "{}: {} lines, {} operators, {} operands, {} dialect operators [{}]",
        path or "<string>",
        lines.loc,
        sum(counts.operators.values()),
        sum(counts.operands.values()),
        sum(counts.dialect_operators.values()),
        "+".join(spec.name for spec in dialects) or "cpp",
    )
    return counts, lines, [spec.name for spec in dialects]


def evaluate(
    sources: list[Path],
    language_dialect: str = AUTO_DIALECT_NAME,
    metrics: list[str] | None = None,
    *,
    diff: bool = False,
    aggregate: bool = False,
    output: Path | None = None,
    csv_separator: str = ",",
    keywords_path: Path | None = None,
    dialects_path: Path | None = None,
) -> pd.DataFrame:
    """Runs the complexity analysis - the library's top-level entry point.

    Args:
        sources: Source files and/or directories to analyse.
        language_dialect: Dialect selection: ``"auto"`` (default) detects the
            dialects per file, ``"cpp"``/``"none"`` analyses plain C++, and
            any dialect name/alias (optionally comma-separated, e.g.
            ``"kokkos,openmp"``) forces those dialects for all files.
        metrics: Metric names/groups to include as columns (e.g.
            ``["halstead_effort", "loc"]``); ``None`` selects all metrics.
        diff: If True, add ``baseline_``/``delta_`` columns comparing the
            full metrics against the metrics without dialect tokens.
        aggregate: If True, append a ``TOTAL`` row aggregating all files
            (operator/operand multisets are merged before recomputing the
            Halstead measures, so distinct counts are program-wide).
        output: Optional CSV destination; written when given.
        csv_separator: Field separator for the CSV output.
        keywords_path: Optional override for the packaged
            ``cpp_keywords.toml``.
        dialects_path: Optional override for the packaged ``dialects.toml``.

    Returns:
        DataFrame with one row per source file (plus the optional ``TOTAL``
        row) and the selected metric columns.

    Raises:
        FileNotFoundError: If a source path does not exist.
        KeyError: If ``language_dialect`` names an unknown dialect.
        ValueError: If ``metrics`` contains an unknown metric name.
    """
    keywords = load_cpp_keywords(keywords_path)
    registry = load_dialects(dialects_path)

    forced_dialects: list[DialectSpec] | None = None
    if language_dialect.strip().lower() != AUTO_DIALECT_NAME:
        forced_dialects = registry.resolve_all(language_dialect)
        logger.info(
            "Analyzing with fixed dialect(s): {}",
            ", ".join(spec.name for spec in forced_dialects) or "cpp (baseline)",
        )
    else:
        logger.info("Analyzing with automatic per-file dialect detection")

    files = collect_source_files(sources)
    if not files:
        logger.warning("No source files found in {}", [str(s) for s in sources])
    logger.info("Analyzing {} source file(s)", len(files))

    rows: list[dict] = []
    total_counts = TokenCounts()
    total_lines = LineMetrics(0, 0, 0, 0)
    all_dialects: set[str] = set()
    for path in files:
        code = path.read_text(encoding="utf-8", errors="replace")
        counts, lines, dialect_names = analyze_source(
            code, path, keywords, registry, forced_dialects
        )
        rows.append(_build_row(str(path), dialect_names, counts, lines, diff))
        if aggregate:
            total_counts.merge(counts)
            total_lines = total_lines.combine(lines)
            all_dialects.update(dialect_names)

    if aggregate and files:
        rows.append(
            _build_row(AGGREGATE_ROW_NAME, sorted(all_dialects), total_counts, total_lines, diff)
        )

    frame = to_dataframe(rows, metrics, diff)
    if output is not None:
        save_csv(frame, Path(output), separator=csv_separator)
    return frame
