"""Loading and representation of the analysis configuration.

The configuration consists of two TOML resources shipped with the package in
``code_complexity/share``:

* ``cpp_keywords.toml`` -- the baseline C++ keyword sets used to distinguish
  operators from operands.
* ``dialects.toml`` -- the registry of GPU/parallel programming *dialects*
  (Kokkos, CUDA, OpenMP, ...) whose constructs are counted as dialect
  operators.

Both files can be overridden with user-supplied paths, e.g. via the CLI.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from loguru import logger

#: Number of ``detect_patterns`` hits in a source file required to activate a
#: dialect during automatic detection (headers, file extensions, pragmas and
#: namespace usage activate a dialect with a single hit instead).
DETECTION_MIN_HITS: int = 3

#: File extensions collected when a directory is passed as a source.
SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".c++",
    ".h", ".hh", ".hpp", ".hxx", ".h++", ".inl", ".inc", ".ipp",
    ".cu", ".cuh", ".hip",
    ".cl",
    ".comp", ".glsl", ".vert", ".frag", ".geom", ".tesc", ".tese",
    ".slang", ".hlsl", ".wgsl", ".metal",
})

#: Names of ``language_dialect`` that mean "plain C++, no dialect".
BASELINE_DIALECT_NAMES: frozenset[str] = frozenset({"cpp", "c++", "none", "baseline"})

#: Name of the pseudo dialect that triggers automatic per-file detection.
AUTO_DIALECT_NAME: str = "auto"


@dataclass(frozen=True)
class CppKeywords:
    """Baseline C++ keyword sets.

    Attributes:
        keywords: Keywords counted as operators (e.g. ``for``, ``const``).
        operand_keywords: Keywords that denote values and therefore count as
            operands (e.g. ``true``, ``nullptr``, ``this``).
    """

    keywords: frozenset[str]
    operand_keywords: frozenset[str]


@dataclass(frozen=True)
class DialectSpec:
    """Description of a single GPU/parallel-programming dialect.

    All regular expressions are pre-compiled; identifier patterns are matched
    with ``fullmatch`` against single identifiers, header and detection
    patterns with ``search``.

    Attributes:
        name: Canonical dialect name (the table name in ``dialects.toml``).
        aliases: Lower-case names accepted for ``language_dialect``.
        namespace_prefixes: Namespace prefixes (as tuples of segments) that
            turn a whole qualified name into a single dialect operator.
        identifier_pattern: Combined regex for dialect identifiers, or
            ``None`` if the dialect defines no identifier patterns.
        keywords: Exact identifiers belonging to the dialect.
        pragma_prefixes: First pragma tokens claiming a ``#pragma`` line.
        pragma_clauses: Identifiers counted as operators inside a claimed
            pragma line (all other pragma identifiers count as operands).
        punctuation: Extra punctuation operators (e.g. CUDA ``<<<``).
        header_patterns: Regexes matched against included header names for
            automatic detection.
        detect_patterns: Regexes searched in the raw source for automatic
            detection (requires :data:`DETECTION_MIN_HITS` total hits).
        extensions: File extensions that imply the dialect.
    """

    name: str
    aliases: frozenset[str]
    namespace_prefixes: tuple[tuple[str, ...], ...] = ()
    identifier_pattern: re.Pattern[str] | None = None
    keywords: frozenset[str] = frozenset()
    pragma_prefixes: frozenset[str] = frozenset()
    pragma_clauses: frozenset[str] = frozenset()
    punctuation: frozenset[str] = frozenset()
    header_patterns: tuple[re.Pattern[str], ...] = ()
    detect_patterns: tuple[re.Pattern[str], ...] = ()
    extensions: frozenset[str] = frozenset()

    def matches_identifier(self, identifier: str) -> bool:
        """Checks whether a single identifier belongs to this dialect.

        Args:
            identifier: The identifier text, e.g. ``"cudaMalloc"``.

        Returns:
            True if the identifier is a dialect keyword or fully matches one
            of the dialect's identifier patterns.
        """
        if identifier in self.keywords:
            return True
        return bool(self.identifier_pattern and self.identifier_pattern.fullmatch(identifier))

    def matches_qualified(self, segments: tuple[str, ...]) -> bool:
        """Checks whether a qualified name belongs to this dialect.

        Args:
            segments: The ``::``-separated name split into segments, e.g.
                ``("Kokkos", "parallel_for")``.

        Returns:
            True if the name starts with one of the dialect's namespace
            prefixes.
        """
        return any(
            segments[: len(prefix)] == prefix
            for prefix in self.namespace_prefixes
            if len(segments) >= len(prefix)
        )


@dataclass(frozen=True)
class DialectRegistry:
    """All known dialects plus lookup helpers.

    Attributes:
        dialects: Mapping from canonical dialect name to its specification.
        alias_map: Mapping from every accepted (lower-case) alias to the
            canonical dialect name.
    """

    dialects: dict[str, DialectSpec]
    alias_map: dict[str, str] = field(default_factory=dict)

    def resolve(self, name: str) -> DialectSpec:
        """Resolves a dialect name or alias to its specification.

        Args:
            name: Dialect name or alias, case-insensitive (e.g. ``"Kokkos"``).

        Returns:
            The matching :class:`DialectSpec`.

        Raises:
            KeyError: If the name is not a known dialect or alias.
        """
        canonical = self.alias_map.get(name.strip().lower())
        if canonical is None:
            known = ", ".join(sorted(self.alias_map))
            raise KeyError(f"Unknown dialect {name!r}. Known dialects/aliases: {known}")
        return self.dialects[canonical]

    def resolve_all(self, names: str | list[str]) -> list[DialectSpec]:
        """Resolves a dialect selection string or list to specifications.

        Args:
            names: Either a single (possibly comma-separated) string such as
                ``"kokkos,openmp"`` or a list of dialect names. Baseline names
                (``"cpp"``, ``"none"``, ...) resolve to an empty selection.

        Returns:
            List of resolved dialect specifications (without duplicates).

        Raises:
            KeyError: If any name is neither a dialect alias nor a baseline
                name. ``"auto"`` must be handled by the caller and also
                raises here.
        """
        if isinstance(names, str):
            names = [part for part in names.split(",") if part.strip()]
        specs: dict[str, DialectSpec] = {}
        for name in names:
            if name.strip().lower() in BASELINE_DIALECT_NAMES:
                continue
            spec = self.resolve(name)
            specs[spec.name] = spec
        return list(specs.values())


def _read_toml(default_resource: str, override: Path | None) -> dict:
    """Reads a TOML configuration, either packaged or user-supplied.

    Args:
        default_resource: File name of the packaged resource in
            ``code_complexity/share``.
        override: Optional path to a user-supplied TOML file replacing the
            packaged one.

    Returns:
        The parsed TOML document as a dictionary.
    """
    if override is not None:
        logger.debug("Loading configuration from override file {}", override)
        return tomllib.loads(Path(override).read_text(encoding="utf-8"))
    resource = resources.files("code_complexity") / "share" / default_resource
    logger.trace("Loading packaged configuration resource {}", default_resource)
    return tomllib.loads(resource.read_text(encoding="utf-8"))


def load_cpp_keywords(path: Path | None = None) -> CppKeywords:
    """Loads the baseline C++ keyword sets.

    Args:
        path: Optional path to a TOML file overriding the packaged
            ``cpp_keywords.toml``.

    Returns:
        The loaded :class:`CppKeywords`.
    """
    data = _read_toml("cpp_keywords.toml", path)
    return CppKeywords(
        keywords=frozenset(data.get("keywords", [])),
        operand_keywords=frozenset(data.get("operand_keywords", [])),
    )


def _compile_dialect(name: str, data: dict) -> DialectSpec:
    """Compiles one dialect table from ``dialects.toml`` into a spec.

    Args:
        name: Canonical dialect name (TOML table name).
        data: The dialect's TOML table contents.

    Returns:
        The compiled :class:`DialectSpec`.
    """
    identifier_patterns = data.get("identifier_patterns", [])
    combined = None
    if identifier_patterns:
        combined = re.compile("|".join(f"(?:{p})" for p in identifier_patterns))
    return DialectSpec(
        name=name,
        aliases=frozenset(alias.lower() for alias in data.get("aliases", [name])),
        namespace_prefixes=tuple(
            tuple(prefix.split("::")) for prefix in data.get("namespace_prefixes", [])
        ),
        identifier_pattern=combined,
        keywords=frozenset(data.get("keywords", [])),
        pragma_prefixes=frozenset(data.get("pragma_prefixes", [])),
        pragma_clauses=frozenset(data.get("pragma_clauses", [])),
        punctuation=frozenset(data.get("punctuation", [])),
        header_patterns=tuple(re.compile(p) for p in data.get("headers", [])),
        detect_patterns=tuple(re.compile(p) for p in data.get("detect_patterns", [])),
        extensions=frozenset(ext.lower() for ext in data.get("extensions", [])),
    )


def load_dialects(path: Path | None = None) -> DialectRegistry:
    """Loads the dialect registry.

    Args:
        path: Optional path to a TOML file overriding the packaged
            ``dialects.toml``.

    Returns:
        The loaded :class:`DialectRegistry` with all regexes compiled.
    """
    data = _read_toml("dialects.toml", path)
    dialects = {
        name: _compile_dialect(name, table)
        for name, table in data.get("dialects", {}).items()
    }
    alias_map: dict[str, str] = {}
    for name, spec in dialects.items():
        alias_map[name.lower()] = name
        for alias in spec.aliases:
            alias_map[alias] = name
    logger.debug("Loaded {} dialects: {}", len(dialects), ", ".join(dialects))
    return DialectRegistry(dialects=dialects, alias_map=alias_map)
