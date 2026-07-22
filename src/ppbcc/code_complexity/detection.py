"""Automatic detection of the GPU dialects used by a source file.

When ``language_dialect="auto"`` is requested, every file is scanned for
strong dialect signals; a dialect is activated for the file if any of the
following matches:

* the file extension is dialect-specific (``.cu``, ``.hip``, ``.cl``, ...),
* an ``#include`` matches one of the dialect's header patterns,
* a ``#pragma`` line starts with one of the dialect's pragma prefixes,
* one of the dialect's namespaces is used (``Kokkos::``, also via a
  namespace alias), or
* at least :data:`ppbcc.code_complexity.config.DETECTION_MIN_HITS` occurrences of
  the dialect's weaker ``detect_patterns`` are found.

A dialect detected only via ``detect_patterns`` is suppressed when a sibling
dialect sharing its keywords (e.g. CUDA and HIP both define ``threadIdx``)
was detected with one of the stronger signals.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from .classification import extract_namespace_aliases
from .config import DETECTION_MIN_HITS, DialectRegistry, DialectSpec

_INCLUDE_RE = re.compile(r'(?m)^\s*#\s*include\s*[<"]([^">\n]+)[">]')


def extract_includes(code: str) -> list[str]:
    """Extracts all included header names from source text.

    Args:
        code: Raw source text.

    Returns:
        Header names as written between ``<>`` or ``""``.
    """
    return _INCLUDE_RE.findall(code)


def _uses_namespace(code: str, spec: DialectSpec) -> bool:
    """Checks whether one of the dialect's namespaces is used in the code.

    Both direct usage (``Kokkos::...``) and namespace aliases
    (``namespace bc = boost::compute;``) are considered.

    Args:
        code: Raw source text.
        spec: Dialect to test.

    Returns:
        True if a namespace prefix of the dialect occurs.
    """
    aliases = extract_namespace_aliases(code)
    for prefix in spec.namespace_prefixes:
        pattern = r"\b" + r"\s*::\s*".join(re.escape(seg) for seg in prefix)
        if len(prefix) == 1:
            pattern += r"\s*::"
        if re.search(pattern, code):
            return True
        if any(target[: len(prefix)] == prefix for target in aliases.values()):
            return True
    return False


def detect_dialects(
    code: str, path: Path | None, registry: DialectRegistry
) -> list[DialectSpec]:
    """Detects the dialects used by one source file.

    Args:
        code: Raw source text.
        path: Path of the file (used for extension matching); may be ``None``.
        registry: The dialect registry to match against.

    Returns:
        The detected dialect specifications, in registry order (possibly
        empty for plain C++ code).
    """
    includes = extract_includes(code)
    extension = path.suffix.lower() if path is not None else ""
    strong: list[DialectSpec] = []
    weak: list[DialectSpec] = []

    for spec in registry.dialects.values():
        reason: str | None = None
        if extension and extension in spec.extensions:
            reason = f"file extension {extension!r}"
        elif any(
            pattern.fullmatch(header) for pattern in spec.header_patterns for header in includes
        ):
            reason = "included header"
        elif any(
            re.search(rf"(?m)^\s*#\s*pragma\s+{re.escape(prefix)}\b", code)
            for prefix in spec.pragma_prefixes
        ):
            reason = "pragma directive"
        elif _uses_namespace(code, spec):
            reason = "namespace usage"
        elif spec.detect_patterns:
            hits = sum(len(pattern.findall(code)) for pattern in spec.detect_patterns)
            if hits >= DETECTION_MIN_HITS:
                weak.append(spec)
                logger.debug(
                    "Weakly detected dialect {} in {} ({} pattern hits)",
                    spec.name, path or "<string>", hits,
                )
                continue

        if reason is not None:
            logger.debug(
                "Detected dialect {} in {} ({})", spec.name, path or "<string>", reason
            )
            strong.append(spec)

    # Sibling dialects (CUDA/HIP) share markers such as '__global__' and
    # 'threadIdx'. When one of them is detected with a strong signal (file
    # extension, header, ...), a sibling detected only via those shared
    # markers is a false positive and dropped.
    detected = list(strong)
    for spec in weak:
        rival = next(
            (other for other in strong if other is not spec and spec.keywords & other.keywords),
            None,
        )
        if rival is not None:
            logger.debug(
                "Dropping weakly detected dialect {} in {} (overlaps with {})",
                spec.name, path or "<string>", rival.name,
            )
            continue
        detected.append(spec)

    return sorted(detected, key=list(registry.dialects.values()).index)
