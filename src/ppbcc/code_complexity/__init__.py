"""Halstead complexity and LOC metrics for C++ and GPU-enriched C++.

The package analyses C++ sources (and the C-like kernel/shading languages
used by GPU paradigms) and computes the Halstead complexity measures as well
as line-based size metrics. Constructs of GPU programming models such as
OpenMP, OpenACC, Kokkos, RAJA, Alpaka, CUDA, HIP, SYCL, OpenCL, Vulkan,
Boost.Compute, WebGPU/WGSL, GLSL, Slang and Metal are recognised and counted
as *dialect operators*, which allows separating the baseline C++ complexity
from the share added by a paradigm.

Typical usage:

    >>> from ppbcc.code_complexity import evaluate
    >>> frame = evaluate([Path("src/")], language_dialect="kokkos",
    ...                  metrics=["halstead", "loc"], diff=True)
"""

from .classification import TokenClassifier, TokenCounts, extract_namespace_aliases
from .config import (
    AUTO_DIALECT_NAME,
    BASELINE_DIALECT_NAMES,
    SOURCE_EXTENSIONS,
    CppKeywords,
    DialectRegistry,
    DialectSpec,
    load_cpp_keywords,
    load_dialects,
)
from .detection import detect_dialects
from .evaluate import analyze_source, collect_source_files, evaluate
from .halstead import HalsteadMetrics
from .loc import LineMetrics, count_lines
from .report import (
    DIALECT_COLUMNS,
    DIFF_COLUMNS,
    HALSTEAD_COLUMNS,
    LOC_COLUMNS,
    METRIC_GROUPS,
    resolve_metric_columns,
    save_csv,
)
from .tokenizer import Token, TokenKind, tokenize

__version__ = "0.1.0"

__all__ = [
    "AUTO_DIALECT_NAME",
    "BASELINE_DIALECT_NAMES",
    "CppKeywords",
    "DIALECT_COLUMNS",
    "DIFF_COLUMNS",
    "DialectRegistry",
    "DialectSpec",
    "HALSTEAD_COLUMNS",
    "HalsteadMetrics",
    "LOC_COLUMNS",
    "LineMetrics",
    "METRIC_GROUPS",
    "SOURCE_EXTENSIONS",
    "Token",
    "TokenClassifier",
    "TokenCounts",
    "TokenKind",
    "analyze_source",
    "collect_source_files",
    "count_lines",
    "detect_dialects",
    "evaluate",
    "extract_namespace_aliases",
    "load_cpp_keywords",
    "load_dialects",
    "resolve_metric_columns",
    "save_csv",
    "tokenize",
    "__version__",
]
