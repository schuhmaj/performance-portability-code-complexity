"""Classification of lexer tokens into Halstead operators and operands.

The classifier consumes the token stream produced by
:mod:`ppbcc.code_complexity.tokenizer` and sorts every countable token into one of
four multisets:

* **operators** -- baseline C++ operators: keywords, punctuation,
  preprocessor directives.
* **operands** -- baseline C++ operands: identifiers, literals.
* **dialect operators** -- constructs contributed by an active GPU dialect:
  dialect keywords (``__global__``), qualified names under a dialect
  namespace (``Kokkos::parallel_for`` counts as *one* operator), dialect
  pragmas and their clauses, kernel-launch punctuation, ...
* **dialect operands** -- values that only occur inside dialect constructs
  (e.g. the variables listed in an OpenMP ``map(...)`` clause).

Counting conventions (documented here once, applied consistently):

* Closing brackets ``)``, ``]``, ``}`` pair with their opening bracket and
  are not counted separately.
* A qualified name that does *not* belong to a dialect is counted classically
  as its identifier segments (operands) joined by ``::`` operators.
* ``true``, ``false``, ``nullptr`` and ``this`` count as operands.
* Namespace aliases (``namespace bc = boost::compute;``) are resolved before
  dialect matching, so ``bc::vector`` is recognised as Boost.Compute.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from loguru import logger

from .config import CppKeywords, DialectSpec
from .tokenizer import CLOSING_BRACKETS, KERNEL_LAUNCH_SPLIT, Token, TokenKind

_NAMESPACE_ALIAS_RE = re.compile(
    r"\bnamespace\s+(\w+)\s*=\s*(?:::)?\s*((?:\w+\s*::\s*)*\w+)\s*;"
)


@dataclass
class TokenCounts:
    """Multisets of classified tokens of one translation unit.

    Attributes:
        operators: Baseline C++ operators keyed by token text.
        operands: Baseline C++ operands keyed by token text.
        dialect_operators: Dialect operators keyed by (qualified) token text.
        dialect_operands: Dialect operands keyed by token text.
    """

    operators: Counter[str] = field(default_factory=Counter)
    operands: Counter[str] = field(default_factory=Counter)
    dialect_operators: Counter[str] = field(default_factory=Counter)
    dialect_operands: Counter[str] = field(default_factory=Counter)

    def merge(self, other: "TokenCounts") -> None:
        """Adds another unit's counts in place (used for aggregation).

        Args:
            other: Counts to merge into this instance.
        """
        self.operators += other.operators
        self.operands += other.operands
        self.dialect_operators += other.dialect_operators
        self.dialect_operands += other.dialect_operands

    @property
    def full_operators(self) -> Counter[str]:
        """Baseline plus dialect operators (the "full" program view)."""
        return self.operators + self.dialect_operators

    @property
    def full_operands(self) -> Counter[str]:
        """Baseline plus dialect operands (the "full" program view)."""
        return self.operands + self.dialect_operands


def extract_namespace_aliases(code: str) -> dict[str, tuple[str, ...]]:
    """Extracts C++ namespace alias definitions from source text.

    Args:
        code: Raw source text.

    Returns:
        Mapping from alias name to the target namespace split into segments,
        e.g. ``{"bc": ("boost", "compute")}``.
    """
    aliases: dict[str, tuple[str, ...]] = {}
    for name, target in _NAMESPACE_ALIAS_RE.findall(code):
        aliases[name] = tuple(seg.strip() for seg in target.split("::"))
    if aliases:
        logger.debug("Found namespace aliases: {}", aliases)
    return aliases


class TokenClassifier:
    """Classifies a token stream for a fixed set of active dialects.

    Args:
        keywords: Baseline C++ keyword sets.
        dialects: Active dialect specifications; their constructs are counted
            as dialect operators. Pass an empty list for plain C++.
        namespace_aliases: Namespace aliases of the translation unit, as
            returned by :func:`extract_namespace_aliases`.
    """

    def __init__(
        self,
        keywords: CppKeywords,
        dialects: list[DialectSpec],
        namespace_aliases: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._keywords = keywords
        self._dialects = list(dialects)
        self._aliases = dict(namespace_aliases or {})
        self._pragma_specs: dict[str, DialectSpec] = {
            prefix: spec for spec in self._dialects for prefix in spec.pragma_prefixes
        }
        self._dialect_punctuation: frozenset[str] = frozenset().union(
            *(spec.punctuation for spec in self._dialects)
        ) if self._dialects else frozenset()

    def classify(self, tokens: list[Token]) -> TokenCounts:
        """Classifies all tokens into operator/operand multisets.

        Args:
            tokens: Token stream of one translation unit (comments included;
                they are skipped here).

        Returns:
            The resulting :class:`TokenCounts`.
        """
        counts = TokenCounts()
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.kind is TokenKind.COMMENT:
                index += 1
                continue
            if token.pragma is not None or (
                token.kind is TokenKind.DIRECTIVE and token.text == "#pragma"
            ):
                index = self._classify_pragma_token(tokens, index, counts)
                continue
            if token.kind is TokenKind.IDENT:
                index = self._classify_identifier_chain(tokens, index, counts)
                continue
            self._classify_simple(token, counts)
            index += 1
        return counts

    def _classify_simple(self, token: Token, counts: TokenCounts) -> None:
        """Classifies a token that needs no lookahead (not ident/pragma).

        Args:
            token: The token to classify.
            counts: Multisets updated in place.
        """
        match token.kind:
            case TokenKind.NUMBER | TokenKind.STRING | TokenKind.CHAR:
                counts.operands[token.text] += 1
            case TokenKind.DIRECTIVE:
                counts.operators[token.text] += 1
            case TokenKind.PUNCT:
                self._classify_punctuation(token.text, counts)
            case _:  # pragma: no cover - defensive
                logger.warning("Unhandled token {!r}", token)

    def _classify_punctuation(self, text: str, counts: TokenCounts) -> None:
        """Classifies a punctuation token.

        Args:
            text: Punctuation text, e.g. ``"("`` or ``"<<<"``.
            counts: Multisets updated in place.
        """
        if text in self._dialect_punctuation:
            # Kernel-launch brackets: '>>>' closes the counted '<<<'.
            if text == ">>>" and "<<<" in self._dialect_punctuation:
                return
            counts.dialect_operators[text] += 1
            return
        if text in KERNEL_LAUNCH_SPLIT:
            # No CUDA/HIP active: '<<<' was over-munched, re-interpret it as
            # the underlying C++ operators ('<<' + '<').
            for part in KERNEL_LAUNCH_SPLIT[text]:
                if part not in CLOSING_BRACKETS:
                    counts.operators[part] += 1
            return
        if text in CLOSING_BRACKETS:
            return
        counts.operators[text] += 1

    def _classify_identifier(self, text: str, counts: TokenCounts) -> None:
        """Classifies a single (unqualified) identifier.

        Dialect keywords take precedence over C++ keywords (relevant e.g. for
        ``volatile`` in GLSL), C++ keywords over dialect identifier patterns.

        Args:
            text: Identifier text.
            counts: Multisets updated in place.
        """
        for spec in self._dialects:
            if text in spec.keywords:
                counts.dialect_operators[text] += 1
                return
        if text in self._keywords.keywords:
            counts.operators[text] += 1
            return
        if text in self._keywords.operand_keywords:
            counts.operands[text] += 1
            return
        for spec in self._dialects:
            if spec.matches_identifier(text):
                logger.trace("{} identifier operator: {}", spec.name, text)
                counts.dialect_operators[text] += 1
                return
        counts.operands[text] += 1

    def _classify_identifier_chain(
        self, tokens: list[Token], index: int, counts: TokenCounts
    ) -> int:
        """Classifies an identifier and any qualified name it starts.

        Consumes the maximal chain ``A::B::...::Z`` starting at ``index``.
        If the (alias-resolved) chain lies in a dialect namespace it is
        counted as one dialect operator; otherwise the segments are counted
        individually with ``::`` operators in between.

        Args:
            tokens: Full token stream.
            index: Index of the leading identifier token.
            counts: Multisets updated in place.

        Returns:
            Index of the first token after the consumed chain.
        """
        segments = [tokens[index].text]
        next_index = index + 1
        while (
            next_index + 1 < len(tokens)
            and tokens[next_index].kind is TokenKind.PUNCT
            and tokens[next_index].text == "::"
            and tokens[next_index + 1].kind is TokenKind.IDENT
            and tokens[next_index + 1].pragma is None
        ):
            segments.append(tokens[next_index + 1].text)
            next_index += 2

        resolved = tuple(segments)
        if resolved[0] in self._aliases:
            resolved = self._aliases[resolved[0]] + resolved[1:]

        for spec in self._dialects:
            if spec.matches_qualified(resolved):
                qualified = "::".join(resolved)
                logger.trace("{} qualified operator: {}", spec.name, qualified)
                counts.dialect_operators[qualified] += 1
                return next_index

        for position, segment in enumerate(segments):
            if position > 0:
                counts.operators["::"] += 1
            self._classify_identifier(segment, counts)
        return next_index

    def _classify_pragma_token(
        self, tokens: list[Token], index: int, counts: TokenCounts
    ) -> int:
        """Classifies one token belonging to a ``#pragma`` line.

        For dialect pragmas (``#pragma omp ...``) the directive and prefix
        are merged into one operator (``"#pragma omp"``), known clauses and
        punctuation count as dialect operators and everything else (loop
        variables, bounds, ...) as dialect operands. Non-dialect pragmas
        (``#pragma once``) are counted as baseline operators/operands.

        Args:
            tokens: Full token stream.
            index: Index of the pragma token.
            counts: Multisets updated in place.

        Returns:
            Index of the next token to process.
        """
        token = tokens[index]
        spec = self._pragma_specs.get(token.pragma) if token.pragma else None

        if spec is None:
            # Not a dialect pragma: classify like regular code.
            if token.kind is TokenKind.IDENT:
                self._classify_identifier(token.text, counts)
            else:
                self._classify_simple(token, counts)
            return index + 1

        if token.kind is TokenKind.DIRECTIVE:
            counts.dialect_operators[f"#pragma {token.pragma}"] += 1
            return index + 1
        if token.kind is TokenKind.IDENT and token.text == token.pragma:
            # The prefix word itself, already merged into '#pragma <prefix>'.
            return index + 1

        match token.kind:
            case TokenKind.IDENT:
                if token.text in spec.pragma_clauses or spec.matches_identifier(token.text):
                    counts.dialect_operators[token.text] += 1
                else:
                    counts.dialect_operands[token.text] += 1
            case TokenKind.NUMBER | TokenKind.STRING | TokenKind.CHAR:
                counts.dialect_operands[token.text] += 1
            case TokenKind.PUNCT:
                if token.text not in CLOSING_BRACKETS:
                    counts.dialect_operators[token.text] += 1
            case _:  # pragma: no cover - defensive
                logger.warning("Unhandled pragma token {!r}", token)
        return index + 1
