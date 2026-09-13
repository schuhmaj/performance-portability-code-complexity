"""Removal of code that should not count towards an implementation's complexity.

A benchmark carries code that is not part of the algorithm it implements, most
prominently profiler instrumentation: region markers around the kernels, and
headers which define them. Such code would be counted as if every implementation
had written it, so it is removed from the source text *before* the analysis,
which then sees the program as if the instrumentation had never been added.

Two kinds of exclusion are supported:

* **Macros**, given as regular expressions matched against the whole macro name.

  - An invocation of such a macro is removed together with its argument list and
    a directly following ``;``, e.g. ``PPB_MARKER_GPU_SCOPE("evaluate");``.
  - A conditional on such a macro (``#ifdef``, ``#ifndef``, ``#if defined(...)``,
    ``#if !defined(...)``, ``#if NAME``) is resolved as if the macro was
    undefined: the branch the preprocessor would drop is removed with the
    directives, the branch it would keep stays. A conditional which combines the
    macro with other conditions is left untouched, with a warning.

* **Headers**, given as glob patterns (e.g. ``common/Marker.h``) matched against
  the spelling of an ``#include`` and against the tail of a file path. The
  ``#include`` directive is removed, and :func:`is_excluded_file` tells callers
  which files to leave out altogether.

A line which held nothing but removed code (and possibly a comment) is removed
entirely, so the line metrics shrink along with the Halstead counts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from loguru import logger

from .tokenizer import Token, TokenKind, tokenize

#: Conditional directives which open a block.
_OPENING_DIRECTIVES = frozenset({"#if", "#ifdef", "#ifndef"})

#: ``defined(NAME)``, ``defined NAME`` or ``NAME``, optionally negated.
_SIMPLE_CONDITION_RE = re.compile(
    r"^(?P<negated>!\s*)?(?:defined\s*\(\s*(?P<parenthesized>\w+)\s*\)|defined\s+(?P<bare>\w+)|(?P<name>\w+))$"
)
_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


@dataclass(frozen=True)
class Exclusions:
    """What to remove from the analysed code.

    Attributes:
        macros: Compiled patterns; a macro is excluded if one matches its whole name.
        headers: Glob patterns for excluded headers.
    """

    macros: tuple[re.Pattern[str], ...] = ()
    headers: tuple[str, ...] = ()

    @classmethod
    def create(cls, macros: Iterable[str] | None = None, headers: Iterable[str] | None = None) -> Exclusions:
        """Builds the exclusions from their textual patterns.

        Args:
            macros: Regular expressions for macro names, e.g. ``PPB_MARKER_\\w+``.
            headers: Glob patterns for headers, e.g. ``common/Marker.h``.

        Returns:
            The exclusions.

        Raises:
            ValueError: If a macro pattern is not a valid regular expression.
        """
        compiled = []
        for pattern in macros or ():
            try:
                compiled.append(re.compile(pattern))
            except re.error as error:
                raise ValueError(f"Invalid macro pattern {pattern!r}: {error}") from error
        return cls(tuple(compiled), tuple(headers or ()))

    def __bool__(self) -> bool:
        return bool(self.macros or self.headers)

    def is_excluded_macro(self, name: str) -> bool:
        """Whether ``name`` is one of the excluded macros."""
        return any(pattern.fullmatch(name) for pattern in self.macros)

    def is_excluded_header(self, spelling: str) -> bool:
        """Whether an ``#include`` of ``spelling`` includes an excluded header."""
        return _matches_tail(spelling, self.headers)


def is_excluded_file(path: Path, exclusions: Exclusions) -> bool:
    """Whether ``path`` is an excluded header.

    A pattern matches the whole path or any trailing part of it, so
    ``common/Marker.h`` matches ``/repo/src/common/Marker.h``.

    Args:
        path: The file in question.
        exclusions: The exclusions in effect.

    Returns:
        True if the file is to be left out of the analysis.
    """
    return _matches_tail(Path(path).as_posix(), exclusions.headers)


def _matches_tail(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(path, pattern) or fnmatchcase(path, f"*/{pattern}") for pattern in patterns)


@dataclass
class _Conditional:
    """An open ``#if`` block while scanning.

    Attributes:
        excluded: Whether the block is conditioned on an excluded macro.
        keeping: For an excluded block, whether the current branch is kept.
        resolved: For an excluded block, whether a kept branch has been seen.
    """

    excluded: bool
    keeping: bool = True
    resolved: bool = False


class _Stripper:
    """Collects the character ranges to delete from one source text."""

    def __init__(self, code: str, exclusions: Exclusions, name: str) -> None:
        self.code = code
        self.exclusions = exclusions
        self.name = name
        self.tokens = tokenize(code)
        # Half-open character ranges which are deleted.
        self.deleted: list[tuple[int, int]] = []
        # Half-open character ranges of whole lines which are dropped, even if blank.
        self.dropped: list[tuple[int, int]] = []
        # Character offsets at which "elif" becomes "if".
        self.renamed: list[int] = []

    # -- helpers ---------------------------------------------------------

    def _line_start(self, offset: int) -> int:
        return self.code.rfind("\n", 0, offset) + 1

    def _logical_line_end(self, offset: int) -> int:
        """Offset of the newline ending the (backslash-continued) line at ``offset``."""
        position = offset
        while True:
            newline = self.code.find("\n", position)
            if newline == -1:
                return len(self.code)
            if self.code[position:newline].rstrip("\r").endswith("\\"):
                position = newline + 1
                continue
            return newline

    def _directive_argument(self, directive: Token) -> str:
        text = self.code[directive.end:self._logical_line_end(directive.start)]
        text = _COMMENT_RE.sub(" ", text.replace("\\\n", " "))
        return " ".join(text.split())

    def _delete_line(self, directive: Token) -> int:
        """Deletes the whole logical line of a directive; returns the offset after it."""
        start = self._line_start(directive.start)
        end = self._logical_line_end(directive.start)
        self.dropped.append((start, end))
        return min(end + 1, len(self.code))

    # -- passes ----------------------------------------------------------

    def run(self) -> str:
        self._strip_conditionals_and_includes()
        self._strip_invocations()
        return self._render()

    def _strip_conditionals_and_includes(self) -> None:
        stack: list[_Conditional] = []
        drop_from: int | None = None  # start of the region currently being dropped

        def dropping() -> bool:
            return drop_from is not None

        for index, token in enumerate(self.tokens):
            if token.kind is not TokenKind.DIRECTIVE:
                continue
            name = token.text

            if name in _OPENING_DIRECTIVES:
                if dropping():
                    stack.append(_Conditional(excluded=False))
                    continue
                macro, negated = self._excluded_condition(token)
                if macro is None:
                    stack.append(_Conditional(excluded=False))
                    continue
                # The macro counts as undefined: '#ifdef X' and '#if defined(X)' are false,
                # '#ifndef X' and '#if !defined(X)' are true.
                keeping = negated or name == "#ifndef"
                stack.append(_Conditional(excluded=True, keeping=keeping, resolved=keeping))
                after = self._delete_line(token)
                if not keeping:
                    drop_from = after
                continue

            if name in ("#elif", "#else"):
                if not stack or not stack[-1].excluded:
                    continue
                block = stack[-1]
                if block.keeping:
                    # The kept branch ends here; every later branch is dropped.
                    block.keeping = False
                    drop_from = self._line_start(token.start)
                    continue
                if block.resolved:
                    continue
                # All branches so far were dropped, so this one takes over.
                self.dropped.append((drop_from, self._line_start(token.start)))
                drop_from = None
                if name == "#else":
                    block.keeping = True
                    block.resolved = True
                    self._delete_line(token)
                else:
                    # '#elif Y' is now the first branch: it becomes '#if Y' and the rest of the
                    # conditional is left to the preprocessor.
                    self.renamed.append(self.code.index("elif", token.start))
                    block.excluded = False
                continue

            if name == "#endif":
                if not stack:
                    continue
                block = stack.pop()
                if not block.excluded:
                    continue
                if dropping() and not any(b.excluded and not b.keeping for b in stack):
                    self.dropped.append((drop_from, self._line_start(token.start)))
                    drop_from = None
                self._delete_line(token)
                continue

            if name == "#include" and not dropping():
                following = self.tokens[index + 1] if index + 1 < len(self.tokens) else None
                if (
                    following is not None
                    and following.kind is TokenKind.STRING
                    and following.line == token.line
                    and self.exclusions.is_excluded_header(following.text.strip('"'))
                ):
                    self._delete_line(token)

        if drop_from is not None:
            logger.warning("{}: unterminated conditional on an excluded macro", self.name)
            self.dropped.append((drop_from, len(self.code)))

    def _excluded_condition(self, directive: Token) -> tuple[str | None, bool]:
        """The excluded macro an opening directive tests, and whether the test is negated."""
        argument = self._directive_argument(directive)
        if directive.text in ("#ifdef", "#ifndef"):
            macro = argument.split(" ", 1)[0] if argument else ""
            return (macro, False) if self.exclusions.is_excluded_macro(macro) else (None, False)
        match = _SIMPLE_CONDITION_RE.match(argument)
        if match is not None:
            macro = match.group("parenthesized") or match.group("bare") or match.group("name")
            if self.exclusions.is_excluded_macro(macro):
                return macro, match.group("negated") is not None
            return None, False
        if any(self.exclusions.is_excluded_macro(word) for word in re.findall(r"\w+", argument)):
            logger.warning(
                "{}:{}: '{} {}' combines an excluded macro with other conditions and is kept",
                self.name, directive.line, directive.text, argument,
            )
        return None, False

    def _strip_invocations(self) -> None:
        directive_lines: set[int] = set()
        code_tokens: list[Token] = []
        for token in self.tokens:
            if token.kind is TokenKind.DIRECTIVE:
                directive_lines.add(token.line)
            elif token.kind is not TokenKind.COMMENT:
                code_tokens.append(token)

        index = 0
        while index < len(code_tokens):
            token = code_tokens[index]
            if (
                token.kind is not TokenKind.IDENT
                or token.line in directive_lines
                or not self.exclusions.is_excluded_macro(token.text)
            ):
                index += 1
                continue
            end = token.end
            index += 1
            if index < len(code_tokens) and code_tokens[index].text == "(":
                depth = 0
                while index < len(code_tokens):
                    text = code_tokens[index].text
                    depth += text == "("
                    depth -= text == ")"
                    end = code_tokens[index].end
                    index += 1
                    if depth == 0:
                        break
            if index < len(code_tokens) and code_tokens[index].text == ";":
                end = code_tokens[index].end
                index += 1
            if not self.code[self._line_start(token.start):token.start].strip():
                # Nothing precedes the removed code on its line, so neither does the indentation after it.
                while end < len(self.code) and self.code[end] in " \t":
                    end += 1
            self.deleted.append((token.start, end))

    def _render(self) -> str:
        code = self.code
        deleted = bytearray(len(code))
        for start, end in self.deleted:
            deleted[start:end] = b"\x01" * (end - start)
        for offset in self.renamed:
            deleted[offset:offset + 2] = b"\x01\x01"
        comment = bytearray(len(code))
        for token in self.tokens:
            if token.kind is TokenKind.COMMENT:
                comment[token.start:token.end] = b"\x01" * (token.end - token.start)

        output: list[str] = []
        line_start = 0
        while line_start <= len(code):
            newline = code.find("\n", line_start)
            line_end = len(code) if newline == -1 else newline
            if line_start == len(code) and newline == -1:
                break
            terminator = "" if newline == -1 else "\n"

            if any(start <= line_start and line_end <= end for start, end in self.dropped):
                line_start = line_end + 1
                continue

            kept = "".join(code[i] for i in range(line_start, line_end) if not deleted[i])
            had_deletion = any(deleted[i] for i in range(line_start, line_end))
            only_comment_left = all(
                code[i].isspace() or comment[i] for i in range(line_start, line_end) if not deleted[i]
            )
            if had_deletion and only_comment_left:
                # The line held removed code, and maybe a comment on it: it goes as a whole.
                line_start = line_end + 1
                continue
            output.append(kept + terminator)
            line_start = line_end + 1
        return "".join(output)


def strip_excluded(code: str, exclusions: Exclusions, name: str = "<string>") -> str:
    """Removes the excluded macros and headers from a source text.

    Args:
        code: Raw source text.
        exclusions: What to remove.
        name: Name of the source, used in warnings.

    Returns:
        The source text without the excluded code; the text itself if there is
        nothing to exclude.
    """
    if not exclusions:
        return code
    return _Stripper(code, exclusions, name).run()
