"""Line-based size metrics (LOC/SLOC/comment/blank lines).

The classification is derived from the token stream so that multi-line
comments, comments behind code and string literals containing ``//`` are all
handled correctly:

* **LOC** -- total number of physical lines.
* **SLOC** -- lines containing at least one code token (a line with code
  *and* a trailing comment counts as source, not comment).
* **comment lines** -- lines whose only content is comment text.
* **blank lines** -- lines with neither code nor comments.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tokenizer import Token, TokenKind


@dataclass(frozen=True)
class LineMetrics:
    """Line counts of one source file.

    Attributes:
        loc: Total number of physical lines (Lines of Code).
        sloc: Number of lines containing code (Source Lines of Code).
        comment_lines: Number of comment-only lines.
        blank_lines: Number of lines without code or comments.
    """

    loc: int
    sloc: int
    comment_lines: int
    blank_lines: int

    def as_dict(self) -> dict[str, int]:
        """Serialises the line counts.

        Returns:
            Mapping from metric name (matching the CSV column names) to its
            value, in a stable order.
        """
        return {
            "loc": self.loc,
            "sloc": self.sloc,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
        }

    def combine(self, other: "LineMetrics") -> "LineMetrics":
        """Adds another file's line counts (used for aggregation).

        Args:
            other: Line counts to add.

        Returns:
            A new :class:`LineMetrics` with the summed counts.
        """
        return LineMetrics(
            loc=self.loc + other.loc,
            sloc=self.sloc + other.sloc,
            comment_lines=self.comment_lines + other.comment_lines,
            blank_lines=self.blank_lines + other.blank_lines,
        )


def count_lines(code: str, tokens: list[Token]) -> LineMetrics:
    """Computes the line metrics of one source file.

    Args:
        code: Raw source text.
        tokens: Token stream of the same text, as produced by
            :func:`ppbcc.code_complexity.tokenizer.tokenize`.

    Returns:
        The :class:`LineMetrics` of the file.
    """
    if not code:
        return LineMetrics(loc=0, sloc=0, comment_lines=0, blank_lines=0)
    total = code.count("\n") + (0 if code.endswith("\n") else 1)

    code_lines: set[int] = set()
    comment_lines: set[int] = set()
    for token in tokens:
        target = comment_lines if token.kind is TokenKind.COMMENT else code_lines
        target.update(range(token.line, token.end_line + 1))

    comment_only = comment_lines - code_lines
    blank = total - len(code_lines | comment_lines)
    return LineMetrics(
        loc=total,
        sloc=len(code_lines),
        comment_lines=len(comment_only),
        blank_lines=blank,
    )
