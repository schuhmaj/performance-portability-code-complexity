"""A lightweight lexer for C++ and C-like GPU/shading languages.

The lexer splits source text into :class:`Token` objects (identifiers,
numbers, literals, punctuation, comments and preprocessor directives). It is
deliberately *not* a full C++ parser: it is precise enough for Halstead
counting and line classification, and it degrades gracefully on the C-like
shading languages (OpenCL C, GLSL, WGSL, Slang, Metal).

Special handling:

* Comments and string/character literals are matched first, so their content
  never produces spurious tokens.
* ``#include <header>`` is normalised so that the header name becomes a
  single (operand) token.
* ``#pragma`` lines are tracked: every token of a pragma line carries the
  pragma's first word (e.g. ``"omp"``) in :attr:`Token.pragma`, which lets
  the classifier attribute OpenMP/OpenACC directives to their dialect.
* Backslash line continuations inside pragma lines are honoured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto

from loguru import logger


class TokenKind(Enum):
    """Lexical category of a token."""

    COMMENT = auto()
    STRING = auto()
    CHAR = auto()
    NUMBER = auto()
    IDENT = auto()
    PUNCT = auto()
    DIRECTIVE = auto()


@dataclass(slots=True)
class Token:
    """A single lexical token.

    Attributes:
        kind: Lexical category.
        text: Exact source text (directives are normalised to ``#name``).
        line: 1-based line number of the token's first character.
        end_line: 1-based line number of the token's last character (differs
            from ``line`` only for multi-line comments/raw strings).
        pragma: First word of the surrounding ``#pragma`` line (e.g.
            ``"omp"``) if the token is part of one, otherwise ``None``.
        start: Offset of the token's first character in the source text
            (for a directive, the offset of its ``#``), or ``-1`` for a token
            built by hand. Not part of the token's equality.
        end: Offset one past the token's last character, or ``-1``. Not part
            of the token's equality.
    """

    kind: TokenKind
    text: str
    line: int
    end_line: int = 0
    pragma: str | None = None
    start: int = field(default=-1, compare=False, repr=False)
    end: int = field(default=-1, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Defaults ``end_line`` to ``line`` when not provided."""
        if self.end_line == 0:
            self.end_line = self.line


#: Multi-character punctuation, longest first (maximal munch). ``<<<`` and
#: ``>>>`` are always lexed as single tokens; the classifier splits them
#: again if no kernel-launch dialect (CUDA/HIP) is active.
_PUNCTUATION: tuple[str, ...] = (
    "<<<", ">>>", "<<=", ">>=", "<=>", "->*", "...",
    "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "++", "--",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "->", ".*", "##", "::",
    "+", "-", "*", "/", "%", "&", "|", "^", "~", "!", "<", ">", "=",
    "?", ":", ";", ",", ".", "(", ")", "[", "]", "{", "}", "#", "@",
)

#: Closing brackets: lexed, but skipped by the classifier because they pair
#: with their (counted) opening bracket.
CLOSING_BRACKETS: frozenset[str] = frozenset({")", "]", "}"})

#: Kernel-launch punctuation with its baseline C++ re-interpretation, used
#: when no CUDA/HIP dialect is active.
KERNEL_LAUNCH_SPLIT: dict[str, tuple[str, ...]] = {"<<<": ("<<", "<"), ">>>": (">>", ">")}

_PUNCT_ALTERNATION = "|".join(re.escape(p) for p in _PUNCTUATION)

_MASTER_RE = re.compile(
    r"""
      (?P<block_comment>/\*.*?\*/)
    | (?P<line_comment>//[^\n]*)
    | (?P<raw_string>(?:u8|u|U|L)?R"(?P<rsdelim>[^()\\\s]{0,16})\(.*?\)(?P=rsdelim)")
    | (?P<string>(?:u8|u|U|L)?"(?:\\.|[^"\\\n])*")
    | (?P<char>(?:u8|u|U|L)?'(?:\\.|[^'\\\n])+')
    | (?P<number>
          0[xX][0-9a-fA-F']+(?:\.[0-9a-fA-F']*)?(?:[pP][+-]?\d+)?[uUlLfFzZ]*
        | (?:\d[\d']*\.[\d']*|\.\d[\d']*|\d[\d']*)(?:[eE][+-]?\d+)?[uUlLfFzZ]*
      )
    | (?P<ident>[A-Za-z_]\w*)
    | (?P<punct>%s)
    | (?P<newline>\n)
    | (?P<ws>[ \t\r\f\v]+)
    | (?P<backslash>\\)
    | (?P<other>.)
    """ % _PUNCT_ALTERNATION,
    re.DOTALL | re.VERBOSE,
)

_INCLUDE_ANGLE_RE = re.compile(r"(?m)^(\s*\#\s*include\s*)<([^>\n]*)>")


def normalize_includes(code: str) -> str:
    """Rewrites ``#include <header>`` to ``#include "header"``.

    The angle-bracket form would otherwise be lexed as comparison operators
    and identifiers; the string form yields a single operand token for the
    header name. The replacement preserves the text length and line numbers.

    Args:
        code: Raw source text.

    Returns:
        The source text with angle-bracket includes converted to the quoted
        form.
    """
    return _INCLUDE_ANGLE_RE.sub(r'\1"\2"', code)


def tokenize(code: str) -> list[Token]:
    """Lexes source text into tokens.

    Args:
        code: Raw source text of a C++/C-like translation unit.

    Returns:
        All tokens including comments (whitespace is dropped). Tokens inside
        ``#pragma`` lines carry the pragma's first word in
        :attr:`Token.pragma`.
    """
    code = normalize_includes(code)
    tokens: list[Token] = []
    line = 1
    at_line_start = True
    hash_pending = False           # a line-initial '#' awaiting its directive name
    hash_line = 0
    hash_start = 0
    pragma_directive: Token | None = None  # the '#pragma' token awaiting its first word
    pragma_prefix: str | None = None       # first word of the active pragma line
    continuation = False           # backslash directly before the newline

    for match in _MASTER_RE.finditer(code):
        kind = match.lastgroup
        text = match.group()
        start_line = line
        line += text.count("\n")

        if kind == "ws":
            continue
        if kind == "newline":
            if continuation:
                continuation = False
            else:
                pragma_directive = None
                pragma_prefix = None
                if hash_pending:  # stray '#' at end of line
                    tokens.append(
                        Token(TokenKind.PUNCT, "#", start_line, start=hash_start, end=hash_start + 1)
                    )
                    hash_pending = False
            at_line_start = True
            continue
        if kind == "backslash":
            continuation = True
            continue
        continuation = False

        if kind in ("block_comment", "line_comment"):
            tokens.append(
                Token(TokenKind.COMMENT, text, start_line, line, start=match.start(), end=match.end())
            )
            continue

        if kind == "punct" and text == "#" and at_line_start:
            hash_pending = True
            hash_line = start_line
            hash_start = match.start()
            at_line_start = False
            continue
        at_line_start = False

        if hash_pending:
            hash_pending = False
            if kind == "ident":
                directive = Token(
                    TokenKind.DIRECTIVE, f"#{text}", hash_line, start=hash_start, end=match.end()
                )
                tokens.append(directive)
                if text == "pragma":
                    pragma_directive = directive
                continue
            # '#' not followed by a name: emit it as plain punctuation.
            tokens.append(Token(TokenKind.PUNCT, "#", hash_line, start=hash_start, end=hash_start + 1))

        token = Token(
            _TOKEN_KIND_BY_GROUP[kind],
            text,
            start_line,
            line,
            pragma=pragma_prefix,
            start=match.start(),
            end=match.end(),
        )
        if kind == "other":
            logger.trace("Unexpected character {!r} in line {}", text, start_line)
            token.kind = TokenKind.PUNCT

        if pragma_directive is not None and kind == "ident":
            # First word of the pragma line, e.g. 'omp' in '#pragma omp ...'.
            pragma_prefix = text
            pragma_directive.pragma = pragma_prefix
            token.pragma = pragma_prefix
            pragma_directive = None

        tokens.append(token)

    return tokens


_TOKEN_KIND_BY_GROUP: dict[str, TokenKind] = {
    "raw_string": TokenKind.STRING,
    "string": TokenKind.STRING,
    "char": TokenKind.CHAR,
    "number": TokenKind.NUMBER,
    "ident": TokenKind.IDENT,
    "punct": TokenKind.PUNCT,
    "other": TokenKind.PUNCT,
}
