"""Tests for :mod:`ppbcc.code_complexity.tokenizer`."""

from ppbcc.code_complexity.tokenizer import Token, TokenKind, tokenize


def kinds(tokens: list[Token]) -> list[TokenKind]:
    """Extracts the token kinds for compact assertions."""
    return [token.kind for token in tokens]


def texts(tokens: list[Token], kind: TokenKind | None = None) -> list[str]:
    """Extracts the token texts, optionally filtered by kind."""
    return [token.text for token in tokens if kind is None or token.kind is kind]


class TestBasicLexing:
    def test_simple_statement(self):
        tokens = tokenize("int a = 1;")
        assert texts(tokens) == ["int", "a", "=", "1", ";"]
        assert kinds(tokens) == [
            TokenKind.IDENT,
            TokenKind.IDENT,
            TokenKind.PUNCT,
            TokenKind.NUMBER,
            TokenKind.PUNCT,
        ]

    def test_line_numbers(self):
        tokens = tokenize("int a;\n\nint b;\n")
        assert [(t.text, t.line) for t in tokens if t.text in ("a", "b")] == [
            ("a", 1),
            ("b", 3),
        ]

    def test_maximal_munch_operators(self):
        tokens = tokenize("a <<= b; c <=> d; e->*f;")
        punct = texts(tokens, TokenKind.PUNCT)
        assert "<<=" in punct and "<=>" in punct and "->*" in punct

    def test_kernel_launch_brackets_lexed_as_one_token(self):
        tokens = tokenize("kernel<<<grid, block>>>(n);")
        punct = texts(tokens, TokenKind.PUNCT)
        assert "<<<" in punct and ">>>" in punct


class TestCommentsAndLiterals:
    def test_comments_are_comment_tokens(self):
        tokens = tokenize("int a; // trailing\n/* multi\nline */ int b;")
        comments = texts(tokens, TokenKind.COMMENT)
        assert comments == ["// trailing", "/* multi\nline */"]
        assert "b" in texts(tokens, TokenKind.IDENT)

    def test_comment_content_produces_no_code_tokens(self):
        tokens = tokenize("// int hidden = 1;\n")
        assert texts(tokens, TokenKind.IDENT) == []
        assert texts(tokens, TokenKind.NUMBER) == []

    def test_string_with_comment_marker(self):
        tokens = tokenize('const char *s = "no // comment";')
        assert texts(tokens, TokenKind.COMMENT) == []
        assert '"no // comment"' in texts(tokens, TokenKind.STRING)

    def test_raw_string(self):
        tokens = tokenize('auto s = R"(x = "1" // raw)"; int b;')
        assert texts(tokens, TokenKind.COMMENT) == []
        assert len(texts(tokens, TokenKind.STRING)) == 1
        assert "b" in texts(tokens, TokenKind.IDENT)

    def test_char_literal(self):
        tokens = tokenize("char c = 'x'; char n = '\\n';")
        assert texts(tokens, TokenKind.CHAR) == ["'x'", "'\\n'"]

    def test_numbers(self):
        code = "a = 1'000 + 0x1F + 1.0e-14 + .5f + 10ULL;"
        numbers = texts(tokenize(code), TokenKind.NUMBER)
        assert numbers == ["1'000", "0x1F", "1.0e-14", ".5f", "10ULL"]


class TestPreprocessor:
    def test_include_angle_normalised_to_operand(self):
        tokens = tokenize("#include <vector>\n")
        assert texts(tokens, TokenKind.DIRECTIVE) == ["#include"]
        assert texts(tokens, TokenKind.STRING) == ['"vector"']

    def test_include_quoted(self):
        tokens = tokenize('#include "my/header.h"\n')
        assert texts(tokens, TokenKind.DIRECTIVE) == ["#include"]
        assert texts(tokens, TokenKind.STRING) == ['"my/header.h"']

    def test_directive_with_space_after_hash(self):
        tokens = tokenize("#  define FOO 1\n")
        assert texts(tokens, TokenKind.DIRECTIVE) == ["#define"]

    def test_hash_not_at_line_start_is_punctuation(self):
        tokens = tokenize("#define CAT(a, b) a##b\n")
        assert "##" in texts(tokens, TokenKind.PUNCT)


class TestPragmaTracking:
    def test_pragma_tokens_carry_prefix(self):
        tokens = tokenize("#pragma omp parallel for\nfor (;;) {}\n")
        directive = next(t for t in tokens if t.kind is TokenKind.DIRECTIVE)
        assert directive.text == "#pragma"
        assert directive.pragma == "omp"
        pragma_idents = [t.text for t in tokens if t.pragma == "omp" and t.kind is TokenKind.IDENT]
        assert pragma_idents == ["omp", "parallel", "for"]

    def test_pragma_ends_at_newline(self):
        tokens = tokenize("#pragma omp parallel\nint a;\n")
        after = [t for t in tokens if t.text in ("int", "a", ";")]
        assert all(t.pragma is None for t in after)

    def test_pragma_backslash_continuation(self):
        tokens = tokenize("#pragma omp parallel \\\n    reduction(+ : x)\nint y;\n")
        reduction = next(t for t in tokens if t.text == "reduction")
        assert reduction.pragma == "omp"
        y = next(t for t in tokens if t.text == "y")
        assert y.pragma is None

    def test_pragma_once(self):
        tokens = tokenize("#pragma once\n")
        directive = next(t for t in tokens if t.kind is TokenKind.DIRECTIVE)
        assert directive.pragma == "once"
