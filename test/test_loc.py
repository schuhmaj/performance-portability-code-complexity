"""Tests for :mod:`ppbcc.code_complexity.loc`."""

from ppbcc.code_complexity.loc import count_lines
from ppbcc.code_complexity.tokenizer import tokenize


def metrics_of(code: str):
    """Tokenises and counts the lines of a snippet."""
    return count_lines(code, tokenize(code))


class TestCountLines:
    def test_empty_file(self):
        metrics = metrics_of("")
        assert metrics.as_dict() == {
            "loc": 0,
            "sloc": 0,
            "comment_lines": 0,
            "blank_lines": 0,
        }

    def test_mixed_file(self):
        code = (
            "// header\n"        # 1: comment only
            "\n"                 # 2: blank
            "int a; // note\n"   # 3: code (trailing comment does not demote)
            "/* multi\n"         # 4: comment only
            "   line */\n"       # 5: comment only
            "int b;\n"           # 6: code
        )
        metrics = metrics_of(code)
        assert metrics.loc == 6
        assert metrics.sloc == 2
        assert metrics.comment_lines == 3
        assert metrics.blank_lines == 1

    def test_missing_trailing_newline(self):
        metrics = metrics_of("int a;\nint b;")
        assert metrics.loc == 2
        assert metrics.sloc == 2

    def test_code_after_block_comment_on_same_line(self):
        metrics = metrics_of("/* c */ int a;\n")
        assert metrics.sloc == 1
        assert metrics.comment_lines == 0

    def test_whitespace_only_line_is_blank(self):
        metrics = metrics_of("int a;\n   \t\nint b;\n")
        assert metrics.blank_lines == 1

    def test_multiline_string_counts_as_code(self):
        metrics = metrics_of('auto s = R"(line1\nline2)";\n')
        assert metrics.loc == 2
        assert metrics.sloc == 2
