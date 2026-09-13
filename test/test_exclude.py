"""Tests for :mod:`ppbcc.code_complexity.exclude`."""

import pytest

from ppbcc.code_complexity.cli import main
from ppbcc.code_complexity.evaluate import evaluate
from ppbcc.code_complexity.exclude import Exclusions, is_excluded_file, strip_excluded
from ppbcc.code_complexity.tokenizer import tokenize

MARKERS = Exclusions.create(macros=[r"PPB_MARKER_\w+", "PPB_PROFILING"], headers=["common/Marker.h"])


def strip(code: str, exclusions: Exclusions = MARKERS) -> str:
    return strip_excluded(code, exclusions)


class TestExclusions:
    def test_macro_patterns_match_whole_names(self):
        assert MARKERS.is_excluded_macro("PPB_MARKER_GPU_SCOPE")
        assert MARKERS.is_excluded_macro("PPB_PROFILING")
        assert not MARKERS.is_excluded_macro("PPB_PROFILING_DEVICE_DEBUG")
        assert not MARKERS.is_excluded_macro("MY_PPB_MARKER_X")

    def test_header_patterns_match_path_tails(self, tmp_path):
        assert MARKERS.is_excluded_header("common/Marker.h")
        assert not MARKERS.is_excluded_header("common/Marker.hpp")
        assert is_excluded_file(tmp_path / "src" / "common" / "Marker.h", MARKERS)
        assert not is_excluded_file(tmp_path / "src" / "other" / "Marker.h", MARKERS)

    def test_empty_exclusions_leave_the_code_untouched(self):
        code = 'PPB_MARKER_GPU_SCOPE("x");\n'
        assert not Exclusions.create()
        assert strip_excluded(code, Exclusions.create()) == code

    def test_invalid_macro_pattern_raises(self):
        with pytest.raises(ValueError):
            Exclusions.create(macros=["("])


class TestInvocations:
    def test_invocation_line_is_removed(self):
        code = 'void f() {\n    PPB_MARKER_GPU_SCOPE("evaluate");\n    run();\n}\n'
        assert strip(code) == "void f() {\n    run();\n}\n"

    def test_nested_parentheses_and_trailing_comment(self):
        code = 'a();\nPPB_MARKER_GPU_START(name(("x"))); // opens the region\nb();\n'
        assert strip(code) == "a();\nb();\n"

    def test_invocation_inside_a_line_keeps_the_rest(self):
        assert strip('a(); PPB_MARKER_CPU_STOP("t"); b();\n') == "a();  b();\n"

    def test_unrelated_identifiers_and_strings_stay(self):
        code = 'const char *s = "PPB_MARKER_GPU_SCOPE(x);";\nPPB_MARKER_X_Y z;\n'
        assert strip(code) == 'const char *s = "PPB_MARKER_GPU_SCOPE(x);";\nz;\n'

    def test_multi_line_invocation(self):
        code = 'a();\nPPB_MARKER_GPU_START(\n    "tag");\nb();\n'
        assert strip(code) == "a();\nb();\n"


class TestConditionals:
    def test_ifdef_block_is_removed(self):
        code = "a();\n#ifdef PPB_MARKER_SYNC_REGION\n// wait\n\nsync();\n#endif\nb();\n"
        assert strip(code) == "a();\nb();\n"

    def test_ifdef_else_keeps_the_else_branch(self):
        code = "#ifdef PPB_PROFILING\nsmall();\n#else\nlarge();\n#endif\n"
        assert strip(code) == "large();\n"

    def test_ifndef_else_keeps_the_first_branch(self):
        code = "#ifndef PPB_PROFILING\nlarge();\n#else\nsmall();\n#endif\ntail();\n"
        assert strip(code) == "large();\ntail();\n"

    @pytest.mark.parametrize(
        ("condition", "kept"),
        [("defined(PPB_PROFILING)", "no"), ("defined PPB_PROFILING", "no"), ("PPB_PROFILING", "no"),
         ("!defined(PPB_PROFILING)", "yes"), ("! defined PPB_PROFILING", "yes")],
    )
    def test_if_defined_forms(self, condition, kept):
        code = f"#if {condition}\nyes();\n#else\nno();\n#endif\n"
        assert strip(code) == f"{kept}();\n"

    def test_elif_takes_over_as_if(self):
        code = "#ifdef PPB_PROFILING\na();\n#elif defined(OTHER)\nb();\n#else\nc();\n#endif\n"
        assert strip(code) == "#if defined(OTHER)\nb();\n#else\nc();\n#endif\n"

    def test_branches_after_a_kept_branch_are_removed(self):
        code = "#ifndef PPB_PROFILING\na();\n#elif defined(OTHER)\nb();\n#else\nc();\n#endif\nd();\n"
        assert strip(code) == "a();\nd();\n"

    def test_nested_conditionals_inside_a_removed_branch(self):
        code = "#ifdef PPB_PROFILING\n#ifdef OTHER\nx();\n#else\ny();\n#endif\n#else\nz();\n#endif\n"
        assert strip(code) == "z();\n"

    def test_foreign_conditional_around_an_excluded_one(self):
        code = "#ifdef OTHER\n#ifdef PPB_PROFILING\nx();\n#endif\ny();\n#endif\n"
        assert strip(code) == "#ifdef OTHER\ny();\n#endif\n"

    def test_combined_condition_is_kept(self):
        code = "#if defined(PPB_PROFILING) || defined(OTHER)\nx();\n#endif\n"
        assert strip(code) == code


class TestIncludes:
    def test_excluded_include_is_removed(self):
        code = '#include "common/Marker.h"\n#include <memory>\n#include "polyhedral/Definitions.h"\n'
        assert strip(code) == '#include <memory>\n#include "polyhedral/Definitions.h"\n'

    def test_angle_bracket_include_is_removed(self):
        assert strip("#include <common/Marker.h>\nint a;\n") == "int a;\n"


INSTRUMENTED = """\
#include "common/Marker.h"
#include <vector>

// Sums the values
int sum(const std::vector<int> &values) {
    PPB_MARKER_CPU_SCOPE("sum");
    int result = 0;
    for (const int value : values) {
        result += value;
    }
#ifdef PPB_MARKER_SYNC_REGION
    synchronize();
#endif
    return result;
}
"""

CLEAN = """\
#include <vector>

// Sums the values
int sum(const std::vector<int> &values) {
    int result = 0;
    for (const int value : values) {
        result += value;
    }
    return result;
}
"""


class TestEvaluate:
    @pytest.fixture
    def trees(self, tmp_path):
        instrumented = tmp_path / "instrumented"
        (instrumented / "common").mkdir(parents=True)
        (instrumented / "impl.cpp").write_text(INSTRUMENTED)
        (instrumented / "common" / "Marker.h").write_text("#define PPB_MARKER_CPU_SCOPE(tag) Region r(tag)\n")
        clean = tmp_path / "clean"
        clean.mkdir()
        (clean / "impl.cpp").write_text(CLEAN)
        return instrumented, clean

    def test_stripped_source_equals_the_clean_source(self):
        assert strip(INSTRUMENTED, Exclusions.create([r"PPB_MARKER_\w+"], ["common/Marker.h"])) == CLEAN

    def test_offsets_do_not_change_token_equality(self):
        assert tokenize("int a;") == tokenize("  int   a ;")

    def test_instrumented_tree_measures_like_the_clean_one(self, trees):
        instrumented, clean = trees
        excluded = evaluate(
            [instrumented], aggregate=True,
            exclude_macros=[r"PPB_MARKER_\w+"], exclude_headers=["common/Marker.h"],
        )
        reference = evaluate([clean], aggregate=True)
        assert len(excluded) == len(reference) == 2
        columns = [column for column in reference.columns if column != "file"]
        assert excluded[columns].reset_index(drop=True).equals(reference[columns].reset_index(drop=True))

    def test_without_exclusions_the_instrumentation_counts(self, trees):
        instrumented, clean = trees
        counted = evaluate([instrumented / "impl.cpp"])
        reference = evaluate([clean / "impl.cpp"])
        assert counted["sloc"].iloc[0] > reference["sloc"].iloc[0]
        assert counted["total_operators"].iloc[0] > reference["total_operators"].iloc[0]

    def test_cli_flags(self, trees, tmp_path):
        instrumented, _ = trees
        output = tmp_path / "report.csv"
        arguments = [str(instrumented), "--exclude-macro", r"PPB_MARKER_\w+",
                     "--exclude-header", "common/Marker.h", "-o", str(output)]
        assert main(arguments) == 0
        assert "Marker.h" not in output.read_text()

    def test_cli_rejects_an_invalid_pattern(self, trees):
        instrumented, _ = trees
        assert main([str(instrumented), "--exclude-macro", "("]) == 1
