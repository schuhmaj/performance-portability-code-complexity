"""Tests for :mod:`ppbcc.code_complexity.evaluate` and :mod:`ppbcc.code_complexity.report`."""

import pandas as pd
import pytest

from ppbcc.code_complexity.evaluate import AGGREGATE_ROW_NAME, collect_source_files, evaluate
from ppbcc.code_complexity.report import format_table, report_columns, resolve_metric_columns

CPP_CODE = """\
#include <vector>

int sum(const std::vector<int> &values) {
    int result = 0;
    for (const int value : values) {
        result += value;
    }
    return result;
}
"""

KOKKOS_CODE = """\
#include <Kokkos_Core.hpp>

double dot(int n) {
    double result = 0.0;
    Kokkos::parallel_reduce("dot", n, KOKKOS_LAMBDA(const int i, double &sum) {
        sum += 1.0;
    }, result);
    return result;
}
"""


@pytest.fixture
def project(tmp_path):
    """Creates a small mixed C++/Kokkos project tree."""
    (tmp_path / "plain.cpp").write_text(CPP_CODE)
    nested = tmp_path / "kokkos"
    nested.mkdir()
    (nested / "impl.cpp").write_text(KOKKOS_CODE)
    (tmp_path / "notes.md").write_text("not a source file")
    return tmp_path


class TestCollectSourceFiles:
    def test_directory_is_searched_recursively(self, project):
        files = collect_source_files([project])
        assert [f.name for f in files] == ["impl.cpp", "plain.cpp"]

    def test_explicit_file_and_no_duplicates(self, project):
        files = collect_source_files([project, project / "plain.cpp"])
        assert len(files) == 2

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            collect_source_files([tmp_path / "does_not_exist"])


class TestEvaluate:
    def test_returns_one_row_per_file(self, project):
        frame = evaluate([project])
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 2
        assert list(frame.columns[:2]) == ["file", "dialect"]

    def test_auto_detection_labels_dialects(self, project):
        frame = evaluate([project]).set_index("file", drop=False)
        by_name = {row["file"].split("/")[-1]: row for _, row in frame.iterrows()}
        assert by_name["plain.cpp"]["dialect"] == "cpp"
        assert by_name["impl.cpp"]["dialect"] == "kokkos"
        assert by_name["plain.cpp"]["dialect_total_operators"] == 0
        assert by_name["impl.cpp"]["dialect_total_operators"] > 0

    def test_forced_dialect(self, project):
        frame = evaluate([project / "kokkos"], language_dialect="kokkos")
        assert frame.iloc[0]["dialect"] == "kokkos"
        assert frame.iloc[0]["dialect_distinct_operators"] >= 2

    def test_baseline_dialect_cpp(self, project):
        frame = evaluate([project / "kokkos"], language_dialect="cpp")
        assert frame.iloc[0]["dialect"] == "cpp"
        assert frame.iloc[0]["dialect_total_operators"] == 0

    def test_unknown_dialect_raises(self, project):
        with pytest.raises(KeyError):
            evaluate([project], language_dialect="fortran")

    def test_metric_selection(self, project):
        frame = evaluate([project], metrics=["Halstead_Effort", "Lines of Code"])
        assert list(frame.columns) == [
            "file", "dialect", "loc", "sloc", "comment_lines", "blank_lines", "effort",
        ]

    def test_diff_columns(self, project):
        frame = evaluate([project], diff=True).set_index("file", drop=False)
        row = next(r for _, r in frame.iterrows() if r["file"].endswith("impl.cpp"))
        assert row["delta_total_operators"] > 0
        assert row["baseline_total_operators"] + row["delta_total_operators"] == (
            row["total_operators"]
        )
        assert row["delta_effort"] > 0
        plain = next(r for _, r in frame.iterrows() if r["file"].endswith("plain.cpp"))
        assert plain["delta_total_operators"] == 0

    def test_aggregate_row(self, project):
        frame = evaluate([project], aggregate=True)
        assert len(frame) == 3
        total = frame.iloc[-1]
        assert total["file"] == AGGREGATE_ROW_NAME
        assert total["loc"] == frame.iloc[0]["loc"] + frame.iloc[1]["loc"]
        assert total["dialect"] == "kokkos"

    def test_csv_output(self, project, tmp_path):
        output = tmp_path / "out" / "report.csv"
        frame = evaluate([project], output=output)
        assert output.exists()
        loaded = pd.read_csv(output)
        assert len(loaded) == len(frame)
        assert list(loaded.columns) == list(frame.columns)


class TestFormatTable:
    def test_contains_headers_rows_and_borders(self, project):
        frame = evaluate([project], metrics=["sloc"])
        table = format_table(frame)
        assert "file" in table and "sloc" in table
        assert "plain.cpp" in table and "impl.cpp" in table
        assert "╭" in table  # rounded_outline border

    def test_alternative_format(self, project):
        frame = evaluate([project], metrics=["sloc"])
        assert "|" in format_table(frame, table_format="github")


class TestMetricResolution:
    def test_default_selects_all(self):
        columns = resolve_metric_columns(None)
        assert "loc" in columns and "effort" in columns and "dialect_total_operators" in columns

    def test_group_and_single_metric(self):
        assert resolve_metric_columns(["sloc"]) == ["sloc"]
        assert resolve_metric_columns(["halstead_volume"]) == ["volume"]
        assert resolve_metric_columns(["volume", "effort"]) == ["volume", "effort"]

    def test_case_and_separator_insensitive(self):
        assert resolve_metric_columns(["Halstead Effort"]) == ["effort"]
        assert resolve_metric_columns(["LINES-OF-CODE"]) == [
            "loc", "sloc", "comment_lines", "blank_lines",
        ]

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            resolve_metric_columns(["cyclomatic"])

    def test_diff_columns_follow_their_metric(self):
        columns = report_columns(["halstead_effort"], diff=True)
        assert columns == ["file", "dialect", "effort", "baseline_effort", "delta_effort"]
