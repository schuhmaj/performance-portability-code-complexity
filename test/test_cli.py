"""Tests for the command line interface (:mod:`code_complexity.__main__`)."""

import pandas as pd
import pytest

from code_complexity.__main__ import build_parser, main

CODE = """\
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    process(i);
}
"""


@pytest.fixture
def source_dir(tmp_path):
    (tmp_path / "impl.cpp").write_text(CODE)
    return tmp_path


class TestArgumentParsing:
    def test_defaults(self):
        args = build_parser().parse_args(["src"])
        assert args.dialect == "auto"
        assert args.metrics is None
        assert args.verbose == 0
        assert not args.diff

    def test_verbosity_counting(self):
        args = build_parser().parse_args(["src", "-vv"])
        assert args.verbose == 2


class TestMain:
    def test_analysis_with_csv_output(self, source_dir, tmp_path, capsys):
        output = tmp_path / "report.csv"
        exit_code = main([str(source_dir), "-o", str(output), "-m", "halstead", "loc"])
        assert exit_code == 0
        frame = pd.read_csv(output)
        assert len(frame) == 1
        assert frame.iloc[0]["dialect"] == "openmp"
        assert "effort" in frame.columns
        # The table is also printed to stdout.
        assert "impl.cpp" in capsys.readouterr().out

    def test_diff_flag(self, source_dir, tmp_path):
        output = tmp_path / "report.csv"
        assert main([str(source_dir), "--diff", "-o", str(output)]) == 0
        frame = pd.read_csv(output)
        assert frame.iloc[0]["delta_total_operators"] > 0

    def test_missing_source_returns_error(self, tmp_path):
        assert main([str(tmp_path / "nope")]) == 1

    def test_unknown_dialect_returns_error(self, source_dir):
        assert main([str(source_dir), "-d", "fortran"]) == 1

    def test_unknown_metric_returns_error(self, source_dir):
        assert main([str(source_dir), "-m", "cyclomatic"]) == 1

    def test_list_dialects(self, capsys):
        assert main(["--list-dialects"]) == 0
        out = capsys.readouterr().out
        assert "kokkos" in out
        assert "cuda" in out
