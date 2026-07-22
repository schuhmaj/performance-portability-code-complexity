"""Tests for benchmark discovery and report normalization."""

import json

import pytest

from ppbcc.benchmark.cli import build_parser
from ppbcc.benchmark.reports import load_reports
from ppbcc.benchmark.runner import find_files
from ppbcc.constants import (
    BENCHMARK_PROBLEM,
    DESCRIPTION,
    HARDWARE,
    PARADIGM,
    PRECISION,
    PROBLEM_SIZE,
    WALL_CLOCK_TIME,
)


def test_find_files_applies_include_exclude_and_executable(tmp_path):
    selected = tmp_path / "nbody_cuda"
    excluded = tmp_path / "nbody_cpp"
    ignored = tmp_path / "matrix_cuda"
    for path in (selected, excluded, ignored):
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)

    matches = find_files(
        [tmp_path],
        [r"nbody"],
        require_executable=True,
        exclude=[r"cpp"],
    )

    assert matches == [selected.resolve()]


def test_benchmark_cli_has_no_chart_options():
    parser = build_parser()
    args = parser.parse_args(["--regex", "nbody"])
    assert not hasattr(args, "chart")
    with pytest.raises(SystemExit):
        parser.parse_args(["--regex", "nbody", "--chart", "heatmap"])


def test_load_reports_normalizes_google_benchmark_json(tmp_path):
    report = tmp_path / "nbody.json"
    report.write_text(
        json.dumps(
            {
                "context": {"paradigm": "Cuda", "float_type": "64"},
                "benchmarks": [
                    {
                        "name": "NBody-Naive/1024",
                        "run_type": "iteration",
                        "real_time": 2.5,
                        "cpu_time": 3.0,
                        "time_unit": "ms",
                        "iterations": 10,
                    },
                    {
                        "name": "NBody-Naive/1024_mean",
                        "run_type": "aggregate",
                        "real_time": 2.5,
                    },
                ],
            }
        )
    )

    frame = load_reports([report], hardware="RTX5080")

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row[BENCHMARK_PROBLEM] == "NBody"
    assert row[DESCRIPTION] == "Naive"
    assert row[PARADIGM] == "Cuda"
    assert row[PRECISION] == "64"
    assert row[HARDWARE] == "RTX5080"
    assert row[PROBLEM_SIZE] == 1024
    assert row[WALL_CLOCK_TIME] == 2.5
