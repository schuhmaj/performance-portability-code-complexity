"""Benchmark discovery, execution, and report consolidation."""

from .reports import load_reports
from .runner import find_files, run_benchmarks

__all__ = ["find_files", "load_reports", "run_benchmarks"]
