"""Batch kernel profiling with Nsight Compute and roofline analysis."""

from .reports import aggregate_kernels, filter_kernels, load_reports
from .runner import find_ncu, run_profiles

__all__ = [
    "aggregate_kernels",
    "filter_kernels",
    "find_ncu",
    "load_reports",
    "run_profiles",
]
