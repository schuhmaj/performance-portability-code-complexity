"""Batch kernel profiling and roofline analysis.

Four backends produce the same tidy table, differing only in what they can see:
:mod:`ppbcc.profiling.runner` drives Nsight Compute from the outside,
:mod:`ppbcc.profiling.nsys` samples the GPU's performance monitors device-wide
(the only one that sees OpenCL and Vulkan), :mod:`ppbcc.profiling.ngfx` traces a
Vulkan queue submission, and :mod:`ppbcc.profiling.likwid` reads the counters
inside regions marked in the benchmark source.
"""

from . import likwid, ngfx, nsys
from .reports import (
    aggregate_kernels,
    filter_regions,
    load_csv,
    load_reports,
    select_regions,
)
from .runner import find_ncu, run_profiles

__all__ = [
    "aggregate_kernels",
    "filter_regions",
    "find_ncu",
    "likwid",
    "load_csv",
    "load_reports",
    "ngfx",
    "nsys",
    "run_profiles",
    "select_regions",
]
