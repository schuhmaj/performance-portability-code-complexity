"""Shared column names and runtime-unit conversions."""

BENCHMARK_PROBLEM = "Benchmark Problem"
PARADIGM = "Paradigm"
DESCRIPTION = "Description"
PRECISION = "Precision"
HARDWARE = "Hardware"
PROBLEM_SIZE = "Problem Size"
ITERATIONS = "Iterations"
WALL_CLOCK_TIME = "Wall Clock Time"
CPU_TIME = "CPU Time"
KERNEL_TIME = "Kernel Time"
NEIGHBOR_SEARCH_TIME = "Neighbor Search Time"
POSITION_UPDATE_TIME = "Position Update Time"
VELOCITY_UPDATE_TIME = "Velocity Update Time"
FORCE_UPDATE_TIME = "Force Update Time"
TIME_UNIT = "Time Unit"

APPLICATION = "Application"
APPLICATION_EFFICIENCY = "Application Efficiency"
PERFORMANCE_PORTABILITY = "Performance Portability"
PROBLEM = "Problem"

RUNTIME_COLUMNS = [
    WALL_CLOCK_TIME,
    CPU_TIME,
    KERNEL_TIME,
    NEIGHBOR_SEARCH_TIME,
    POSITION_UPDATE_TIME,
    VELOCITY_UPDATE_TIME,
    FORCE_UPDATE_TIME,
]

TIME_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

COLUMN_LIST = [
    BENCHMARK_PROBLEM,
    PARADIGM,
    DESCRIPTION,
    PRECISION,
    HARDWARE,
    PROBLEM_SIZE,
    ITERATIONS,
    WALL_CLOCK_TIME,
    CPU_TIME,
    KERNEL_TIME,
    NEIGHBOR_SEARCH_TIME,
    POSITION_UPDATE_TIME,
    VELOCITY_UPDATE_TIME,
    FORCE_UPDATE_TIME,
    TIME_UNIT,
]

RAW_KEY_TO_COLUMN = {
    "real_time": WALL_CLOCK_TIME,
    "cpu_time": CPU_TIME,
    "time_unit": TIME_UNIT,
    "iterations": ITERATIONS,
    "kernel_time": KERNEL_TIME,
    "neighbor_search": NEIGHBOR_SEARCH_TIME,
    "position_update_reset": POSITION_UPDATE_TIME,
    "velocity_update": VELOCITY_UPDATE_TIME,
    "force_update": FORCE_UPDATE_TIME,
}

__all__ = [name for name in globals() if name.isupper()]
