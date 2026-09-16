"""Nsight Compute metric set and the roofline quantities derived from it.

The metric names follow Nvidia's own roofline recipe: floating-point work is
counted from the executed SASS instructions (an FMA counts twice), memory
traffic is read at a selectable level of the hierarchy, and the two ceilings are
*measured* rather than looked up in a datasheet — every ``.peak_sustained``
metric is a per-cycle rate that becomes an absolute rate once multiplied with
the clock frequency that unit actually ran at during the kernel.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Metrics requested from ncu
# --------------------------------------------------------------------------- #

#: Wall-clock duration and the clock rates the ceilings are scaled with.
TIMING_METRICS = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.avg.per_second",
]

#: Executed floating-point SASS instructions, per precision and operation.
FLOP_METRICS = [
    f"sm__sass_thread_inst_executed_op_{precision}{operation}_pred_on.sum"
    for precision in ("f", "d", "h")
    for operation in ("add", "mul", "fma")
]

#: Tensor-core instructions; reported for context, never counted as FLOPs.
TENSOR_METRICS = ["sm__inst_executed_pipe_tensor.sum"]

#: Bytes moved and the achievable bytes/cycle, per level of the hierarchy.
MEMORY_METRICS = [
    "dram__bytes.sum",
    "dram__bytes.sum.peak_sustained",
    "dram__cycles_elapsed.avg.per_second",
    "lts__t_bytes.sum",
    "lts__t_bytes.sum.peak_sustained",
    "lts__cycles_elapsed.avg.per_second",
    "l1tex__t_bytes.sum",
    "l1tex__t_bytes.sum.peak_sustained",
    "l1tex__cycles_elapsed.avg.per_second",
]

#: Achievable floating-point instructions per cycle (the compute ceilings).
PEAK_METRICS = [
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained",
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained",
]

#: Launch configuration, useful to explain a point that sits far below the roof.
LAUNCH_METRICS = [
    "launch__grid_size",
    "launch__block_size",
    "launch__waves_per_multiprocessor",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
]

#: Every metric collected by ``ppbcc profile``.
ROOFLINE_METRICS = (
    TIMING_METRICS
    + FLOP_METRICS
    + TENSOR_METRICS
    + MEMORY_METRICS
    + PEAK_METRICS
    + LAUNCH_METRICS
)

# --------------------------------------------------------------------------- #
# Precisions and memory levels
# --------------------------------------------------------------------------- #

#: Precision key -> (ncu instruction infix, human-readable label).
PRECISIONS = {
    "fp32": ("f", "FP32"),
    "fp64": ("d", "FP64"),
    "fp16": ("h", "FP16"),
}

#: Memory level key -> (ncu counter base, human-readable label).
MEMORY_LEVELS = {
    "dram": ("dram__bytes", "DRAM"),
    "l2": ("lts__t_bytes", "L2"),
    "l1": ("l1tex__t_bytes", "L1/TEX"),
}

#: ``dram__bytes`` is clocked by ``dram__cycles_elapsed``, the others by the
#: unit that shares their name prefix.
MEMORY_LEVEL_CLOCK = {
    "dram": "dram__cycles_elapsed.avg.per_second",
    "l2": "lts__cycles_elapsed.avg.per_second",
    "l1": "l1tex__cycles_elapsed.avg.per_second",
}


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #

#: Units of the derived roofline columns, as written into the CSV header.
DERIVED_UNITS = {
    "Grid Size": "blocks",
    "Block Size": "threads/block",
    "Duration": "s",
    "FLOP": "FLOP",
    "FLOP FP32": "FLOP",
    "FLOP FP64": "FLOP",
    "FLOP FP16": "FLOP",
    "Memory Traffic": "Byte",
    "Arithmetic Intensity": "FLOP/Byte",
    "Performance": "FLOP/s",
    "Peak Performance": "FLOP/s",
    "Peak Bandwidth": "Byte/s",
}


def column_unit(column: str) -> str | None:
    """Return the unit of a profiling column.

    Raw metrics follow Nsight Compute's naming scheme, from which their unit
    follows (checked against ``IMetric.unit()`` of the reports): ``.sum`` of
    bytes or instructions is a count, ``.peak_sustained`` the same per clock
    cycle, ``.per_second`` a clock rate, ``pct_of_*`` a percentage.

    Args:
        column: Column name without a unit.

    Returns:
        The unit, or ``None`` for identifying and textual columns.
    """
    if column in DERIVED_UNITS:
        return DERIVED_UNITS[column]
    if "__" not in column:
        return None
    if column == "gpu__time_duration.sum":
        return "ns"
    if "pct_of_peak" in column:
        return "%"
    if column.endswith(".per_second"):
        return "cycle/s"
    if column.startswith("launch__"):
        return {"launch__grid_size": "blocks", "launch__block_size": "threads/block"}.get(
            column, "1"
        )
    quantity = "Byte" if "bytes" in column else "inst" if "inst" in column else "cycle"
    return f"{quantity}/cycle" if column.endswith(".peak_sustained") else quantity


def flop_columns(precision: str) -> tuple[str, str, str]:
    """Return the add/mul/fma metric names of one precision.

    Args:
        precision: Key of :data:`PRECISIONS` (``fp32``, ``fp64`` or ``fp16``).

    Returns:
        The ``add``, ``mul`` and ``fma`` metric names, in that order.
    """
    infix = PRECISIONS[precision][0]
    return tuple(  # type: ignore[return-value]
        f"sm__sass_thread_inst_executed_op_{infix}{operation}_pred_on.sum"
        for operation in ("add", "mul", "fma")
    )


def peak_flop_column(precision: str) -> str:
    """Return the ``peak_sustained`` FMA metric name of one precision.

    Args:
        precision: Key of :data:`PRECISIONS`.

    Returns:
        The metric name holding the achievable FMA instructions per cycle.
    """
    infix = PRECISIONS[precision][0]
    return (
        f"sm__sass_thread_inst_executed_op_{infix}fma_pred_on.sum.peak_sustained"
    )
