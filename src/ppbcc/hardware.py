"""Published peak floating-point performance of the benchmarked platforms.

Every value is the *datasheet* peak in FLOP/s: execution units times the
FMA operations per unit times the (boost) core clock, as the sources below
tabulate it. Where a source gives a base and a boost clock value, the boost
value is used, since it is the peak a card reaches under sustained load.
Sources were accessed on 2026-09-16.

These are not measured ceilings. ``ppbcc profile`` scales Nsight Compute's
``peak_sustained`` counters with the clock a card actually ran at, which for
the RTX 5080 of the study gives 57.4 TFLOP/s against the tabulated 56.3.

Caveats worth knowing before comparing platforms:

* The GeForce FP64 figures of the sources are one to two orders of magnitude
  below FP32. They are reproduced as tabulated, without explanation there.
* The source labels the Instinct MI210 column *Vector TFLOPS*, which is not
  necessarily the same unit of work as the NVIDIA per-core FMA figure, so its
  value may not be directly comparable with the NVIDIA ones.
* The Intel Max 1550 has two Xe-HPC stacks, and SYCL/Level Zero exposes each
  as its own device. The benchmark runs on one stack, so the published
  whole-card figure is halved.
"""

from __future__ import annotations

import re

#: Peak performance in FLOP/s, keyed by the ``Hardware`` label of the benchmark
#: CSVs and then by floating-point precision in bits.
PEAK_PERFORMANCE: dict[str, dict[int, float]] = {
    # GeForce RTX 3080 (10 GB), boost clock.
    # https://en.wikipedia.org/wiki/GeForce_RTX_30_series
    "NVIDIA RTX3080": {32: 29.77e12, 64: 0.465e12},
    # GeForce RTX 4060, boost clock.
    # https://en.wikipedia.org/wiki/GeForce_RTX_40_series
    "NVIDIA RTX4060": {32: 15.11e12, 64: 0.236e12},
    # GeForce RTX 5080, computed from the boost core clock.
    # https://en.wikipedia.org/wiki/GeForce_RTX_50_series
    "NVIDIA RTX5080": {32: 56.3e12, 64: 0.88e12},
    # GH200 Grace Hopper: its GPU is the H100 SXM.
    # https://en.wikipedia.org/wiki/Nvidia_Tesla
    "NVIDIA GH200": {32: 66.9e12, 64: 33.5e12},
    # AMD Instinct MI210 (Aldebaran), boost clock, vector TFLOPS.
    # https://en.wikipedia.org/wiki/AMD_Instinct
    "AMD MI210": {32: 181.0e12, 64: 22.63e12},
    # Intel Data Center GPU Max 1550: 52 TFLOP/s FP32 and FP64 for the card,
    # but two stacks --> halved
    # https://flopper.io/gpu/intel-data-center-gpu-max-1550-128gb
    "Intel Max 1550": {32: 26.0e12, 64: 26.0e12},
}


def _token(label: str) -> str:
    """Reduce a hardware label to lowercase alphanumerics for matching."""
    return re.sub(r"[^a-z0-9]+", "", label.casefold())


def peak_performance(hardware: str, precision: int) -> float:
    """Return the published peak performance of one platform.

    Args:
        hardware: ``Hardware`` label as it appears in the benchmark CSVs.
            Matching ignores case, spaces and punctuation.
        precision: Floating-point precision in bits (32 or 64).

    Returns:
        Peak performance in FLOP/s.

    Raises:
        ValueError: If the platform or the precision is not tabulated.
    """
    matches = [
        values
        for label, values in PEAK_PERFORMANCE.items()
        if _token(label) == _token(hardware)
    ]
    if not matches:
        raise ValueError(
            f"No peak performance is tabulated for hardware {hardware!r}. "
            f"Known platforms: {sorted(PEAK_PERFORMANCE)}"
        )
    values = matches[0]
    if precision not in values:
        raise ValueError(
            f"No FP{precision} peak performance is tabulated for {hardware!r}."
        )
    return values[precision]
