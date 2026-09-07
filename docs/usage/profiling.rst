.. _profiling:

Kernel Profiling & Roofline
===========================

``ppbcc profile`` runs `Nvidia Nsight Compute <https://developer.nvidia.com/nsight-compute>`__
(``ncu``) over a set of benchmark executables, one after another, consolidates
the per-kernel hardware counters into one tidy CSV, and draws a roofline model
from them.

.. note::

    Nsight Compute only sees **CUDA** kernels. Executables whose paradigm runs
    on OpenCL, Vulkan or the host (``*_ocl``, ``*_boost``, ``*_vulkan``,
    ``*_slang_vulkan``, ``*_cpp``) produce no report; they are reported as such
    and skipped.

Building the executables for profiling
--------------------------------------

A profiler replays every kernel launch it sees, so a benchmark that sweeps a
range of problem sizes and repeats each of them takes hours under ``ncu`` and
produces thousands of near-identical kernels. The
`performance-portability-benchmark <https://github.com/schuhmaj/performance-portability-benchmark>`__
repository therefore has a dedicated build switch:

.. code-block:: bash

    cmake --preset cuda-llvm-profiling      # = cuda-llvm plus -DPPB_PROFILING=ON
    cmake --build build-cuda-llvm-profiling -j

``PPB_PROFILING=ON`` reduces every executable to a **single input** running a
**single iteration and repetition**, so profiling one binary takes about a
second and every implementation is profiled on the same input:

=========================== =========================
Benchmark problem           Input under PPB_PROFILING
=========================== =========================
Vector addition             :math:`10^8` elements
Matrix multiplication       :math:`16384 \times 16384`
N-body simulation           :math:`10^5` particles
Polyhedral gravity model    the Eros mesh
=========================== =========================

Profiling
---------

As with :doc:`benchmark`, start with ``--dry-run``:

.. code-block:: bash

    ppbcc profile -b build-cuda-llvm-profiling -p src -r "polyhedral_.*" --dry-run

Drop it to profile. Each run writes ``<report-dir>/<executable>.ncu-rep`` next
to the Google-Benchmark report ``<report-dir>/<executable>.json`` — the profiler
does not know which paradigm a kernel was written in, so the JSON report
supplies the paradigm and the precision:

.. code-block:: bash

    ppbcc profile -b build-cuda-llvm-profiling -p src -r "polyhedral_.*" \
      -d profiling -H "NVIDIA RTX5080" -o Profiling_NVIDIA_RTX5080 --roofline

Existing reports are reused unless ``--force`` is given, and ``--skip-profile``
runs nothing at all and only re-parses the reports already in ``--report-dir``.
That splits nicely into a collection pass per toolchain (``--no-csv``, writing
into a shared ``--report-dir``) and one consolidation pass over all of them.

What is measured
----------------

The collected metric set follows Nvidia's own roofline recipe:

* **Work** is counted from executed SASS instructions, per precision:
  :math:`\mathrm{FLOP} = \mathrm{add} + \mathrm{mul} + 2 \cdot \mathrm{fma}`.
  All three precisions are always reported (``FLOP FP32``, ``FLOP FP64``,
  ``FLOP FP16``); ``--precision`` selects which one the roofline uses, with
  ``auto`` following the precision the binary was built with.
* **Traffic** is read at a selectable level of the hierarchy — ``--memory-level``
  picks ``dram`` (default), ``l2`` or ``l1``.
* **Both ceilings are measured, not looked up.** Every ``.peak_sustained``
  metric is a per-cycle rate; multiplied with the clock the corresponding unit
  actually ran at during the kernel it becomes an absolute rate. The highest
  value observed across all profiled kernels becomes the roof.

Aggregation
-----------

One run launches several kernels. ``--aggregate`` decides what a point is:

``sum`` (default)
    One point per implementation: work, traffic and kernel time of every launch
    are added up.
``dominant``
    One point per implementation, its longest-running kernel only.
``none``
    One point per kernel launch.

Framework bootstrap kernels — Kokkos' architecture query, desul's lock-array
initialization — are launched once by the runtime and have nothing to do with
the algorithm, so they would distort a ``sum``. ``--exclude-kernel`` drops them
from the plot (the CSV always keeps every launch):

.. code-block:: bash

    ppbcc profile ... --roofline \
      -k "init_lock_arrays" -k "query_cuda_kernel_arch"

Output
------

The CSV holds one row per profiled kernel launch: the identifying columns
(problem, paradigm, precision, hardware, executable, kernel, grid/block size),
the derived roofline quantities (duration, FLOP, traffic, arithmetic intensity,
performance, and both ceilings), and behind them every raw ``ncu`` metric.

The roofline chart uses the same paradigm colors as :doc:`p3analysis`, so a
point can be matched to its paradigm across both figures.
