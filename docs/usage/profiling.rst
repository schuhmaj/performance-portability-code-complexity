.. _profiling:

Kernel Profiling & Roofline
===========================

``ppbcc profile`` runs a profiler over a set of benchmark executables, one
after another, consolidates the hardware counters into one tidy CSV, and draws a
roofline model from them. ``--profiler`` selects the backend:

============== ================================================================ ==========================
``--profiler`` Tool                                                             Row granularity
============== ================================================================ ==========================
``ncu``        `Nsight Compute <https://developer.nvidia.com/nsight-compute>`__  one kernel launch
``likwid``     `LIKWID <https://github.com/RRZE-HPC/likwid>`__ NvMarker          one marked region
``nsys``       `Nsight Systems <https://developer.nvidia.com/nsight-systems>`__ GPU metrics one marked region
``ngfx``       `Nsight Graphics <https://developer.nvidia.com/nsight-graphics>`__ GPU Trace one queue submission
============== ================================================================ ==========================

``ncu`` is the default and needs no instrumentation: it attaches from the
outside and replays every launch it sees. ``likwid`` reads the counters from
inside the application, around the regions marked in the benchmark source, which
keeps several kernels of one call together in a single row. All backends write
the same columns, so the CSV and the plot are identical in shape, and
``--from-csv`` merges the tables of several of them into one roofline.

.. note::

    Nsight Compute and LIKWID both read their counters through CUPTI, which
    needs a *current CUDA context*. Executables whose paradigm builds its own
    (``*_ocl``, ``*_boost``, ``*_vulkan``, ``*_slang_vulkan``) or that never
    reach the GPU (``*_cpp``) produce no report; they are reported as such and
    skipped. :mod:`ppbcc.profiling.nsys` is the backend for those.

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

=========================== ==================================
Benchmark problem           Input under PPB_PROFILING
=========================== ==================================
Vector addition             :math:`10^8` elements
Matrix multiplication       :math:`4096 \times 4096`
N-body simulation           :math:`10^5` particles
Polyhedral gravity model    ``SHAPE_SFM_3M_v20180804`` (3.1 M faces)
=========================== ==================================

It is the largest input except where that would defeat the profiler: one Nsight
Compute replay pass at :math:`16384^2` has to snapshot several GB around every
launch, and Eros' polyhedral kernels are only a few microseconds long — too
short for a counter- or sample-based tool to resolve.

``PPB_PROFILING=ON`` also switches on ``PPB_ENABLE_NVTX``, which names the
kernels (see `Regions`_ below), and compiles the targets with line tables so a
profiler can map an instruction back to a source line. The optimization flags
are untouched: a roofline measured from a differently optimized binary
describes a different program.

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

The LIKWID backend
------------------

LIKWID reads the counters through its NvMarker API, from regions that the
benchmark source opens itself. Build the benchmark with the markers compiled in:

.. code-block:: bash

    cmake --preset cuda-llvm-profiling -B build-likwid \
      -DPPB_ENABLE_LIKWID=ON -DLIKWID_ROOT="$LIKWID_PREFIX"
    cmake --build build-likwid -j

Then point ``ppbcc profile`` at it. LIKWID exposes no ``peak_sustained``
counters, so the two ceilings have to be supplied; they are a property of the
hardware, and an ``ncu`` run on the same machine measures them:

.. code-block:: bash

    ppbcc profile --profiler likwid -b build-likwid -p src -r "polyhedral_.*" \
      -O lib="$LIKWID_PREFIX/lib" -d ../profiling-likwid \
      --peak-performance 5.74e13 --peak-bandwidth 9.59e11 \
      -H "NVIDIA RTX5080" --roofline

Three properties of LIKWID 5.5.1 are worth knowing.

*The* ``likwid-perfctr`` *wrapper is bypassed.* It programs the counters from a
separate process, which fails on Blackwell with
``CUPTI_ERROR_INVALID_PARAMETER`` regardless of the CUDA version. ``ppbcc``
therefore sets ``LIKWID_NVMON_GPUS``, ``LIKWID_NVMON_EVENTS`` and
``LIKWID_NVMON_FILEPATH`` and runs the binary directly, so the instrumented
process programs its own counters.

*Every executable runs twice.* The SMSP counters (floating-point instructions)
and the DRAM counter cannot be programmed in the same pass, so the events are
split into the two groups of
:data:`~ppbcc.profiling.likwid.EVENT_GROUPS` and merged afterwards.

The two runs do not cost the same. Programming three SMSP counters perturbs a
kernel far more than a single DRAM counter -- on the 16384\ :sup:`2` matrix
multiplication the same region takes about 36\ s in the first group and 3.6\ s
in the second. The counters themselves are unaffected by that overhead, but the
region time is, so the reported duration is the **smallest** of the two, which is
the least perturbed measurement of the kernel. Run at ``-v`` to see the spread.

*The counters are corrected by a factor of two.* On this hardware LIKWID reports
every Nvidia counter at exactly half its true value. The factor was established
against kernels with an analytically known instruction count: for ``FADD``,
``FMUL``, ``FFMA`` and ``DRAM_BYTES_SUM`` alike, ``ncu`` reproduces the analytic
value exactly and LIKWID returns half of it, independently of the counter domain
and of how many launches a region contains.
:data:`~ppbcc.profiling.likwid.COUNTER_SCALE` compensates for it. Note that
arithmetic intensity is a ratio of two equally scaled counters and is therefore
unaffected either way; only the absolute rates need the correction.

.. warning::

    NvMarker needs a current **CUDA** context, so it sees the same paradigms
    Nsight Compute does. OpenCL and Vulkan build their own contexts and produce
    no marker file. The markers around those kernels are still compiled in, so
    the gap shows up as an explicit warning rather than silently.

The Nsight Systems backend
--------------------------

``--gpu-metrics-devices`` programs the GPU's performance monitors in *time-based
sampling* mode: device-wide, no context filter, no kernel boundaries. That is
what makes it the only backend here that sees OpenCL and Vulkan. What it gives
up is attribution — a sample knows *when*, not *which kernel* — and the
benchmark buys that back: ``--trace=nvtx`` records the marked regions with the
same clock as the samples, so a window is named by the region it falls in and
the stretches where compute warps are in flight inside it are the kernel.

.. code-block:: bash

    ppbcc profile --profiler nsys -b build-cuda-llvm-profiling -p src \
      -r "matMul_ocl$" "matMul_vulkan$" -d ../profiling-nsys \
      -O iterations=20 --peak-performance 5.74e13 --peak-bandwidth 9.592e11 \
      --analytic-flop "matMul_.*=137438953472" -H "NVIDIA RTX5080"

Three options carry the method.

``--analytic-flop``
    The metric sets available on consumer hardware carry pipe *utilisations*,
    not instruction counts, so the FLOP count has to come from the work model.
    It is not a guess where it matters: Nsight Compute counts exactly
    :math:`2 \cdot M \cdot N \cdot K` for every CUDA-backed matrix
    multiplication.
``-O iterations``
    A lazily initialising backend compiles pipelines and fills buffers on its
    first call, and the sampler cannot tell that from a kernel launch. Every
    binary is therefore profiled twice, at 1 and at *N* iterations; the
    difference over *N-1* iterations cancels the setup both runs paid.
``-O activity-ratio``
    Some runtimes move buffers with a compute shader rather than a copy engine,
    which shows up as compute-active windows that are not the kernel — and,
    because the marked region encloses everything the benchmark times, inside
    the same region. A window below this fraction of the busiest one's
    occupancy in that region is dropped.

``--peak-bandwidth`` is required: the sampler reports DRAM traffic as a
percentage of peak, so the peak is what turns it into bytes. Both ceilings are
properties of the hardware, and an ``ncu`` run on the same machine measures them.

The Nsight Graphics backend
---------------------------

GPU Trace reads the same performance monitors as Nsight Compute — as counter
sums rather than samples — and it is the one Nvidia tool that does so for a
Vulkan workload, which makes it the independent check on the sampled numbers.
Its unit of attribution is the *trace*: with no swapchain there are no frames,
and the per-regime table stays empty. The region is bounded by submit index
instead, and ``-O submit=auto`` traces the first ``-O probes`` submissions and
keeps the one with the most compute cycles. A Vulkan submission carries no NVTX
range, so this is the one backend whose rows are unnamed.

.. warning::

    ``auto`` picks the busiest submission, which is the wrong one whenever the
    first call also does the lazy initialisation — it is then longer than the
    steady-state dispatch. Pass ``-O submit=<index>`` explicitly in that case.

Regions
-------

A profiler names a kernel whatever the compiler called it, which for several
paradigms is nothing useful: AdaptiveCpp launches four kernels all called
``__acpp_sscp_kernel``, Kokkos eleven called
``Kokkos::cuda_parallel_launch_local_memory``. The benchmark therefore names the
work itself. ``src/common/Marker.h`` brackets each phase with an NVTX range —
``matmul`` for the matrix multiplication, ``init`` and ``evaluate`` for the
polyhedral gravity — and every backend reports which range a row belongs to in
the ``Region`` column. ``-DPPB_PROFILING=ON`` turns that on.

Everything outside those ranges is the runtime setting itself up: Kokkos'
architecture query, desul's lock arrays, a framework's buffer staging. Those
launches say nothing about the algorithm, so they are dropped before the CSV is
written; ``--all-kernels`` keeps them. An executable with no range at all is
kept unchanged, with a warning, since that means the binary was built without
``PPB_ENABLE_NVTX`` rather than that its kernels are uninteresting.

``--region`` restricts the *plot* to some of them; the CSV always keeps every
region:

.. code-block:: bash

    # The polyhedral kernel without its one-off initialization.
    ppbcc profile ... --roofline --region evaluate

Aggregation
-----------

One region may launch several kernels. ``--aggregate`` decides what a point is:

``sum`` (default)
    One point per implementation and region: work, traffic and kernel time of
    every launch in it are added up.
``dominant``
    One point per implementation and region, its longest-running kernel only.
``none``
    One point per kernel launch.

Output
------

The CSV holds one row per profiled kernel launch: the identifying columns
(problem, paradigm, precision, hardware, executable, region, kernel, grid/block
size),
the derived roofline quantities (duration, FLOP, traffic, arithmetic intensity,
performance, and both ceilings), and behind them every raw ``ncu`` metric.

The roofline chart uses the same paradigm colors as :doc:`p3analysis`, so a
point can be matched to its paradigm across both figures.
