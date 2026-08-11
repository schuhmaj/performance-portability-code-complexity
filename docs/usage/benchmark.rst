.. _benchmark:

Benchmarking
============

``ppbcc benchmark`` discovers `Google Benchmark <https://github.com/google/benchmark>`__
executables in a build folder, runs them, and consolidates their JSON reports
into a single tidy CSV that :doc:`p3analysis` consumes.

.. note::

    This workflow expects the build tree of the
    `performance-portability-benchmark <https://github.com/schuhmaj/performance-portability-benchmark>`__
    repository, or any other project whose targets emit Google Benchmark JSON.

Running benchmarks
------------------

Always start with ``--dry-run`` to check which executables the regexes match:

.. code-block:: bash

    cd <build>
    ppbcc benchmark -p src -H "INTEL Data Center GPU Max 1550" \
      -r "vec_.*" "matMul_.*" "nbody_.*" "polyhedral_.*" -x ".*_cpp" --dry-run

Drop ``--dry-run`` to actually run them. Each executable writes a JSON report
next to itself; existing reports are reused unless ``--force`` is given.

Consolidating existing reports
------------------------------

With ``--skip-benchmark`` nothing is executed — ``--path``/``--regex`` then
select the JSON reports directly. This is how the result CSVs in the benchmark
repository's ``results/`` folder are produced, one per hardware platform:

.. code-block:: bash

    ppbcc benchmark -p ./nvidia-rtx5080 -r ".*\.json" --skip-benchmark \
      -H "NVIDIA RTX5080" -o "Results_NVIDIA_RTX5080"
    ppbcc benchmark -p ./nvidia-gh200 -r ".*\.json" --skip-benchmark \
      -H "NVIDIA GH200" -o "Results_NVIDIA_GH200"
    ppbcc benchmark -p ./amd-mi210 -r ".*\.json" --skip-benchmark \
      -H "AMD Instinct MI210" -o "Results_AMD_Instinct_MI210"

The ``-H/--hardware`` value ends up in the ``Hardware`` column and is what
``ppbcc p3analysis`` treats as a *platform* when computing performance
portability — so keep it consistent across runs.

Output format
-------------

The consolidated CSV contains one row per benchmark observation:

+-----------------------------+-------------------------------------------------------+
| Column                      | Meaning                                               |
+=============================+=======================================================+
| ``Benchmark Problem``       | Problem name, e.g. ``MatrixMultiplication``           |
+-----------------------------+-------------------------------------------------------+
| ``Paradigm``                | Implementation label, e.g. ``Kokkos``                 |
+-----------------------------+-------------------------------------------------------+
| ``Description``             | Variant detail, e.g. ``SharedMemory``                 |
+-----------------------------+-------------------------------------------------------+
| ``Precision``               | Floating-point precision (32 or 64)                   |
+-----------------------------+-------------------------------------------------------+
| ``Hardware``                | Platform identifier from ``-H``                       |
+-----------------------------+-------------------------------------------------------+
| ``Problem Size``            | Workload size                                         |
+-----------------------------+-------------------------------------------------------+
| ``Iterations``              | Google Benchmark repetition count                     |
+-----------------------------+-------------------------------------------------------+
| ``Wall Clock Time``,        | Measured timings; ``Time Unit`` records their unit    |
| ``CPU Time``,               | and the loader normalises them to a common base       |
| ``Kernel Time``, ...        |                                                       |
+-----------------------------+-------------------------------------------------------+

A concrete row:

.. code-block:: text

    Benchmark Problem,Paradigm,Description,Precision,Hardware,Problem Size,Iterations,Wall Clock Time,...,Time Unit
    MatrixMultiplication,OpenACC,,32,NVIDIA RTX5080,32.0,31702.0,21740.16,...,ns

Python API
----------

.. code-block:: python

    from pathlib import Path
    from ppbcc.benchmark import find_files, load_reports, run_benchmarks

    executables = find_files(
        [Path("build/src")], [r"nbody_.*"], require_executable=True, exclude=[r".*_cpp"]
    )
    reports = run_benchmarks(executables)
    frame = load_reports(reports, hardware="NVIDIA RTX5080")

See :doc:`../api/benchmark` for the full signatures.
