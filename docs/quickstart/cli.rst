.. _cli:

Command Line Interface
======================

Everything is reachable through the single ``ppbcc`` executable:

.. code-block:: bash

    ppbcc <command> [options]

    ppbcc code-complexity SOURCES...   # Halstead / LOC metrics
    ppbcc benchmark -r REGEX...        # run and consolidate benchmarks
    ppbcc profile -r REGEX...          # profile kernels with ncu, roofline model
    ppbcc p3analysis NAME CSV...       # efficiency, portability, plots

Each command is also installed as a prefixed executable
(``ppbcc-code-complexity``, ``ppbcc-benchmark``, ``ppbcc-profile``,
``ppbcc-p3analysis``), and the
package can be run as a module:

.. code-block:: bash

    python -m ppbcc code-complexity src/ -o report.csv

.. tip::

    The tables below list the options you reach for most often, not every
    option. Run ``ppbcc <command> --help`` (or ``-h``) for the exhaustive,
    always-current list.

``ppbcc code-complexity``
-------------------------

Halstead complexity and LOC/SLOC metrics for C++ and GPU-enriched C++.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``sources``
     - Files and/or directories; directories are searched recursively for known source extensions
   * - ``-d, --dialect``
     - ``auto`` (default, per-file detection), ``cpp`` (baseline only), or names like ``kokkos,openmp``
   * - ``-m, --metrics``
     - Metric selection, e.g. ``halstead``, ``loc``, ``sloc``, ``dialect`` (default: all)
   * - ``--diff``
     - Add ``baseline_*``/``delta_*`` columns: metrics with all dialect tokens removed, and the delta
   * - ``--aggregate``
     - Append a ``TOTAL`` row; operator/operand sets are merged before recomputing
   * - ``-o, --output``
     - Write the table to this CSV file
   * - ``--list-dialects``
     - List all known dialects and their aliases, then exit

Details and examples: :doc:`../usage/code_complexity`.

``ppbcc benchmark``
-------------------

Run Google Benchmark targets and consolidate their JSON reports into one CSV.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``-b, --build-dir``
     - Build folder used as the working directory for the whole pipeline
   * - ``-p, --path``
     - Directories searched recursively, relative to ``--build-dir``
   * - ``-r, --regex``
     - **Required.** Pattern(s) matched against file paths to select executables (or JSON reports)
   * - ``-x, --exclude``
     - Pattern(s) excluding matched paths, e.g. ``'.*_cpp'``
   * - ``-s, --skip-benchmark``
     - Do not run anything; locate existing JSON reports and only consolidate them
   * - ``-n, --dry-run``
     - List the matched executables (or reports) and exit
   * - ``-H, --hardware``
     - Value stored in the ``Hardware`` column, e.g. ``"NVIDIA RTX5080"``
   * - ``-o, --output``
     - Base name of the consolidated CSV (defaults to a timestamped name)

Details and examples: :doc:`../usage/benchmark`.

``ppbcc profile``
-----------------

Batch-profile CUDA kernels with Nsight Compute and draw a roofline model.
Expects executables built with ``-DPPB_PROFILING=ON``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``-b, --build-dir``
     - Build folder used as the working directory for the whole pipeline
   * - ``-p, --path`` / ``-r, --regex`` / ``-x, --exclude``
     - Executable discovery, exactly as for ``ppbcc benchmark`` (``--regex`` is required)
   * - ``-d, --report-dir``
     - Where the ``<executable>.ncu-rep`` reports are written (default: ``profiling``)
   * - ``-s, --skip-profile``
     - Run nothing; only re-parse the reports already in ``--report-dir``
   * - ``-m, --memory-level``
     - Level the arithmetic intensity refers to: ``dram`` (default), ``l2``, ``l1``
   * - ``--precision``
     - ``auto`` (default, follows the build), ``fp32``, ``fp64``, ``fp16``
   * - ``-a, --aggregate``
     - ``sum`` (default), ``dominant``, or ``none`` — how kernels become plot points
   * - ``-k, --exclude-kernel``
     - Drop kernels matching a pattern from the plot, e.g. framework bootstrap kernels
   * - ``--roofline`` / ``--roofline-output``
     - Render the roofline chart, optionally to a given path
   * - ``--no-csv``
     - Skip the consolidated CSV (for a pure collection run)
   * - ``-H, --hardware`` / ``-o, --output``
     - Hardware label and base name of the consolidated CSV

Details and examples: :doc:`../usage/profiling`.

``ppbcc p3analysis``
--------------------

Application efficiency, performance portability, and plots from benchmark CSVs.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``NAME``
     - Benchmark problem to plot; exact case-insensitive match preferred, unique substring okay
   * - ``CSV``
     - One or more CSVs produced by ``ppbcc benchmark``
   * - ``-c, --chart``
     - ``cascade`` (default), ``navchart``, ``combined``, ``heatmap``, ``boxplot``
   * - ``--complexity``
     - Code-complexity CSV; **required** for ``navchart`` and ``combined``
   * - ``--complexity-metric``
     - SLOC, Halstead vocabulary/length/volume/difficulty/effort (default: ``halstead-effort``)
   * - ``--normalize`` / ``--additive``
     - Divide by, or subtract, the plain-C++ complexity score (mutually exclusive)
   * - ``-s, --size``
     - ``all`` (default), an exact size, or ``avg``/``best``/``worst``
   * - ``--non-zero-pp``
     - Compute :math:`\Phi` over supported platforms only
   * - ``-x, --exclude`` / ``-i, --include``
     - Regex filters on the ``Description`` column
   * - ``--remove-description``
     - Drop bracketed labels such as ``[Naive]``; efficiency charts combine variants by paradigm
   * - ``-l, --legend``
     - Save the legend as a separate PDF (add ``--legend--vertical`` for a single column)
   * - ``-e, --export-to-csv``
     - Also export efficiency and portability data as CSV
   * - ``-o, --output``
     - Output path; defaults to ``<problem>_<chart>.pdf``

Details and examples: :doc:`../usage/p3analysis` and :doc:`../usage/plots`.

Option compatibility
~~~~~~~~~~~~~~~~~~~~

Not every option applies to every chart, and the CLI rejects invalid
combinations rather than silently ignoring them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Valid for
   * - ``--complexity``
     - ``navchart``, ``combined`` (warned and ignored elsewhere)
   * - ``--normalize``/``--additive``
     - ``navchart``, ``combined``, together with ``--complexity``
   * - ``--log-complexity``
     - ``navchart``, ``combined``
   * - ``--log-size``
     - ``combined`` only
   * - ``-H, --hardware``
     - ``boxplot`` only
   * - ``-s avg``/``best``/``worst``
     - every chart except ``boxplot`` (needs ``all`` or a numeric size)

Logging
-------

All three commands log through ``loguru`` at ``INFO`` by default. ``-v`` raises
the level to ``DEBUG``, ``-vv`` to ``TRACE``.
