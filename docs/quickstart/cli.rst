.. _cli:

Command Line Interface
======================

Everything is reachable through the single ``ppbcc`` executable:

.. code-block:: bash

    ppbcc <command> [options]

    ppbcc code-complexity SOURCES...   # Halstead / LOC metrics
    ppbcc benchmark -r REGEX...        # run and consolidate benchmarks
    ppbcc profile -r REGEX...          # profile kernels (ncu or likwid), roofline model
    ppbcc p2analysis PLOT CSV...       # efficiency and portability plots
    ppbcc p3analysis PLOT COMPLEXITY_CSV CSV...  # portability over code complexity

The package can also be run as a module:

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

Batch-profile GPU kernels and draw a roofline model. Expects executables built
with ``-DPPB_PROFILING=ON``. ``--profiler`` picks the backend: ``ncu`` (default,
one row per kernel launch) or ``likwid`` (one row per marked region, needs
``-DPPB_ENABLE_LIKWID=ON`` too).

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
     - Where the profiler artefacts are written (default: ``profiling``)
   * - ``--profiler``
     - ``ncu`` (default), ``nsys``, ``ngfx`` or ``likwid``
   * - ``--profiler-path``
     - Path to that backend's CLI; found automatically otherwise
   * - ``-O, --option``
     - ``NAME=VALUE`` setting of the selected backend; ``--help`` lists the names
   * - ``--from-csv``
     - Skip profiling and plot from consolidated CSVs, merging several backends
   * - ``-s, --skip-profile``
     - Run nothing; only re-parse the artefacts already in ``--report-dir``
   * - ``--timeout``
     - Wall-clock limit per executable; a run that hits it is skipped
   * - ``-m, --memory-level``
     - Level the arithmetic intensity refers to: ``dram`` (default), ``l2``, ``l1``
   * - ``--precision``
     - ``auto`` (default, follows the build), ``fp32``, ``fp64``, ``fp16``
   * - ``--region``
     - Plot only the regions matching a pattern, e.g. ``--region evaluate``
   * - ``--all-kernels``
     - Keep the launches outside every named region (dropped by default)
   * - ``-a, --aggregate``
     - ``sum`` (default), ``dominant``, or ``none`` — how launches become plot points
   * - ``--roofline [PATH]``
     - Render the roofline chart, optionally to a given path
   * - ``-l, --no-legend``
     - Leave the paradigm legend out of the roofline chart
   * - ``--no-csv``
     - Skip the consolidated CSV (for a pure collection run)
   * - ``--peak-performance`` / ``--peak-bandwidth``
     - Roofline ceilings for the backends without ``peak_sustained`` counters
   * - ``--analytic-flop``
     - ``REGEX=FLOP`` work model for ``nsys``/``ngfx``
   * - ``-H, --hardware`` / ``-o, --output``
     - Hardware label and base name of the consolidated CSV

Details and examples: :doc:`../usage/profiling`.

``ppbcc p2analysis`` / ``ppbcc p3analysis``
-------------------------------------------

Application efficiency, performance portability, and plots from benchmark CSVs.
``p2analysis`` renders the charts that need benchmark results only,
``p3analysis`` those that also need code complexity.

.. code-block:: bash

    ppbcc p2analysis PLOT CSV [CSV ...] [options]
    ppbcc p3analysis PLOT COMPLEXITY_CSV CSV [CSV ...] [options]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``PLOT``
     - ``p2analysis``: ``cascade``, ``heatmap``, ``double-heatmap``, ``boxplot``, ``time-barplot``;
       ``p3analysis``: ``navchart``, ``combined``, ``complexity-comparison``,
       ``rank-correlation``
   * - ``COMPLEXITY_CSV``
     - ``p3analysis`` only: code-complexity CSV with one row per implementation
   * - ``CSV``
     - One or more CSVs produced by ``ppbcc benchmark``
   * - ``-n, --name``
     - Benchmark problem to plot; exact case-insensitive match preferred, unique substring okay.
       May be omitted when the CSVs contain exactly one problem. ``rank-correlation``
       takes a comma-separated list and defaults to every problem in the CSVs
   * - ``-c, --complexity-metric``
     - ``p3analysis`` only: SLOC, Halstead vocabulary/length/volume/difficulty/effort
       (default: ``halstead-difficulty``)
   * - ``--complexity-metric-absolute``
     - ``p3analysis`` only: plot absolute values instead of a percentage of the plain-C++ score
   * - ``--compare-metric``
     - x-axis metric of ``complexity-comparison`` (default: ``sloc``); ``-c`` is its y axis
   * - ``--log-complexity``
     - ``p3analysis`` only: logarithmic complexity axes
   * - ``--correlation``
     - ``rank-correlation`` only: the variable whose paradigm ordering is
       correlated — ``pp`` (default) or a complexity metric such as
       ``halstead-difficulty`` or ``sloc``
   * - ``-s, --size``
     - ``all`` (default), an exact size, or ``avg``/``best``/``worst``
   * - ``--average-over``
     - With ``-s avg``: average :math:`\Phi` over sizes (``pp``, default) or compute :math:`\Phi` from size-averaged efficiencies (``efficiency``)
   * - ``--non-zero-pp``
     - Compute :math:`\Phi` over supported platforms only
   * - ``-x, --exclude`` / ``-i, --include``
     - Regex filters on the ``Description`` column
   * - ``-H, --hardware``
     - ``p2analysis`` only: restrict a ``boxplot`` or ``time-barplot`` to one platform
   * - ``-t, --time``
     - ``time-barplot`` only: ``wall-clock`` (default), ``kernel``, ``force-update``, ``neighbor-search``
   * - ``--normalize-time-to-peak``
     - ``time-barplot`` only: multiply runtimes by the platform's published peak FLOP/s
   * - ``--remove-description``
     - Drop bracketed labels such as ``[Naive]``; efficiency charts combine variants by paradigm
   * - ``-l, --legend``
     - Save the legend as a separate PDF (add ``--legend--vertical`` for a single column)
   * - ``-e, --export-to-csv``
     - Also export efficiency and portability data as CSV; ``p3analysis`` adds every complexity metric
   * - ``-o, --output``
     - Output path; defaults to ``<problem>_<chart>.pdf``, or ``.csv`` for
       ``rank-correlation``

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
   * - ``--complexity-metric-absolute``
     - ``navchart``, ``combined``; ``complexity-comparison`` always compares
       percentages of the plain-C++ baseline and rejects it
   * - ``--compare-metric``
     - ``complexity-comparison`` only
   * - ``--correlation``
     - ``rank-correlation`` only
   * - ``-l, --legend`` / ``--remove-description`` / ``--log-complexity``
     - every chart except ``rank-correlation``, which writes a CSV table
       instead of a figure and always ranks paradigms
   * - ``-H, --hardware``
     - ``boxplot``, ``time-barplot``
   * - ``-t, --time`` / ``--normalize-time-to-peak``
     - ``time-barplot`` only
   * - ``-s avg``/``best``/``worst``
     - every chart except ``boxplot`` (needs ``all`` or a numeric size) and
       ``time-barplot`` (needs a numeric size; ``all`` means the largest)

Logging
-------

All commands log through ``loguru`` at ``INFO`` by default. ``-v`` raises
the level to ``DEBUG``, ``-vv`` to ``TRACE``.
