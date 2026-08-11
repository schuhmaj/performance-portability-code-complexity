.. _p3analysis:

P3 Analysis
===========

``ppbcc p3analysis`` turns benchmark CSVs (and, optionally, a code-complexity
CSV) into application-efficiency and performance-portability figures, then
renders them as one of five charts. The chart gallery lives in :doc:`plots`;
this page covers the metrics and the data selection behind them.

.. code-block:: bash

    ppbcc p3analysis NAME CSV [CSV ...] [options]

``NAME`` selects the benchmark problem. An exact case-insensitive match wins;
a unique substring is accepted, so ``Polyhedral`` resolves to
``PolyhedralGravity``.

Metrics
-------

Application efficiency
~~~~~~~~~~~~~~~~~~~~~~

For every hardware platform and workload, an implementation's runtime is
compared against the best runtime observed for that platform and workload:

.. math::

    e_A(a, p) = \frac{\min_{a' \in A} t(a', p)}{t(a, p)}

Duplicate measurements are reduced to their median runtime first, and the
per-workload efficiencies are then averaged arithmetically.

Performance portability :math:`\Phi`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\Phi` is the harmonic mean of the application efficiencies over the set
of platforms :math:`H`:

.. math::

    \Phi(a, H) = \frac{|H|}{\sum_{p \in H} \frac{1}{e_A(a, p)}}

By default, a platform an implementation does not support contributes zero and
therefore drives :math:`\Phi` to zero — the strict reading, which rewards
implementations that run everywhere. ``--non-zero-pp`` restricts the harmonic
mean to the platforms that actually produced a result; the
application-efficiency output still retains the zeros, so the heatmap and the
Cascade platform ranking stay honest about the gaps.

.. note::

    Both definitions and the Cascade/Navchart layouts follow the
    `P3 Analysis Library <https://github.com/P3HPC/p3-analysis-library>`__,
    which inspired this part of ``ppbcc``.

Selecting data
--------------

Problem sizes
~~~~~~~~~~~~~

``-s/--size`` controls how the size axis is collapsed:

+-------------------------+---------------------------------------------------------------------+
| Value                   | Behaviour                                                           |
+=========================+=====================================================================+
| ``all`` (default)       | Use every measured size                                             |
+-------------------------+---------------------------------------------------------------------+
| exact number            | Restrict everything to that one size                                |
+-------------------------+---------------------------------------------------------------------+
| ``avg``/``average``/    | Compute both metrics independently per size, then average           |
| ``mean``                | arithmetically — every size gets equal weight                       |
+-------------------------+---------------------------------------------------------------------+
| ``best`` / ``worst``    | Take the per-application maximum / minimum over sizes               |
+-------------------------+---------------------------------------------------------------------+

Boxplots accept only ``all`` or an exact numeric size, because the summary
modes remove the very distribution the boxplot displays. In the combined
chart, the scaling panel always uses all sizes regardless of ``-s``.

Filtering implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

``-i/--include`` and ``-x/--exclude`` are regular expressions matched against
the ``Description`` column. Excluding vendor-optimised references is a common
move, since they otherwise define the efficiency baseline:

.. code-block:: bash

    ppbcc p3analysis MatrixMultiplication ./Results_* -x "Cublas"

``--remove-description`` drops bracketed labels such as ``[Naive]`` from
labels and legends; efficiency charts then combine variants by paradigm.

``-p/--precision`` keeps only 32- or 64-bit results.

Joining code complexity
-----------------------

``--complexity`` takes the implementation-level CSV described in
:ref:`code-complexity-applied-example`. Its ``Name``/``Framework`` columns are
matched against the benchmark problem and paradigm labels, and its raw Halstead
counts (``n1``, ``n2``, ``N1``, ``N2``) are used to derive the metric selected
by ``--complexity-metric``:

+---------------------------+-------------------------------+---------------------------+
| Metric                    | Accepted aliases              | Definition                |
+===========================+===============================+===========================+
| ``sloc``                  | —                             | source lines of code      |
+---------------------------+-------------------------------+---------------------------+
| ``halstead-vocabulary``   | ``vocabulary``, ``eta``       | :math:`n_1 + n_2`         |
+---------------------------+-------------------------------+---------------------------+
| ``halstead-length``       | ``halstead-program-length``,  | :math:`N_1 + N_2`         |
|                           | ``program-length``,           |                           |
|                           | ``length``, ``n``             |                           |
+---------------------------+-------------------------------+---------------------------+
| ``halstead-volume``       | ``volume``, ``v``             | :math:`N \log_2 \eta`     |
+---------------------------+-------------------------------+---------------------------+
| ``halstead-difficulty``   | ``difficulty``, ``d``         | :math:`\frac{n_1}{2}      |
|                           |                               | \cdot \frac{N_2}{n_2}`    |
+---------------------------+-------------------------------+---------------------------+
| ``halstead-effort``       | ``effort``, ``e``             | :math:`D \cdot V`         |
| (default)                 |                               |                           |
+---------------------------+-------------------------------+---------------------------+

Matching is case-insensitive and treats ``_``, ``-``, and spaces alike, so
``Halstead Difficulty`` and ``halstead_difficulty`` both work.

Two options put the numbers into perspective relative to the plain-C++
implementation:

* ``--normalize`` divides every score by the CPP score, so CPP sits at 100 %.
* ``--additive`` subtracts the CPP score, showing the complexity a paradigm
  *adds* over the sequential baseline.

.. warning::

    ``--additive`` can produce non-positive values when a paradigm is terser
    than the CPP reference. Do not combine it with ``--log-complexity`` in that
    case.

CSV export
----------

``-e/--export-to-csv`` writes application-efficiency and
performance-portability data next to the plot as
``<plot-prefix>_application_efficiency.csv`` and
``<plot-prefix>_performance_portability.csv``. The export keeps separate rows
per size and precision *plus* averaged rows, and includes every available
complexity metric when ``--complexity`` was supplied — useful when you want the
underlying numbers rather than the figure.

Python API
----------

.. code-block:: python

    from pathlib import Path
    from ppbcc.performance_portability import (
        calculate_metrics,
        load_benchmark_csvs,
        select_problem_rows,
    )

    frame = load_benchmark_csvs([Path("Results_NVIDIA_RTX5080.csv")])
    rows, title, description_is_workload = select_problem_rows(
        frame,
        "MatrixMultiplication",
        description_include=None,
        description_exclude="Cublas",
        problem_size=None,
        precision=None,
    )
    efficiency, portability = calculate_metrics(
        rows, description_is_workload, non_zero_pp=True
    )

See :doc:`../api/performance_portability` for the full signatures.
