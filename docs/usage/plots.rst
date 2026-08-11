.. _plots:

Available Plots
===============

``ppbcc p3analysis -c/--chart`` renders five chart types. All of them are shown
below for the same benchmark problem, so they can be compared directly.

.. admonition:: Where this data comes from
   :class: note

   Every figure on this page was produced from the measurement data in the
   ``results/`` folder of the companion repository
   `performance-portability-benchmark <https://github.com/schuhmaj/performance-portability-benchmark>`__.

   * **Runtimes** — the per-platform ``Results_*.csv`` files
     (``Results_AMD_Instinct_MI210.csv``,
     ``Results_INTEL_Data_Center_GPU_Max_1550.csv``,
     ``Results_NVIDIA_GH200.csv``, ``Results_NVIDIA_RTX3080.csv``,
     ``Results_NVIDIA_RTX4060.csv``, ``Results_NVIDIA_RTX5080.csv``), each
     consolidated from Google Benchmark JSON reports by
     :doc:`ppbcc benchmark <benchmark>`.
   * **Complexity** — ``results/code-complexity/code-complexity.csv``, produced
     by ``scripts/generate_code_complexity.py`` on top of
     :doc:`ppbcc code-complexity <code_complexity>` (see
     :ref:`code-complexity-applied-example`).

   The shown problem is **matrix multiplication** across six GPU platforms,
   averaged over benchmark sizes, with the cuBLAS reference excluded so the
   vendor library does not define the efficiency baseline on its own.
   The source PDFs live in the ``examples/`` folder of this repository.

Chart overview
--------------

+----------------+--------------------------------------------------+-------------------+
| Chart          | Shows                                            | Needs complexity? |
+================+==================================================+===================+
| ``cascade``    | Efficiency decay across ranked platforms + PP    | no                |
+----------------+--------------------------------------------------+-------------------+
| ``navchart``   | PP against code complexity                       | **yes**           |
+----------------+--------------------------------------------------+-------------------+
| ``combined``   | Cascade + Navchart + size scaling in one figure  | **yes**           |
+----------------+--------------------------------------------------+-------------------+
| ``heatmap``    | Efficiency per paradigm and platform             | no                |
+----------------+--------------------------------------------------+-------------------+
| ``boxplot``    | Efficiency distribution per paradigm             | no                |
+----------------+--------------------------------------------------+-------------------+

Cascade plot
------------

The Cascade plot is the canonical performance-portability figure. The left
panel sorts platforms per implementation from highest to lowest application
efficiency, so a flat line means an implementation performs consistently while
a steep drop exposes a platform it struggles on. The right panel shows the
resulting performance portability :math:`\Phi`, and the tile matrix below
identifies which device sits at which rank for each paradigm.

.. figure:: ../figures/matrixmultiplication_cascade.svg
   :alt: Cascade plot for matrix multiplication
   :width: 100%
   :class: plot-figure

   ``-c cascade`` — efficiency decay, :math:`\Phi`, and the platform ranking.

.. code-block:: bash

    ppbcc p3analysis MatrixMultiplication ./Results_* -c cascade \
      --non-zero-pp --remove-description -s avg -x "Cublas"

Navchart
--------

The Navchart places each implementation at its code complexity (x-axis) and its
performance portability (y-axis). The upper-left corner is the goal: portable
*and* cheap to write. Pick the complexity axis with ``--complexity-metric``,
and use ``--normalize``/``--additive`` to express complexity relative to the
plain-C++ implementation.

.. figure:: ../figures/matrixmultiplication_navchart.svg
   :alt: Navchart of performance portability against Halstead difficulty
   :width: 80%
   :class: plot-figure

   ``-c navchart`` — :math:`\Phi` against absolute Halstead difficulty.

.. code-block:: bash

    ppbcc p3analysis MatrixMultiplication ./Results_* \
      --complexity ./code-complexity/code-complexity.csv -c navchart \
      --complexity-metric halstead-difficulty \
      --non-zero-pp --remove-description -s avg -x "Cublas"

Combined chart
--------------

The combined chart is the summary figure: Cascade panels, the Navchart, and a
problem-size scaling panel in one layout, sharing a single legend. It is the
one to put in a paper when you have room for exactly one figure.

.. figure:: ../figures/matrixmultiplication_combined.svg
   :alt: Combined cascade, navchart, and scaling plot
   :width: 100%
   :class: plot-figure

   ``-c combined`` — Cascade, Navchart (additive complexity), and scaling over
   problem size.

.. code-block:: bash

    ppbcc p3analysis MatrixMultiplication ./Results_* \
      --complexity ./code-complexity/code-complexity.csv -c combined \
      --complexity-metric halstead-difficulty --additive --log-size \
      --non-zero-pp --remove-description -s avg -x "Cublas"

``--additive`` puts the Navchart panel on a *complexity added over plain C++*
axis. ``--log-size`` applies only to the scaling panel and is therefore valid
for the combined chart only.

Efficiency heatmap
------------------

The heatmap trades the ranking abstraction for raw numbers: one cell per
paradigm and platform, annotated with the application efficiency. Zero cells
are exactly the platforms an implementation does not support — which is what
drives :math:`\Phi` to zero without ``--non-zero-pp``. The color scale is fixed
to :math:`[0, 1]`, so heatmaps of different problems remain comparable.

.. figure:: ../figures/matrixmultiplication_heatmap.svg
   :alt: Application-efficiency heatmap by paradigm and platform
   :width: 100%
   :class: plot-figure

   ``-c heatmap -s 16384`` — application efficiency at one problem size.

.. code-block:: bash

    ppbcc p3analysis MatrixMultiplication ./Results_* -c heatmap \
      --non-zero-pp --remove-description -s 16384 -x "Cublas"

Efficiency boxplot
------------------

The boxplot shows the *distribution* of application efficiency for each
paradigm across all platforms and problem sizes, with the individual
observations overlaid. A tall box means the paradigm's efficiency depends
strongly on platform or size; a compact box high up means dependable
performance. Paradigms are sorted alphabetically.

.. figure:: ../figures/matrixmultiplication_boxplot.svg
   :alt: Application-efficiency boxplot by paradigm
   :width: 100%
   :class: plot-figure

   ``-c boxplot -s all`` — efficiency spread across platforms and sizes.

.. code-block:: bash

    ppbcc p3analysis MatrixMultiplication ./Results_* -c boxplot \
      --non-zero-pp --remove-description -s all -x "Cublas"

Boxplots accept only ``-s all`` or an exact numeric size — the ``avg``,
``best``, and ``worst`` modes collapse the distribution the chart exists to
show. Use ``-H/--hardware`` to restrict a boxplot to a single platform; that
option is an error for every other chart type.

Legends and output
------------------

``-l/--legend`` moves the legend out of the figure and saves it as a separate
PDF next to the plot (``<name>_legend.pdf``) with four columns; add
``--legend--vertical`` for a single column. This keeps the plot area usable
when a problem has many paradigms.

``-o/--output`` sets the output path; without a suffix, ``.pdf`` is appended.
The default name is ``<problem>_<chart>.pdf``.
