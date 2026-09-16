.. _plots:

Available Plots
===============

``ppbcc p2analysis`` and ``ppbcc p3analysis`` render six chart types, selected
by their first argument. All of them are shown below, all but one for the same
benchmark problem, so they can be compared directly.

.. admonition:: Where this data comes from
   :class: note

   Every figure on this page was produced from the measurement data in the
   ``results/`` folder of the companion repository
   `performance-portability-benchmark <https://github.com/schuhmaj/performance-portability-benchmark>`__.

   * **Runtimes** — the per-platform ``Results_*.csv`` files
     (``Results_AMD_MI210.csv``, ``Results_Intel_Max_1550.csv``,
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
   The complexity comparison shows **vector addition** instead, whose small
   implementations make the difference between the two metrics easy to read.
   The source PDFs live in the ``examples/`` folder of this repository.

Chart overview
--------------

+---------------------------+-------------------------------------------------+-------------------+
| Chart                     | Shows                                           | Command           |
+===========================+=================================================+===================+
| ``cascade``               | Efficiency decay across ranked platforms + PP   | ``p2analysis``    |
+---------------------------+-------------------------------------------------+-------------------+
| ``navchart``              | PP against code complexity                      | ``p3analysis``    |
+---------------------------+-------------------------------------------------+-------------------+
| ``combined``              | Cascade + Navchart + size scaling in one figure | ``p3analysis``    |
+---------------------------+-------------------------------------------------+-------------------+
| ``complexity-comparison`` | Two complexity metrics against each other       | ``p3analysis``    |
+---------------------------+-------------------------------------------------+-------------------+
| ``heatmap``               | Efficiency per paradigm and platform            | ``p2analysis``    |
+---------------------------+-------------------------------------------------+-------------------+
| ``boxplot``               | Efficiency distribution per paradigm            | ``p2analysis``    |
+---------------------------+-------------------------------------------------+-------------------+

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

   ``cascade`` — efficiency decay, :math:`\Phi`, and the platform ranking.

.. code-block:: bash

    ppbcc p2analysis cascade ./Results_* -n MatrixMultiplication \
      --non-zero-pp --remove-description -s avg -x "Cublas"

Navchart
--------

The Navchart places each implementation at its code complexity (x-axis) and its
performance portability (y-axis). The upper-left corner is the goal: portable
*and* cheap to write. Pick the complexity axis with ``-c/--complexity-metric``.
Complexity is expressed as a percentage of the plain-C++ implementation unless
``--complexity-metric-absolute`` is given.

.. figure:: ../figures/matrixmultiplication_navchart.svg
   :alt: Navchart of performance portability against Halstead difficulty
   :width: 80%
   :class: plot-figure

   ``navchart`` — :math:`\Phi` against absolute Halstead difficulty.

.. code-block:: bash

    ppbcc p3analysis navchart ./code-complexity/code-complexity.csv ./Results_* \
      -n MatrixMultiplication -c halstead-difficulty --complexity-metric-absolute \
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

   ``combined`` — Cascade, Navchart (normalized complexity), and performance
   portability over problem size.

.. code-block:: bash

    ppbcc p3analysis combined ./code-complexity/code-complexity.csv ./Results_* \
      -n MatrixMultiplication -c halstead-difficulty --log-complexity \
      --non-zero-pp --remove-description -s avg -x "Cublas"

The Navchart panel sits on a *percentage of plain C++* axis. Every value is
positive, which is what makes ``--log-complexity`` usable — and the log axis
matters whenever one implementation dwarfs the rest, as the two CUDA polyhedral
implementations do at roughly 500 % of the baseline.

The scaling panel is a **heatmap** over the benchmark sizes, one row per
implementation and one column per size, because a line per implementation
becomes unreadable beyond a handful of paradigms. Its rows keep the
descending-:math:`\Phi` order of the platform-ranking panel beside it, so the
two lower panels line up row for row.

Column labels become exponents whenever every size is an exact power of one
base — :math:`2^5 \ldots 2^{14}`, :math:`10^1 \ldots 10^8` — which is short
enough to carry the same font size as the cell values. A sweep that is not a
clean power sequence keeps decimal labels at a smaller size, since upright
labels that long would grow the figure's bounding box.

Complexity comparison
---------------------

``complexity-comparison`` answers whether two complexity metrics rank the
paradigms differently. Each paradigm is one marker, both axes are relative to
the plain-C++ baseline, and the identity line splits the plane: above it the
y-axis metric charges a paradigm more than the x-axis one — *dense lines* — and
below it the x-axis metric charges more — *verbose code*, which is how the two
half-planes are captioned.

.. figure:: ../figures/vecadd_complexity_comparison.svg
   :alt: Halstead difficulty against SLOC for vector addition
   :width: 80%
   :class: plot-figure

   ``complexity-comparison`` — Halstead difficulty against SLOC for vector
   addition, both relative to plain C++.

.. code-block:: bash

    ppbcc p3analysis complexity-comparison ./code-complexity/code-complexity.csv \
      ./Results_* -n VecAdd -c halstead-difficulty --compare-metric sloc \
      --log-complexity --non-zero-pp -s avg -x "Cublas" --remove-description

``--compare-metric`` names the x-axis metric and ``-c/--complexity-metric`` the
y axis; both take the usual metric names and aliases. Because the identity line
only means something on a shared relative scale, the chart rejects
``--complexity-metric-absolute``. ``-l`` suppresses the in-plot legend *and* the
per-point paradigm labels, for placing the chart next to a shared legend.

A key on the right states the absolute plain-C++ values behind the 100 % of
both axes, so the chart can be read without the running text. It hangs from
half height into the lower-right corner, which the markers leave free because a
paradigm below the identity line on one axis is rarely far below it on the
other. A chart spanning several problems has several baselines and therefore no
key.

Spearman's :math:`\rho` and Kendall's :math:`\tau` are not drawn; they belong
in the running text, where they can be given to three decimals and discussed.

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

   ``heatmap -s 16384`` — application efficiency at one problem size.

.. code-block:: bash

    ppbcc p2analysis heatmap ./Results_* -n MatrixMultiplication \
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

   ``boxplot -s all`` — efficiency spread across platforms and sizes.

.. code-block:: bash

    ppbcc p2analysis boxplot ./Results_* -n MatrixMultiplication \
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
