.. _code-complexity:

Code Complexity
===============

The code-complexity workflow computes the classic Halstead measures
[Halstead1977]_ (volume, difficulty, effort, language level, ...) and
line-based size metrics for a set of C++ and GPU source files. All formulas
follow Halstead's original definitions; see :ref:`code-complexity-references`
for the source and :ref:`code-complexity-metric-definitions` for the concrete
expressions used here.

Its distinguishing feature is that it understands
the constructs of GPU/parallel programming paradigms and counts them as
**dialect operators**, so you can quantify how much syntactic surface a
paradigm adds on top of the plain C++ baseline.

.. note::

    This workflow is **stand-alone**. It has no dependency on the benchmark
    or plotting parts of ``ppbcc`` and works on any C++/GPU source tree.

Command line
------------

.. code-block:: bash

    # Analyse a directory recursively, auto-detect the dialect of every file,
    # print the table and save it as CSV:
    ppbcc code-complexity path/to/src -o report.csv

    # Force a dialect (several may be combined), restrict the metrics, and
    # additionally report the plain-C++ baseline and the paradigm's delta:
    ppbcc code-complexity src/kokkos -d kokkos -m halstead loc --diff -o kokkos.csv

    # Aggregate all files into an additional TOTAL row:
    ppbcc code-complexity src/cuda -d cuda --aggregate

    # More logging: -v (DEBUG), -vv (TRACE); the default level is INFO.
    ppbcc code-complexity src -vv

    # List all known dialects and their aliases:
    ppbcc code-complexity --list-dialects

A run on a single Kokkos implementation looks like this:

.. code-block:: text

    $ ppbcc code-complexity src/matrixMultiplication/kokkos \
        -m sloc halstead_volume halstead_difficulty dialect --table-format github

    INFO | Analyzing with automatic per-file dialect detection
    INFO | Analyzing 3 source file(s)

    | file                     | dialect | sloc | volume  | difficulty | dialect_distinct_operators | dialect_total_operators |
    |--------------------------|---------|------|---------|------------|----------------------------|-------------------------|
    | .../kokkos/Impl_Kokkos.cpp | kokkos  |   41 | 2581.89 |    60.2727 |                         12 |                      30 |
    | .../kokkos/Impl_Kokkos.h   | cpp     |   14 |  412.26 |    16.3333 |                          0 |                       0 |
    | .../kokkos/main.cpp        | kokkos  |   20 |  616.56 |    12.6452 |                          1 |                       1 |

Note that ``Impl_Kokkos.h`` is auto-detected as plain ``cpp`` — automatic
detection is per file, not per directory.

The full option list is in :ref:`cli`, or run ``ppbcc code-complexity --help``.

Python API
----------

.. code-block:: python

    from pathlib import Path
    from ppbcc.code_complexity import evaluate

    frame = evaluate(
        sources=[Path("src/")],            # files and/or directories
        language_dialect="auto",           # or "cpp", "kokkos", "cuda,openmp", ...
        metrics=["halstead", "loc"],       # None = all metrics
        diff=True,                         # add baseline_*/delta_* columns
        aggregate=False,                   # add a TOTAL row
        output=Path("report.csv"),         # optionally write the CSV directly
    )
    print(frame[["file", "dialect", "effort", "delta_effort"]])

The result is a :class:`pandas.DataFrame` with one row per file.

.. _code-complexity-metric-definitions:

Halstead metric definitions
---------------------------

Let :math:`n_1` and :math:`n_2` be the number of *distinct* operators and
operands, and :math:`N_1` and :math:`N_2` their total number of occurrences.
``ppbcc`` derives Halstead's measures [Halstead1977]_ exactly as originally
defined:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Measure
     - Formula
     - Column
   * - Vocabulary :math:`\eta`
     - :math:`n_1 + n_2`
     - ``vocabulary``
   * - Length :math:`N`
     - :math:`N_1 + N_2`
     - ``length``
   * - Calculated length :math:`\hat{N}`
     - :math:`n_1 \log_2 n_1 + n_2 \log_2 n_2`
     - ``calculated_length``
   * - Volume :math:`V`
     - :math:`N \log_2 \eta`
     - ``volume``
   * - Difficulty :math:`D`
     - :math:`\frac{n_1}{2} \cdot \frac{N_2}{n_2}`
     - ``difficulty``
   * - Effort :math:`E`
     - :math:`D \cdot V`
     - ``effort``
   * - Time :math:`T`
     - :math:`E / 18` seconds
     - ``time_seconds``
   * - Delivered bugs :math:`B`
     - :math:`V / 3000`
     - ``delivered_bugs``
   * - Program level :math:`L`
     - :math:`1 / D`
     - ``program_level``
   * - Language level :math:`\lambda`
     - :math:`L^2 \cdot V`
     - ``language_level``

Degenerate inputs are handled defensively: an empty vocabulary yields a volume
of zero, and a program without operands yields a difficulty — and therefore a
program level — of zero rather than a division by zero.

.. note::

    :math:`T = E/18` and :math:`B = V/3000` carry Halstead's original empirical
    constants (the Stroud number and the bug-rate estimate). They are reported
    for completeness; for comparing paradigms, prefer ``volume``,
    ``difficulty``, or ``effort``, which are what the
    :doc:`Navchart and combined charts <plots>` consume.

Result columns
--------------

+---------------------------------+--------------------------------------------------------------+
| Column group                    | Columns                                                      |
+=================================+==============================================================+
| Identification                  | ``file``, ``dialect``                                        |
+---------------------------------+--------------------------------------------------------------+
| Line metrics                    | ``loc``, ``sloc``, ``comment_lines``, ``blank_lines``        |
+---------------------------------+--------------------------------------------------------------+
| Halstead base counts            | ``distinct_operators`` (:math:`n_1`),                        |
|                                 | ``distinct_operands`` (:math:`n_2`),                         |
|                                 | ``total_operators`` (:math:`N_1`),                           |
|                                 | ``total_operands`` (:math:`N_2`)                             |
+---------------------------------+--------------------------------------------------------------+
| Halstead derived                | ``vocabulary``, ``length``, ``calculated_length``,           |
|                                 | ``volume``, ``difficulty``, ``effort``, ``time_seconds``,    |
|                                 | ``delivered_bugs``, ``program_level``, ``language_level``    |
+---------------------------------+--------------------------------------------------------------+
| Dialect share                   | ``dialect_distinct_operators``, ``dialect_total_operators``, |
|                                 | ``dialect_distinct_operands``, ``dialect_total_operands``    |
+---------------------------------+--------------------------------------------------------------+
| With ``--diff``                 | ``baseline_<metric>`` and ``delta_<metric>`` for the key     |
|                                 | Halstead columns                                             |
+---------------------------------+--------------------------------------------------------------+

How tokens are counted
----------------------

* The lexer strips comments and treats string/char literals, numbers, and
  identifiers as operands; keywords, punctuation, and preprocessor directives
  are operators. Closing brackets ``)``, ``]``, ``}`` pair with their (already
  counted) opening bracket and are not counted again.
* Everything recognised as belonging to a dialect becomes a *dialect operator*:

  * qualified names under a dialect namespace count as **one** operator
    (``Kokkos::parallel_for``),
  * dialect keywords (``__global__``, ``threadIdx``),
  * API identifiers matched by patterns (``cudaMalloc``, ``cl_mem``,
    ``vkCreateBuffer``),
  * CUDA/HIP kernel-launch brackets ``<<< >>>``,
  * dialect pragmas (``#pragma omp ...``), whose clauses are operators and
    whose referenced variables become dialect operands.

* Namespace aliases are resolved: ``namespace bc = boost::compute;`` makes
  ``bc::vector`` a Boost.Compute operator.
* In ``auto`` mode, dialects are detected per file via file extensions
  (``.cu``, ``.hip``, ``.cl``, ``.comp``, ``.wgsl``, ``.slang``, ...), included
  headers, pragma prefixes, namespace usage, and characteristic identifier
  patterns.

.. _code-complexity-configuration:

Configuration
-------------

Keyword and dialect definitions live in TOML files packaged under
``src/ppbcc/code_complexity/share/`` (``cpp_keywords.toml``,
``dialects.toml``). They ship inside the package so pip-installed wheels carry
them. Both can be replaced at runtime with ``--keywords-config`` /
``--dialects-config`` (CLI) or ``keywords_path`` / ``dialects_path`` (API).

Adding a new dialect is a matter of adding a ``[dialects.<name>]`` table — no
code changes required.

.. _code-complexity-applied-example:

Applied example: analysing a whole benchmark suite
--------------------------------------------------

The CLI works well for a directory of files, but a real study usually needs
finer control: several implementations may share one directory, a shader may
belong to two hosts at once, and each *logical implementation* — not each file
— should become one row.

The companion repository
`performance-portability-benchmark <https://github.com/schuhmaj/performance-portability-benchmark>`__
solves this with a script (``scripts/generate_code_complexity.py``) built on
the Python API. It is a good template for your own study. The pattern has four
parts.

**1. An explicit implementation manifest.** Every logical implementation names
the files it owns, so variants sharing a directory stay separate:

.. code-block:: python

    @dataclass(frozen=True)
    class Implementation:
        problem: str
        framework: str
        dialect: str
        sources: tuple[str, ...]

    entries = [
        impl("MatrixMultiplication", "Cuda[Naive]", "cuda",
             "matrixMultiplication/cuda/Impl_CudaNaive.cu",
             "matrixMultiplication/cuda/Impl_CudaNaive.cuh"),
        impl("MatrixMultiplication", "Cuda[SharedMemory]", "cuda",
             "matrixMultiplication/cuda/Impl_Cuda.cu",
             "matrixMultiplication/cuda/Impl_Cuda.cuh"),
        # The same Slang shader is counted once with its CUDA host …
        impl("MatrixMultiplication", "Slang-Cuda", "slang,cuda",
             "matrixMultiplication/slang/Impl_SlangCuda.cu",
             "matrixMultiplication/slang/MatrixMultiplicationShader.slang"),
        # … and once with its Vulkan host.
        impl("MatrixMultiplication", "Slang-Vulkan", "slang,vulkan",
             "matrixMultiplication/slang/Impl_SlangVulkan.cpp",
             "matrixMultiplication/slang/MatrixMultiplicationShader.slang"),
    ]

Note how ``dialect`` may combine paradigms (``"slang,cuda"``): a Slang host
program written against the CUDA runtime is both.

**2. Transitive local includes.** Seed files are expanded by following quoted
``#include`` directives, so shared definition headers are attributed to every
implementation that pulls them in:

.. code-block:: python

    def local_dependencies(source: Path, seed_sources: tuple[str, ...]) -> tuple[Path, ...]:
        """Resolve repository-local quoted includes and common implementation units."""
        pending = [(source / relative).resolve() for relative in seed_sources]
        selected: set[Path] = set()
        while pending:
            path = pending.pop()
            ...  # parse `#include "..."` lines, resolve, and push onto `pending`
        return tuple(sorted(selected))

**3. One aggregated row per implementation.** ``evaluate`` is called per
implementation with ``aggregate=True`` and the raw Halstead counts. The
``TOTAL`` row merges operator/operand *sets* before recomputing, so distinct
counts are implementation-wide instead of a sum of per-file counts:

.. code-block:: python

    from ppbcc.code_complexity import AUTO_DIALECT_NAME, evaluate, save_csv

    AGGREGATE_METRICS = (
        "sloc",
        "distinct_operators",
        "distinct_operands",
        "total_operators",
        "total_operands",
    )

    frame = evaluate(
        sources=list(sources),
        language_dialect=AUTO_DIALECT_NAME if dialect is None else dialect,
        metrics=list(AGGREGATE_METRICS),
        aggregate=True,
    )
    total = frame.loc[frame["file"] == "TOTAL"].iloc[0]

.. important::

    Emit the raw counts (:math:`n_1, n_2, N_1, N_2`) rather than derived
    figures. ``ppbcc p3analysis`` derives volume, difficulty, and effort from
    them, so all complexity metrics stay selectable at plotting time via
    ``--complexity-metric``.

**4. Two CSV products.** A file-level report for browsing, and the
implementation-level report consumed by ``ppbcc p3analysis``:

.. code-block:: text

    Name,Framework,SLOC,n1,n2,N1,N2
    MatrixMultiplication,Kokkos,548,90,313,2643,1683
    MatrixMultiplication,Cuda[Naive],579,100,312,2844,1781
    MatrixMultiplication,Slang-Cuda,732,123,402,3136,2027
    ...

Run it with:

.. code-block:: bash

    # inside the performance-portability-benchmark repository
    python scripts/generate_code_complexity.py --source src --output results/code-complexity

    # preview every analysis without executing it
    python scripts/generate_code_complexity.py --dry-run

The resulting ``results/code-complexity/code-complexity.csv`` is exactly the
file passed to ``ppbcc p3analysis --complexity`` in :doc:`plots`.

.. _code-complexity-references:

References
----------

.. [Halstead1977] M. H. Halstead, *Elements of Software Science*, in Operating
   and Programming Systems Series. USA: Elsevier Science Inc., 1977.
   `Catalogue entry <https://search.ub.tum.de/vufind/Record/DE-604.BV002283430>`__.

The performance-portability side of ``ppbcc`` builds on a separate body of
work; see :ref:`p3analysis-references`.
