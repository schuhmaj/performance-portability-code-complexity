.. _overview:

Overview
========

Scope
-----

``ppbcc`` was written to answer one question quantitatively:

    *How much syntactic surface does a GPU programming paradigm add on top of
    plain C++, and what does that buy you in performance portability?*

The implementations under study live in the companion repository
`performance-portability-benchmark <https://github.com/schuhmaj/performance-portability-benchmark>`__.
That repository contains the same four algorithms — vector addition, matrix
multiplication, an n-body simulation, and a polyhedral gravity model —
re-implemented in every paradigm under test, plus the raw measurement results
in its ``results/`` folder. ``ppbcc`` is the analysis tooling for that data,
but the code-complexity part works on any C++/GPU source tree.

The three workflows
-------------------

+-----------------------------+---------------------------------------------------------------+-------------------+
| Workflow                    | What it does                                                  | Needs benchmarks? |
+=============================+===============================================================+===================+
| ``ppbcc code-complexity``   | Halstead + LOC/SLOC metrics, dialect-aware token counting     | no (stand-alone)  |
+-----------------------------+---------------------------------------------------------------+-------------------+
| ``ppbcc benchmark``         | Run Google Benchmark targets, consolidate JSON into one CSV   | yes               |
+-----------------------------+---------------------------------------------------------------+-------------------+
| ``ppbcc p3analysis``        | Application efficiency, performance portability, five charts  | yes (CSV input)   |
+-----------------------------+---------------------------------------------------------------+-------------------+

The data flow
-------------

.. code-block:: text

    C++/GPU sources ──► ppbcc code-complexity ──► code-complexity.csv ─┐
                                                                      │
    build/ (Google Benchmark) ──► ppbcc benchmark ──► Results_*.csv ──┴─► ppbcc p3analysis ──► plots + exports

The two CSV products are joined by ``ppbcc p3analysis``: benchmark rows supply
application efficiency and performance portability, and the code-complexity
rows supply the complexity axis of the Navchart and combined charts.

Key concepts
------------

Dialect operators
~~~~~~~~~~~~~~~~~

The code-complexity lexer recognises constructs belonging to a GPU paradigm and
counts them as **dialect operators** rather than plain C++ tokens. A qualified
name under a dialect namespace counts as *one* operator
(``Kokkos::parallel_for``), as do dialect keywords (``__global__``,
``threadIdx``), API identifiers matched by patterns (``cudaMalloc``,
``vkCreateBuffer``), CUDA/HIP launch brackets ``<<< >>>``, and dialect pragmas
(``#pragma omp ...``). With ``--diff`` you additionally get the plain-C++
baseline (all dialect tokens removed) and the delta against it.

Application efficiency
~~~~~~~~~~~~~~~~~~~~~~

For every hardware platform and workload, each implementation's runtime is
compared against the best runtime observed on that platform, yielding an
efficiency in :math:`[0, 1]`. Duplicate measurements are reduced to their
median first.

Performance portability :math:`\Phi`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\Phi` is the harmonic mean of the application efficiencies across all
selected platforms. By default an unsupported platform contributes zero and
therefore drives :math:`\Phi` to zero; ``--non-zero-pp`` restricts the harmonic
mean to the platforms an implementation actually supports.

.. note::

    These definitions are those of Pennycook et al. [Pennycook2019]_
    [Pennycook2021]_, and the Cascade/Navchart layouts follow the
    `P3 Analysis Library <https://github.com/P3HPC/p3-analysis-library>`__ that
    accompanies that work, which inspired this part of ``ppbcc``. The Halstead
    measures come from [Halstead1977]_. Full citations:
    :ref:`p3analysis-references` and :ref:`code-complexity-references`.

Supported dialects
------------------

OpenMP, OpenACC, Kokkos, RAJA, Alpaka, CUDA, HIP, Thrust, pCUDA
(`AdaptiveCpp's portable CUDA dialect <https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/pcuda.md>`__),
SYCL (AdaptiveCpp/DPC++), OpenCL (host API and ``.cl`` kernels), Vulkan,
Boost.Compute, WebGPU (host API and WGSL), GLSL compute shaders, Slang/HLSL,
Metal, and ``stdpar`` (``std::execution``).

Run ``ppbcc code-complexity --list-dialects`` for the authoritative list with
all aliases. Dialects are defined in TOML and can be extended without code
changes — see :ref:`code-complexity-configuration`.
