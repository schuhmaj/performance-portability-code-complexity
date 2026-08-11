ppbcc — Performance Portability, Benchmarking & Code Complexity
===============================================================

.. image:: https://img.shields.io/badge/python-3.12%2B-blue
   :alt: Python 3.12+
.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: MIT License

**ppbcc** is the tooling half of a performance-portability study of GPU
programming paradigms. The benchmark implementations it analyses live in the
companion repository
`performance-portability-benchmark <https://github.com/schuhmaj/performance-portability-benchmark>`__,
which implements the same four algorithms — vector addition, matrix
multiplication, an n-body simulation, and a polyhedral gravity model — across
CUDA, HIP, SYCL, Kokkos, RAJA, Alpaka, OpenMP, OpenACC, OpenCL, Vulkan,
Boost.Compute, WebGPU, Slang, Metal, and ``stdpar``.

This package provides three workflows:

* :doc:`Code complexity <usage/code_complexity>` — Halstead and LOC/SLOC
  metrics for C++ and *GPU-enriched* C++. This part is **stand-alone**: it
  needs nothing but the source files you point it at.
* :doc:`Benchmarking <usage/benchmark>` — discover and run Google Benchmark
  executables and consolidate their JSON reports into one tidy CSV.
* :doc:`P3 analysis and plotting <usage/plots>` — application efficiency,
  performance portability, and five chart types that relate both to code
  complexity.

The performance-portability metrics and the Cascade/Navchart layouts follow —
and are inspired by — the
`P3 Analysis Library <https://github.com/P3HPC/p3-analysis-library>`__ by the
P3HPC community.

.. toctree::
   :caption: INSTALLATION & QUICK START
   :maxdepth: 2

   quickstart/installation
   quickstart/overview
   quickstart/cli

.. toctree::
   :caption: USAGE
   :maxdepth: 2

   usage/code_complexity
   usage/benchmark
   usage/p3analysis
   usage/plots

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2

   api/code_complexity
   api/benchmark
   api/performance_portability
   api/plot

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
