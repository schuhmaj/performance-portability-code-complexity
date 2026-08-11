.. _installation:

Installation
============

``ppbcc`` requires **Python 3.12 or newer** (it uses :mod:`tomllib` and modern
typing syntax).

From source with pip
--------------------

.. code-block:: bash

    git clone https://github.com/schuhmaj/performance-portability-code-complexity.git
    cd performance-portability-code-complexity
    pip install .

For development, install in editable mode together with the test extra:

.. code-block:: bash

    pip install -e ".[test]"
    pytest

With conda
----------

The repository ships a conda environment file that pulls all runtime and test
dependencies from ``conda-forge``:

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate ppbcc
    pip install -e .

Dependencies
------------

+------------------+-----------------------------------------------------------+
| Package          | Used for                                                  |
+==================+===========================================================+
| ``pandas``       | result tables and CSV export                              |
+------------------+-----------------------------------------------------------+
| ``numpy``        | numerical metric and plot processing                      |
+------------------+-----------------------------------------------------------+
| ``matplotlib``   | plot rendering                                            |
+------------------+-----------------------------------------------------------+
| ``seaborn``      | heatmap and boxplot styling                               |
+------------------+-----------------------------------------------------------+
| ``loguru``       | logging                                                   |
+------------------+-----------------------------------------------------------+
| ``tabulate``     | pretty-printing result tables on stdout                   |
+------------------+-----------------------------------------------------------+

.. note::

    The :doc:`code-complexity workflow <../usage/code_complexity>` is
    stand-alone — it only needs ``pandas``, ``loguru``, and ``tabulate``. The
    plotting dependencies are only exercised by ``ppbcc p3analysis``.

Verifying the installation
--------------------------

.. code-block:: bash

    ppbcc --version
    ppbcc code-complexity --list-dialects

Building the documentation
--------------------------

.. code-block:: bash

    pip install -r docs/requirements.txt
    cd docs
    make html
    # open _build/html/index.html
