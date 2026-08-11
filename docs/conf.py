# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# ``ppbcc`` is a pure Python package. ``sphinx.ext.autodoc`` imports it, so it
# must be importable. Either install it (``pip install -e ..``, which is what
# the GitHub Pages workflow does) or rely on the ``src`` layout path below.
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "ppbcc"
copyright = "2025 - 2026, Jonas Schuhmacher"
author = "Jonas Schuhmacher"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Docstrings are written in Google style, which is what napoleon parses here.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
# Render ``Attributes:`` sections as a field list instead of separate object
# descriptions; otherwise autodoc documents every dataclass field twice.
napoleon_use_ivar = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# The example plots are vector graphics with an opaque white canvas, so they
# stay legible when the theme switches to dark mode.
html_css_files = ["figures.css"]

# -- Options for Theme 'sphinx_book_theme' -----------------------------------
# https://sphinx-book-theme.readthedocs.io/en/latest/tutorials/get-started.html
html_theme_options = {
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/schuhmaj/performance-portability-code-complexity",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Benchmark Suite",
            "url": "https://github.com/schuhmaj/performance-portability-benchmark",
            "icon": "fa-solid fa-flask",
            "type": "fontawesome",
        },
    ],
}

html_context = {
    "github_url": "https://github.com",
    "github_user": "schuhmaj",
    "github_repo": "performance-portability-code-complexity",
    "github_version": "main",
    "doc_path": "docs",
}
