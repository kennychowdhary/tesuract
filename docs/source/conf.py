# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os

sys.path.append(os.path.abspath("."))
import doc

# -- Project information -----------------------------------------------------

project = "tesuract"
copyright = "2020, K. Chowdhary"
author = "K. Chowdhary"

# The full version, including alpha/beta/rc tags
release = "0.1.2"

# add path
sys.path.append(os.path.abspath("../../"))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary',
#               'sphinx.ext.doctest',
#               'sphinx.ext.mathjax',
#               'sphinx.ext.viewcode',
#               'sphinx_rtd_theme',
#               'sphinx.ext.todo',
#               'numpydoc'
# ]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgconverter",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# napoleon_google_docstring = False
# napolean_numpy_docstring = True
todo_include_todos = True
# napoleon_include_init_with_doc = True # True = show init
# napoleon_use_admonition_for_examples = True
# napoleon_use_admonition_for_notes = False
# napoleon_use_ivar = False
# napoleon_use_param = True
numpydoc_class_members_toctree = True

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# pygments_style = 'monokai'
# html_theme = 'sphinx_rtd_theme'

# book theme
html_theme = "sphinx_book_theme"

html_theme_options = {
    "collapse_navigation": False,
    "repository_url": "https://github.com/kennychowdhary/tesuract",
    "use_repository_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_short_title = "tesuract"
