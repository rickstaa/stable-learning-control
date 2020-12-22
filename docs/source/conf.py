#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# NOTE: This documentation is build upon the documentation provided by the spinningup
# project https://spinningup.openai.com/en/latest/.
#
# Machine Learning Control documentation build configuration file, created by
# sphinx-quickstart on Wed Aug 15 04:21:07 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Make sure mlc is accessible without going through setup.py
dirname = os.path.dirname
sys.path.insert(0, dirname(dirname(__file__)))

# Mock mpi4py to get around having to install it on RTD server (which fails)
# Also to mock PyTorch, because it is too large for the RTD server to download
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = [
    "mpi4py",
    "torch",
    "torch.optim",
    "torch.nn",
    "torch.distributions",
    "torch.distributions.normal",
    "torch.distributions.categorical",
    "torch.nn.functional",
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# Finish imports
import machine_learning_control
from recommonmark.parser import CommonMarkParser


source_parsers = {
    ".md": CommonMarkParser,
}


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
]

# 'sphinx.ext.mathjax', ??

# imgmath settings
imgmath_image_format = "svg"
imgmath_font_size = 14

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]
# source_suffix = '.rst'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Machine Learning Control"
copyright = "2020, Rick Staa"
author = "Rick Staa"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.1.0"
# The full version, including alpha/beta/rc tags.
release = "0.1.0"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogues.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"  # 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customise the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "images/logo.svg"
html_theme_options = {"logo_only": True}
html_favicon = "_static/favicon.ico"

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "MachineLearningControldoc"

# -- Options for LaTeX output ---------------------------------------------

imgmath_latex_preamble = r'\input{_latex/img_math_latex_preabmle.tex.txt'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    "preamble": r'\input{_latex/latex_elements_preamble.tex.txt'
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "MachineLearningControl.tex",
        "Machine Learning Control Documentation",
        "Joshua Achiam",
        "manual",
    ),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "machine_learning_control",
        "Machine Learning Control documentation",
        [author],
        1
    ),
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "MachineLearningControl",
        "Machine Learning Control Documentation",
        author,
        "MachineLearningControl",
        "Framework that enables you to automatically create, train and deploy RL control algorithms from data.",
        "Miscellaneous",
    ),
]

# -- External links dictionary -----------------------------------------------
# Here you will find some often used global url definitions.
extlinks = {
    "mlc": ("https://github.com/rickstaa/panda_openai_sim/%s", None),
    "issue": ("https://github.com/rickstaa/machine-learning-control/issues/%s", None),
}

# -- Add extra style sheets --------------------------------------------------
def setup(app):
    app.add_stylesheet("css/modify.css")
