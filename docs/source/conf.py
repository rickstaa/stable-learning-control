#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# -- Make sure mlc is accessible without going through setup.py -----------
import os.path as osp
import sys

dirname = osp.dirname
top_folder = dirname(dirname(dirname(__file__)))
sys.path.insert(0, osp.join(top_folder, "machine_learning_control"))
sys.path.insert(0, osp.join(top_folder, "scripts"))
sys.path.insert(0, osp.join(top_folder, "machine_learning_control", "control"))
sys.path.insert(0, osp.join(top_folder, "machine_learning_control", "hardware"))
sys.path.insert(0, osp.join(top_folder, "machine_learning_control", "modeling"))
sys.path.insert(0, osp.join(top_folder, "machine_learning_control", "simzoo"))

# -- General configuration ------------------------------------------------

# Mock mpi4py to get around having to install it on RTD server (which fails)
# Also to mock PyTorch, because it is too large for the RTD server to download
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = [
    "mpi4py",
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "3.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "recommonmark",
]

# Extension settings
autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_member_order = "bysource"
autosummary_imported_members = True

# imgmath settings
imgmath_image_format = "svg"
imgmath_font_size = 14

# Add mappings
intersphinx_mapping = {
    "python3": ("http://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "tf": (
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/mr-ubik/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Map suffix(es) to parsers
# source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "machine-learning-control"
copyright = "2020, Rick Staa"
author = "Rick Staa"
git_user_name = "rickstaa"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.5.0"
# The full version, including alpha/beta/rc tags.
release = "0.5.0"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogues.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "TODO.*", "README.*"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"  # 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "images/logo.svg"
html_favicon = "_static/favicon.ico"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {"logo_only": True}
html_context = {
    "display_github": True,  # Add 'Edit on Github' link instead of 'View page source'
    "github_user": git_user_name,
    "github_repo": project,
    "github_version": "master",
    "conf_py_path": "/docs/source/",  # needs leading and trailing slashes!
}

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "MachineLearningControldoc"

# -- Options for LaTeX output ---------------------------------------------

imgmath_latex_preamble = r"""
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amsmath}
\usepackage{cancel}

\usepackage[verbose=true,letterpaper]{geometry}
\geometry{
    textheight=12in,
    textwidth=6.5in,
    top=1in,
    headheight=12pt,
    headsep=25pt,
    footskip=30pt
    }

\newcommand{\E}{{\mathrm E}}

\newcommand{\underE}[2]{\underset{\begin{subarray}{c}#1 \end{subarray}}{\E}\left[ #2 \right]}

\newcommand{\Epi}[1]{\underset{\begin{subarray}{c}\tau \sim \pi \end{subarray}}{\E}\left[ #1 \right]}
"""

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    "preamble": r"""
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amsmath}
\usepackage{cancel}


\newcommand{\E}{{\mathrm E}}

\newcommand{\underE}[2]{\underset{\begin{subarray}{c}#1 \end{subarray}}{\E}\left[ #2 \right]}

\newcommand{\Epi}[1]{\underset{\begin{subarray}{c}\tau \sim \pi \end{subarray}}{\E}\left[ #1 \right]}
""",
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
        1,
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
    "mlc": ("https://github.com/rickstaa/machine-learning-control/%s", None),
    "issue": ("https://github.com/rickstaa/machine-learning-control/issues/%s", None),
}

# -- Add extra style sheets --------------------------------------------------
def setup(app):
    app.add_css_file("css/modify.css")
