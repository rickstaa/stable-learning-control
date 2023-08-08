# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from datetime import datetime
from importlib.metadata import version

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

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "stable-learning-control"
copyright = f"{datetime.now().year}, Rick Staa"
author = "Rick Staa"
release = version("stable_learning_control")
version = ".".join(release.split(".")[:3])
print("Doc release: ", release)
print("Doc version: ", version)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "3.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",  # Add google docstring support.
    "sphinx.ext.extlinks",  # Simplify linking to external documentation.
    "sphinx.ext.githubpages",  # Allow GitHub Pages rendering.
    "sphinx.ext.intersphinx",  # Link to other Sphinx documentation.
    "sphinx.ext.viewcode",  # Add a link to the Python source code for python objects.
    "myst_parser",  # Support for MyST Markdown syntax.
    "autoapi.extension",  # Generate API documentation from code.
    "sphinx.ext.autodoc",  # Include documentation from docstrings.
    "sphinx.ext.imgmath",  # Render math as images.
    "sphinx.ext.todo",  # Support for todo items.
]
autoapi_dirs = ["../../stable_learning_control"]
autoapi_python_class_content = "both"
myst_heading_anchors = 2  # Add anchors to headings.

# Extensions settings.
autodoc_member_order = "bysource"

# imgmath settings.
imgmath_image_format = "svg"
imgmath_font_size = 14

latex_packages = r"""
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amsmath}
\usepackage{cancel}
\usepackage{physics}
"""
latex_macros = r"""
\newcommand{\E}{{\mathrm E}}
\newcommand{\underE}[2]{\underset{\begin{subarray}{c}#1 \end{subarray}}{\E}\left[ #2 \right]}
\newcommand{\Epi}[1]{\underset{\begin{subarray}{c}\tau \sim \pi \end{subarray}}{\E}\left[ #1 \right]}
"""
imgmath_latex_overwrites = r"""
\usepackage[verbose=true,letterpaper]{geometry}
\geometry{
    textheight=12in,
    textwidth=6.5in,
    top=1in,
    headheight=12pt,
    headsep=25pt,
    footskip=30pt
    }
"""
imgmath_latex_preamble = latex_packages + imgmath_latex_overwrites + latex_macros

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Add mappings.
intersphinx_mapping = {
    "gymnasium": ("https://www.gymlibrary.dev/", None),
    "python3": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "tf": (
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/mr-ubik/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "stable_gym": ("https://rickstaa.dev/stable-gym", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

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
    "github_user": "rickstaa",
    "github_repo": "stable-learning-control",
    "github_version": "main",
    "conf_py_path": "/docs/source/",  # needs leading and trailing slashes!
}

# -- External links dictionary -----------------------------------------------
# Here you will find some often used global url definitions.
extlinks = {
    "slc": ("https://github.com/rickstaa/stable-learning-control/%s", None),
    "slc-docs": ("https://stable-learning-control.readthedocs.io/en/latest/%s", None),
    "python": ("https://docs.python.org/3/%s", None),
    "anaconda": ("https://docs.anaconda.com/free/anaconda/install/index.html/%s", None),
    "torch": ("https://pytorch.org/%s", None),
    "tensorflow": ("https://www.tensorflow.org/%s", None),
    "stable-gym": ("https://github.com/rickstaa/stable-gym/%s", None),
    "ros-gazebo-gym": ("https://github.com/rickstaa/ros-gazebo-gym/%s", None),
    "gymnasium": ("https://gymnasium.farama.org/%s", None),
    "tf2": ("https://www.tensorflow.org/api_docs/python/tf/%s", None),
    "tb": ("https://www.tensorflow.org/tensorboard/%s", None),
    "wandb": ("https://docs.wandb.ai/%s", None),
}


# -- Add extra style sheets --------------------------------------------------
def setup(app):
    app.add_css_file("css/modify.css")
