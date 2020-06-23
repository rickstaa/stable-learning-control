"""Setup file for the 'simzoo' python package.
"""

# Standard library imports
import logging
import os
from setuptools import setup
import sys
import re
from distutils.sysconfig import get_python_lib

# Get the relative path for including (data) files with the package
relative_site_packages = get_python_lib().split(sys.prefix + os.sep)[1]
date_files_relative_path = os.path.join(relative_site_packages, "simzoo")

# Additional python requirements that could not be specified in the package.xml
requirements = ["gym", "matplotlib"]

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#################################################
# Setup script ##################################
#################################################

# Get current package version
__version__ = re.sub(
    r"[^\d.]",
    "",
    open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "version.py")
    ).read(),
)

# Parse readme.md
with open("README.md") as f:
    readme = f.read()

# Run python setup
setup(
    name="simzoo",
    version=__version__,
    description=("A python package containing several openai gym environments."),
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Rick Staa",
    author_email="rick.staa@outlook.com",
    license="Rick Staa copyright",
    url="https://github.com/rickstaa/machine_learning_control",
    keywords="rl, openai gym",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=requirements,
    extras_require={
        "docs": [
            "sphinx",
            "sphinxcontrib-napoleon",
            "sphinx_rtd_theme",
            "sphinx-navtree",
        ],
        "dev": ["bumpversion", "flake8", "black"],
    },
    include_package_data=True,
    data_files=[(date_files_relative_path, ["README.md"])],
)
