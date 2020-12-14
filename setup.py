"""Setup file for the 'machine_learning_control' python package.
"""

# Standard library imports
import logging
import os
from setuptools import setup, find_namespace_packages
import sys
import re
from distutils.sysconfig import get_python_lib

# Get the relative path for including (data) files with the package
relative_site_packages = get_python_lib().split(sys.prefix + os.sep)[1]
date_files_relative_path = os.path.join(
    relative_site_packages, "machine_learning_control"
)

# Add extra virtual shortened package for each stand-alone namespace package
# NOTE: This only works if you don't have a __init__.py file in your parent folder and
# stand alone_ns_pkgs folder.
PACKAGES = find_namespace_packages(
    include=["machine_learning_control*"], exclude=["*tests"]
)
redundant_namespaces = [
    pkg
    for pkg in PACKAGES
    if pkg in [PACKAGES[0] + "." + item + "." + item for item in stand_alone_ns_pkgs]
]

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#################################################
# Setup script ##################################
#################################################

# Parse readme.md
with open("README.md") as f:
    readme = f.read()

# Retrieve package list
PACKAGES = find_namespace_packages(include=["machine_learning_control*"])

# Add extra virtual shortened package for each of namespace_pkgs
namespace_pkgs = ["simzoo"]
exclusions = r"|".join(
    [r"\." + item + r"\.(?=" + item + r".)" for item in namespace_pkgs]
)
PACKAGE_DIR = {}
for package in PACKAGES:
    sub_tmp = re.sub(exclusions, ".", package)
    if sub_tmp is not package:
        PACKAGE_DIR[sub_tmp] = package.replace(".", "/")
PACKAGES.extend(PACKAGE_DIR.keys())

# Run python setup
setup(
    name="machine_learning_control",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    version="0.0.0",
    description=(
        "A python package for performing the whole machine_learning_control pipeline."
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Rick Staa",
    author_email="rick.staa@outlook.com",
    license="Rick Staa copyright",
    url="https://github.com/rickstaa/machine_learning_control",
    keywords="rl, openai gym, control",
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
        "dev": ["pytest", "pytest-xdist", "bumpversion", "flake8", "black"],
    },
    include_package_data=True,
    data_files=[(date_files_relative_path, ["README.md"])],
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
)
