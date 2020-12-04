"""Setup file for the 'machine_learning_control' python package.
"""

from setuptools import setup, find_namespace_packages
import re

with open("README.md") as f:
    readme = f.read()

# Add extra virtual shortened package for each of namespace_pkgs
PACKAGES = find_namespace_packages(include=["machine_learning_control*"])
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
    packages=PACKAGES, package_dir=PACKAGE_DIR,
)
