"""Setup file for the 'machine_learning_control' python package.
"""

import os.path as osp
import re
import sys

from setuptools import find_namespace_packages, setup

# Script settings
stand_alone_ns_pkgs = ["simzoo"]


def submodules_available(submodules):
    """Throws warning and stops the script if any of the submodules is not present."""
    for submodule in submodules:
        submodule_setup_path = osp.join(
            osp.abspath(osp.dirname(__file__)),
            "machine_learning_control",
            submodule,
            "setup.py",
        )

        if not osp.exists(submodule_setup_path):
            print("Could not find {}".format(submodule_setup_path))
            print("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)


# Add extra virtual shortened package for each stand-alone namespace package
# NOTE: This only works if you don't have a __init__.py file in your parent folder and
# stand alone_ns_pkgs folder.
PACKAGES = find_namespace_packages(
    include=["machine_learning_control*"],
    exclude=["*.tests*", "*.pytest*", "*.node_modules*"],
)
redundant_namespaces = [
    pkg
    for pkg in PACKAGES
    if pkg in [PACKAGES[0] + "." + item + "." + item for item in stand_alone_ns_pkgs]
]
PACKAGE_DIR = {}
for ns in redundant_namespaces:
    short_ns = re.sub(
        r"\." + ns.split(".")[-1] + r"\.(?=" + ns.split(".")[-1] + r")", ".", ns
    )
    PACKAGE_DIR[short_ns] = ns.replace(".", "/")
    children = [pkg for pkg in PACKAGES if ns in pkg and ns]
    for child in children:
        PACKAGE_DIR[child] = child.replace(".", "/")
        short_child = short_ns + re.sub(ns, "", child)
        if short_child not in PACKAGES:
            PACKAGES.append(short_child)

# Throw warning if submodules were not pulled
submodules_available(stand_alone_ns_pkgs)

setup(
    packages=PACKAGES, package_dir=PACKAGE_DIR,
)
