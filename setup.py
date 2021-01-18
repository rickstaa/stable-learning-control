"""Setup file for the 'machine_learning_control' python package.
"""

import re

from setuptools import find_namespace_packages, setup

stand_alone_ns_pkgs = ["simzoo"]

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

# Run python setup
setup(
    packages=PACKAGES, package_dir=PACKAGE_DIR,
)
