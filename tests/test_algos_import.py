"""This script is used to test whether the algorithms in the `algos` folder can be
successfully imported.
"""

import pytest

ALGOS = [
    "bayesian_learning_control.control.algos.pytorch.lac.lac",
    "bayesian_learning_control.control.algos.pytorch.sac.sac",
    "bayesian_learning_control.control.algos.tf2.sac.sac",
    "bayesian_learning_control.control.algos.tf2.sac.sac",
]


@pytest.mark.parametrize("module_name", ALGOS)
def test_import(module_name):
    __import__(module_name)
