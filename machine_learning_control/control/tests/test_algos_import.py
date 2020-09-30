"""This script is used to test whether the algorithms in the `algos` folder can be
successfully imported.
"""

# Main python imports
import pytest

# Script Parameters
ALGOS = [
    "machine_learning_control.control.algos.lac.lac",
    "machine_learning_control.control.algos.sac.sac",
]


#################################################
# Test script ###################################
#################################################
@pytest.mark.parametrize("module_name", ALGOS)
def test_import(module_name):
    __import__(module_name)
