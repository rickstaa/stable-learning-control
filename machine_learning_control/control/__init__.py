"""The set of supported RL algorithms.

This module is based on the algos module found in the the
`Spinning Up repository <https://github.com/openai/spinningup>`_.
"""

from machine_learning_control.version import __version__

# Import algorithms onto namespace
from machine_learning_control.control.algos.pytorch.sac.sac import sac as sac_pytorch
from machine_learning_control.control.algos.pytorch.lac.lac import lac as lac_pytorch
