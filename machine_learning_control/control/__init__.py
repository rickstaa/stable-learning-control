"""The set of supported RL algorithms.

This module is based on the algos module found in the the
`Spinning Up repository <https://github.com/openai/spinningup>`_.
"""

from machine_learning_control.control.algos.pytorch.lac.lac import lac as lac_pytorch
from machine_learning_control.control.algos.pytorch.sac.sac import sac as sac_pytorch
from machine_learning_control.control.algos.tf2.lac.lac import lac as lac_tf
from machine_learning_control.control.algos.tf2.sac.sac import sac as sac_tf
from machine_learning_control.version import __version__
