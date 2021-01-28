"""The set of supported RL algorithms.

This module is based on the algos module found in the the
`Spinning Up repository <https://github.com/openai/spinningup>`_.
"""

from machine_learning_control.version import __version__

# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
# FIXME: Move to top folder
# FIXME: ONLY: Invoke if tensorflow installed.
# FIXME: Create TORCH TENSORFLOW DEPS.
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Import algorithms onto namespace
# from machine_learning_control.control.algos.tf2.sac.sac import sac as sac_tf2
from machine_learning_control.control.algos.pytorch.sac.sac import sac as sac_pytorch
from machine_learning_control.control.algos.pytorch.lac.lac import lac as lac_pytorch
