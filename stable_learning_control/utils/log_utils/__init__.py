"""Several utilities and helper functions used for logging.

.. note::
    This module was based on
    `spinningup repository <https://github.com/openai/spinningup/tree/master/spinup/utils/logx.py>`_.
"""

from stable_learning_control.utils.log_utils.helpers import (
    colorize,
    dict_to_mdtable,
    log_to_std_out,
    setup_logger_kwargs,
)
from stable_learning_control.utils.log_utils.logx import EpochLogger
