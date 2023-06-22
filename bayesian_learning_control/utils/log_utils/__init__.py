"""Several utilities and helper functions used for logging.

.. note::
    This module was based on
    `spinningup repository <https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py>`_.
"""

from bayesian_learning_control.utils.log_utils.helpers import (
    colorize,
    friendly_err,
    log_to_std_out,
    setup_logger_kwargs,
)
from bayesian_learning_control.utils.log_utils.logx import EpochLogger, Logger