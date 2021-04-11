"""File used for storing several package configuration values.

.. literalinclude:: /../../bayesian_learning_control/user_config.py
   :language: python
   :linenos:
   :lines: 9-
"""

import os.path as osp

# Default neural network backend for each algo
# (Must be either 'tf1' or 'pytorch')
DEFAULT_BACKEND = {
    "lac": "pytorch",
    "lac2": "pytorch",
    "lac3": "pytorch",
    "lac4": "pytorch",
    "lac5": "pytorch",
    "lac6": "pytorch",
    "sac": "pytorch",
    "sac2": "pytorch",
}

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), "data")

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching
# experiments.
WAIT_BEFORE_LAUNCH = 5

# Logger std out output type
# NOTE:The format in which the statistics are displayed to the terminal. Options are
# "table"  which supplies them as a table and "line" which prints them in one line
DEFAULT_STD_OUT_TYPE = "line"
