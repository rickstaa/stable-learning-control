"""Module used for storing several configuration values.

This module was based on the `user_config` module of the
`spinningup repository <https://github.com/openai/spinningup/blob/master/spinup/user_config.py>`_.
"""

import os.path as osp

# Default neural network backend for each algo
# (Must be either 'tf1' or 'pytorch')
DEFAULT_BACKEND = {
    "lac": "pytorch",
    "sac": "pytorch",
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

# Logger output type
# NOTE (rickstaa): LOGGER_OUTPUT_TYPE - The format in which the statistics are displayed
# to the terminal. Options are "tab" which supplies them as a table and "line"
# which prints them in one line. Defaults to "tab".
# NOTE (rickstaa): LOG_IGNORE - Dictionary containing the keys for which you don't want
# the statistics to be printed to the terminal.
# IMPROVE: Replace by commmand line arguments
# IMPROVE: Ability to turn of and one std, avg, max, min
LOG_FMT = "line"
LOG_IGNORE = {
    "MinQ2Vals",
    "StdEpRet",
    "MaxEpRet",
    "MinEpRet",
    "StdTestEpRet",
    "MaxTestEpRet",
    "MinTestEpRet",
    "AverageQ1Vals",
    'AverageQ1Vals',
    "StdQ1Vals",
    "MaxQ1Vals",
    "MinQ1Vals",
    'AverageQ2Vals',
    "StdQ2Vals",
    "MaxQ2Vals",
    "MinQ2Vals",
    "StdLogPi",
    "MaxLogPi",
    "MinLogPi",
    "StdAlpha"
    "StdLabda"
}
