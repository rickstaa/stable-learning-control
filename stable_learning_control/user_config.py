"""File used for storing several configuration values for the SLC package.

.. literalinclude:: /../../stable_learning_control/user_config.py
   :language: python
   :linenos:
   :lines: 8-
"""

import os.path as osp

# Default neural network backend for each algo (Must be either 'tf2' or 'pytorch').
DEFAULT_BACKEND = {
    "lac": "pytorch",
    "latc": "pytorch",
    "sac": "pytorch",
}

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), "data")

# Whether to automatically insert a date and time stamp into the names of save
# directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching experiments.
WAIT_BEFORE_LAUNCH = 5

# Print experiment config to terminal.
PRINT_CONFIG = False

# Logger stdout output type.
# NOTE:The format in which the training diagnostics are displayed to the terminal.
# Options are "table"  which supplies them as a table and "line" which prints them in
# one line.
DEFAULT_STD_OUT_TYPE = "line"

# Weights & Biases default job type and project.
DEFAULT_WANDB_JOB_TYPE = "train"
DEFAULT_WANDB_PROJECT = "stable-learning-control"

# TensorBoard parameters.
TB_HPARAMS_FILTER = [
    "epochs",
    "num_of_test_episodes",
    "seed",
    "device",
    "save_freq",
    "start_policy",
    "export",
]  # Config keys to filter out when writing hyperparameters to TensorBoard.
TB_HPARAMS_METRICS = [
    "AverageEpRet",
    "AverageTestEpRet",
]  # The metrics to be tracked in TensorBoard.
