import time
import os

import numpy as np

from machine_learning_control.control.utils.logx import EpochLogger
from machine_learning_control.control.utils.run_utils import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs("test", np.random.seed())
logger_kwargs["output_dir"] = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        f"../../../data/sac/runs/run_{int(time.time())}",
    )
)
logger = EpochLogger(**logger_kwargs)
print(logger.use_tensorboard)
logger.use_tensorboard = True
print(logger.use_tensorboard)
logger.use_tensorboard = False
print(logger.use_tensorboard)
logger.use_tensorboard = True
print(logger.use_tensorboard)
logger.use_tensorboard = True
print(logger.use_tensorboard)
print("jan")
