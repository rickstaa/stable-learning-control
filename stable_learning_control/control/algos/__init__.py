"""Contains the Pytorch and TensorFlow RL/IL algorithms.

.. warning::
    Due to being more friendly, the Pytorch implementation was eventually used during
    research. As a result, the TensorFlow implementation has yet to be thoroughly tested,
    and no guarantees can be given about the correctness of these algorithms.
"""
from stable_learning_control.control.algos.pytorch.lac.lac import LAC as LAC_pytorch
from stable_learning_control.control.algos.pytorch.sac.sac import SAC as SAC_pytorch
from stable_learning_control.utils.import_utils import import_tf

if import_tf(dry_run=True, frail=False):
    from stable_learning_control.control.algos.tf2.lac.lac import LAC as LAC_tf
    from stable_learning_control.control.algos.tf2.sac.sac import SAC as SAC_tf
