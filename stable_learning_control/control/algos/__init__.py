"""Contains the Pytorch and Tensorflow RL/IL algorithms.
"""

from stable_learning_control.utils.import_utils import import_tf

# Put algorithms on namespace
from stable_learning_control.control.algos.pytorch.lac.lac import LAC as LAC_pytorch
from stable_learning_control.control.algos.pytorch.sac.sac import SAC as SAC_pytorch

if import_tf(dry_run=True, frail=False):
    from stable_learning_control.control.algos.tf2.lac.lac import LAC as LAC_tf
    from stable_learning_control.control.algos.tf2.sac.sac import SAC as SAC_tf
