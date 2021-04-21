"""Contains the Pytorch and Tensorflow RL/IL algorithms.
"""

# Put algorithms on namespace
from bayesian_learning_control.control.algos.pytorch.lac.lac import LAC as LAC_pytorch
from bayesian_learning_control.control.algos.pytorch.sac.sac import SAC as SAC_pytorch
from bayesian_learning_control.utils.import_utils import import_tf

if import_tf(dry_run=True, frail=False):
    from bayesian_learning_control.control.algos.tf2.lac.lac import LAC as LAC_tf
    from bayesian_learning_control.control.algos.tf2.sac.sac import SAC as SAC_tf
