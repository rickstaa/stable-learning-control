"""Contains the BLC Reinforcement Learning (RL) and Imitation learning (IL) algorithms.
"""

from bayesian_learning_control.control.algos.pytorch.lac.lac import lac as lac_pytorch
from bayesian_learning_control.control.algos.pytorch.sac.sac import sac as sac_pytorch
)

from bayesian_learning_control.utils.import_utils import import_tf
from bayesian_learning_control.version import __version__

if import_tf(dry_run=True, frail=False):
    from bayesian_learning_control.control.algos.tf2.lac.lac import lac as lac_tf2
    from bayesian_learning_control.control.algos.tf2.sac.sac import sac as sac_tf2
