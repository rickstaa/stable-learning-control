"""Contains the MLC Reinforcement Learning (RL) and Imitation learning (IL) algorithms.
"""

from machine_learning_control.control.algos.pytorch.lac.lac import lac as lac_pytorch
from machine_learning_control.control.algos.pytorch.sac.sac import sac as sac_pytorch

from machine_learning_control.control.utils.import_tf import import_tf
from machine_learning_control.version import __version__

if import_tf(dry_run=True, frail=False):
    from machine_learning_control.control.algos.tf2.lac.lac import lac as lac_tf2
    from machine_learning_control.control.algos.tf2.sac.sac import sac as sac_tf2
