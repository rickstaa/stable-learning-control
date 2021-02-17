"""The set of supported RL algorithms and Policy Classes.
"""

from machine_learning_control.control.utils import import_tf

if import_tf(dry_run=True, frail=False):
    from machine_learning_control.control.algos.tf2.sac.sac import SAC
    from machine_learning_control.control.algos.tf2.lac.lac import LAC
