"""Imports algorithms onto namespace.
"""

from machine_learning_control.control.algos.tf2.sac.sac import sac as sac_tf
# from machine_learning_control.control.algos.tf.lac.lac import lac as lac_tf

from machine_learning_control.control.algos.pytorch.sac.sac import sac as sac_pytorch
from machine_learning_control.control.algos.pytorch.lac.lac import lac as lac_pytorch
