"""Contains the Pytorch and Tensorflow RL/IL algorithms.
"""

# Put algorithms on namespace for easy loading in the test_policy utility
from bayesian_learning_control.control.algos.pytorch.lac.lac import LAC
from bayesian_learning_control.control.algos.pytorch.sac.sac import SAC
