"""Contains the Tensorflow 2.x implementations of the RL/IL algorithms.
"""

# Put algorithms on namespace for easy loading in the test_policy utility
from bayesian_learning_control.control.algos.tf2.lac.lac import LAC
from bayesian_learning_control.control.algos.tf2.sac.sac import SAC
