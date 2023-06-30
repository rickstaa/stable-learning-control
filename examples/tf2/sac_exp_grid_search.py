"""Example script that shows you how to use the ExperimentGrid utility for a tf2 algorithm.

This utility can helps you to find the right hyperparameters for you agent and environment using a
grid search. This is equivalent to running the ``python -m stable_learning_control.run``
command line command with multiple parameters.

You can modify this script to your liking to run a grid search for your algorithm
and environment.

Taken almost without modification from the Spinning Up example script in the
`SpinningUp documentation`_.

.. _`SpinningUp documentation`: https://spinningup.openai.com/en/latest/user/running.html#using-experimentgrid
"""  # noqa
import argparse

import stable_gym  # Imports the in this example used environment  # noqa: F401
import tensorflow as tf

# Import the RL agent you want to perform the grid search for.
from stable_learning_control.algos.tf2.sac import sac
from stable_learning_control.utils.run_utils import ExperimentGrid

# Scriptparameters.
ENV_NAME = "Oscillator-v1"  # The environment on which you want to train the agent.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=1)
    args = parser.parse_args()

    # Setup Grid search parameters.
    # NOTE: Here you can add the algorithm parameters you want using their name.
    eg = ExperimentGrid(name="sac-grid-search")
    eg.add("env_name", "Oscillator-v1", "", True)
    eg.add("seed", [10 * i for i in range(args.num_runs)])
    eg.add("epochs", 100)
    eg.add("steps_per_epoch", 4000)
    eg.add("ac_kwargs:hidden_sizes", [(32,), (64, 64)], "hid")
    eg.add("ac_kwargs:activation", [tf.nn.relu, tf.nn.relu], "")

    # Run the grid search.
    eg.run(sac, num_cpu=args.cpu)
