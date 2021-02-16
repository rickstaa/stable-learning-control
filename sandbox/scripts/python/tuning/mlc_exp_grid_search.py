"""Script that allows you to find the right hyperparameters for you Agent and
Environment using a grid search. This is equivalent to running the
`python -m machine_learning_control.run` command line command with multiple
parameters.

You can modify this script to your liking to run a grid search for your algorithm
and environment.

Taken almost without modification from the Spinning Up example script in the
documentation `spinup_docs`_.

.. _`spinup_docs`: https://spinningup.openai.com/en/latest/user/running.html#using-experimentgrid
"""  # noqa

import argparse

import torch
from machine_learning_control.control.utils.run_utils import ExperimentGrid

# Import the RL agent you want to perform the grid search for
from machine_learning_control.control.algos import sac_pytorch

# Script parametesr
ENV_NAME = "Oscillator-v1"  # The environment on which you want to train the agent.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=1)
    args = parser.parse_args()

    # Setup Grid search parameters.
    # NOTE: Here you can add the algorithm parameters you want using their name.
    eg = ExperimentGrid(name="sac-grid-search")
    eg.add("env_name", "Oscillator-v0", "", True)
    eg.add("seed", [10 * i for i in range(args.num_runs)])
    eg.add("epochs", 100)
    eg.add("steps_per_epoch", 4000)
    eg.add("ac_kwargs:hidden_sizes", [(32,), (64, 64)], "hid")
    eg.add("ac_kwargs:activation", [torch.nn.Tanh, torch.nn.ReLU], "")

    # Run the grid search
    eg.run(sac_pytorch, num_cpu=args.cpu)
