"""Example script that shows you how you can use the Ray hyperparameter Tuner on a tf2
algorithm.

Can be used to tune the hyper parameters of any of the Machine Learning Control RL
or IL agents using the
`Ray tuning package <https://docs.ray.io/en/latest/tune/index.htm>`_.

How to use
----------

The results of using this tuner are placed in the ``./data/ray_results`` folder and can
be viewed in tensorboard using the ``tensorboard --logdir="./data/ray_results`` command.
For more information on how to use this package see the `Ray tuning documentation`_.

.. _`Ray tuning documentation`: https://docs.ray.io/en/latest/tune/index.html
"""

import os.path as osp
import sys

import gym
import numpy as np

try:
    import ray
except ImportError:
    raise ImportError(
        "The ray package appears to be missing. Did you run the `pip install .[tuning]`"
        " command?"
    )
from hyperopt import hp

# Import the algorithm we want to tune
from machine_learning_control.control.algos.pytorch.sac.sac import sac
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch


def train_sac(config):
    """Wrapper function that unpacks the config provided by the RAY tuner and converts
    them into the format the learning algorithm expects.

    Args:
        config (dict): The Ray tuning configuration dictionary.
    """

    # Unpack trainable arguments
    env_name = config.pop("env_name")

    # Run algorithm training
    sac(
        lambda: gym.make(env_name),
        **config,
    )


if __name__ == "__main__":

    # Pass system arguments to ray
    if len(sys.argv) > 1:
        ray.init(redis_address=sys.argv[1])

    # Setup the logging dir
    dirname = osp.dirname(__file__)
    log_path = osp.abspath(osp.join(dirname, "../data/ray_results"))

    # Setup hyperparameter search starting point
    current_best_params = [
        {
            "gamma": 0.995,
            "lr_a": 1e-4,
            "alpha3": 0.2,
        }
    ]

    # Setup the parameter space for you hyperparameter search
    # NOTE: This script uses the hyperopt search algorithm for efficient hyperparameter
    # selection. For more information see
    # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html?highlight=hyperopt.
    # For the options see https://github.com/hyperopt/hyperopt/wiki/FMin.
    search_space = {
        "env_name": "Oscillator-v1",
        "opt_type": "minimize",
        "gamma": hp.uniform("gamma", 0.9, 0.999),
        "lr_a": hp.loguniform("pi_lr", np.log(1e-6), np.log(1e-3)),
        "alpha3": hp.uniform("alpha3", 0.0, 1.0),
    }
    hyperopt_search = HyperOptSearch(
        search_space,
        metric="mean_ep_ret",
        mode="min",
        points_to_evaluate=current_best_params,
    )

    # Start the hyperparameter tuning job
    # NOTE: We use the ASHA job scheduler to early terminate bad trials, pause trials,
    # clone trials, and alter hyperparameters of a running trial. For more information
    # see https://docs.ray.io/en/master/tune/api_docs/schedulers.html.
    analysis = tune.run(
        train_sac,
        name="tune_sac_oscillator_1",
        num_samples=200,
        scheduler=ASHAScheduler(
            time_attr="epoch",
            metric="mean_ep_ret",
            mode="min",
            max_t=200,
            grace_period=40,
        ),
        config=search_space,
        search_alg=hyperopt_search,
        local_dir=log_path,
    )

    # Print the best trail
    best_trial = analysis.get_best_trial(metric="mean_ep_ret", mode="min", scope="all")
    best_path = analysis.get_best_logdir(metric="mean_ep_ret", mode="min", scope="all")
    best_config = analysis.get_best_config(
        metric="mean_ep_ret", mode="min", scope="all"
    )
    best_result = analysis.fetch_trial_dataframes()[best_path]["mean_ep_len"].min()
    print("The hyperparameter tuning job has finished.")
    print(f"Best trail: {best_trial}")
    print(f"Best result: {best_result}")
    print(f"Path: {best_path}")
    print(f"Best hyperparameters found were: {best_config}")
