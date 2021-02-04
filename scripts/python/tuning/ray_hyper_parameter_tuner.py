"""Script that can be used to tune the hyper parameters of any of the Machine Learning
Control RL agents using the`ray tuning package`_. For more information on this package
see the `Ray tuning documentation`_.

The results of this tuning is placed in the ./data/ray_results folder and can be viewed
in tensorboard using the `tensorboard --logdir="./data/ray_results` command.

.. `ray tuning package`: https://docs.ray.io/en/latest/tune/index.html
.. `ray tuning documentation`: https://docs.ray.io/en/latest/tune/index.html
"""

import os.path as osp
import sys

import numpy as np

try:
    import ray
except ImportError:
    raise ImportError(
        "The ray package appears to be missing. Did you run the `pip install .[tuning]`"
        " command?"
    )
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
from hyperopt.pyll.base import scope

# Import the algorithm we want to tune
from machine_learning_control.control.algos.pytorch.sac.sac import train_sac


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
            "gamma": 0.997,
            "lam": 0.9971,
            "pi_lr": 8.2599e-4,
            "vf_lr": 1.5316e-4,
            "labda_lr": 7.0384e-6,
            "sigma": 0.492,
            "alpha3": 0.21087,
            "c_ba": 0.60526,
            "traj_per_epoch": 39,
        }
    ]

    # Setup the parameter space for you hyperparameter search
    # NOTE (rickstaa): This script uses the hyperopt search algorithm for efficient
    # hyperparameter selection. For more information see
    # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html?highlight=hyperopt.
    # For the options see https://github.com/hyperopt/hyperopt/wiki/FMin.
    search_space = {
        "env_name": "Oscillator-v0",
        "opt_type": "minimize",
        "gamma": hp.uniform("gamma", 0.9, 0.999),
        "lam": hp.uniform("lam", 0.9, 0.99),
        "pi_lr": hp.loguniform("pi_lr", np.log(1e-6), np.log(1e-3)),
        "vf_lr": hp.loguniform("vf_lr", np.log(1e-6), np.log(1e-3)),
        "labda_lr": hp.loguniform("labda_lr", np.log(1e-6), np.log(1e-3)),
        "sigma": hp.uniform("sigma", 0.0, 1.0),
        "alpha3": hp.uniform("alpha3", 0.0, 1.0),
        "c_ba": hp.uniform("c_ba", 0.0, 1.0),
        "traj_per_epoch": scope.int(hp.randint("traj_per_epoch", 5, 50)),
    }
    hyperopt_search = HyperOptSearch(
        search_space,
        metric="mean_ep_ret",
        mode="min",
        points_to_evaluate=current_best_params,
    )

    # Start the hyperparameter tuning job
    # NOTE (rickstaa): We use the ASHA job scheduler to  early terminate bad trials,
    # pause trials, clone trials, and alter hyperparameters of a running trial. For
    # more information see https://docs.ray.io/en/master/tune/api_docs/schedulers.html.
    analysis = tune.run(
        train_sac,
        name="tune_elpg_oscillator_1",
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
