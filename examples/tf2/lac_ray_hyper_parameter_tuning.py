"""Example script that shows you how you can use the Ray hyperparameter Tuner on a
the Tensorflow version of the Lyapunov Actor Critic (LAC) algorithm. It uses the
``Oscillator-v1`` environment in the :stable_gym:`stable_gym <>` package.

It can tune the hyperparameters of any Stable Learning Control RL agent
using the `Ray tuning package <https://docs.ray.io/en/latest/tune/index.htm>`_. This
example uses the `HyperOpt`_ search algorithm to optimize the hyperparameters search
and the `ASHA`_ scheduler to terminate bad trials, pause trials, clone trials, and alter
hyperparameters of a running trial. It was based on the `tutorial`_ provided by the
Ray team.

.. note::
    This example also contains a :cont:`USE_WANDB` flag that allows you to log the
    results of the hyperparameter search to `Weights & Biases`_ (WandB). For more
    information on how to use WandB, see the `Weights & Biases documentation`_.

How to use
----------

The results of using this tuner are placed in the ``./data/ray_results`` folder and can
be viewed in TensorBoard using the ``tensorboard --logdir="./data/ray_results`` command.
For more information on how to use this package, see the `Ray tuning documentation`_.

.. _`Ray tuning documentation`: https://docs.ray.io/en/latest/tune/index.html
.. _HyperOpt: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-hyperopt
.. _ASHA: https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-schedulers-asha
.. _tutorial: https://docs.ray.io/en/latest/tune/examples/hyperopt_example.html
.. _`Weights & Biases`: https://wandb.ai/site
.. _`Weights & Biases documentation`: https://docs.wandb.ai/
"""  # noqa: E501

import os.path as osp

import gymnasium as gym
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

# Import the algorithm we want to tune.
from stable_learning_control.algos.tf2.lac import lac

# Script parameters.
USE_WANDB = False


def train_lac(config):
    """Wrapper function that unpacks the config provided by the RAY tuner and converts
    them into the format the learning algorithm expects.

    Args:
        config (dict): The Ray tuning configuration dictionary.
    """
    # Unpack trainable arguments.
    env_name = config.pop("env_name")

    # Run algorithm training.
    lac(
        lambda: gym.make(env_name),
        **config,
    )


if __name__ == "__main__":
    # NOTE: Uncomment if you want to debug the code.
    # import ray
    # ray.init(local_mode=True)

    # Setup Weights & Biases logging.
    ray_callbacks = []
    if USE_WANDB:
        from ray.air.integrations.wandb import WandbLoggerCallback

        ray_callbacks.append(
            WandbLoggerCallback(
                job_type="tune",
                project="stable-learning-control",
                name="lac_ray_hyper_parameter_tuning_example",
            )
        )

    # Setup the logging dir.
    dirname = osp.dirname(__file__)
    log_path = osp.abspath(osp.join(dirname, "../../data/ray_results"))

    # Setup hyperparameter search starting point.
    initial_params = [{"gamma": 0.995, "lr_a": 1e-4, "alpha3": 0.2}]

    # Setup the parameter space for you hyperparameter search.
    search_space = {
        "env_name": "stable_gym:Oscillator-v1",
        "opt_type": "minimize",
        "gamma": tune.uniform(0.9, 0.999),
        "lr_a": tune.loguniform(1e-6, 1e-3),
        "alpha3": tune.uniform(0.0, 1.0),
    }

    # Initialize the hyperparameter tuning job.
    # NOTE: Available algorithm metrics are found in the `progress.csv` by the SLC CLI.
    trainable_with_cpu_gpu = tune.with_resources(
        train_lac,
        {"cpu": 12, "gpu": 1},
    )
    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="AverageEpRet",
            mode="min",  # NOTE: Should be equal to the 'opt_type'
            search_alg=HyperOptSearch(
                points_to_evaluate=initial_params,
            ),
            scheduler=ASHAScheduler(
                time_attr="epoch",
                max_t=200,
                grace_period=10,
            ),
            num_samples=20,
        ),
        run_config=air.RunConfig(
            storage_path=log_path,
            name=f"tune_lac_{search_space['env_name'].replace(':', '_')}",
            callbacks=ray_callbacks,
        ),
    )

    # Start the hyperparameter tuning job.
    results = tuner.fit()

    # Print the best trail.
    best_trial = results.get_best_trial(metric="AverageEpRet", mode="min", scope="all")
    best_path = results.get_best_logdir(metric="AverageEpRet", mode="min", scope="all")
    best_config = results.get_best_config(
        metric="AverageEpRet", mode="min", scope="all"
    )
    best_result = results.fetch_trial_dataframes()[best_path]["AverageEpLen"].min()
    print("The hyperparameter tuning job has finished.")
    print(f"Best trail: {best_trial}")
    print(f"Best result: {best_result}")
    print(f"Path: {best_path}")
    print(f"Best hyperparameters found were: {best_config}")
