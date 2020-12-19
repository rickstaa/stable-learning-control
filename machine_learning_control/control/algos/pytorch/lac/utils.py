import copy
from collections import OrderedDict

import numpy as np
import torch.nn as nn

from variant import ENV_PARAMS, TRAIN_PARAMS


def get_env_from_name(name, ENV_SEED):
    """Initialises the gym environment with the given name

    Args:
        name (str): The name of the gym environment you want to initialise.

    Returns:
        gym.Env: The gym environment.
    """
    if name == "oscillator":
        from envs.oscillator import oscillator as env

        env = env()
        env = env.unwrapped
    elif name == "Ex3_EKF":
        from envs.Ex3_EKF import Ex3_EKF as env

        env = env()
        env = env.unwrapped
    elif name == "Ex4_EKF":
        from envs.Ex4_EKF import Ex4_EKF as env

        env = env()
        env = env.unwrapped
    if ENV_SEED is not None:
        env.seed(ENV_SEED)
    return env


def evaluate_training_rollouts(paths):
    """Evaluate the performance of the policy in the training rollouts."""
    data = copy.deepcopy(paths)
    if len(data) < 1:
        return None
    try:
        diagnostics = OrderedDict(
            (
                ("return", np.mean([np.sum(path["rewards"]) for path in data])),
                ("length", np.mean([len(p["rewards"]) for p in data])),
            )
        )
    except KeyError:
        return
    [path.pop("rewards") for path in data]
    for key in data[0].keys():
        result = [np.mean(path[key]) for path in data]
        diagnostics.update({key: np.mean(result)})

    return diagnostics


def training_evaluation(env, policy):
    """Evaluates the performance of the current policy in
    several test rollouts.

    Args:
        env (gym.Env): The gym environment you want to use.
        policy (object): The current policy.

    Returns:
        [type]: [description]
    """
    # Retrieve action space bounds from env
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    # Training setting
    total_cost = []
    episode_length = []
    die_count = 0
    seed_average_cost = []

    # Perform roolouts to evaluate performance
    for i in range(TRAIN_PARAMS["num_of_evaluation_paths"]):
        cost = 0
        s = env.reset()
        for j in range(ENV_PARAMS["max_ep_steps"]):
            if ENV_PARAMS["eval_render"]:
                env.render()
            a = policy.choose_action(s, True)
            action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2
            s_, r, done, _ = env.step(action)
            cost += r
            if j == ENV_PARAMS["max_ep_steps"] - 1:
                done = True
            s = s_
            if done:
                seed_average_cost.append(cost)
                episode_length.append(j)
                if j < ENV_PARAMS["max_ep_steps"] - 1:
                    die_count += 1
                break

    # Save evaluation results
    total_cost.append(np.mean(seed_average_cost))
    total_cost_mean = np.average(total_cost)
    average_length = np.average(episode_length)

    # Return evaluation results
    diagnostic = {
        "return": total_cost_mean,
        "average_length": average_length,
    }
    return diagnostic


def mlp(sizes, activation, output_activation=nn.Identity):
    """Create a multi-layered perceptron using pytorch.

    Args:
        sizes (list): The size of each of the layers.

        activation (torch.nn.modules.activation): The activation function used for the
            hidden layers.

        output_activation (torch.nn.modules.activation, optional): The activation
            function used for the output layers. Defaults to torch.nn.Identity.

    Returns:
        torch.nn.modules.container.Sequential: The multi-layered perceptron.
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)
