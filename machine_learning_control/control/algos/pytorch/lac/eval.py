"""Evaluate trained LAC agent."""

import argparse
import os
import sys

import gym

# Cuda Settings
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import machine_learning_control.simzoo.simzoo.envs

# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import torch

import logger
from utils import get_env_from_name
from variant import ALG_PARAMS, ENV_NAME, ENV_PARAMS, ENV_SEED, EVAL_PARAMS

# from machine_learning_control.control.algos.lac.lac import LAC


def get_disturbance_function(env_name):
    """Retrieve disturbance function for a given environment.

    Args:
        env_name (str): Environment you want to use.

    Raises:
        NameError: If disturbance does not exist for the given environment.

    Returns:
        object: Disturbance function.
    """
    if "oscillator" in env_name:
        disturbance_step = oscillator_disturber
    elif "Ex3_EKF" in env_name:
        disturbance_step = Ex3_EKF_disturber
    elif "Ex4_EKF" in env_name:
        disturbance_step = Ex4_EKF_disturber
    else:
        print("no disturber designed for " + env_name)
        raise NameError
    return disturbance_step


def oscillator_disturber(time, s, action, env, eval_params, disturber=None):
    """Disturbance function used for evaluating the oscillator."""
    d = np.zeros_like(action)
    s_, r, done, info = env.step(action + d)
    done = False
    return s_, r, done, info


def Ex3_EKF_disturber(time, s, action, env, eval_params, disturber=None):
    """Disturbance function used for evaluating the Ex3_EKF environment."""
    d = np.zeros_like(action)
    s_, r, done, info = env.step(action + d)
    done = False
    return s_, r, done, info


def Ex4_EKF_disturber(time, s, action, env, eval_params, disturber=None):
    """Disturbance function used for evaluating the Ex3_EKF environment."""
    d = np.zeros_like(action)
    s_, r, done, info = env.step(action + d)
    done = False
    return s_, r, done, info


def dynamic(policy_path, env_name, env_params, alg_params, eval_params):
    """Performs dynamic robustness evaluation.

    Args:
        policy_path (str): Log path.
        env_name (str): The gym environment you want to use.
        alg_params (dict): Dictionary containing the algorithm parameters.
    """

    # Retrieve environment
    env = gym.make(env_name)

    # Get trained policy
    # s_dim = env.observation_space.shape[0]
    # a_dim = env.action_space.shape[0]
    policy = torch.load(policy_path + "/pyt_save/model.pt")

    # Configure logger
    log_path = policy_path + "/eval/dynamic/" + eval_params["additional_description"]
    eval_params.update({"magnitude": 0})
    logger.configure(dir=log_path, format_strs=["csv"])

    # Evaluate policy results
    _, paths = evaluation(policy_path, env_name, env, env_params, eval_params, policy)
    max_len = 0
    print(len(paths))
    for path in paths["s"]:
        path_length = len(path)
        if path_length > max_len:
            max_len = path_length
    average_path = np.average(np.array(paths["s"]), axis=0)
    std_path = np.std(np.array(paths["s"]), axis=0)
    for i in range(max_len):
        logger.logkv("average_path", average_path[i])
        logger.logkv("std_path", std_path[i])
        logger.logkv("reference", paths["reference"][0][i])
        logger.dumpkvs()
    if eval_params["directly_show"]:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        if eval_params["plot_average"]:
            t = range(max_len)
            ax.plot(t, average_path, color="red")
            ax.fill_between(
                t,
                average_path - std_path,
                average_path + std_path,
                color="red",
                alpha=0.1,
            )
            plt.show()
        else:
            for path in paths["s"]:
                path_length = len(path)
                print(path_length)
                t = range(path_length)
                path = np.array(path)

                # Ex3_EKF
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(t, path, color="red")
                plt.show()
                ax.plot(t, np.array(path), color="blue", label="0.1")
                plt.show()


def evaluation(
    policy_path, env_name, env, env_params, eval_params, policy, disturber=None
):

    # Retrieve disturber and action space dimention
    disturbance_step = get_disturbance_function(env_name)
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    # Training setting
    total_cost = []
    death_rates = []
    trial_list = os.listdir(policy_path)
    episode_length = []
    cost_paths = []
    value_paths = []
    state_paths = []
    ref_paths = []

    # Evalute policy in several rollouts
    print(trial_list)

    # Check if agent is present
    if len(trial_list) == 1:
        print(
            "The agent you specified for evaluation does not exist please check the "
            "'eval_list' parameter."
        )
        sys.exit(0)

    # Loop through agents
    for trial in trial_list:
        if trial == "eval":
            continue
        if trial not in eval_params["trials_for_eval"]:
            continue
        success_load = policy.restore(os.path.join(policy_path, trial) + "/policy")
        if not success_load:
            continue
        die_count = 0
        seed_average_cost = []
        for i in range(
            int(np.ceil(eval_params["num_of_paths"] / (len(trial_list) - 1)))
        ):
            path = []
            state_path = []
            value_path = []
            ref_path = []
            cost = 0
            s = env.reset()
            global initial_pos
            initial_pos = np.random.uniform(0.0, np.pi, size=[a_dim])
            for j in range(env_params["max_ep_steps"]):

                if env_params["eval_render"]:
                    env.render()
                a = policy.choose_action(s, True)
                action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2
                s_, r, done, info = disturbance_step(j, s, action, env, eval_params)
                path.append(r)
                cost += r
                if "reference" in info.keys():
                    ref_path.append(info["reference"])
                if "state_of_interest" in info.keys():
                    state_path.append(info["state_of_interest"])
                if j == env_params["max_ep_steps"] - 1:
                    done = True
                s = s_
                if done:
                    seed_average_cost.append(cost)
                    episode_length.append(j)
                    if j < env_params["max_ep_steps"] - 1:
                        die_count += 1
                    break
            cost_paths.append(path)
            value_paths.append(value_path)
            state_paths.append(state_path)
            ref_paths.append(ref_path)
        death_rates.append(die_count / (i + 1) * 100)
        total_cost.append(np.mean(seed_average_cost))
    total_cost_std = np.std(total_cost, axis=0)
    total_cost_mean = np.average(total_cost)
    death_rate = np.mean(death_rates)
    death_rate_std = np.std(death_rates, axis=0)
    average_length = np.average(episode_length)
    diagnostic = {
        "return": total_cost_mean,
        "return_std": total_cost_std,
        "death_rate": death_rate,
        "death_rate_std": death_rate_std,
        "average_length": average_length,
    }
    path_dict = {"c": cost_paths, "v": value_paths}
    if "reference" in info.keys():
        path_dict.update({"reference": ref_paths})
    if "state_of_interest" in info.keys():
        path_dict.update({"s": state_paths})
    return diagnostic, path_dict


###############################################
# Main function ###############################
###############################################
if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the LAC agent in a given environment."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=EVAL_PARAMS["eval_list"],
        help="The name of the model you want to evaluate.",
    )
    args = parser.parse_args()

    # Evaluate robustness
    eval_agents = (
        [args.model_name] if not isinstance(args.model_name, list) else args.model_name
    )
    for name in eval_agents:
        dirname = os.path.dirname(__file__)
        LOG_PATH = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                f"../../../../data/lac/{ENV_NAME.lower()}/runs/" + name,
            )
        )
        print("evaluating " + name)
        dynamic(LOG_PATH, ENV_NAME, ENV_PARAMS, ALG_PARAMS, EVAL_PARAMS)
        # tf.compat.v1.reset_default_graph()
