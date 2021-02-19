"""A set of functions that can be used to see a algorithm perform in the environment
it was trained on.
"""

import glob
import os
import os.path as osp
import time

import joblib
import machine_learning_control.control.utils.log_utils as log_utils
import torch
from machine_learning_control.control.utils import import_tf
from machine_learning_control.control.utils.log_utils.logx import EpochLogger
from machine_learning_control.control.utils.serialization_utils import load_from_json


def load_policy_and_env(fpath, itr="last"):
    """Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Args:
        fpath (str): The path where the model is found.
        itr (str, optional): The current policy iteration. Defaults to "last".
        deterministic (bool, optional): Whether you want the action from the policy to
            be deterministic. Defaults to ``False``.

    Returns:
        (tuple): tuple containing:

            env(gym.env): The gym environment.
            get_action (func): The policy get_action function.
    """

    # determine if tf save or pytorch save
    if any(["tf_save" in x for x in os.listdir(fpath)]):
        backend = "tf"
    else:
        backend = "pytorch"

    # handle which epoch to load from
    if itr == "last":
        if backend == "tf":
            tf_save_path = osp.join(fpath, "tf_save")
            saves = [
                int(osp.basename(item).split(".")[0][18:])
                for item in glob.glob(
                    osp.join(tf_save_path, "weights_checkpoint*.index")
                )
                if len(osp.basename(item).split(".")[0]) > 18
            ]
        elif backend == "pytorch":
            torch_save_path = osp.join(fpath, "torch_save")
            saves = [
                int(osp.basename(item).split(".")[0][11:])
                for item in glob.glob(osp.join(torch_save_path, "model_state*.pt"))
                if len(osp.basename(item).split(".")[0]) > 11
            ]
        itr = "%d" % max(saves) if len(saves) > 0 else ""

    else:
        assert isinstance(
            itr, int
        ), "Bad value provided for itr (needs to be int or 'last')."
        itr = "%d" % itr

    # try to load environment from save
    # NOTE: Sometimes this will fail because the environment could not be pickled.
    try:
        state = joblib.load(osp.join(fpath, "vars" + itr + ".pkl"))
        env = state["env"]
    except Exception:
        env = None

    # load the get_action function
    if backend == "tf":
        policy = load_tf_policy(fpath, itr, env)
    else:
        policy = load_pytorch_policy(fpath, itr, env)

    return env, policy


def load_tf_policy(fpath, itr, env=None):
    """Load a tensorflow policy saved with Machine Learning Control Logger.

    Args:
        fpath (str): The path where the model is found.
        itr (str, optional): The current policy iteration. Defaults to "last".
        env (gym.env): The gym environment in which you want to test the policy.

    Returns:
        (tf.nn.Module): The policy.
    """
    tf = import_tf()  # Import tf if installed otherwise throw warning
    fname = osp.join(fpath, "tf_save" + itr)
    print("\n")
    log_utils.log("Loading from %s.\n\n" % fname, type="info")

    # Retrieve get_action method
    save_info = load_from_json(osp.join(fname, "save_info.json"))
    import machine_learning_control.control.algos.tf2 as tf2_algos

    model = getattr(tf2_algos, save_info["class_name"])(env=env)
    latest = tf.train.latest_checkpoint(fname)  # Restore latest checkpoint
    model.load_weights(latest)
    # return model.get_action
    return model


def load_pytorch_policy(fpath, itr, env=None):
    """Load a pytorch policy saved with Machine Learning Control Logger.

    Args:
        fpath (str): The path where the model is found.
        itr (str, optional): The current policy iteration. Defaults to "last".
        env (gym.env): The gym environment in which you want to test the policy.

    Returns:
        (tf.keras.Model): The policy.
    """
    fname = osp.join(
        fpath,
        "torch_save",
        "model_state" + itr + ".pt",
    )
    log_utils.log("\n\nLoading from %s.\n\n" % fname, type="info")
    model_data = torch.load(fname)

    # Retrieve get_action method
    import machine_learning_control.control.algos.pytorch as torch_algos

    model = getattr(torch_algos, model_data["class_name"])(env=env)
    model.load_state_dict(model_data)  # Retore model parameters
    return model


def run_policy(
    env, get_action, max_ep_len=None, num_episodes=100, render=True, deterministic=False
):
    """Evaluates a policy inside a given gym environment.

    Args:
        env gym.env): The gym environment.
        policy (union[tf.keras.Model, torch.nn.Module]): The policy.
        max_ep_len (int, optional): The maximum episode length. Defaults to None.
        num_episodes (int, optional): Number of episodes you want to perform in the
            environment. Defaults to 100.
        deterministic (bool, optional): Whether you want the action from the policy to
            be deterministic. Defaults to ``False``.
        render (bool, optional): Whether you want to render the episode to the screen.
            Defaults to ``True``.
    """
    assert env is not None, (
        "Environment not found!\n\n It looks like the environment wasn't saved, "
        + "and we can't run the agent in it. :( \n\n Check out the readthedocs "
        + "page on Experiment Outputs for how to handle this situation."
    )

    logger = EpochLogger(verbose_fmt="table")
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    supports_deterministic = True  # Only supported with gaussian algorithms
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

            if supports_deterministic:
                try:
                    a = policy.get_action(o, deterministic=deterministic)
                except TypeError:
                    log_utils.log(
                        "Input argument 'deterministic' ignored as the algorithm does "
                        "not support deterministic actions. This is only supported for "
                        "gaussian  algorithms.",
                        type="warning",
                    )
                    a = policy.get_action(o)
            else:
                a = policy.get_action(o)
            supports_deterministic = False
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            logger.log("Episode %d \t EpRet %.3f \t EpLen %d" % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    print("")
    logger.log_tabular("EpRet", with_min_and_max=True)
    logger.log_tabular("EpLen", average_only=True)
    logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str)
    parser.add_argument("--len", "-l", type=int, default=0)
    parser.add_argument("--episodes", "-n", type=int, default=100)
    parser.add_argument("--norender", "-nr", action="store_true")
    parser.add_argument("--itr", "-i", type=int, default=-1)
    parser.add_argument("--deterministic", "-d", action="store_true")
    args = parser.parse_args()
    env, policy = load_policy_and_env(
        args.fpath, args.itr if args.itr >= 0 else "last"
    )
    run_policy(env, policy, args.len, args.episodes, not (args.norender))
