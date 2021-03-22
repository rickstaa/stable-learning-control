"""A set of functions that can be used to see a algorithm perform in the environment
it was trained on.
"""

import glob
import os
import os.path as osp
from pathlib import Path
import time

import joblib
import torch
from bayesian_learning_control.utils.import_utils import import_tf
from bayesian_learning_control.utils.serialization_utils import load_from_json
from bayesian_learning_control.utils.log_utils import EpochLogger, log_to_std_out


def _retrieve_iter_folder(fpath, itr):
    """Retrieves the path of the requested model iteration.

    Args:
        fpath (str): The path where the model is found.
        itr (int): The current policy iteration (checkpoint).

    Raises:
        IOError: Raised if the model is corrupt.
        FileNotFoundError: Raised if the model path did not exist.

    Returns:
        str: The model iteration path.
    """
    cpath = Path(fpath).joinpath("checkpoints", f"iter{itr}")
    if osp.isdir(cpath):
        log_to_std_out(f"Using model iteration {itr}.", type="info")
        return cpath
    else:
        log_to_std_out(
            f"Iteration {itr} not found inside the supplied model path '{fpath}'. "
            "Last iteration used instead.",
            type="warning",
        )
        return fpath


def _retrieve_model_folder(fpath):
    """Tries to retrieve the model folder and backend from the given path.

    Args:
        fpath (str): The path where the model is found.

    Raises:
        IOError: Raised if the model is corrupt.
        FileNotFoundError: Raised if the model path did not exist.

    Returns:
        (tuple): tuple containing:

            - model_folder (:obj:`func`): The model folder.
            - backend (:obj:`str`): The inferred backend. Options are ``tf`` and
                ``torch``.
    """
    data_folders = glob.glob(fpath + r"/*_save")
    if ("tf2_save" in fpath and "torch_save" in fpath) or len(data_folders) > 1:
        raise IOError(
            "Policy could not be loaded as the model as the specified model folder "
            f"'{fpath}' seems to be corrupted. It contains both a 'torch_save' and "
            "'tf2_save' folder. Please check your model path (fpath) and try again."
        )
    elif "tf2_save" in fpath:
        model_path = os.sep.join(
            fpath.split(os.sep)[: fpath.split(os.sep).index("tf2_save") + 1]
        )
        return model_path, "tf"
    elif "torch_save" in fpath:
        model_path = os.sep.join(
            fpath.split(os.sep)[: fpath.split(os.sep).index("torch_save") + 1]
        )
        return model_path, "torch"
    else:  # Check
        if len(data_folders) == 0:
            raise FileNotFoundError(
                f"No model was found inside the supplied model path '{fpath}'. Please "
                "check your model path (fpath) and try again."
            )
        else:
            return data_folders[0], (
                "torch" if "torch_save" in data_folders[0] else "tf"
            )


def load_policy_and_env(fpath, itr="last"):
    """Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Args:
        fpath (str): The path where the model is found.
        itr (str, optional): The current policy iteration (checkpoint). Defaults to
            ``last``.
        deterministic (bool, optional): Whether you want the action from the policy to
            be deterministic. Defaults to ``False``.

    Returns:
        (tuple): tuple containing:

            - env (:obj:`gym.env`): The gym environment.
            - get_action (:obj:`func`): The policy get_action function.
    """
    if not os.path.isdir(fpath):
        raise FileNotFoundError(
            f"The model folder you specified '{fpath}' does not exist. Please specify "
            "a valid model folder (fpath) and try again."
        )

    # Retrieve model path and backend
    fpath, backend = _retrieve_model_folder(fpath)

    if itr != "last":
        assert isinstance(
            itr, int
        ), "Bad value provided for itr (needs to be int or 'last')."
        itr = "%d" % itr

    # try to load environment from save
    # NOTE: Sometimes this will fail because the environment could not be pickled.
    try:
        state = joblib.load(Path(fpath).parent.joinpath("vars.pkl"))
        env = state["env"]
    except Exception:
        env = None

    # load the get_action function
    if backend == "tf":
        policy = load_tf_policy(fpath, itr, env)
    else:
        policy = load_pytorch_policy(fpath, itr, env)

    return env, policy


def load_tf_policy(fpath, itr="last", env=None):
    """Load a tensorflow policy saved with Bayesian Learning Control Logger.

    Args:
        fpath (str): The path where the model is found.
        itr (str, optional): The current policy iteration. Defaults to "last".
        env (:obj:`gym.env`): The gym environment in which you want to test the policy.

    Returns:
        tf.keras.Model: The policy.
    """
    if itr != "last":
        model_path = _retrieve_iter_folder(fpath, itr)
    else:
        model_path = fpath
    tf = import_tf()  # Import tf if installed otherwise throw warning
    print("\n")
    log_to_std_out("Loading model from '%s'.\n\n" % fpath, type="info")

    # Retrieve get_action method
    save_info = load_from_json(Path(fpath).joinpath("save_info.json"))
    import bayesian_learning_control.control.algos.tf2 as tf2_algos

    model = getattr(tf2_algos, save_info["alg_name"])(env=env)
    latest = tf.train.latest_checkpoint(model_path)  # Restore latest checkpoint
    model.load_weights(latest)

    # return model.get_action
    return model


def load_pytorch_policy(fpath, itr="last", env=None):
    """Load a pytorch policy saved with Bayesian Learning Control Logger.

    Args:
        fpath (str): The path where the model is found.
        itr (str, optional): The current policy iteration. Defaults to "last".
        env (:obj:`gym.env`): The gym environment in which you want to test the policy.

    Returns:
        torch.nn.Module: The policy.
    """

    if itr != "last":
        fpath = _retrieve_iter_folder(fpath, itr)
    model_file = Path(fpath).joinpath(
        "model_state.pt",
    )
    print("\n")
    log_to_std_out("Loading model from '%s'.\n\n" % model_file, type="info")

    # Retrieve get_action method
    import bayesian_learning_control.control.algos.pytorch as torch_algos

    model_data = torch.load(model_file)
    model = getattr(torch_algos, model_data["alg_name"])(env=env)
    model.load_state_dict(model_data)  # Retore model parameters
    return model


def run_policy(
    env, policy, max_ep_len=None, num_episodes=100, render=True, deterministic=False
):
    """Evaluates a policy inside a given gym environment.

    Args:
        env (:obj:`gym.env`): The gym environment.
        policy (Union[tf.keras.Model, torch.nn.Module]): The policy.
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
    # TODO: What happended here?

    logger = EpochLogger(verbose_fmt="table")
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    supports_deterministic = True  # Only supported with gaussian algorithms
    render_error = False
    while n < num_episodes:
        # Render env if requested
        if render and not render_error:
            try:
                env.render()
                time.sleep(1e-3)
            except NotImplementedError:
                render_error = True
                log_to_std_out(
                    (
                        "WARNING: Nothing was rendered since no render method was "
                        f"implemented for the '{env.unwrapped.spec.id}' environment."
                    ),
                    type="warning",
                )

        # Retrieve action
        if deterministic and supports_deterministic:
            try:
                a = policy.get_action(o, deterministic=deterministic)
            except TypeError:
                supports_deterministic = False
                log_to_std_out(
                    "Input argument 'deterministic' ignored as the algorithm does "
                    "not support deterministic actions. This is only supported for "
                    "gaussian  algorithms.",
                    type="warning",
                )
                a = policy.get_action(o)
        else:
            a = policy.get_action(o)

        # Perform action in the environment and store result
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

    # TODO: Add help
    # TODO: Test for multiple policies!
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str)
    parser.add_argument("--len", "-l", type=int, default=0)
    parser.add_argument("--episodes", "-n", type=int, default=100)
    parser.add_argument("--norender", "-nr", action="store_true")
    parser.add_argument("--itr", "-i", type=int, default=-1)
    parser.add_argument("--deterministic", "-d", action="store_true")
    args = parser.parse_args()
    env, policy = load_policy_and_env(args.fpath, args.itr if args.itr >= 0 else "last")
    run_policy(env, policy, args.len, args.episodes, not (args.norender))

# TODO: Add warning for when something goes wrong!
