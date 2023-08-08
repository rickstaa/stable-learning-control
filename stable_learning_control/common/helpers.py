"""Contains several helper functions that are used throughout the SLC package."""
import itertools
import re
import string
from collections.abc import Iterable, MutableMapping

import gymnasium as gym
import numpy as np
import torch


def atleast_2d(array, axis=1):
    """Similar to :meth:`numpy.atleast_2d` but with an additional ``axis`` argument
    which can be used to specify where the extra dimension should be-added.

    Args:
        array (numpy.ndarray): [description]
        axis (int, optional): Position in the expanded axes where the new axis (or axes)
            is placed if the dimension is smaller than 2. Defaults to ``1``.

    Returns:
        numpy.ndarray: The 2D numpy array.
    """
    # IMPROVE: Can be replaced with numpy.atleast_2d when
    # https://github.com/numpy/numpy/pull/18386 is merged.
    if array.ndim < 2:
        array = np.expand_dims(array, axis=axis)
    return array


def convert_to_tuple(input_var):
    """Converts input into a tuple.

    Args:
        input_arg (Union[int, float, list]): A input variable.

    Returns:
        tuple: A tuple.
    """

    return (
        input_var
        if isinstance(input_var, tuple)
        else tuple(input_var)
        if isinstance(input_var, list)
        else (input_var,)
    )


def flatten(items):
    """Flatten a list with any nested iterable.

    Args:
        items (list): A nested list.

    Returns:
        list: A flattened version of the list.
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def get_unique_list(input_list, trim=True):
    """Removes non-unique items from a list.

    Args:
        input_list (list): The input list.
        trim (list, optional): Trim empty items. Defaults to ``True``.

    Returns:
        list: The new list containing only unique items.
    """
    if trim:
        return list({item for item in input_list if item != ""})
    else:
        return list({item for item in input_list})


def combine_shapes(*args):
    """Combines multiple tuples/ints/floats into one tuple.

    Args:
        *args (Union[tuple,int,float]): Input arguments to combine

    Returns:
        Tuple: A tuple in which al the input arguments are combined.
    """
    return tuple(
        itertools.chain(
            *[[item] if isinstance(item, (int, float)) else list(item) for item in args]
        )
    )


def strict_dict_update(input_dict, update_obj):
    """Updates a dictionary with values supplied in another :obj:`dict` or python
    :class:`object`. This function performs a strict update, meaning it does not add
    new keys to the original dictionary.Additionally, if a Python object is supplied, it
    will be applied to all keys in the dictionary.

    Args:
        input_dict (dict): The input dictionary.
        update_dict (Union[dict, list]): Dictionary or list containing the update
            values.

    Returns:
        (tuple): tuple containing:

            - input_dict(:obj:`list`): The new updated dictionary.
            - ignored (:obj:`str`): The ignored keys.
    """
    ignored = []
    if isinstance(update_obj, dict):
        for key, val in update_obj.items():
            if key in input_dict.keys():
                input_dict[key] = val
            else:
                ignored.append(key)  # Ignore keys that are not in the dictionary.
    else:  # Apply object to all keys in the dictionary.
        for key in input_dict:
            input_dict[key] = update_obj
    return input_dict, ignored


def valid_str(v):
    """Convert a value or values to a string which could go in a filepath.

    .. note::
        Partly based on `this gist`_.

        .. _`this gist`: https://gist.github.com/seanh/93666

    Args:
        v (list): List with values.
    """
    if hasattr(v, "__name__"):
        return valid_str(v.__name__)

    if isinstance(v, tuple) or isinstance(v, list):
        return "-".join([valid_str(x) for x in v])

    # Valid characters are '-', '_', and alphanumeric. Replace invalid chars
    # with '-'.
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = "".join(c if c in valid_chars else "-" for c in str_v)
    return str_v


def all_bools(vals):
    """Check if list contains only strings.

    Args:
        vals (list): List with values.

    Returns:
        bool: Boolean specifying the result.
    """
    return all([isinstance(v, bool) for v in vals])


def is_scalar(obj):
    """Recursive function that checks whether a input

    Args:
        obj (object): Object for which you want to check if it is a scalar.

    Returns:
        boole: Boolean specifying whether the object is a scalar.
    """
    if isinstance(obj, (int, float)):
        return True
    elif np.isscalar(obj):
        if isinstance(obj, str):
            try:
                float(obj)
                return True
            except ValueError:
                return False
        else:
            return True
    elif isinstance(obj, np.array):
        if obj.shape == (1,):
            return is_scalar(obj[0])
        else:
            return False
    elif isinstance(obj, torch.Tensor):
        if len(obj.shape) == 0:
            try:
                float(obj)
                return True
            except ValueError:
                return False
        elif len(obj.shape) <= 1:
            if obj.shape[0] <= 1:
                return is_scalar(obj[0])
            else:
                return False
        else:
            return False
    elif isinstance(obj, str):
        try:
            float(obj)
            return True
        except ValueError:
            return False
    else:
        return False


def get_env_id(env):
    """Returns the environment id of a given environment.

    Args:
        env (:obj:`gym.Env`): The environment.

    Returns:
        str: The environment id.
    """
    if isinstance(env, gym.Env):
        return (
            env.unwrapped.spec.id
            if hasattr(env.unwrapped.spec, "id")
            else type(env.unwrapped).__name__
        )
    return env


def get_env_class(env):
    """Get the environment class.

    Args:
        env (:obj:`gym.Env`): The environment.

    Returns:
        str: The environment class.
    """
    if isinstance(env, gym.Env):
        return "{}.{}".format(
            env.unwrapped.__module__, env.unwrapped.__class__.__name__
        )
    return env


def parse_config_env_key(config):
    """Replace environment objects (i.e. gym.Env) with their id and class path if they
    are present in the config. Also removes the 'env_fn' from the config.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: The parsed configuration dictionary.
    """
    parsed_config = {}
    for key, val in config.items():
        if key == "env" and isinstance(val, gym.Env):
            parsed_config[key] = get_env_id(val)
            parsed_config["env_class"] = get_env_class(val)
        elif key == "env_fn":  # Remove env_fn from config.
            continue
        else:
            parsed_config[key] = val
    return parsed_config


def convert_to_snake_case(input_str):
    """Converts a string from camel/pascal case to snake case.

    Args:
        input_str (str): The input string.

    Returns:
        str: The converted string.
    """
    return re.sub(
        "([a-z0-9])([A-Z])",
        r"\1_\2",
        re.sub("(.)([A-Z][a-z]+)", r"\1_\2", input_str),
    ).lower()


def friendly_err(err_msg, prepend=True, append=True):
    """Add whitespace line to error message to make it more readable.

    Args:
        err_msg (str): Error message.
        prepend (bool, optional): whether to prepend empty whitespace line before the
            string. Defaults to ``True``.
        append (bool, optional): Whether to append empty whitespace line after the
            string. Defaults to ``True``.

    Returns:
        str: Error message with extra whitespace line.
    """
    return ("\n\n" if prepend else "") + err_msg + ("\n\n" if append else "")


def flatten_dict(d, parent_key="", sep="."):
    """Flattens a nested dictionary.

    Args:
        d (dict): The input dictionary.
        parent_key (str, optional): The parent key. Defaults to ``""``.
        sep (str, optional): The separator. Defaults to ``"."``.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_to_wandb_config(config):
    """Transform the config to a format that looks better on Weights & Biases.

    Args:
        config (dict): The config that should be transformed.

    Returns:
        dict: The transformed config.
    """
    wandb_config = {}
    for key, value in config.items():
        if (
            key
            in [
                "env_fn",
                "output_dir",
                "use_wandb",
                "wandb_job_type",
                "wandb_project",
                "wandb_group",
                "wandb_run_name",
            ]
            or value is None
        ):  # Filter keys.
            continue
        elif key in ["policy", "disturber"]:  # Transform policy object to policy id.
            value = "{}.{}".format(value.__module__, value.__class__.__name__)
        elif key == "env" and isinstance(value, gym.Env):
            wandb_config["env_class"] = get_env_class(value)
            value = get_env_id(value)
        wandb_config[key] = value
    return wandb_config


def convert_to_tb_config(config):
    """Transform the config to a format that looks better on TensorBoard.

    Args:
        config (dict): The config that should be transformed.

    Returns:
        dict: The transformed config.
    """
    tb_config = {}
    for key, value in config.items():
        if key in ["env_fn"]:  # Skip env_fn.
            continue
        elif key == "env" and isinstance(value, gym.Env):
            tb_config["env_class"] = get_env_class(value)
            value = get_env_id(value)
        tb_config[key] = value
    return flatten_dict(tb_config)
