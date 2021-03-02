"""Functions that are used in multiple Pytorch and Tensorflow algorithms.
"""

import importlib

import numpy as np
import torch  # noqa:F401
from machine_learning_control.control.utils.gym_utils import (
    is_continuous_space,
    is_discrete_space,
)
from machine_learning_control.control.utils.import_tf import import_tf

tf = import_tf(frail=False)
tensorflow = tf


def heuristic_target_entropy(action_space):
    """Returns a heuristic target entropy for a given action space using the method
    explained in `Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.

    Args:
        action_space (:obj:`:obj:`gym.spaces``): The action space.

    Raises:
        NotImplementedError: If no heuristic target entropy has yet been implemented
            for the given action space.

    Returns:
        :obj:`numpy.int64`: The target entropy.
    """
    if is_continuous_space(action_space):
        heuristic_target_entropy = -np.prod(
            action_space.shape
        )  # Maximum information (bits) contained in action space
    elif is_discrete_space(action_space):
        raise NotImplementedError(
            "The heuristic target entropy is not yet implement for discrete spaces."
            "Please open a feature/pull request on "
            "https://github.com/rickstaa/machine-learning-control/issues if you need "
            "this."
        )
    else:
        raise NotImplementedError(
            "The heuristic target entropy is not yet implement for "
            f"{type(action_space)} action spaces. Please open a feature/pull request"
            "on https://github.com/rickstaa/machine-learning-control/issues if you"
            "need this."
        )
    return heuristic_target_entropy


def get_activation_function(activation_fn_name, backend="torch"):
    """Get a given torch activation function.

    Args:
        activation_fn_name (str): The name of the activation function you want to
            retrieve.
        backend (str): The machine learning backend you want to use. By default
            ``None``, meaning no backend is assumed.

    Raises:
        ValueError: Thrown if the activation function does not exist withing the
            backend.

    Returns:
        :obj:`torch.nn.modules.activation`: The torch activation function.
    """
    if backend.lower() in ["tf", "tf2", "tensorflow"]:
        backend_prefix = ["tensorflow", "nn"]
    else:
        backend_prefix = ["torch", "nn"]

    # Retrieve activation function
    if len(activation_fn_name.split(".")) == 1:
        activation_fn_name = ".".join(backend_prefix) + "." + activation_fn_name
    elif len(activation_fn_name.split(".")) == 2:
        if activation_fn_name.split(".")[0] == "nn":
            activation_fn_name = backend_prefix[0] + "." + activation_fn_name
    try:
        return getattr(
            importlib.import_module(".".join(activation_fn_name.split(".")[:-1])),
            activation_fn_name.split(".")[-1],
        )
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(
            "'{}' is not a valid '{}' activation function.".format(
                activation_fn_name, backend_prefix[0]
            )
        )
