"""Functions that are used in multiple Pytorch and TensorFlow algorithms."""
import importlib

import numpy as np
import scipy
import torch  # noqa:F401

from stable_learning_control.utils.gym_utils import (
    is_continuous_space,
    is_discrete_space,
)
from stable_learning_control.utils.import_utils import import_tf

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
            "https://github.com/rickstaa/stable-learning-control/issues if you need "
            "this."
        )
    else:
        raise NotImplementedError(
            "The heuristic target entropy is not yet implement for "
            f"{type(action_space)} action spaces. Please open a feature/pull request"
            "on https://github.com/rickstaa/stable-learning-control/issues if you"
            "need this."
        )
    return heuristic_target_entropy


def get_activation_function(activation_fn_name, backend="torch"):
    """Get a given torch activation function.

    Args:
        activation_fn_name (str): The name of the activation function you want to
            retrieve.
        backend (str): The machine learning backend you want to use. Options are
            ``torch`` or ``tf2``. By default ``torch``.

    Raises:
        ValueError: Thrown if the activation function does not exist within the
            backend.

    Returns:
        :obj:`torch.nn.modules.activation`: The torch activation function.
    """
    if backend.lower() == "tf2":
        backend_prefix = ["tensorflow", "nn"]
    else:
        backend_prefix = ["torch", "nn"]

    # Retrieve activation function.
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


def discount_cumsum(x, discount):
    """Calculate the discounted cumsum.

    .. seealso::
        Magic from rllab for computing discounted cumulative sums of vectors.

    Input:
        vector x: [x0, x1, x2]

    Output:
        [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
