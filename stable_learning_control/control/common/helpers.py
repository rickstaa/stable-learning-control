"""Functions that are used in multiple Pytorch and Tensorflow algorithms.
"""

import importlib

import numpy as np
import scipy.signal
import torch  # noqa:F401
from stable_learning_control.control.utils.gym_utils import (
    is_continuous_space,
    is_discrete_space,
)
from stable_learning_control.utils.import_utils import import_tf

from stable_learning_control.utils.log_utils import log_to_std_out

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
        backend (str): The machine learning backend you want to use. By default
            ``None``, meaning no backend is assumed.

    Raises:
        ValueError: Thrown if the activation function does not exist within the
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


def validate_observations(observations, obs_dataframe):
    """Checks if the request observations exist in the ``obs_dataframe`` displays a
    warning if they do not.

    Args:
        observations (list): The requested observations.
        obs_dataframe (pandas.DataFrame): The dataframe with the observations that are
            present.

    Returns:
        list: List with the observations that are present in the dataframe.
    """
    valid_vals = obs_dataframe.observation.unique()
    if observations is None:
        return list(valid_vals)
    else:
        invalid_vals = [obs for obs in map(int, observations) if obs not in valid_vals]
        valid_observations = [
            obs for obs in map(int, observations) if obs in valid_vals
        ]
        if len(observations) == len(invalid_vals):
            log_to_std_out(
                "{} not valid. All observations plotted instead.".format(
                    f"Observations {invalid_vals} are"
                    if len(invalid_vals) > 1
                    else f"Observation {invalid_vals[0]} is"
                ),
                type="warning",
            )
            valid_observations = list(valid_vals)
        elif invalid_vals:
            log_to_std_out(
                "{} not valid.".format(
                    f"Observations {invalid_vals} could not plotted as they are"
                    if len(invalid_vals) > 1
                    else f"Observation {invalid_vals[0]} could not be plotted as it is"
                ),
                type="warning",
            )
        return valid_observations
