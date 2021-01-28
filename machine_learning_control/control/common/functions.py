"""Functions that are used in multiple Pytorch and Tensorflow algorithms.
"""
# TODO: Clean up utils and common folder
import numpy as np

from machine_learning_control.control.utils.gym_utils import (
    is_continuous_space,
    is_discrete_space,
)


def heuristic_target_entropy(action_space):
    """Returns a heuristic target entropy for a given action space using the method
    explained in `Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.

    Args:
        action_space (gym.spaces): The action space.
    Raises:
        NotImplementedError: If no heuristic target entropy has yet been implemented
            for the given action space.
    Returns:
        [type]: [description]
    """
    if is_continuous_space(action_space):
        heuristic_target_entropy = -np.prod(
            action_space.shape
        )  # Maximum information (bits) contained in action space
    elif is_discrete_space(action_space):
        raise NotImplementedError(
            "The heuristic target entropy is not yet implement for discrete spaces."
        )
    else:
        raise NotImplementedError(
            "The heuristic target entropy is not yet implement for "
            f"{type(action_space)} action spaces."
        )
    return heuristic_target_entropy
