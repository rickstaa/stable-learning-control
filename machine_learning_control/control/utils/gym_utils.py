"""Module that contains utilities that can be used with the
`openai gym package <https://github.com/openai/gym>`_.
"""

from gym import spaces

DISCRETE_SPACES = (
    spaces.Discrete,
    spaces.MultiBinary,
    spaces.MultiDiscrete,
)
CONTINUOUS_SPACES = (spaces.Box,)


def is_continuous_space(space):
    """Checks whether a given space is continuous.

    Args:
        space (gym.spaces): The gym space object.

    Returns:
        bool: Boolean specifying whether the space is discrete.
    """
    return isinstance(space, CONTINUOUS_SPACES)


def is_discrete_space(space):
    """Checks whether a given space is discrete.

    Args:
        space (gym.spaces): The gym space object.

    Returns:
        bool: Boolean specifying whether the space is discrete.
    """
    return isinstance(space, DISCRETE_SPACES)
