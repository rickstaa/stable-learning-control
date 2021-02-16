"""Module that contains utilities that can be used with the
`openai gym package <https://github.com/openai/gym>`_.
"""

from textwrap import dedent

import gym
import machine_learning_control.control.utils.log_utils as log_utils
from gym import spaces

DISCRETE_SPACES = (
    spaces.Discrete,
    spaces.MultiBinary,
    spaces.MultiDiscrete,
)
CONTINUOUS_SPACES = (spaces.Box,)


def is_gym_env(env):
    """Checks whether object is a gym environment.

    Args:
        env (object): A python object.

    Returns:
        bool: Boolean specifying whether object is gym environment.
    """
    return isinstance(env, gym.core.Env)


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


def validate_gym_env(arg_dict):
    """Make sures that env_name is a real, registered gym environment.

    Args:
        cmd (dict): The cmd dictionary.

    Raises:
        AssertError: Raised when a environment is supplied that is not a valid gym
            environment.
    """

    valid_envs = [e.id for e in list(gym.envs.registry.all())]
    assert "env_name" in arg_dict, log_utils.friendly_err(
        "You did not give a value for --env_name! Add one and try again."
    )
    for env_name in arg_dict["env_name"]:
        err_msg = dedent(
            """

            %s is not registered with Gym.

            Recommendations:

                * Check for a typo (did you include the version tag?)

                * View the complete list of valid Gym environments at

                    https://gym.openai.com/envs/

            """
            % env_name
        )
        assert env_name in valid_envs, err_msg
