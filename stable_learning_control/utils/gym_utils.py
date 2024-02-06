"""Contains utilities that can be used with the
`gymnasium package <https://gymnasium.farama.org/>`_.
"""

import importlib

import gymnasium as gym
from gymnasium import spaces

from stable_learning_control.common.helpers import friendly_err

# from textwrap import dedent


DISCRETE_SPACES = (
    spaces.Discrete,
    spaces.MultiBinary,
    spaces.MultiDiscrete,
)
CONTINUOUS_SPACES = (spaces.Box,)


def is_gym_env(env):
    """Checks whether object is a gymnasium environment.

    Args:
        env (object): A python object.

    Returns:
        bool: Boolean specifying whether object is gymnasium environment.
    """
    return isinstance(env, gym.Env)


def is_continuous_space(space):
    """Checks whether a given space is continuous.

    Args:
        space (:obj:`gym.spaces`): The gymnasium space object.

    Returns:
        bool: Boolean specifying whether the space is discrete.
    """
    return isinstance(space, CONTINUOUS_SPACES)


def is_discrete_space(space):
    """Checks whether a given space is discrete.

    Args:
        space (:obj:`gym.spaces`): The gymnasium space object.

    Returns:
        bool: Boolean specifying whether the space is discrete.
    """
    return isinstance(space, DISCRETE_SPACES)


def validate_gym_env(arg_dict):
    """Make sure that env_name is a real, registered gymnasium environment.

    Args:
        cmd (dict): The cmd dictionary.

    Raises:
        AssertError: Raised when a environment is supplied that is not a valid gymnasium
            environment.
    """
    # Special handling for environment: make sure that env_name is a real,
    # registered gymnasium environment.
    assert "env_name" in arg_dict, friendly_err(
        "You did not give a valid value for --env_name! Please try again."
    )

    # Check if the environment is a valid gymnasium environment.
    for env_name in arg_dict["env_name"]:
        if ":" in env_name:
            # Try to import the custom gymnasium environment package.
            try:
                importlib.import_module(env_name.split(":")[0])
                env_name = env_name.split(":")[1]
            except ImportError as e:
                raise ImportError(
                    friendly_err(
                        "Could not import custom gymnasium environment package: "
                        + str(e)
                    )
                )

        # Check if the environment is a valid gymnasium environment.
        if env_name not in gym.envs.registry:
            err_msg = friendly_err(
                """
                %s is not registered with gymnasium.

                Recommendations:
                    * Check for a typo (did you include the version tag?)

                    * Gymnasium environments: View the complete list of valid gymnasium
                      environments at https://gymnasium.farama.org/api/env/

                    * Custom environments: Ensure the custom environment is installed
                      and you specify the module prefix (e.g. `custom_module:env_name`).
                """
                % (env_name)
            )
            assert False, err_msg
