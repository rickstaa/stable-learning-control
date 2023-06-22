"""Contains utilities that can be used with the
`gymnasium package <https://gymnasium.farama.org/>`_.
"""

import importlib
import sys

import gymnasium as gym
from gymnasium import spaces

from stable_learning_control.utils.log_utils import friendly_err

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
    return isinstance(env, gym.core.Env)


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
    # Import gymnasium environments
    # import gymnasium as gym

    # Import environment configuration file. This file can be used to inject
    # custom gymnasium environments into the slc package.
    try:
        import stable_learning_control.env_config  # noqa: F401
    except Exception as e:
        raise Exception(
            friendly_err(
                "Something went wrong when trying to import the 'env_config' " " file."
            )
        ) from e

    # Special handling for environment: make sure that env_name is a real,
    # registered gymnasium environment.
    assert "env_name" in arg_dict, friendly_err(
        "You did not give a valid value for --env_name! Please try again."
    )
    # TODO: Remove this when having checked why this was here in the first page as gym
    # handles this?
    # for env_name in arg_dict["env_name"]:
    #     if env_name not in gym.envs.registry:
    #         err_msg = dedent(
    #             """
    #             %s is not registered with gymnasium.

    #             Recommendations:
    #                 * Check for a typo (did you include the version tag?)

    #                 * View the complete list of valid gymnasium environments at
    #                     https://gymnasium.farama.org/api/env/
    #             """
    #             % (env_name)
    #         )
    #         assert False, err_msg


def import_gym_env_pkg(module_name, frail=True, dry_run=False):
    """Tries to import the custom gymnasium environment package.

    Args:
        module_name (str): The python module you want to import.
        frail (bool, optional): Throw ImportError when tensorflow can not be imported.
            Defaults to ``true``.
        dry_run (bool, optional): Do not actually import tensorflow if available.
            Defaults to ``False``.

    Raises:
        ImportError: A import error if the package could not be imported.

    Returns:
        Union[:obj:`gym.env`, bool]:
            - Custom env package if ``dry_run`` is set to ``False``.
            - Returns a success bool if ``dry_run`` is set to ``True``.
    """
    module_name = module_name[0] if isinstance(module_name, list) else module_name
    try:
        if module_name in sys.modules:
            if not dry_run:
                return sys.modules[module_name]
            else:
                return True
        elif importlib.util.find_spec(module_name) is not None:
            if not dry_run:
                return importlib.import_module(module_name)
            else:
                return True
        else:
            if frail:
                raise ImportError(
                    friendly_err("No module named '{}'.".format(module_name))
                )
            return False
    except (ImportError, KeyError, AttributeError) as e:
        if ImportError:
            if not frail:
                return False
        raise e
