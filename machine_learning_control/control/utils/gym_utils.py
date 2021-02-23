"""Module that contains utilities that can be used with the
`openai gym package <https://github.com/openai/gym>`_.
"""

import importlib
import sys
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
    """Make sure that env_name is a real, registered gym environment.

    Args:
        cmd (dict): The cmd dictionary.

    Raises:
        AssertError: Raised when a environment is supplied that is not a valid gym
            environment.
    """

    # Import main gym environments
    import gym
    import machine_learning_control.simzoo.simzoo  # noqa: F401

    # Try to import custom environments
    try:
        import machine_learning_control.env_config  # noqa: F401
    except Exception as e:
        raise Exception(
            "Something went wrong when trying to import the 'env_config' file."
        ) from e
    env_pkg_err_msg = ""
    if "env_pkg" in arg_dict.keys():
        try:
            import_gym_env_pkg(arg_dict["env_pkg"], frail=False)
        except ImportError:
            env_pkg_err_msg = (
                "\n\t\t* Make sure the package you supplied in the 'env_pkg' contains a "
                "a valid gym environment.\n"
            )

    # Special handling for environment: make sure that env_name is a real,
    # registered gym environment.
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
                %s
            """
            % (env_name, env_pkg_err_msg)
        )
        assert env_name in valid_envs, err_msg


def import_gym_env_pkg(module_name, frail=True, dry_run=False):
    """Tries to import the custom gym environment package.

    Args:
        module_name (str): The python module you want to import.
        frail (bool, optional): Throw ImportError when tensorflow can not be imported.
            Defaults to ``true``.
        dry_run (bool, optional): Do not actually import tensorflow if available.
            Defaults to ``False``.

    Raises:
        ImportError: A import error if the package could not be imported.

    Returns:
        union[tf, bool]: Custom env package if ``dry_run`` is set to ``False``.
            Returns a success bool if ``dry_run`` is set to ``True``.
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
                raise ImportError("No module named '{}'.".format(module_name))
            return False
    except (ImportError, KeyError, AttributeError) as e:
        if ImportError:
            if not frail:
                return False
        raise e
