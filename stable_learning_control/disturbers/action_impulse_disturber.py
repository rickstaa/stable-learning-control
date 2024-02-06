"""Contains a small gymnasium wrapper that can be used to disturb the action of
a gymnasium environment with a impulse applied at a certain time step (i.e.
``magnitude`` and ``time``).
"""

import gymnasium as gym
import numpy as np

from stable_learning_control.common.helpers import friendly_err
from stable_learning_control.utils.log_utils.helpers import log_to_std_out


def get_time_attribute(env):
    """Get the time attribute of the environment.

    Args:
        env (gym.Env): The gymnasium environment.

    Returns:
        str: The time attribute of the environment.
    """
    if hasattr(env.unwrapped, "time") or hasattr(env.unwrapped, "t"):
        return "time" if hasattr(env.unwrapped, "time") else "t"
    return None


def get_time_step_attribute(env):
    """Get the time step attribute of the environment.

    Args:
        env (gym.Env): The gymnasium environment.

    Returns:
        str: The time step attribute of the environment.
    """
    if hasattr(env.unwrapped, "dt") or hasattr(env.unwrapped, "time_step"):
        return "dt" if hasattr(env.unwrapped, "dt") else "time_step"
    elif hasattr(env.unwrapped, "timestep"):
        return "timestep"
    elif hasattr(env.unwrapped, "tau"):
        return "tau"
    return None


class ActionImpulseDisturber(gym.ActionWrapper):
    """A gymnasium wrapper that can be used to disturb the action of a gymnasium
    environment with a impulse applied at a certain time step.

    Attributes:
        impulse_magnitude (float): The magnitude of the impulse.
        impulse_time (float): The time (s) at which to apply the impulse.
    """

    time_warning = False
    time_step_warning = False

    def __init__(self, env, magnitude, time):
        """Initialise the ActionImpulseDisturber object.

        Args:
            env (gym.Env): The gymnasium environment.
            magnitude (float): The impulse magnitude to apply.
            time (float): The time (s) at which to apply the impulse.
        """
        self._track_time = False
        self._time_step_attr = None
        self._time_attr = None
        super().__init__(env)

        self.magnitude = magnitude
        self.impulse_time = time

        # Check if the environment contains the time and or time step attributes.
        self._time_attr = get_time_attribute(env)
        self._time_step_attr = get_time_step_attribute(env)
        if self._time_attr is None:
            if not ActionImpulseDisturber.time_warning:
                log_to_std_out(
                    (
                        "The environment does not contain a 'time' or 't' attribute. "
                        "As a result, the time will be tracked within the "
                        f"'{self.__class__.__name__}' disturber."
                    ),
                    type="warning",
                )
                ActionImpulseDisturber.time_warning = True
            self._track_time = True
            self.t = 0
            self._time_attr = "t"
            if self._time_step_attr is None:
                if not ActionImpulseDisturber.time_step_warning:
                    log_to_std_out(
                        (
                            "The environment does not contain a 'dt', 'time_step', "
                            "'timestep', or 'tau' attribute. As a result, the time "
                            "step will be assumed to be '1'."
                        ),
                        type="warning",
                    )
                    ActionImpulseDisturber.time_step_warning = True
                self._time_step_attr = "dt"
                self.dt = 1

        # Throw warning if the time_step is not within the environment's time.
        max_episode_steps = getattr(self.env.env, "_max_episode_steps", None)
        if max_episode_steps is not None:
            if self._time_step_attr is None:
                if self._time_attr is None and self.impulse_time > max_episode_steps:
                    raise ValueError(
                        friendly_err(
                            f"The '{self.__class__.__name__}' disturber's time step "
                            f"({self.impulse_time}) is larger than the "
                            "environment's maximum episode steps "
                            f"({max_episode_steps}).",
                            prepend=False,
                        )
                    )
                elif self._time_attr is not None:
                    log_to_std_out(
                        (
                            "The environment does not contain a 'dt', 'time_step', "
                            "'timestep', or 'tau' attribute. As a result, the maximum "
                            "episode time could not be determined. Please ensure that "
                            "the time step is less than the maximum episode time."
                        ),
                        type="warning",
                    )
            else:
                impulse_time_step = np.ceil(
                    self.impulse_time
                    / getattr(self.env.unwrapped, self._time_step_attr)
                )
                if impulse_time_step >= max_episode_steps:
                    raise ValueError(
                        friendly_err(
                            f"The '{self.__class__.__name__}' disturber's time step "
                            f"({impulse_time_step}) is larger than the environment's "
                            f"maximum episode steps ({max_episode_steps}).",
                            prepend=False,
                        )
                    )

    def track_time(self):
        """Track the time of the environment."""
        if self._track_time:
            self.t += getattr(self.env.unwrapped, self._time_step_attr)

    def action(self, action):
        """Add a impulse to the action.

        Args:
            action (np.ndarray): The action.

        Returns:
            np.ndarray: The action with added impulse.
        """
        self.track_time()

        # If time is greater than the the impulse time
        if getattr(self.env.unwrapped, self._time_attr) >= self.impulse_time and (
            getattr(self.env.unwrapped, self._time_attr)
            <= self.impulse_time + getattr(self.env.unwrapped, self._time_step_attr)
        ):
            return action + self.magnitude
        return action
