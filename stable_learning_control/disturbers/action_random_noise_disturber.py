"""Contains a small gymnasium wrapper that can be used to disturb the action of
a gymnasium environment with normally distributed random noise (i.e. ``mean`` and
``std``).
"""

import gymnasium as gym


class ActionRandomNoiseDisturber(gym.ActionWrapper):
    """A gymnasium wrapper that can be used to disturb the action of a gymnasium
    environment with normally distributed random noise.

    Attributes:
        mean (float): The mean of the noise normal distribution.
        std (float): The standard deviation of the noise normal distribution.
    """

    def __init__(self, env, mean, std):
        """Initialise the ActionRandomNoiseDisturber object.

        Args:
            env (gym.Env): The gymnasium environment.
            mean (float): The mean of the noise normal distribution.
            std (float): The standard deviation of the noise normal distribution.
        """
        super().__init__(env)

        self.mean = mean
        self.std = std

    def action(self, action):
        """Add normally distributed random noise to the action.

        Args:
            action (np.ndarray): The action.

        Returns:
            np.ndarray: The action with added noise.
        """
        return action + super().np_random.normal(self.mean, self.std, action.shape)
