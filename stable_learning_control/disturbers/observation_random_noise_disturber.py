"""Contains a small gymnasium wrapper that can be used to disturb the observation of
a gymnasium environment with normally distributed random noise (i.e. ``mean`` and
``std``).
"""

import gymnasium as gym


class ObservationRandomNoiseDisturber(gym.ObservationWrapper):
    """A gymnasium wrapper that can be used to disturb the observation of a gymnasium
    environment with normally distributed random noise.

    Attributes:
        mean (float): The mean of the noise normal distribution.
        std (float): The standard deviation of the noise normal distribution.
    """

    def __init__(self, env, mean, std):
        """Initialise the ObservationRandomNoiseDisturber object.

        Args:
            env (gym.Env): The gymnasium environment.
            mean (float): The mean of the noise normal distribution.
            std (float): The standard deviation of the noise normal distribution.
        """
        super().__init__(env)

        self.mean = mean
        self.std = std

    def observation(self, observation):
        """Add normally distributed random noise to the observation.

        Args:
            observation (np.ndarray): The observation.

        Returns:
            np.ndarray: The observation with added noise.
        """
        return observation + super().np_random.normal(
            self.mean, self.std, observation.shape
        )
