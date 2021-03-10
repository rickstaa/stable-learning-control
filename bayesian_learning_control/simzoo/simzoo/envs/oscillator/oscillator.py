"""A gym environment for a synthetic oscillatory network of transcriptional regulators
called a repressilator. A repressilator is a three-gene regulatory network where the
dynamics of mRNA and proteins follow an oscillatory behaviour
(see https://www-nature-com.tudelft.idm.oclc.org/articles/35002125).
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding

__VERSION__ = "1.3.4"  # Oscillator version

RANDOM_STEP = False  # Use random steps in __main__

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.

    This function was originally written by John Schulman.

    Args:
        string (str): The string you want to colorize.
        color (str): The color you want to use.
        bold (bool, optional): Whether you want the text to be bold text has to be bold.
        highlight (bool, optional):  Whether you want to highlight the text. Defaults to
            False.

    Returns:
        str: Colorized string.
    """
    if color:
        attr = []
        num = color2num[color]
        if highlight:
            num += 10
        attr.append(str(num))
        if bold:
            attr.append("1")
        return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)
    else:
        return string


class Oscillator(gym.Env):
    """Synthetic oscillatory network
    The goal of the agent in the oscillator environment is to act in such a way that
    one of the proteins of the synthetic oscillatory network follows a supplied
    reference signal.

    Observations:
        - m1: Lacl mRNA concentration.
        - m2: tetR mRNA concentration.
        - m3: CI mRNA concentration.
        - p1: lacI (repressor) protein concentration (Inhibits transcription tetR gene).
        - p2: tetR (repressor) protein concentration (Inhibits transcription CI).
        - p3: CI (repressor) protein concentration (Inhibits transcription of lacI).
        - r: The value of the reference for protein 1.
        - e: The error between the current value of protean 1 and the reference.

    Action space:
        - u1: Number of Lacl proteins produced during continuous growth under repressor
          saturation (Leakiness).
        - u1: Number of tetR proteins produced during continuous growth under repressor
          saturation (Leakiness).
        - u1: Number of CI proteins produced during continuous growth under repressor
          saturation (Leakiness).

    Attributes:
        state (numpy.ndarray): The current system state.
        t (float): The current time step.
        dt (float): The environment step size.
        sigma (float): The variance of the system noise.
    """

    def __init__(self, reference_type="periodic", seed=None):
        """Constructs all the necessary attributes for the oscillator object.

        Args:
            reference_type (str, optional): The type of reference you want to use
                (``constant`` or ``periodic``), by default ``periodic``.
            seed (int, optional): A random seed for the environment. By default
                ``None``.
        """

        self.reference_type = reference_type
        self.t = 0
        self.dt = 1.0
        self.sigma = 0.0
        self.__logged_render_warning = False
        self.__init_state = np.array(
            [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
        )  # Initial state when random is disabled

        # Set oscillator network parameters
        self._K = 1.0
        self._c1 = 1.6
        self._c2 = 0.16
        self._c3 = 0.16
        self._c4 = 0.06
        self._b1 = 1.0
        self._b2 = 1.0
        self._b3 = 1.0

        # Set angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([100, 100, 100, 100, 100, 100, 100, 100], dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.reward_range = spaces.Box(
            np.array([0.0], dtype=np.float32),
            np.array([100], dtype=np.float32),
            dtype=np.float32,
        )

        # Create random seed and set gym environment parameters
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        """Return random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Take step into the environment.

        Args:
            action (numpy.ndarray): The action we want to perform in the environment.

        Returns:
            (tuple): tuple containing:

                - obs (:obj:`numpy.ndarray`): The current state
                - cost (:obj:`numpy.float64`): The current cost.
                - done (:obj:`bool`): Whether the episode was done.
                - info_dict (:obj:`dict`): Dictionary with additional information.
        """

        # Perform action in the environment and return the new state
        # NOTE: The new state is found by solving 3 first-order differential equations.
        u1, u2, u3 = action
        m1, m2, m3, p1, p2, p3 = self.state
        m1_dot = self._c1 / (self._K + np.square(p3)) - self._c2 * m1 + self._b1 * u1
        p1_dot = self._c3 * m1 - self._c4 * p1
        m2_dot = self._c1 / (self._K + np.square(p1)) - self._c2 * m2 + self._b2 * u2
        p2_dot = self._c3 * m2 - self._c4 * p2
        m3_dot = self._c1 / (self._K + np.square(p2)) - self._c2 * m3 + self._b3 * u3
        p3_dot = self._c3 * m3 - self._c4 * p3

        # Calculate mRNA concentrations
        # Note: Use max to make sure concentrations can not be negative.
        m1 = np.max(
            [
                m1
                + m1_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        m2 = np.max(
            [
                m2
                + m2_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        m3 = np.max(
            [
                m3
                + m3_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )

        # Calculate protein concentrations
        # Note: Use max to make sure concentrations can not be negative.
        p1 = np.max(
            [
                p1
                + p1_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        p2 = np.max(
            [
                p2
                + p2_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        p3 = np.max(
            [
                p3
                + p3_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )

        # Retrieve state
        self.state = np.array([m1, m2, m3, p1, p2, p3])
        self.t = self.t + self.dt  # Increment time step # Question: This used to be 1

        # Calculate cost
        r1 = self.reference(self.t)
        cost = np.square(p1 - r1)
        # cost = (abs(p1 - r1)) ** 0.2

        # Define stopping criteria
        if cost > self.reward_range.high or cost < self.reward_range.low:
            done = True
        else:
            done = False

        # Return state, cost, done and reference
        return (
            np.array([m1, m2, m3, p1, p2, p3, r1, p1 - r1]),
            cost,
            done,
            dict(reference=r1, state_of_interest=p1 - r1),
        )

    def reset(self, random=True):
        """Reset gym environment.

        Args:
            random (bool, optional): Whether we want to randomly initialise the
                environment. By default True.

        Returns:
            numpy.ndarray: Array containing the current observations.
        """

        # Return random initial state
        self.state = (
            self.np_random.uniform(low=0, high=1, size=(6,))
            if random
            else self.__init_state
        )
        self.t = 0
        m1, m2, m3, p1, p2, p3 = self.state
        r1 = self.reference(self.t)
        return np.array([m1, m2, m3, p1, p2, p3, r1, p1 - r1])

    def reference(self, t):
        """Returns the current value of the periodic reference signal that is tracked by
        the Synthetic oscillatory network.

        Args:
            t (float): The current time step.

        Returns:
            float: The current reference value.
        """
        if self.reference_type == "periodic":
            return 8 + 7 * np.sin((2 * np.pi) * t / 200)
        else:
            return 8

    def render(self, mode="human"):
        """Render one frame of the environment.

        Args:
            mode (str, optional): Gym rendering mode. The default mode will do something
                human friendly, such as pop up a window.

        Note:
            This currently is not yet implemented.
        """
        if not self.__logged_render_warning:
            print(
                colorize(
                    (
                        "WARNING: Nothing was rendered as the oscillator environment "
                        "doesn't have a render method."
                    ),
                    color="yellow",
                    bold=True,
                )
            )
            self.__logged_render_warning = True
        return


if __name__ == "__main__":

    print("Settting up oscillator environment.")
    env = Oscillator()

    # Take T steps in the environment
    T = 600
    path = []
    t1 = []
    s = env.reset()
    print(f"Taking {T} steps in the oscillator environment.")
    for i in range(int(T / env.dt)):
        action = (
            env.np_random.uniform(
                env.action_space.low, env.action_space.high, env.action_space.shape
            )
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, done, info = env.step(action)
        path.append(s)
        t1.append(i * env.dt)
    print("Finished oscillator environment simulation.")

    # Plot results
    print("Plot results.")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.plot(t1, np.array(path)[:, 0], color="orange", label="mRNA1")
    # ax.plot(t1, np.array(path)[:, 1], color="magenta", label="mRNA2")
    # ax.plot(t1, np.array(path)[:, 2], color="sienna", label="mRNA3")
    ax.plot(t1, np.array(path)[:, 3], color="blue", label="protein1")
    # ax.plot(t1, np.array(path)[:, 4], color="cyan", label="protein2")
    # ax.plot(t1, np.array(path)[:, 5], color="green", label="protein3")
    # ax.plot(t1, np.array(path)[:, 0:3], color="blue", label="mRNA")
    # ax.plot(t1, np.array(path)[:, 3:6], color="blue", label="protein")
    ax.plot(t1, np.array(path)[:, 6], color="yellow", label="reference")
    ax.plot(t1, np.array(path)[:, 7], color="red", label="error")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print("Done")
