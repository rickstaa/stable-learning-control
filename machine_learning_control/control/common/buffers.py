"""This module contains several replay buffers that are used in multiple Pytorch and
Tensorflow algorithms.
"""

import numpy as np
from machine_learning_control.common.helpers import combine_shapes


class ReplayBuffer:
    """A simple FIFO experience replay buffer.

    Attributes:
        obs_buf (numpy.ndarray): Buffer containing the current state.
        obs_next_buf (numpy.ndarray): Buffer containing the next state.
        act_buf (numpy.ndarray): Buffer containing the current action.
        rew_buf (numpy.ndarray): Buffer containing the current reward.
        done_buf (numpy.ndarray): Buffer containing information whether the episode was
            terminated after the action was taken.
    """

    def __init__(self, obs_dim, act_dim, rew_dim, size):
        """Constructs all the necessary attributes for the FIFO ReplayBuffer object.

        Args:
            obs_dim (tuple): The size of the observation space.
            act_dim (tuple): The size of the action space.
            rew_dim (tuple): The size of the reward space.
            size (int): The replay buffer size.
        """
        # Preallocate memory for experience buffer (s, s', a, r, d)
        # NOTE: Squeeze is needed to remove length 1 axis.
        self.obs_buf = np.zeros(
            combine_shapes(size, obs_dim), dtype=np.float32
        ).squeeze()
        self.obs_next_buf = np.zeros(
            combine_shapes(size, obs_dim), dtype=np.float32
        ).squeeze()
        self.act_buf = np.zeros(
            combine_shapes(size, act_dim), dtype=np.float32
        ).squeeze()
        self.rew_buf = np.zeros(
            combine_shapes(size, rew_dim), dtype=np.float32
        ).squeeze()
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """Add experience tuple to buffer.

        Args:
            obs (numpy.ndarray): Start state (observation).
            act (numpy.ndarray): Action.
            rew (:obj:`numpy.float64`): Reward.
            next_obs (numpy.ndarray): Next state (observation)
            done (bool): Boolean specifying whether the terminal state was reached.
        """
        try:
            self.obs_buf[self.ptr] = obs
            self.obs_next_buf[self.ptr] = next_obs
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure you set the "
                "ReplayBuffer 'obs_dim' equal to your environment 'observation_space'."
            )
            raise ValueError(error_msg)
        try:
            self.act_buf[self.ptr] = act
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure you set the "
                "ReplayBuffer 'act_dim' equal to your environment 'action_space'."
            )
            raise ValueError(error_msg)
        try:
            self.rew_buf[self.ptr] = rew
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure you set the "
                "ReplayBuffer 'rew_dim' equal to your environment 'reward_space'."
            )
            raise ValueError(error_msg)
        try:
            self.done_buf[self.ptr] = done
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure your 'done' ReplayBuffer"
                "element is of dimension 1."
            )
            raise ValueError(error_msg)

        # Increase pointers based on buffer size (first in first out)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        """Retrieve a batch of experiences from buffer.

        Args:
            batch_size (int, optional): The batch size. Defaults to ``32``.
        Returns:
            dict: A batch of experiences.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs_next=self.obs_next_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return batch
