import numpy as np
import torch

from control.utils.helpers import combined_shape


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        # QUESTION: Why is self.size needed what does it track?

        # Preallocate memory for experience buffer (s,s',a,r,d)
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)  # S
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)  # S'
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)  # A
        self.rew_buf = np.zeros(size, dtype=np.float32)  # R
        self.done_buf = np.zeros(size, dtype=np.float32)  # d

        # Initiate position, size and max size tracking variables
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """Store experience tuple elements."""
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (
            self.ptr + 1
        ) % self.max_size  # Increase pointer based on buffer size (first in first out)
        self.size = min(self.size + 1, self.max_size)  # Increase current size

    def sample_batch(self, batch_size=32):
        """Return batch of experiences"""
        idxs = np.random.randint(
            0, self.size, size=batch_size
        )  # Choice random experiences
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()
        }  # Make sure they are a torch tensor not numpy
