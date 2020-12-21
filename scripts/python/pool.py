"""Contains the replay buffer class.
"""

from collections import deque

import numpy as np
import torch


class Pool(object):
    """Memory buffer class.

    Attributes:
        self.memory_capacity (int): The current memory capacity.

        paths (collections.deque): A storage bucket for storing paths.

        memory (dict): The replay memory storage bucket.

        min_memory_size (np.float32): The minimum memory size before we start to sample
            from the memory buffer.

        memory_pointer (): The number of experiences that are currently stored in the
            replay buffer.

        device (str): The device the sampled experiences are placed on (CPU or GPU).
    """

    def __init__(
        self,
        s_dim,
        a_dim,
        memory_capacity,
        store_last_n_paths,
        min_memory_size,
        device="cpu",
    ):
        """Initializes memory buffer object.

        Args:
            s_dim (int): The observation space dimension.

            a_dim (int): The action space dimension.

            memory_capacity (int): The size of the memory buffer.

            store_last_n_paths (int): How many paths you want to store in the replay
                buffer.

            min_memory_size (int): The minimum memory size before we start to sample
                from the memory buffer.

            device (str): Whether you want to return the sample on the CPU or GPU.
        """
        self.memory_capacity = memory_capacity
        self.reset()
        self.device = device
        self.memory = {
            "s": torch.zeros([1, s_dim], dtype=torch.float32),
            "a": torch.zeros([1, a_dim], dtype=torch.float32),
            "r": torch.zeros([1, 1], dtype=torch.float32),
            "terminal": torch.zeros([1, 1], dtype=torch.float32),
            "s_": torch.zeros([1, s_dim], dtype=torch.float32),
        }
        self.memory_pointer = 0
        self.min_memory_size = min_memory_size

    def reset(self):
        """Reset memory buffer.
        """
        self.current_path = {
            "s": torch.tensor([], dtype=torch.float32),
            "a": torch.tensor([], dtype=torch.float32),
            "r": torch.tensor([], dtype=torch.float32),
            "terminal": torch.tensor([], dtype=torch.float32),
            "s_": torch.tensor([], dtype=torch.float32),
        }

    def store(self, s, a, r, terminal, s_):
        """Stores experience tuple.

        Args:
            s (numpy.ndarray): State.

            a (numpy.ndarray): Action.

            r (numpy.ndarray): Reward.

            terminal (numpy.ndarray): Whether the terminal state was reached.

            s_ (numpy.ndarray): Next state.

        Returns:
            int: The current memory buffer size.
        """

        # Store experience in memory buffer
        transition = {
            "s": torch.as_tensor(s, dtype=torch.float32),
            "a": torch.as_tensor(a, dtype=torch.float32),
            "r": torch.as_tensor([r], dtype=torch.float32),
            "terminal": torch.as_tensor([terminal], dtype=torch.float32),
            "s_": torch.as_tensor(s_, dtype=torch.float32),
        }
        if len(self.current_path["s"]) < 1:
            for key in transition.keys():
                self.current_path[key] = transition[key].unsqueeze(dim=0)
        else:
            for key in transition.keys():
                self.current_path[key] = torch.cat(
                    [self.current_path[key], transition[key].unsqueeze(dim=0)], axis=0
                )
        if terminal == 1.0:
            for key in self.current_path.keys():
                self.memory[key] = torch.cat(
                    [self.memory[key], self.current_path[key]], axis=0
                )
            self.reset()
            self.memory_pointer = len(self.memory["s"])

        # Return current memory buffer size
        return self.memory_pointer

    def sample(self, batch_size):
        """Samples from memory buffer.

        Args:
            batch_size (int): The memory buffer sample size.

        Returns:
            numpy.ndarray: The batch of experiences.
        """
        if self.memory_pointer < self.min_memory_size:
            return None
        else:

            # Sample a random batch of experiences
            indices = np.random.choice(
                min(self.memory_pointer, self.memory_capacity) - 1,
                size=batch_size,
                replace=False,
            ) + max(1, 1 + self.memory_pointer - self.memory_capacity) * np.ones(
                [batch_size], np.int
            )
            batch = {}
            for key in self.memory.keys():
                if "s" in key:
                    sample = self.memory[key][indices].to(self.device)
                    batch.update({key: sample})
                else:
                    batch.update({key: self.memory[key][indices].to(self.device)})
            return batch
