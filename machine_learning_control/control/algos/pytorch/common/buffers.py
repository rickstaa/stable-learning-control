"""This module contains several replay buffers used in the Pytorch algorithms.
"""

import torch
from machine_learning_control.control.common.buffers import ReplayBuffer


class ReplayBuffer(ReplayBuffer):
    """A wrapper around the general FIFO :class:`ReplayBuffer` which makes sure a
    torch.tensor is returned when sampling.

    Attributes:
        device (str): The device the experiences are placed on (CPU or GPU).
    """

    def __init__(self, device="cpu", *args, **kwargs):
        """Constructs all the necessary attributes for the FIFO ReplayBuffer object.

        Args:
            device (str, optional): The computational device to put the sampled
                experiences on.
            *args: All args to pass to the ReplayBuffer parent class.
            **kwargs: All kwargs to pass to the ReplayBuffer parent class.
        """
        self.device = device
        super().__init__(*args, **kwargs)

    def sample_batch(self, batch_size=32):
        """Retrieve a batch of experiences from buffer.

        Args:
            batch_size (int, optional): The batch size. Defaults to 32.
        Returns:
            dict: A batch of experiences.
        """
        batch = super().sample_batch(batch_size)
        return {
            k: torch.as_tensor(v, dtype=torch.float32).to(device=self.device)
            for k, v in batch.items()
        }  # Make sure output is a torch tensor
