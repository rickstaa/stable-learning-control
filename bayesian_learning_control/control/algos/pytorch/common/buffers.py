"""Contains several replay buffers used in the Pytorch algorithms.
"""

import torch
from bayesian_learning_control.control.common.buffers import (
    ReplayBuffer,
    TrajectoryBuffer,
)


class ReplayBuffer(ReplayBuffer):
    """Wrapper around the general FIFO
    :obj:`~bayesian_learning_control.control.common.buffers.ReplayBuffer` which makes
    sure a :obj:`torch.tensor` is returned when sampling.

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
            batch_size (int, optional): The batch size. Defaults to ``32``.

        Returns:
            dict: A batch of experiences.
        """
        batch = super().sample_batch(batch_size=batch_size)
        return {
            k: torch.as_tensor(v, dtype=torch.float32).to(device=self.device)
            for k, v in batch.items()
        }  # Make sure output is a torch tensor


class TrajectoryBuffer(TrajectoryBuffer):
    """Wrapper around the general
    :obj:`~bayesian_learning_control.control.common.buffers.TrajectoryBuffer` which
    makes sure a :obj:`torch.tensor` is returned when sampling.

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

    def get(self, flat=False):
        """Retrieve the trajectory buffer.

        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.

        Args:
            flat (bool, optional): Retrieve a flat buffer (i.e. the trajectories are
                concatenated). Defaults to ``False``.

        .. note:
            If you set flat to `True` all the trajectories will be concatenated into one
            array. You can use the :attr:`~TrajectoryBuffer.traj_lengths` or
            :attr:`traj_ptrs` attributes to split this array into distinct
            trajectories.

        Returns:
            dict: The trajectory buffer.
        """
        batch = super().get(flat=flat)
        return {
            k: torch.as_tensor(v, dtype=torch.float32).to(device=self.device)
            for k, v in batch.items()
        }  # Make sure output is a torch tensor
