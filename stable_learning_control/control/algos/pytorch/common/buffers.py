"""Contains several replay buffers used in the Pytorch algorithms.
"""

import torch
from stable_learning_control.control.algos.pytorch.common.helpers import np_to_torch
from stable_learning_control.control.common.buffers import (
    ReplayBuffer,
    TrajectoryBuffer,
)


class ReplayBuffer(ReplayBuffer):
    """Wrapper around the general FIFO
    :obj:`~stable_learning_control.control.common.buffers.ReplayBuffer` which makes
    sure a :obj:`torch.tensor` is returned when sampling.

    Attributes:
        device (str): The device the experiences are placed on (CPU or GPU).
    """

    def __init__(self, device="cpu", *args, **kwargs):
        """Constructs all the necessary attributes for the FIFO ReplayBuffer object.

        Args:
            device (str, optional): The computational device to put the sampled
                experiences on.
            *args: All args to pass to the :class:`ReplayBuffer` parent class.
            **kwargs: All kwargs to pass to the class:`ReplayBuffer` parent class.
        """
        self.device = device
        super().__init__(*args, **kwargs)

    def sample_batch(self, *args, **kwargs):
        """Retrieve a batch of experiences from buffer.

        Args:
            *args: All args to pass to the :meth:`~ReplayBuffer.get` parent method.
            **kwargs: All kwargs to pass to the :meth:`~ReplayBuffer.get` parent method.

        Returns:
            dict: A batch of experiences.
        """
        return np_to_torch(
            super().sample_batch(*args, **kwargs),
            dtype=torch.float32,
            device=self.device,
        )  # Make sure output is a torch tensor.


class TrajectoryBuffer(TrajectoryBuffer):
    """Wrapper around the general
    :obj:`~stable_learning_control.control.common.buffers.TrajectoryBuffer` which
    makes sure a :obj:`torch.tensor` is returned when sampling.

    Attributes:
        device (str): The device the experiences are placed on (CPU or GPU).
    """

    def __init__(self, device="cpu", *args, **kwargs):
        """Constructs all the necessary attributes for the FIFO ReplayBuffer object.

        Args:
            device (str, optional): The computational device to put the sampled
                experiences on.
            *args: All args to pass to the :class:`TrajectoryBuffer` parent class.
            **kwargs: All kwargs to pass to the :class:`TrajectoryBuffer` parent class.
        """
        self.device = device
        super().__init__(*args, **kwargs)

    def get(self, *args, **kwargs):
        """Retrieve the trajectory buffer.

        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.

        Args:
            *args: All args to pass to the :meth:`~TrajectoryBuffer.get` parent method.
            **kwargs: All kwargs to pass to the :meth:`~TrajectoryBuffer.get` parent
                method.

        Returns:
            dict: The trajectory buffer.
        """
        return np_to_torch(
            super().get(*args, **kwargs), dtype=torch.float32, device=self.device
        )  # Make sure output is a torch tensor.
