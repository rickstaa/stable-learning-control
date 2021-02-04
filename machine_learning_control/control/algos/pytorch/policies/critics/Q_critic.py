"""Lyapunov actor critic policy.

This module contains a Pytorch implementation of the Q Critic policy of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""

import torch
import torch.nn as nn
from machine_learning_control.control.algos.pytorch.common.helpers import mlp


class QCritic(nn.Module):
    """Soft Q critic network.

    Attributes:
        Q (torch.nn.modules.container.Sequential): The layers of the network.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=nn.ReLU,
        output_activation=nn.Identity,
    ):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (torch.nn.modules.activation): The activation function. Defaults
                to torch.nn.ReLU.
            output_activation (torch.nn.modules.activation, optional): The activation
                function used for the output layers. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.Q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            activation,
            output_activation,
        )

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.
            act (torch.Tensor): The tensor of actions.
        Returns:
            torch.Tensor: The tensor containing the Q values of the input observations
                and actions.
        """
        return torch.squeeze(
            self.Q(torch.cat([obs, act], dim=-1)), -1
        )  # NOTE: Squeeze is critical to ensure q has right shape.
