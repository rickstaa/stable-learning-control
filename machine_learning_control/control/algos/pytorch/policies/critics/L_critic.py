"""Lyapunov critic policy.

This module contains a Pytorch implementation of the Lyapunov Critic policy of
`Han et al. 2020 <http://arxiv.org/abs/2004.14288>`_.
"""

import torch
import torch.nn as nn
from machine_learning_control.control.algos.pytorch.common.helpers import mlp


class LCritic(nn.Module):
    """Soft Lyapunov critic Network.

    Attributes:
        L (torch.nn.modules.container.Sequential): The layers of the network.
    """

    def __init__(
        self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU,
    ):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (torch.nn.modules.activation): The activation function. Defaults
                to torch.nn.ReLU.
        """
        super().__init__()
        self.L = mlp([obs_dim + act_dim] + list(hidden_sizes), activation, activation,)

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.
            act (torch.Tensor): The tensor of actions.
        Returns:
            torch.Tensor: The tensor containing the lyapunov values of the input
                observations and actions.
        """
        L_hid_out = self.L(torch.cat([obs, act], dim=-1))
        L_out = torch.square(L_hid_out)
        L_out = torch.sum(L_out, dim=1)
        return torch.squeeze(
            L_out, -1
        )  # NOTE: Squeeze is critical to ensure L has right shape.
