"""Contains the Q-critic class."""

import torch
from torch import nn
from utils import mlp


class QCritic(nn.Module):
    """Soft Q critic network.

    Attributes:
        q (torch.nn.modules.container.Sequential): The layers of the network.
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

            activation (torch.nn.modules.activation): The hidden layer activation
                function.

            output_activation (torch.nn.modules.activation, optional): The activation
                function used for the output layers. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            activation,
            output_activation,
        )

    def forward(self, obs, act):
        """Performs forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor: The tensor containing the Q values of the input observations
                and actions.
        """
        q_out = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(
            q_out, -1
        )  # NOTE (rickstaa) Critical to ensure Q(s,a) has right shape.
