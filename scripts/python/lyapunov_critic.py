"""Contains the Lyapunov Critic Class.
"""

import torch
import torch.nn as nn

from utils import mlp


class LyapunovCritic(nn.Module):
    """Soft Lyapunov critic Network.

    Attributes:
        lya (torch.nn.modules.container.Sequential): The layers of the network.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=nn.ReLU,
        output_activation=nn.ReLU,  # DEPLOY: Put back to identity when deploy
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
        self.lya = mlp(
            [obs_dim + act_dim] + list(hidden_sizes), activation, output_activation
        )

    def forward(self, obs, act):
        """Performs forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor: The tensor containing the Lyapunov values of the input
                observations and actions.
        """
        # DEPLOY: Make squaring layer from class so it shows up named in the graph!
        # https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
        l_out = self.lya(torch.cat([obs, act], dim=-1))
        l_out_squared = torch.square(l_out)
        l_out_summed = torch.sum(l_out_squared, dim=1)
        return l_out_summed.unsqueeze(dim=1)  # L(s,a)
