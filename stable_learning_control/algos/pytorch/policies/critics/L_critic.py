"""Lyapunov critic policy.

This module contains a Pytorch implementation of the Lyapunov Critic policy of
`Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_.
"""

import torch
import torch.nn as nn

from stable_learning_control.algos.pytorch.common.helpers import mlp
from stable_learning_control.utils.log_utils.helpers import log_to_std_out


class LCritic(nn.Module):
    """Soft Lyapunov critic Network.

    Attributes:
        L (torch.nn.Sequential): The layers of the network.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=nn.ReLU,
    ):
        """Initialise the LCritic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (:obj:`torch.nn.modules.activation`, optional): The activation
                function. Defaults to :obj:`torch.nn.ReLU`.
        """
        super().__init__()
        self.__device_warning_logged = False
        self._obs_same_device = False
        self._act_same_device = False
        self.L = mlp([obs_dim + act_dim] + list(hidden_sizes), activation, activation)

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.
            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor:
                The tensor containing the lyapunov values of the input observations and
                actions.
        """
        # Make sure the observations and actions are on the right device.
        self._obs_same_device = obs.device != self.L[0].weight.device
        self._act_same_device = act.device != self.L[0].weight.device
        if self._obs_same_device or self._act_same_device:
            if not self.__device_warning_logged:
                device_warn_strs = (
                    ("observations and actions", obs.device)
                    if (self._obs_same_device or self._act_same_device)
                    else (
                        ("observations", obs.device)
                        if self._obs_same_device
                        else ("actions", act.device)
                    )
                )
                device_warn_msg = (
                    "The {} were automatically moved from ".format(device_warn_strs[0])
                    + "'{}' to '{}' during the '{}' forward pass.".format(
                        device_warn_strs[1],
                        self.L[0].weight.device,
                        self.__class__.__name__,
                    )
                    + "Please place your observations on the '{}' ".format(
                        self.L[0].weight.device
                    )
                    + "before calling the '{}' as converting them ".format(
                        self.__class__.__name__
                    )
                    + "during the forward pass slows down the algorithm."
                )
                log_to_std_out(device_warn_msg, type="warning")
                self.__device_warning_logged = True
            obs = (
                obs.to(self.L[0].weight.device)
                if obs.device != self.L[0].weight.device
                else obs
            )
            act = (
                act.to(self.L[0].weight.device)
                if act.device != self.L[0].weight.device
                else act
            )

        L_hid_out = self.L(torch.cat([obs, act], dim=-1))
        L_out = torch.square(L_hid_out)
        L_out = torch.sum(L_out, dim=-1)

        return L_out
