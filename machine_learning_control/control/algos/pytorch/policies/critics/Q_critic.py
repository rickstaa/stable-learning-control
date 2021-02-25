"""Lyapunov actor critic policy.

This module contains a Pytorch implementation of the Q Critic policy of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""

import machine_learning_control.control.utils.log_utils as log_utils
import torch
import torch.nn as nn
from machine_learning_control.control.algos.pytorch.common.helpers import mlp


class QCritic(nn.Module):
    """Soft Q critic network.

    Attributes:
        Q (torch.nn.Sequential): The layers of the network.
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
            activation (:obj:`torch.nn.modules.activation`): The activation function.
                Defaults to torch.nn.ReLU.
            output_activation (:obj:`torch.nn.modules.activation`, optional): The
                activation function used for the output layers. Defaults to
                :mod:`torch.nn.Identity`.
        """
        super().__init__()
        self.__device_warning_logged = False
        self._obs_same_device = False
        self._act_same_device = False
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
            torch.Tensor:
                The tensor containing the Q values of the input observations and
                actions.
        """
        # Make sure the observations and actions are on the right device
        self._obs_same_device = obs.device != self.Q[0].weight.device
        self._act_same_device = act.device != self.Q[0].weight.device
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
                        self.Q[0].weight.device,
                        self.__class__.__name__,
                    )
                    + "Please place your observations on the '{}' ".format(
                        self.Q[0].weight.device
                    )
                    + "before calling the '{}' as converting them ".format(
                        self.__class__.__name__
                    )
                    + "during the forward pass slows down the algorithm."
                )
                log_utils.log(device_warn_msg, type="warning")
                self.__device_warning_logged = True
            obs = (
                obs.to(self.L[0].weight.device)
                if obs.device != self.Q[0].weight.device
                else obs
            )
            act = (
                act.to(self.L[0].weight.device)
                if act.device != self.Q[0].weight.device
                else act
            )
        return torch.squeeze(
            self.Q(torch.cat([obs, act], dim=-1)), -1
        )  # NOTE: Squeeze is critical to ensure q has right shape.
