"""Squashed Gaussian Actor policy.

This module contains a Pytorch implementation of the Squashed Gaussian Actor policy of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from machine_learning_control.control.utils.helpers import clamp, mlp
from torch.distributions.normal import Normal


class SquashedGaussianActor(nn.Module):
    """The squashed gaussian actor network.

    Attributes:
        net (torch.nn.modules.container.Sequential): The input/hidden layers of the
            network.
        mu (torch.nn.modules.linear.Linear): The output layer which returns the mean of
            the actions.
        log_std_layer (torch.nn.modules.linear.Linear): The output layer which returns
            the log standard deviation of the actions.
        act_limits (dict, optional): The "high" and "low" action bounds of the
            environment. Used for rescaling the actions that comes out of network
            from (-1, 1) to (low, high). No scaling will be applied if left empty.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=nn.ReLU,
        output_activation=nn.ReLU,
        act_limits=None,
        log_std_min=-20,
        log_std_max=2.0,
    ):
        """Constructs all the necessary attributes for the Squashed Gaussian Actor
        object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (torch.nn.modules.activation): The activation function. Defaults
                to torch.nn.ReLU.
            output_activation (torch.nn.modules.activation, optional): The activation
                function used for the output layers. Defaults to torch.nn.ReLU.
            act_limits (dict): The "high" and "low" action bounds of the environment.
                Used for rescaling the actions that comes out of network from (-1, 1)
                to (low, high).
            log_std_min (int, optional): The minimum log standard deviation. Defaults
                to -20.
            log_std_max (float, optional): The maximum log standard deviation. Defaults
                to 2.0.
        """
        super().__init__()
        self.act_limits = act_limits
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        self.net = mlp([obs_dim] + list(hidden_sizes), activation, output_activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.
            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If false the action is sampled from the stochastic
                policy. Defaults to False.
            with_logprob (bool, optional): Whether we want to return the log probability
                of an action. Defaults to True.

        Returns:
            torch.Tensor,  torch.Tensor: The actions given by the policy, the log
            probabilities of each of these actions.
        """
        # Calculate mean action and standard deviation
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = (
                pi_distribution.rsample()
            )  # Sample while using the parameterization trick

        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        if with_logprob:
            # NOTE: The correction formula is a little bit magic. To get an
            # understanding of where it comes from, check out the original SAC paper
            # (arXiv 1801.01290) and look in appendix C. This is a more
            # numerically-stable equivalent to Eq 21. Try deriving it yourself as a
            # (very difficult) exercise. :)
            sum_axis = 0 if obs.shape.__len__() == 1 else 1
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=sum_axis
            )
        else:
            logp_pi = None

        # Calculate scaled action and return the action and its log probability
        pi_action = torch.tanh(pi_action)  # Squash gaussian to be between -1 and 1

        # Clamp the actions such that they are in range of the environment
        if self.act_limits is not None:
            pi_action = clamp(
                pi_action,
                min_bound=self.act_limits["low"],
                max_bound=self.act_limits["high"],
            )

        # Return action and log likelihood
        return pi_action, logp_pi
