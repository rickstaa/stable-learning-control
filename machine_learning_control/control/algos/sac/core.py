"""SAC core components

This module contains the core components of the SAC algorithm of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_. This implementation is based
on the one found in the the
`Spinning Up repository <https://github.com/openai/spinningup>`_.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from machine_learning_control.control.utils.helpers import mlp


class SquashedGaussianMLPActor(nn.Module):
    """

    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation,
        act_limit,
        log_std_min=-20,
        log_std_max=2.0,
    ):
        """

        Args:
            obs_dim ([type]): [description]
            act_dim ([type]): [description]
            hidden_sizes ([type]): [description]
            activation ([type]): [description]
            act_limit ([type]): [description]
            log_std_min (int, optional): [description]. Defaults to -20.
            log_std_max (float, optional): [description]. Defaults to 2.0.
        """
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

    def forward(self, obs, deterministic=False, with_logprob=True):
        """

        Args:
            obs ([type]): [description]
            deterministic (bool, optional): [description]. Defaults to False.
            with_logprob (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
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
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):
    """

    Args:
        nn ([type]): [description]
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        """[summary]

        Args:
            obs ([type]): [description]
            act ([type]): [description]

        Returns:
            [type]: [description]
        """
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    """

    Args:
        nn ([type]): [description]
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        """

        Args:
            observation_space ([type]): [description]
            action_space ([type]): [description]
            hidden_sizes (tuple, optional): [description]. Defaults to (256, 256).
            activation ([type], optional): [description]. Defaults to nn.ReLU.
        """
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit
        )
        self.q1 = MLPQFunction(
            obs_dim, act_dim, hidden_sizes, activation
        )  # Use min-clipping
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        """

        Args:
            obs ([type]): [description]
            deterministic (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        with torch.no_grad():  # Disable gradient update
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
