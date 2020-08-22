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
    """The squashed gaussian actor network.

    Attributes:
        net (torch.nn.modules.container.Sequential): The input/hidden layers of the
            network.

        mu (torch.nn.modules.linear.Linear): The output layer which returns the mean of
            the actions.

        log_std_layer (torch.nn.modules.linear.Linear): The output layer which returns
            the log standard deviation of the actions.

        act_limit (np.float32): Scaling factor used for the actions that come out of
            the network.
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
        """Constructs all the necessary attributes for the Squashed Gaussian Actor
        object.

        Args:
            obs_dim (int): Dimension of the observation space.

            act_dim (int): Dimension of the action space.

            hidden_sizes (list): Sizes of the hidden layers.

            activation (torch.nn.modules.activation): The activation function.

            act_limit (np.float32): Scaling factor used for rescaling the actions that
                comes out of network from (-1, 1) to (action_limit * -1, action_limit
                * 1).

            log_std_min (int, optional): The minimum log standard deviation. Defaults
                to -20.

            log_std_max (float, optional): The maximum log standard deviation. Defaults
                to 2.0.
        """
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

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

        # Calculate required variables
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

        # Compute log probability in squashed gaussian
        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh
            # squashing. NOTE: The correction formula is a little bit magic. To get an
            # understanding of where it comes from, check out the original SAC paper
            # (arXiv 1801.01290) and look in appendix C. This is a more
            # numerically-stable equivalent to Eq 21. Try deriving it yourself as a
            # (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        # Calculate scaled action and return the action and its log probability
        pi_action = torch.tanh(pi_action)  # Squash gaussian to be between -1 and 1
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi


class MLPQFunction(nn.Module):
    """Soft Q-Network.

    Attributes:
        q (torch.nn.modules.container.Sequential): The layers of the network.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (torch.nn.modules.activation): The activation function.
        """
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor: The tensor containing the Q values of the input observations
                and actions.
        """
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    """Soft actor-critic network.

    Attributes:
        self.pi (:obj:`.SquashedGaussianMLPActor`): The squashed gaussian policy network
            (actor).

        self.q1 (:obj:`.MLPQFunction`): The first soft q-network (critic).

        self.q1 (:obj:`.MLPQFunction`); The second soft q-network (crictic).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        """Constructs all the necessary attributes for the soft actor-critic network
        object.

        Args:
            observation_space (gym.space.box.Box): A gym observation space.

            action_space (gym.space.box.Box): A gym action space.

            hidden_sizes (tuple, optional): Sizes of the hidden layers. Defaults to
                (256, 256).

            activation (torch.nn.modules.activation, optional): The activation function.
                Defaults to nn.ReLU.
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
        """Returns the action from the current state given the current policy.

        Args:
            obs (torch.Tensor): The current observation (state).

            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If false the action is sampled from the stochastic
                policy. Defaults to False.

        Returns:
            numpy.ndarray: The action from the current state given the current
            policy.
        """
        with torch.no_grad():  # Disable gradient update
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
