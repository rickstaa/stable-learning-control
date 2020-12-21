"""Contains the Gaussian actor class.
"""

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils import mlp, clamp

# Script parameters
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SquashedGaussianActor(nn.Module):
    """The squashed gaussian actor network.

    Attributes:
        net (torch.nn.modules.container.Sequential): The fully connected hidden layers
            of the network.

        mu (torch.nn.modules.linear.Linear): The output layer which returns the mean of
            the actions.

        log_sigma (torch.nn.modules.linear.Linear): The output layer which returns
            the log standard deviation of the actions.

        act_limits (dict, optional): The "high" and "low" action bounds of the
            environment. Used for rescaling the actions that comes out of network
            from (-1, 1) to (low, high).
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        act_limits=None,
        activation=nn.ReLU,
        output_activation=nn.ReLU,
    ):
        """Constructs all the necessary attributes for the Squashed Gaussian Actor
        object.

        Args:
            obs_dim (int): Dimension of the observation space.

            act_dim (int): Dimension of the action space.

            hidden_sizes (list): Sizes of the hidden layers.

            activation (torch.nn.modules.activation): The hidden layer activation
                function.

            output_activation (torch.nn.modules.activation, optional): The activation
                function used for the output layers. Defaults to torch.nn.Identity.

            act_limits (dict or , optional): The "high" and "low" action bounds of the
                environment. Used for rescaling the actions that comes out of network
                from (-1, 1) to (low, high). Defaults to (-1, 1).
        """
        super().__init__()
        # Set class attributes
        self.act_limits = act_limits

        # Create networks
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, output_activation)
        self.mu = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_sigma = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        """Performs forward pass through the network.

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
        mu = self.mu(net_out)
        log_std = self.log_sigma(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)  # Transform to standard deviation

        # Check summing axis
        sum_axis = 0 if obs.shape.__len__() == 1 else 1

        # Use the re-parameterization trick to sample a action from the pre-squashed
        # distribution
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu  # Determinstic action used at test time.
        else:
            pi_action = (
                pi_distribution.rsample()
            )  # Sample while using the parameterization trick

            # Compute log probability of the sampled action in  the squashed gaussian
        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh
            # squashing. NOTE: The correction formula is a little bit magic. To get an
            # understanding of where it comes from, check out the original SAC paper
            # (arXiv 1801.01290) and look in appendix C. This is a more
            # numerically-stable equivalent to Eq 21. See
            # https://github.com/openai/spinningup/issues/279 for the derivation.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=sum_axis
            )
        else:
            logp_pi = None

        # Squash the action between (-1 and 1)
        pi_action = torch.tanh(pi_action)

        #  Clamp the actions such that they are in range of the environment
        if self.act_limits:
            pi_action = clamp(
                pi_action,
                min_bound=self.act_limits["low"],
                max_bound=self.act_limits["high"],
            )

        # Return action and log likelihood
        return pi_action, logp_pi
