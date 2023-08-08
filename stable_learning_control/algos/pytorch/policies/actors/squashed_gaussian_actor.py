"""Squashed Gaussian Actor policy.

This module contains a Pytorch implementation of the Squashed Gaussian Actor policy of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from stable_learning_control.algos.pytorch.common.helpers import mlp, rescale
from stable_learning_control.utils.log_utils.helpers import log_to_std_out


class SquashedGaussianActor(nn.Module):
    """The squashed gaussian actor network.

    Attributes:
        net (torch.nn.Sequential): The input/hidden layers of the
            network.
        mu (torch.nn.Linear): The output layer which returns the mean of
            the actions.
        log_std_layer (torch.nn.Linear): The output layer which returns
            the log standard deviation of the actions.
        act_limits (dict, optional): The ``high`` and ``low`` action bounds of the
            environment. Used for rescaling the actions that comes out of network
            from ``(-1, 1)`` to ``(low, high)``. No scaling will be applied if left
            empty.
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
        """Initialise the SquashedGaussianActor object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (:obj:`torch.nn.modules.activation`): The activation function.
                Defaults to :obj:`torch.nn.ReLU`.
            output_activation (:obj:`torch.nn.modules.activation`, optional): The
                activation function used for the output layers. Defaults to
                :obj:`torch.nn.ReLU`.
            act_limits (dict): The ``high`` and ``low`` action bounds of the
                environment. Used for rescaling the actions that comes out of network
                from ``(-1, 1)`` to ``(low, high)``.
            log_std_min (int, optional): The minimum log standard deviation. Defaults
                to ``-20``.
            log_std_max (float, optional): The maximum log standard deviation. Defaults
                to ``2.0``.
        """
        super().__init__()
        self.__device_warning_logged = False
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
                policy. Defaults to ``False``.
            with_logprob (bool, optional): Whether we want to return the log probability
                of an action. Defaults to ``True``.

        Returns:
            (tuple): tuple containing:

                - pi_action (:obj:`torch.Tensor`): The actions given by the policy.
                - logp_pi (:obj:`torch.Tensor`): The log probabilities of each of these actions.
        """  # noqa: E501
        # Make sure the observations are on the right device.
        if obs.device != self.net[0].weight.device:
            if not self.__device_warning_logged:
                device_warn_msg = (
                    "The observations were automatically moved from "
                    "'{}' to '{}' during the '{}' forward pass.".format(
                        obs.device, self.net[0].weight.device, self.__class__.__name__
                    )
                    + "Please place your observations on the '{}' ".format(
                        self.net[0].weight.device
                    )
                    + "before calling the '{}' as converting them ".format(
                        self.__class__.__name__
                    )
                    + "during the forward pass slows down the algorithm."
                )
                log_to_std_out(device_warn_msg, type="warning")
                self.__device_warning_logged = True
            obs = obs.to(self.net[0].weight.device)

        # Calculate mean action and standard deviation.
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
            )  # Sample while using the parameterization trick.

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

        # Calculate scaled action and return the action and its log probability.
        pi_action = torch.tanh(pi_action)  # Squash gaussian to be between -1 and 1

        # Rescale the normalized actions such that they are in range of the environment.
        if self.act_limits is not None:
            pi_action = rescale(
                pi_action,
                min_bound=self.act_limits["low"],
                max_bound=self.act_limits["high"],
            )

        return pi_action, logp_pi

    def act(self, obs, deterministic=False):
        """Returns the action from the current state given the current policy.

        Args:
            obs (torch.Tensor): The current observation (state).
            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If ``False`` the action is sampled from the
                stochastic policy. Defaults to ``False``.

        Returns:
            numpy.ndarray: The action from the current state given the current
            policy.
        """
        with torch.no_grad():
            a, _ = self(obs, deterministic, False)
            return a.cpu().numpy()

    def get_action(self, obs, deterministic=False):
        """Simple warpper for making the :meth:`act` method available under the
        'get_action' alias.

        Args:
            obs (torch.Tensor): The current observation (state).
            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If ``False`` the action is sampled from the
                stochastic policy. Defaults to ``False``.

        Returns:
            numpy.ndarray: The action from the current state given the current
                policy.
        """
        return self.act(obs, deterministic=deterministic)
