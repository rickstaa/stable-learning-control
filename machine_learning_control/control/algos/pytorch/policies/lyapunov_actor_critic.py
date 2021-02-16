"""Lyapunov actor critic policy.

This module contains a Pytorch implementation of the Lyapunov Actor Critic policy of
`Han et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""

import torch
import torch.nn as nn
from machine_learning_control.control.algos.pytorch.policies.actors import (
    SquashedGaussianActor,
)
from machine_learning_control.control.algos.pytorch.policies.critics import LCritic
from machine_learning_control.control.common.helpers import strict_dict_update
from machine_learning_control.control.utils.log_utils import colorize

HIDDEN_SIZES_DEFAULT = {"actor": (64, 64), "critic": (128, 128)}
ACTIVATION_DEFAULT = {"actor": nn.ReLU, "critic": nn.ReLU}
OUTPUT_ACTIVATION_DEFAULT = {
    "actor": nn.ReLU,
}


class LyapunovActorCritic(nn.Module):
    """Lyapunov (soft) Actor-Critic network.

    Attributes:
        self.pi (:obj:`SquashedGaussianActor`): The squashed gaussian policy network
            (actor).
        self.L (:obj:`LCritic`); The soft L-network (critic).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=HIDDEN_SIZES_DEFAULT,
        activation=ACTIVATION_DEFAULT,
        output_activation=OUTPUT_ACTIVATION_DEFAULT,
    ):
        """Constructs all the necessary attributes for the lyapunov (soft) actor-critic
        network object.

        Args:
            observation_space (gym.space.box.Box): A gym observation space.
            action_space (gym.space.box.Box): A gym action space.
            hidden_sizes (Union[dict, tuple, list], optional): Sizes of the hidden
                layers for the actor. Defaults to (256, 256).
            activation (Union[dict, torch.nn.modules.activation], optional): The (actor
                and critic) hidden layers activation function. Defaults to
                torch.nn.ReLU.
            output_activation (Union[dict, torch.nn.modules.activation], optional): The
                actor  output activation function. Defaults to torch.nn.ReLU.

        .. note::
            It is currently not possible to set the critic output activation function
            when using the LyapunovActorCritic. This is since it by design requires the
            critic output activation to by of type :meth:`torch.square`.
        """
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # Parse hidden sizes, activation inputs arguments and action_limits
        hidden_sizes, _ = strict_dict_update(HIDDEN_SIZES_DEFAULT, hidden_sizes)
        activation, _ = strict_dict_update(ACTIVATION_DEFAULT, activation)
        output_activation, ignored = strict_dict_update(
            OUTPUT_ACTIVATION_DEFAULT, output_activation
        )
        act_limits = {"low": action_space.low, "high": action_space.high}

        if "critic" in ignored:
            print(
                colorize(
                    "WARN: Critic output activation function ignored since it is "
                    "not possible to set the critic output activation function when "
                    "using the LyapunovActorCritic architecture. This is since it by "
                    "design requires the critic output activation to by of type "
                    "'torch.square'.",
                    "yellow",
                    bold=True,
                )
            )

        self.pi = SquashedGaussianActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes["actor"],
            activation=activation["actor"],
            output_activation=output_activation["actor"],
            act_limits=act_limits,
        )
        self.L = LCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes["critic"],
            activation=activation["critic"],
        )

    def forward(self, obs, act):
        """Performs a forward pass through all the networks (Actor and L critic).

        Args:
            obs (torch.Tensor): The tensor of observations.
            act (torch.Tensor): The tensor of actions.
        Returns:
            (tuple): tuple containing:

                pi_action (torch.Tensor): The actions given by the policy
                logp_pi (torch.Tensor): The log probabilities of each of these
                    actions.
                L (torch.Tensor): Critic L values.

        .. note::
            Usefull for when you want to print out the full network graph using
            tensorboard.
        """
        pi_action, logp_pi = self.pi(obs)
        L = self.L(obs, act)
        return pi_action, logp_pi, L

    def act(self, obs, deterministic=False):
        """Returns the action from the current state given the current policy.

        Args:
            obs (torch.Tensor): The current observation (state).
            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If false the action is sampled from the stochastic
                policy. Defaults to ``False``.
        Returns:
            numpy.ndarray: The action from the current state given the current
            policy.
        """
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
