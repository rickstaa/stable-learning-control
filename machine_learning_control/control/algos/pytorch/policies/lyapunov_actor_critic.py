"""Lyapunov actor critic policy.

This module contains a Pytorch implementation of the Lyapunov Actor Critic policy of
`Han et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""

import torch
import torch.nn as nn
from machine_learning_control.control.algos.pytorch.common.helpers import (
    parse_network_structure,
)
from machine_learning_control.control.algos.pytorch.policies.actors import (
    SquashedGaussianActor,
)
from machine_learning_control.control.algos.pytorch.policies.critics import LCritic


class LyapunovActorCritic(nn.Module):
    """Lyapunov (soft) Actor-Critic network.

    Attributes:
        self.pi (:obj:`.SquashedGaussianActor`): The squashed gaussian policy network
            (actor).
        self.L (:obj:`.LCritic`); The soft L-network (critic).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes={"actor": (64, 64), "critic": (128, 128)},
        activation={"actor": nn.ReLU, "critic": nn.ReLU},
        output_activation={"actor": nn.ReLU, "critic": nn.ReLU},
    ):
        """Constructs all the necessary attributes for the lyapunov (soft) actor-critic network
        object.

        Args:
            observation_space (gym.space.box.Box): A gym observation space.
            action_space (gym.space.box.Box): A gym action space.
            hidden_sizes (Union[dict, tuple, list], optional): Sizes of the hidden
                layers for the actor. Defaults to (256, 256).
            activation (Union[dict, torch.nn.modules.activation], optional): The (actor
                and critic) hidden layers activation function. Defaults to
                torch.nn.ReLU.
            output_activation (Union[dict, torch.nn.modules.activation], optional): The
                (actor and critic)  output activation function. Defaults to
                torch.nn.ReLU.
        """
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # Parse hidden sizes and activation inputs
        hidden_sizes, activation, output_activation = parse_network_structure(
            hidden_sizes, activation, output_activation
        )

        # Action limit for clamping
        act_limits = {"low": action_space.low, "high": action_space.high}

        # build policy and value functions
        self.pi = SquashedGaussianActor(
            obs_dim,
            act_dim,
            hidden_sizes["actor"],
            activation["actor"],
            output_activation["actor"],
            act_limits,
        )
        self.L = LCritic(
            obs_dim,
            act_dim,
            hidden_sizes["critic"],
            activation["critic"],
            output_activation["critic"],
        )

    def forward(self, obs, act):
        """Perform a forward pass through all the networks.

        Args:
            obs (torch.Tensor): The tensor of observations.
            act (torch.Tensor): The tensor of actions.
        Returns:
            tuple: actor_action, log probability of the action, critic 1 q value,
            critic 2 q value.

        .. note::
            Usefull for when you want to print out the full network graph using
            tensorboard.
        """
        # Perform a forward pass through all the networks (Actor, critic1 and critic2)
        # and return the results
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
                policy. Defaults to False.
        Returns:
            numpy.ndarray: The action from the current state given the current
            policy.
        """
        with torch.no_grad():  # Disable gradient update
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
