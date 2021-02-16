"""Soft actor critic policy.

This module contains a Tensorflow 2.x implementation of the Soft Actor Critic policy of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""

import tensorflow as tf
from machine_learning_control.control.algos.tf2.policies.actors import (
    SquashedGaussianActor,
)
from machine_learning_control.control.algos.tf2.policies.critics import QCritic
from machine_learning_control.control.common.helpers import strict_dict_update
from tensorflow import nn

HIDDEN_SIZES_DEFAULT = {"actor": (256, 256), "critic": (256, 256)}
ACTIVATION_DEFAULT = {"actor": nn.relu, "critic": nn.relu}
OUTPUT_ACTIVATION_DEFAULT = {"actor": nn.relu, "critic": None}


class SoftActorCritic(tf.keras.Model):
    """(Soft) Actor-Critic network.

    Attributes:
        self.pi (:obj:`SquashedGaussianActor`): The squashed gaussian policy network
            (actor).
        self.Q1 (:obj:`QCritic`): The first soft Q-network (critic).
        self.Q1 (:obj:`QCritic`); The second soft Q-network (critic).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=HIDDEN_SIZES_DEFAULT,
        activation=ACTIVATION_DEFAULT,
        output_activation=OUTPUT_ACTIVATION_DEFAULT,
        name="soft_actor_critic",
    ):
        """Constructs all the necessary attributes for the (soft) actor-critic network
        object.

        Args:
            observation_space (gym.space.box.Box): A gym observation space.
            action_space (gym.space.box.Box): A gym action space.
            hidden_sizes (Union[dict, tuple, list], optional): Sizes of the hidden
                layers for the actor. Defaults to (256, 256).
            activation (Union[dict, tf.keras.activations], optional): The (actor
                and critic) hidden layers activation function. Defaults to nn.relu.
            output_activation (Union[dict, tf.keras.activations], optional): The
                (actor and critic)  output activation function. Defaults to nn.relu for
                the actor and nn.Identity for the critic.
            name (str, optional): The name given to the SoftActorCritic. Defaults to
                "soft_actor_critic".
        """
        super().__init__(name=name)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # Parse hidden sizes, activation inputs arguments and action_limits
        hidden_sizes, _ = strict_dict_update(HIDDEN_SIZES_DEFAULT, hidden_sizes)
        activation, _ = strict_dict_update(ACTIVATION_DEFAULT, activation)
        output_activation, _ = strict_dict_update(
            OUTPUT_ACTIVATION_DEFAULT, output_activation
        )
        act_limits = {"low": action_space.low, "high": action_space.high}

        self.pi = SquashedGaussianActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes["actor"],
            activation=activation["actor"],
            output_activation=output_activation["actor"],
            act_limits=act_limits,
        )
        self.Q1 = QCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes["critic"],
            activation=activation["critic"],
            output_activation=output_activation["critic"],
            name="q_critic_1",
        )
        self.Q2 = QCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes["critic"],
            activation=activation["critic"],
            output_activation=output_activation["critic"],
            name="q_critic_2",
        )

    @tf.function
    def call(self, inputs):
        """Performs a forward pass through all the networks (Actor, Q critic 1 and Q
        critic 2).

        Args:
            inputs (tuple/list): The network inputs:

                obs (tf.Tensor): The tensor of observations.
                act (tf.Tensor): The tensor of actions.
        Returns:
            (tuple): tuple containing:

                pi_action (tf.Tensor): The actions given by the policy
                logp_pi (tf.Tensor): The log probabilities of each of these
                    actions.
                Q1 (tf.Tensor): Q-values of the first critic.
                Q2 (tf.Tensor): Q-values of the second critic.

        .. note::
            Useful for when you want to print out the full network graph using
            tensorboard.
        """
        obs, act = inputs
        pi_action, logp_pi = self.pi(obs)
        Q1 = self.Q1([obs, act])
        Q2 = self.Q2([obs, act])
        return pi_action, logp_pi, Q1, Q2

    @tf.function
    def act(self, obs, deterministic=False):
        """Returns the action from the current state given the current policy.

        Args:
            obs (tf.Tensor): The current observation (state).
            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If false the action is sampled from the stochastic
                policy. Defaults to ``False``.
        Returns:
            numpy.ndarray: The action from the current state given the current
            policy.
        """
        # Make sure the batch dimension is present (Required by tf.keras.layers.Dense)
        if obs.shape.ndims == 1:
            obs = tf.reshape(obs, (1, -1))

        a, _ = self.pi(obs, deterministic, False)
        return a