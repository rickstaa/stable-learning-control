"""Soft actor critic policy.

This module contains a TensorFlow 2.x implementation of the Soft Actor Critic policy of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""

import tensorflow as tf
from tensorflow import nn

from stable_learning_control.algos.tf2.policies.actors.squashed_gaussian_actor import (
    SquashedGaussianActor,
)
from stable_learning_control.algos.tf2.policies.critics.Q_critic import QCritic
from stable_learning_control.common.helpers import strict_dict_update

HIDDEN_SIZES_DEFAULT = {"actor": (256, 256), "critic": (256, 256)}
ACTIVATION_DEFAULT = {"actor": nn.relu, "critic": nn.relu}
OUTPUT_ACTIVATION_DEFAULT = {"actor": nn.relu, "critic": None}


class SoftActorCritic(tf.keras.Model):
    """Soft Actor-Critic network.

    Attributes:
        self.pi (:class:`~stable_learning_control.algos.tf2.policies.actors.SquashedGaussianActor`):
            The squashed gaussian policy network (actor).
        self.Q1 (:class:`~stable_learning_control.algos.tf2.policies.critics.QCritic`): The first soft Q-network (critic).
        self.Q1 (:class:`~stable_learning_control.algos.tf2.policies.critics.QCritic`); The second soft Q-network (critic).
    """  # noqa: E501

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=HIDDEN_SIZES_DEFAULT,
        activation=ACTIVATION_DEFAULT,
        output_activation=OUTPUT_ACTIVATION_DEFAULT,
        name="soft_actor_critic",
    ):
        """Initialise the SoftActorCritic object.

        Args:
            observation_space (:obj:`gym.space.box.Box`): A gymnasium observation space.
            action_space (:obj:`gym.space.box.Box`): A gymnasium action space.
            hidden_sizes (Union[dict, tuple, list], optional): Sizes of the hidden
                layers for the actor. Defaults to ``(256, 256)``.
            activation (Union[:obj:`dict`, :obj:`tf.keras.activations`], optional): The
                (actor and critic) hidden layers activation function. Defaults to
                :obj:`tf.nn.relu`.
            output_activation (Union[:obj:`dict`, :obj:`tf.keras.activations`], optional):
                The (actor and critic)  output activation function. Defaults to
                :obj:`tf.nn.relu` for the actor and the Identity function for the
                critic.
            name (str, optional): The name given to the SoftActorCritic. Defaults to
                "soft_actor_critic".
        """  # noqa: E501
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

        # Perform one forward pass to initialise the networks.
        # NOTE: Done because TF doesn't support multiple positional arguments when using
        # the tf.function decorator, and autograph doesn't support list unpacking.
        obs_dummy = tf.random.uniform((1, obs_dim), dtype=tf.float32)
        act_dummy = tf.random.uniform((1, act_dim), dtype=tf.float32)
        self([obs_dummy, act_dummy])

    @tf.function
    def call(self, inputs, deterministic=False, with_logprob=True):
        """Performs a forward pass through all the networks (Actor, Q critic 1 and Q
        critic 2).

        Args:
            inputs (tuple): tuple containing:

                -   obs (tf.Tensor): The tensor of observations.
                -   act (tf.Tensor): The tensor of actions.

            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If false the action is sampled from the stochastic
                policy. Defaults to ``False``.
            with_logprob (bool, optional): Whether we want to return the log probability
                of an action. Defaults to ``True``.

        Returns:
            (tuple): tuple containing:

                - pi_action (:obj:`tensorflow.Tensor`): The actions given by the policy.
                - logp_pi (:obj:`tensorflow.Tensor`): The log probabilities of each of these actions.
                - Q1(:obj:`tensorflow.Tensor`): Q-values of the first critic.
                - Q2(:obj:`tensorflow.Tensor`): Q-values of the second critic.

        .. note::
            Useful for when you want to print out the full network graph using
            TensorBoard.
        """  # noqa: E501
        obs, act = inputs
        pi_action, logp_pi = self.pi(
            obs, deterministic=deterministic, with_logprob=with_logprob
        )
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
                policy is returned. If ``False`` the action is sampled from the
                stochastic policy. Defaults to ``False``.

        Returns:
            numpy.ndarray: The action from the current state given the current
            policy.
        """
        # Make sure the batch dimension is present (Required by tf.keras.layers.Dense)
        if obs.shape.ndims == 1:
            obs = tf.reshape(obs, (1, -1))

        a, _ = self.pi(obs, deterministic, False)

        return tf.squeeze(
            a, axis=0
        )  # NOTE: Squeeze is critical to ensure a has right shape.
