"""Lyapunov actor critic policy.

This module contains a TensorFlow 2.x implementation of the Q Critic policy of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""

import tensorflow as tf
from tensorflow import nn

from stable_learning_control.algos.tf2.common.helpers import mlp


class QCritic(tf.keras.Model):
    """Soft Q critic network.

    Attributes:
        Q (tf.keras.Sequential): The layers of the network.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=nn.relu,
        output_activation=None,
        name="q_critic",
        **kwargs,
    ):
        """Initialise the QCritic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (:obj:`tf.keras.activations`, optional): The activation function.
                Defaults to :obj:`tf.nn.relu`.
            output_activation (:obj:`tf.keras.activations`, optional): The activation
                function used for the output layers. Defaults to ``None`` which is
                equivalent to using the Identity activation function.
            name (str, optional): The Lyapunov critic name. Defaults to ``q_critic``.
            **kwargs: All kwargs to pass to the :mod:`tf.keras.Model`. Can be used to
                add additional inputs or outputs.
        """
        super().__init__(name=name, **kwargs)
        self.Q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            activation,
            output_activation,
            name=name,
        )

        # Build the model to initialise the (trainable) variables.
        self.build((None, obs_dim + act_dim))

    @tf.function
    def call(self, inputs):
        """Perform forward pass through the network.

        Args:
            inputs (tuple): tuple containing:

                - obs (tf.Tensor): The tensor of observations.
                - act (tf.Tensor): The tensor of actions.

        Returns:
            tf.Tensor:
                The tensor containing the Q values of the input observations and
                actions.
        """
        return tf.squeeze(
            self.Q(tf.concat(inputs, axis=-1)), axis=-1
        )  # NOTE: Squeeze is critical to ensure q has right shape.
