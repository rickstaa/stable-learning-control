"""Lyapunov critic policy.

This module contains a TensorFlow 2.x implementation of the Lyapunov Critic policy of
`Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_.
"""

import tensorflow as tf
from tensorflow import nn

from stable_learning_control.algos.tf2.common.helpers import mlp


class LCritic(tf.keras.Model):
    """Soft Lyapunov critic Network.

    Attributes:
        L (tf.keras.Sequential): The layers of the network.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=nn.relu,
        name="lyapunov_critic",
        **kwargs,
    ):
        """Initialise the LCritic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (:obj:`tf.keras.activations`, optional): The activation
                function. Defaults to :obj:`tf.nn.relu`.
            name (str, optional): The Lyapunov critic name. Defaults to
                ``lyapunov_critic``.
            **kwargs: All kwargs to pass to the :mod:`tf.keras.Model`. Can be used to
                add additional inputs or outputs.
        """
        super().__init__(name=name, **kwargs)
        self.L = mlp(
            [obs_dim + act_dim] + list(hidden_sizes), activation, activation, name=name
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
                The tensor containing the lyapunov values of the input observations and
                actions.
        """
        L_hid_out = self.L(tf.concat(inputs, axis=-1))
        L_out = tf.math.square(L_hid_out)
        L_out = tf.reduce_sum(L_out, axis=1)

        return L_out
