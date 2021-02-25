"""Lyapunov critic policy.

This module contains a Tensorflow 2.x implementation of the Lyapunov Critic policy of
`Han et al. 2020 <http://arxiv.org/abs/2004.14288>`_.
"""

import tensorflow as tf
from machine_learning_control.control.algos.tf2.common.helpers import mlp
from tensorflow import nn


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
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (:obj:`tf.keras.activations`): The activation function. Defaults
                to :obj:`tf.nn.relu`.
            name (str, optional): The Lyapunov critic name. Defaults to
                ``lyapunov_critic``.
            **kwargs: All kwargs to pass to the :mod:`tf.keras.Model`. Can be used
                to add additional inputs or outputs.
        """
        super().__init__(name=name, **kwargs)
        self.L = mlp(
            [obs_dim + act_dim] + list(hidden_sizes), activation, activation, name=name
        )

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
