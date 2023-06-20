"""Squashed Gaussian Actor policy.

This module contains a Tensorflow 2.x implementation of the Squashed Gaussian Actor
policy of `Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import nn

from bayesian_learning_control.control.algos.tf2.common.bijectors import SquashBijector
from bayesian_learning_control.control.algos.tf2.common.helpers import clamp, mlp


class SquashedGaussianActor(tf.keras.Model):
    """The squashed gaussian actor network.

    Attributes:
        net (tf.keras.Sequential): The input/hidden layers of the
            network.
        mu (tf.keras.Sequential): The output layer which returns the mean of
            the actions.
        log_std_layer (tf.keras.Sequential): The output layer which returns
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
        activation=nn.relu,
        output_activation=nn.relu,
        act_limits=None,
        log_std_min=-20,
        log_std_max=2.0,
        name="gaussian_actor",
        **kwargs,
    ):
        """Constructs all the necessary attributes for the Squashed Gaussian Actor
        object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (:obj:`tf.keras.activations`): The activation function. Defaults
                to :obj:`tf.nn.relu`.
            output_activation (:obj:`tf.keras.activations`, optional): The activation
                function used for the output layers. Defaults to :obj:`tf.nn.relu`.
            act_limits (dict): The ``high`` and ``low`` action bounds of the
                environment. Used for rescaling the actions that comes out of network
                from ``(-1, 1)`` to ``(low, high)``.
            log_std_min (int, optional): The minimum log standard deviation. Defaults
                to ``-20``.
            log_std_max (float, optional): The maximum log standard deviation. Defaults
                to ``2.0``.
            name (str, optional): The Lyapunov critic name. Defaults to
                ``gaussian_actor``.
            **kwargs: All kwargs to pass to the :mod:`tf.keras.Model`. Can be used
                to add additional inputs or outputs.
        """
        super().__init__(name=name, **kwargs)
        self.act_limits = act_limits
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # Create squash bijector, and normal distribution (Used in the
        # re-parameterization trick)
        self._squash_bijector = SquashBijector()
        self._normal_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(act_dim), scale_diag=tf.ones(act_dim)
        )

        self.net = mlp(
            [obs_dim] + list(hidden_sizes),
            activation,
            output_activation,
            name=name,
        )
        self.mu_layer = tf.keras.layers.Dense(
            act_dim,
            input_shape=(hidden_sizes[-1],),
            activation=None,
            name=name + "/mu",
        )
        self.log_std_layer = tf.keras.layers.Dense(
            act_dim,
            input_shape=(hidden_sizes[-1],),
            activation=None,
            name=name + "/log_std",
        )

    @tf.function
    def call(self, obs, deterministic=False, with_logprob=True):
        """Perform forward pass through the network.

        Args:
            obs (numpy.ndarray): The tensor of observations.
            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If ``False`` the action is sampled from the
                stochastic policy. Defaults to ``False``.
            with_logprob (bool, optional): Whether we want to return the log probability
                of an action. Defaults to ``True``.

        Returns:
            (tuple): tuple containing:

                - pi_action (:obj:`tensorflow.Tensor`): The actions given by the policy.
                - logp_pi (:obj:`tensorflow.Tensor`): The log probabilities of each of these actions.
        """  # noqa: E501
        # Calculate mean action and standard deviation
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = tf.clip_by_value(log_std, self._log_std_min, self._log_std_max)
        std = tf.exp(log_std)

        # Create affine bijector (Used in the re-parameterization trick)
        affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(std))

        # Pre-squash distribution and sample
        if deterministic:
            pi_action = mu  # determinestic action used at test time.
        else:
            # Sample from the normal distribution and calculate the action
            batch_size = tf.shape(input=obs)[0]
            epsilon = self._normal_distribution.sample(batch_size)
            pi_action = affine_bijector.forward(
                epsilon
            )  # Transform action as it was sampled from the policy distribution

        # Squash the action between (-1 and 1)
        pi_action = self._squash_bijector.forward(pi_action)

        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        if with_logprob:
            # Transform base_distribution to the policy distribution
            reparm_trick_bijector = tfp.bijectors.Chain(
                (self._squash_bijector, affine_bijector)
            )
            pi_distribution = tfp.distributions.TransformedDistribution(
                distribution=self._normal_distribution, bijector=reparm_trick_bijector
            )
            logp_pi = pi_distribution.log_prob(pi_action)
        else:
            logp_pi = None

        # Clamp the actions such that they are in range of the environment
        if self.act_limits is not None:
            pi_action = clamp(
                pi_action,
                min_bound=self.act_limits["low"],
                max_bound=self.act_limits["high"],
            )

        return pi_action, logp_pi

    @tf.function
    def act(self, obs, deterministic=False):
        """Returns the action from the current state given the current policy.

        Args:
            obs (numpy.ndarray): The current observation (state).
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

        a, _ = self(obs, deterministic, False)
        return a

    @tf.function
    def get_action(self, obs, deterministic=False):
        """Simple wrapper for making the :meth:`.act` method available under the
        'get_action' alias.

        Args:
            obs (numpy.ndarray): The current observation (state).
            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If ``False`` the action is sampled from the
                stochastic policy. Defaults to ``False``.
        Returns:
            numpy.ndarray: The action from the current state given the current
            policy.
        """
        return self.act(obs, deterministic=deterministic)
