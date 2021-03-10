"""Module that contains several tensorflow
`bijectors <https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector>`.
For more information on Bijectors see
https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector and
https://stackoverflow.com/questions/56425301/what-is-bijectors-in-layman-terms-in-tensorflow-probability.
"""  # noqa: E501

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import nn


class SquashBijector(tfp.bijectors.Bijector):
    """A squash bijector used to keeps track of the distribution properties when the
    distribution is transformed using the tanh squash function."""

    def __init__(self, validate_args=False, name="tanh"):
        """Initiate squashed bijector object.

        Args:
            validate_args (bool, optional): Whether to validate input with asserts. If
                validate_args is False, and the inputs are invalid, correct behavior is
                not guaranteed. Defaults to False.
            name (str, optional): The name to give Ops created by the initializer.
                Defaults to "tanh".
        """
        super().__init__(
            forward_min_event_ndims=0, validate_args=validate_args, name=name
        )

    def _forward(self, x):
        """Forward function. Useful for turning one random outcome into another random
        outcome from a different distribution.

        Args:
            x (structure): The input to the 'forward' evaluation.

        Returns:
            structure: Returns the forward Bijector evaluation, i.e., X = g(Y).
        """
        return nn.tanh(x)
        # return x

    def _inverse(self, y):
        """Inverse bijection function. Useful for 'reversing' a transformation to
        compute one probability in terms of another.

        Args:
            y (structure): The input to the 'inverse' evaluation.

        Returns:
            structure: Return tensor if this bijector is injective. If not
            injective, returns the k-tuple containing the unique k points (x1, ..., xk)
            such that g(xi) = y.
        """
        return tf.math.atanh(y)

    def _forward_log_det_jacobian(self, x):
        """The log of the absolute value of the determinant of the matrix of all
        first-order partial derivatives of the inverse function. Useful for inverting a
        transformation to compute one probability in terms of another. Geometrically,
        the Jacobian determinant is the volume of the transformation and is used to
        scale the probability.

        Args:
            x (structure): The input to the 'forward' Jacobian determinant evaluation.

        Returns:
            structure: Result tensor if this bijector is injective. If not
                injective this is not implemented.
        """
        return 2.0 * (np.log(2.0) - x - nn.softplus(-2.0 * x))
