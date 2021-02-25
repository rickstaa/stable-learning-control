"""Several Tensorflow helper functions.
"""

import numpy as np
import machine_learning_control.control.utils.log_utils as log_utils
import tensorflow as tf
from machine_learning_control.control.common.helpers import convert_to_tuple


def set_device(device_type="cpu"):
    """Sets the computational device given a device type.

    Args:
        device_type (str): The device type (options: ``cpu`` and
            ``gpu``). Defaults to ``cpu``.

    Returns:
        str: The type of device that is used.
    """
    if device_type.lower() == "cpu":
        tf.config.set_visible_devices([], "GPU")  # Force disable GPU
    log_utils.log(f"Tensorflow is using the {device_type.upper()}.", type="info")
    return device_type.lower()


def mlp(sizes, activation, output_activation=None, name=""):
    """Create a multi-layered perceptron using Tensorflow.

    Args:
        sizes (list): The size of each of the layers.
        activation (:obj:`tf.keras.activations`): The activation function used for the
            hidden layers.
        output_activation (:obj:`tf.keras.activations`, optional): The activation
            function used for the output layers. Defaults to ``None``.
        name (str, optional): A nameprefix that is added before the layer name. Defaults
            to an empty string.

    Returns:
        tf.keras.Sequential: The multi-layered perceptron.
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [
            tf.keras.layers.Dense(
                sizes[j + 1],
                input_shape=convert_to_tuple(sizes[j]),
                activation=act,
                name=name + "/l{}".format(j + 1),
            )
        ]
    return tf.keras.Sequential(layers)


def count_vars(module):
    """Returns the total number of parameters of a tensorflow module.

    Args:
        module (Union[tf.keras.Model, tf.module]): The tensorflow model.

    Returns:
        numpy.int64: The total number of parameters inside the module.
    """
    return sum([np.prod(p.shape) for p in module.trainable_variables])


def clamp(data, min_bound, max_bound):
    """Clamp all the values of a input to be between the min and max boundaries.

    Args:
        data (Union[numpy.ndarray, list]): Input data.
        min_bound (Union[numpy.ndarray, list]): Array containing the desired minimum
            values.
        max_bound (Union[numpy.ndarray, list]): Array containing the desired maximum
            values.

    Returns:
        numpy.ndarray: Array which has it values clamped between the min and max
            boundaries.
    """
    return (data + 1.0) * (max_bound - min_bound) / 2 + min_bound
