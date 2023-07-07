"""Several TensorFlow helper functions.
"""
import numpy as np
import tensorflow as tf

from stable_learning_control.algos.common.helpers import get_activation_function
from stable_learning_control.common.helpers import convert_to_tuple
from stable_learning_control.utils.log_utils.helpers import log_to_std_out


def set_device(device_type="cpu"):
    """Sets the computational device given a device type.

    Args:
        device_type (str): The device type (options: ``cpu`` and
            ``gpu``). Defaults to ``cpu``.

    Returns:
        str: The type of device that is used.
    """
    if device_type.lower() == "cpu":
        tf.config.set_visible_devices([], "GPU")  # Force disable GPU.
    log_to_std_out(f"TensorFlow is using the {device_type.upper()}.", type="info")
    return device_type.lower()


def mlp(sizes, activation, output_activation=None, name=""):
    """Create a multi-layered perceptron using TensorFlow.

    Args:
        sizes (list): The size of each of the layers.
        activation (union[:obj:`tf.keras.activations`, :obj:`str`]): The activation
            function used for the hidden layers.
        output_activation (union[:obj:`tf.keras.activations`, :obj:`str`], optional):
            The activation function used for the output layers. Defaults to ``None``.
        name (str, optional): A nameprefix that is added before the layer name. Defaults
            to an empty string.

    Returns:
        tf.keras.Sequential: The multi-layered perceptron.
    """
    if isinstance(activation, str):
        activation = get_activation_function(activation, backend="tf2")
    if isinstance(output_activation, str):
        output_activation = get_activation_function(output_activation, backend="tf2")

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
    """Returns the total number of parameters of a TensorFlow module.

    Args:
        module (Union[tf.keras.Model, tf.module]): The TensorFlow model.

    Returns:
        numpy.int64: The total number of parameters inside the module.
    """
    return sum([np.prod(p.shape) for p in module.trainable_variables])


def rescale(data, min_bound, max_bound):
    """Rescale normalized data (i.e. between ``-1`` and ``1``) to a desired range.

    Args:
        data (Union[numpy.ndarray, list]): Normalized input data.
        min_bound (Union[numpy.ndarray, list]): Array containing the minimum value of
            the desired range.
        max_bound (Union[numpy.ndarray, list]): Array containing the maximum value of
            the desired range.

    Returns:
        numpy.ndarray: Array which has it values scaled between the min and max
            boundaries.
    """
    return (data + 1.0) * (max_bound - min_bound) / 2 + min_bound


def full_model_summary(model):
    """Prints a full summary of all the layers of a TensorFlow model.

    Args:
        layer (:tf:`keras.layers`): The model to print the full summary of.
    """
    if hasattr(model, "layers"):
        model.summary()
        print("\n\n")
        for layer in model.layers:
            full_model_summary(layer)