"""A number of usefull helper functions.
"""

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, activation, output_activation=nn.Identity):
    """Create a multi-layered perceptron using pytorch.

    Args:
        sizes (list): The size of each of the layers.

        activation (torch.nn.modules.activation): The activation function used for the
            hidden layers.

        output_activation (torch.nn.modules.activation, optional): The activation
            function used for the output layers. Defaults to torch.nn.Identity.

    Returns:
        torch.nn.modules.container.Sequential: The multi-layered perceptron.
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def combined_shape(length, shape=None):
    """Combines length and shape objects into one tuple.

    Args:
        length (int, optional): The length of an object. Defaults to None.

        shape (tuple): The shape of an object.

    Returns:
        Tuple: A tuple in which the length and shape are combined.
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    """Returns the total number of parameters of a pytorch module.

    Args:
        module (torch.nn.module): The module.

    Returns:
        numpy.int64: The total number of parameters inside the module.
    """
    return sum([np.prod(p.shape) for p in module.parameters()])


def is_scalar(obj):
    """Recursive function that checks whether a input

    Args:
        obj (object): Object for which you want to check if it is a scalar.

    Returns:
        boole: Boolean specifying whether the object is a scalar.
    """

    # Check if obj is scalar
    if type(obj) in [int, float]:
        return True
    elif np.isscalar(obj):
        if type(obj) == np.str_:
            try:
                float(obj)
                return True
            except ValueError:
                return False
        else:
            return True
    elif type(obj) == np.ndarray:
        if obj.shape == (1,):
            return is_scalar(obj[0])
        else:
            return False
    elif type(obj) == torch.Tensor:
        if len(obj.shape) == 0:
            try:
                float(obj)
                return True
            except ValueError:
                return False
        elif len(obj.shape) <= 1:
            if obj.shape[0] <= 1:
                return is_scalar(obj[0])
            else:
                return False
        else:
            return False
    elif type(obj) == str:
        try:
            float(obj)
            return True
        except ValueError:
            return False
    else:
        return False
