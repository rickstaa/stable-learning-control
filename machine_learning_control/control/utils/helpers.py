"""A number of useful helper functions.
"""

import decimal

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
        length (int): The length of an object.
        shape (tuple, optional): The shape of an object. Only uses length if not
            supplied.

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


def compare_models(model_1, model_2):
    """Compares two models to see if the weights are equal.

    Args:
        model_1 (torch.nn.module): The first Pytorch model.
        model_2 (torch.nn.module): The second Pytorch model.

    Raises:
        Exception: Raises Key error if the graph of the two models is different.

    Returns:
        Bool: Bool specifying whether the weights of two models are equal.
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismatch found at", key_item_1[0])
            else:
                raise KeyError(
                    "Model weights could not be compared between the two models as "
                    f"the two models appear to be different. Parameter {key_item_1[0]} "
                    f"which is found in model 1, does not exist in the model 2."
                )

    # Return result
    if models_differ == 0:
        print("Models match perfectly! :)")
        return True
    else:
        return False


def clamp(data, min_bound, max_bound):
    """Clamp all the values of a input to be between the min and max boundaries.

    Args:
        data (np.ndarray/list): Input data.
        min_bound (np.ndarray/list): Array containing the desired minimum values.
        max_bound (np.ndarray/list): Array containing the desired maximum values.

    Returns:
        np.ndarray: Array which has it values clamped between the min and max
            boundaries.
    """

    # Convert arguments to torch tensor if not already
    data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
    min_bound = (
        torch.tensor(min_bound)
        if not isinstance(min_bound, torch.Tensor)
        else min_bound
    )
    max_bound = (
        torch.tensor(max_bound)
        if not isinstance(max_bound, torch.Tensor)
        else max_bound
    )

    # Clamp all actions to be within the boundaries
    return (data + 1.0) * (max_bound - min_bound) / 2 + min_bound


def calc_gamma_lr_decay(lr_init, lr_final, epoch):
    """Returns the exponential decay factor (gamma) needed to achieve a given final
    learning rate at a certain epoch. This decay factor can for example be used with a
    :py:class:`torch.optim.lr_scheduler.LambdaLR` scheduler. Keep in mind that this
    function assumes the following formula for the learning rate decay.

    .. math::
        lr_{terminal} = lr_{init} \cdot  \gamma^{epoch}

    Args:
        lr_init (float): The initial learning rate.
        lr_final (float): The final learning rate you want to achieve.
        epoch (int): The eposide at which you want to achieve this learning rate.

    Returns:
        decimal.Decimal: Exponential learning rate decay factor (gamma).
    """
    gamma_a = (decimal.Decimal(lr_final) / decimal.Decimal(lr_init)) ** (
        decimal.Decimal(1.0) / decimal.Decimal(epoch)
    )
    return gamma_a


def calc_linear_lr_decay(lr_init, lr_final, epoch):
    """Returns the linear decay factor (G) needed to achieve a given final learning
    rate at a certain epoch. This decay factor can for example be used with a
    :py:class:`torch.optim.lr_scheduler.LambdaLR` scheduler. Keep in mind that this
    function assumes the following formula for the learning rate decay.

    .. math::
        lr_{terminal} = lr_{init} * (1.0 - G \cdot epoch)

    Args:
        lr_init (float): The initial learning rate.
        lr_final (float): The final learning rate you want to achieve.
        epoch (int): The eposide at which you want to achieve this learning rate.

    Returns:
        decimal.Decimal: Linear learning rate decay factor (G)
    """
    return -(
        ((decimal.Decimal(lr_final) / decimal.Decimal(lr_init)) - decimal.Decimal(1.0))
        / (decimal.Decimal(epoch) - decimal.Decimal(1.0))
    )
