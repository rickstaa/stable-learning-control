"""Module used for creating torch learning rate schedulers.
"""

import decimal

import numpy as np
import torch

# TEST: Validate that it works.


def _calc_gamma_lr_decay(lr_init, lr_final, epoch):
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
    """  # noqa: W605
    gamma_a = (decimal.Decimal(lr_final) / decimal.Decimal(lr_init)) ** (
        decimal.Decimal(1.0) / decimal.Decimal(epoch)
    )
    return gamma_a


def _calc_linear_lr_decay(lr_init, lr_final, epoch):
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
    """  # noqa: W605
    return -(
        ((decimal.Decimal(lr_final) / decimal.Decimal(lr_init)) - decimal.Decimal(1.0))
        / (max(decimal.Decimal(epoch) - decimal.Decimal(1.0), 1))
    )


def get_lr_scheduler(optimizer, decaying_lr_type, lr_start, lr_final, epochs):
    """Creates a learning rate scheduler.

    Args:
        optimizer (torch.optim.Adam): Wrapped optimizer.
        decaying_lr_type (string): The learning rate decay type that is used (
        options are: ``linear`` and ``exponential`` and ``constant``).
        lr_start (float): Initial learning rate.
        lr_end (float): Final learning rate.
        epochs (int, optional): Number of epochs used in the training.

    Returns:
        torch.optim.lr_scheduler: A learning rate scheduler object.
    """

    # Check for constant learning rate
    if lr_start == lr_final or decaying_lr_type.lower() == "constant":
        decaying_lr = False
    else:
        decaying_lr = True

    # Create learning rate scheduler
    if decaying_lr_type.lower() == "exponential":
        gamma = np.longdouble(
            (_calc_gamma_lr_decay(lr_start, lr_final, epochs) if decaying_lr else 1.0)
        )  # The decay exponent
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    else:
        lr_decay_a = (
            (
                lambda epoch: np.longdouble(
                    decimal.Decimal(1.0)
                    - (
                        _calc_linear_lr_decay(lr_start, lr_final, epochs)
                        * decimal.Decimal(epoch)
                    )
                )
            )
            if decaying_lr
            else lambda epoch: 1.0
        )  # Linear decay rate
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_decay_a
        )
    return lr_scheduler
