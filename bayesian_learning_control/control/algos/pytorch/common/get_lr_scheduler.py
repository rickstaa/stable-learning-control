"""Contains functions used for creating Pytorch learning rate schedulers.

.. rubric:: Functions

.. autofunction:: get_exponential_decay_rate
.. autofunction:: calc_linear_decay_rate
.. autofunction:: get_lr_scheduler
"""  # NOTE: Manual autofunction request was added because of bug https://github.com/sphinx-doc/sphinx/issues/7912#issuecomment-786011464  # noqa: E501

from decimal import Decimal

import numpy as np
import torch


def get_exponential_decay_rate(lr_start, lr_final, steps):
    """Calculates the exponential decay rate needed to go from a initial learning rate
    to a final learning rate in N steps.

    Args:
        lr_start (float): The starting learning rate.
        lr_final (float): The final learning rate.
        steps (int): The number of steps.

    Returns:
        decimal.Decimal: The exponential decay rate (high precision).
    """
    gamma = (Decimal(lr_final) / Decimal(lr_start)) ** (Decimal(1.0) / Decimal(steps))
    return gamma


def calc_linear_decay_rate(lr_init, lr_final, steps):
    """Returns the linear decay factor (G) needed to achieve a given final learning
    rate at a certain step. This decay factor can for example be used with a
    :py:class:`torch.optim.lr_scheduler.LambdaLR` scheduler. Keep in mind that this
    function assumes the following formula for the learning rate decay.

    .. math::
        lr_{terminal} = lr_{init} * (1.0 - G \cdot step)

    Args:
        lr_init (float): The initial learning rate.
        lr_final (float): The final learning rate you want to achieve.
        steps (int): The step/epoch at which you want to achieve this learning rate.

    Returns:
        decimal.Decimal: Linear learning rate decay factor (G)
    """  # noqa: W605
    return -(
        ((Decimal(lr_final) / Decimal(lr_init)) - Decimal(1.0)) / Decimal(max(steps, 1))
    )


def get_lr_scheduler(optimizer, decaying_lr_type, lr_start, lr_final, steps):
    """Creates a learning rate scheduler.

    Args:
        optimizer (torch.optim.Adam): Wrapped optimizer.
        decaying_lr_type (str): The learning rate decay type that is used
            (options are: ``linear`` and ``exponential`` and ``constant``).
        lr_start (float): Initial learning rate.
        lr_end (float): Final learning rate.
        steps (int, optional): Number of steps/epochs used in the training.  This
            includes the starting step.

    Returns:
        :obj:`torch.optim.lr_scheduler`: A learning rate scheduler object.

    .. seealso::
        See the `pytorch <https://pytorch.org/docs/stable/optim.html>`_ documentation on
        how to implement other decay options.
    """  # noqa: E501
    if decaying_lr_type.lower() != "constant" and lr_start != lr_final:
        if decaying_lr_type.lower() == "exponential":
            exponential_decay_rate = get_exponential_decay_rate(
                lr_start, lr_final, (steps - 1.0)
            )
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, np.float64(exponential_decay_rate)
            )
        else:

            def lr_multiplier_function(step):
                """Returns a learning rate multiplier at each steps that makes the
                learning rate decay linearly.

                Returns:
                    numpy.longdouble: A learning rate multiplier.
                """
                return np.longdouble(
                    Decimal(1.0)
                    - (
                        calc_linear_decay_rate(lr_start, lr_final, (steps - 1.0))
                        * Decimal(step)
                    )
                )

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_multiplier_function
            )
        return lr_scheduler
    else:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: np.longdouble(1.0)
        )  # Return a constant function
