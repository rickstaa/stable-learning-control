"""Contains functions used for creating Pytorch learning rate schedulers."""

from decimal import Decimal

import numpy as np
import torch

import torch.optim


class ConstantLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    """A learning rate scheduler that keeps the learning rate constant."""

    def __init__(self, optimizer):
        """Initialize the constant learning rate scheduler.

        Args:
            optimizer (:class:`torch.optim.Optimizer`): The wrapped optimizer.
        """
        super().__init__(optimizer, lr_lambda=lambda step: np.longdouble(1.0))


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


def get_linear_decay_rate(lr_init, lr_final, steps):
    r"""Returns a linear decay factor (G) that enables a learning rate to transition
    from an initial value (`lr_init`) at step 0 to a final value (`lr_final`) at a
    specified step (N). This decay factor is compatible with the
    :class:`torch.optim.lr_scheduler.LambdaLR` scheduler. The decay factor is calculated
    using the following formula:

    .. math::
        lr_{terminal} = lr_{init} * (1.0 - G \cdot step)

    Args:
        lr_init (float): The initial learning rate.
        lr_final (float): The final learning rate you want to achieve.
        steps (int): The number of steps/epochs over which the learning rate should
            decay. This is equal to epochs -1.

    Returns:
        decimal.Decimal: Linear learning rate decay factor (G).
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
        lr_final (float): Final learning rate.
        steps (int, optional): Number of steps/epochs used in the training. This
            includes the starting step/epoch.

    Returns:
        :obj:`torch.optim.lr_scheduler`: A learning rate scheduler object.

    .. seealso::
        See the :torch:`pytorch <docs/stable/optim.html>` documentation on how to
        implement other decay options.
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
                        get_linear_decay_rate(lr_start, lr_final, (steps - 1.0))
                        * Decimal(step)
                    )
                )

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_multiplier_function
            )
        return lr_scheduler
    else:
        return ConstantLRScheduler(optimizer)


def estimate_step_learning_rate(
    lr_scheduler, lr_start, lr_final, update_after, total_steps, step
):
    """Estimates the learning rate at a given step.

    This function estimates the learning rate for a specific training step. It differs
    from the `get_last_lr` method of the learning rate scheduler, which returns the
    learning rate at the last scheduler step, not necessarily the current training step.

    Args:
        lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        lr_start (float): The initial learning rate.
        update_after (int): The step number after which the learning rate should start
            decreasing.
        lr_final (float): The final learning rate.
        total_steps (int): The total number of steps/epochs in the training process.
            Excludes the initial step.
        step (int): The current step number. Excludes the initial step.

    Returns:
        float: The learning rate at the given step.
    """
    if step < update_after or isinstance(lr_scheduler, ConstantLRScheduler):
        return lr_start

    # Estimate the learning rate at a given step for the lt_scheduler type.
    adjusted_step = step - update_after
    adjusted_total_steps = total_steps - update_after
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR):
        decay_rate = get_linear_decay_rate(lr_start, lr_final, adjusted_total_steps)
        lr = float(
            Decimal(lr_start) * (Decimal(1.0) - decay_rate * Decimal(adjusted_step))
        )
    elif isinstance(lr_scheduler, torch.optim.lr_scheduler.ExponentialLR):
        decay_rate = get_exponential_decay_rate(
            lr_start, lr_final, adjusted_total_steps
        )
        lr = float(
            Decimal(lr_start) * (Decimal(decay_rate) ** Decimal(adjusted_step))
        )
    else:
        supported_schedulers = ["LambdaLR", "ExponentialLR"]
        raise ValueError(
            f"The learning rate scheduler is not supported for this function. "
            f"Supported schedulers are: {', '.join(supported_schedulers)}"
        )
    return max(lr, lr_final)
