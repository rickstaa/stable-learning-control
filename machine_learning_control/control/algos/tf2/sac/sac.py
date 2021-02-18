"""(soft) Actor-Critic algorithm

This module contains a wrapper that can be used for running a Tensorflow 2.x
implementation of the SAC algorithm of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_. This wrapper runs the
`:module:lac` algorithm with the ``use_lyapunov`` input set to false. This results in a
agent that is equivalent to the SAC agent.
"""

import os.path as osp

import machine_learning_control.control.utils.log_utils as log_utils
from machine_learning_control.control.algos.tf2.lac import LAC, lac
from machine_learning_control.control.algos.tf2.policies import SoftActorCritic
from machine_learning_control.control.utils import import_tf

nn = import_tf(module_name="tensorflow.nn")


def apply_sac_defaults(args, kwargs):
    """Function that applies the :art:`sac` defaults to the input arguments and returns
    them.

    Args:
        args (list): The args list.
        kwargs (dict): The kwargs dictionary.

    Returns:
        (tuple): tuple containing:

            args (list): The args list.
            kwargs (dict): The kwargs dictionary.
    """
    kwargs["use_lyapunov"] = False
    kwargs["actor_critic"] = (
        kwargs["actor_critic"] if "actor_critic" in kwargs.keys() else SoftActorCritic
    )
    kwargs["ac_kwargs"] = (
        kwargs["ac_kwargs"]
        if "ac_kwargs" in kwargs.keys()
        else dict(
            hidden_sizes=[256, 256],
            activation={"actor": nn.relu, "critic": nn.relu},
            output_activation={"actor": nn.relu, "critic": None},
        )
    )
    return args, kwargs


class SAC(LAC):
    """Wrapper class used to create the SAC policy from the LAC policy."""

    def __init__(self, *args, **kwargs):
        """Calls the lac superclass using the sac defaults.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        args, kwargs = apply_sac_defaults(args, kwargs)
        super().__init__(*args, **kwargs)


def sac(*args, **kwargs):
    """Wrapper that calles the `:module:lac` algorithm with ``use_lyapunov`` false. It
    also sets up some :atr:`sac` related default arguments.

    Args:
        *args: All args to pass to thunk.
        **kwargs: All kwargs to pass to thunk.
    """
    args, kwargs = apply_sac_defaults(args, kwargs)
    lac(*args, **kwargs)


if __name__ == "__main__":
    file_name = osp.basename(__file__)
    log_utils.log(
        (
            f"Running the '{file_name}' sac algorithm directly "
            "is currently not supported. In order to train the SAC algorithm "
            "your advised to use the CLI or run the LAC algorithm while setting "
            "the 'use_lyapunov' flag to False."
        ),
        type="warning",
    )
