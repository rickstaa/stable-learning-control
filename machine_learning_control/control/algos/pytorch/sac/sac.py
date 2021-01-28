"""(soft) Actor-Critic algorithm

This module contains a wrapper that can be used for running a Pytorch implementation of
the SAC algorithm of `Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_. This
wrapper runs the `:module:lac` algorithm with the ``use_lyapunov`` input set to false.
This results in a agent that is equivalent to the SAC agent.
"""

import os.path as osp

from machine_learning_control.control.algos.pytorch.lac import lac
from machine_learning_control.control.algos.pytorch.policies import SoftActorCritic
from machine_learning_control.control.utils.logx import colorize


def sac(*args, **kwargs):
    """Wrapper that calles the `:module:lac` algorithm with ``use_lyapunov`` false."""

    # Set sac related arguments
    kwargs["use_lyapunov"] = False
    kwargs["actor_critic"] = SoftActorCritic

    # Train sac agent
    lac(*args, **kwargs)
    # TODO: Parse activations


if __name__ == "__main__":
    file_name = osp.basename(__file__)
    print(
        colorize(
            (
                f"WARN: Running the '{file_name}' sac algorithm directly "
                "is currently not supported. In order to train the SAC algorithm "
                "your advised to use the CLI or run the LAC algorithm while setting "
                "the 'use_lyapunov' flag to False."
            ),
            "yellow",
        )
    )
