"""
Soft Actor-Critic algorithm
===========================

This module contains a wrapper that can be used for running a Tensorflow 2.x
implementation of the SAC algorithm of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_. This wrapper runs the
:mod:`~bayesian_learning_control.control.algos.tf2.lac.lac.lac` algorithm with the
``use_lyapunov`` input set to false. This results in a agent that is equivalent to the
SAC agent.
"""

import argparse
import os.path as osp
import time

import gym
from bayesian_learning_control.control.algos.tf2.lac.lac import LAC, lac
from bayesian_learning_control.control.algos.tf2.policies.soft_actor_critic import (
    SoftActorCritic,
)  # noqa: E501
from bayesian_learning_control.control.utils import safer_eval
from bayesian_learning_control.utils.import_utils import import_tf
from bayesian_learning_control.utils.log_utils import setup_logger_kwargs

nn = import_tf(module_name="tensorflow.nn")

# Script settings
STD_OUT_LOG_VARS_DEFAULT = [
    "Epoch",
    "TotalEnvInteracts",
    "AverageEpRet",
    "AverageTestEpRet",
    "AverageEpLen",
    "AverageTestEpLen",
    "AverageAlpha",
    "AverageLambda",
    "AverageLossAlpha",
    "AverageLossLambda",
    "AverageLossPi",
    "AverageEntropy",
]


def apply_sac_defaults(args, kwargs):
    """Function that applies the
    :mod:`bayesian_learning_control.control.algos.tf2.sac.sac` defaults to the input
    arguments and returns them.

    Args:
        args (list): The args list.
        kwargs (dict): The kwargs dictionary.

    Returns:
        (tuple): tuple containing:

            - args (:obj:`list`): The args list.
            - kwargs (:obj:`dict`): The kwargs dictionary.
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
    kwargs["opt_type"] = (
        kwargs["opt_type"] if "opt_type" in kwargs.keys() else "maximize"
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
    """Wrapper that calles the
    :mod:`~bayesian_learning_control.control.algos.tf2.lac.lac`  algorithm with
    ``use_lyapunov`` false. It also sets up some :attr:`sac` related default arguments.

    Args:
        *args: All args to pass to thunk.
        **kwargs: All kwargs to pass to thunk.
    """
    args, kwargs = apply_sac_defaults(args, kwargs)
    lac(*args, **kwargs)


if __name__ == "__main__":

    # Import gym environments
    import bayesian_learning_control.simzoo.simzoo.envs  # noqa: F401

    parser = argparse.ArgumentParser(
        description="Trains a SAC agent in a given environment."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Oscillator-v1",
        help="the gym env (default: Oscillator-v1)",
    )
    parser.add_argument(
        "--hid_a",
        type=int,
        default=256,
        help="hidden layer size of the actor (default: 256)",
    )
    parser.add_argument(
        "--hid_c",
        type=int,
        default=256,
        help="hidden layer size of the lyapunov critic (default: 256)",
    )
    parser.add_argument(
        "--l_a",
        type=int,
        default=2,
        help="number of hidden layer in the actor (default: 2)",
    )
    parser.add_argument(
        "--l_c",
        type=int,
        default=2,
        help="number of hidden layer in the critic (default: 2)",
    )
    parser.add_argument(
        "--act_a",
        type=str,
        default="nn.relu",
        help="the hidden layer activation function of the actor (default: nn.relu)",
    )
    parser.add_argument(
        "--act_c",
        type=str,
        default="nn.relu",
        help="the hidden layer activation function of the critic (default: nn.relu)",
    )
    parser.add_argument(
        "--act_out_a",
        type=str,
        default="nn.relu",
        help="the output activation function of the actor (default: nn.relu)",
    )
    parser.add_argument(
        "--act_out_c",
        type=str,
        default="None",
        help="the output activation function of the critic (default: None)",
    )
    parser.add_argument(
        "--opt_type",
        type=str,
        default="minimize",
        help="algorithm optimization type (default: minimize)",
    )
    parser.add_argument(
        "--max_ep_len",
        type=int,
        default=500,
        help="maximum episode length (default: 500)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="the number of epochs (default: 50)"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=2048,
        help="the number of steps per epoch (default: 2048)",
    )
    parser.add_argument(
        "--start_steps",
        type=int,
        default=0,
        help="the number of random exploration steps (default: 0)",
    )
    parser.add_argument(
        "--update_every",
        type=int,
        default=100,
        help="the number of steps for each SGD update (default: 100)",
    )
    parser.add_argument(
        "--update_after",
        type=int,
        default=1000,
        help="the number of steps before starting the SGD (default: 1000)",
    )
    parser.add_argument(
        "--steps_per_update",
        type=int,
        default=100,
        help=(
            "the number of gradient descent steps that are"
            "performed for each SGD update (default: 100)"
        ),
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=10,
        help="the number of episodes for the performance analysis (default: 10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="the entropy regularization coefficient (default: 0.99)",
    )
    parser.add_argument(
        "--alpha3",
        type=float,
        default=0.2,
        help="the Lyapunov constraint error boundary (default: 0.2)",
    )
    parser.add_argument(
        "--labda",
        type=float,
        default=0.99,
        help="the Lyapunov lagrance multiplier (default: 0.99)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.995, help="discount factor (default: 0.995)"
    )
    parser.add_argument(
        "--polyak",
        type=float,
        default=0.995,
        help="the interpolation factor in polyak averaging (default: 0.995)",
    )
    parser.add_argument(
        "--target_entropy",
        type=float,
        default=None,
        help="the initial target entropy (default: -action_space)",
    )
    parser.add_argument(
        "--adaptive_temperature",
        type=bool,
        default=True,
        help="the boolean for enabling automating Entropy Adjustment (default: True)",
    )
    parser.add_argument(
        "--lr_a", type=float, default=1e-4, help="actor learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--lr_c", type=float, default=3e-4, help="critic learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--lr_a_final",
        type=float,
        default=1e-10,
        help="the finalactor learning rate (default: 1e-10)",
    )
    parser.add_argument(
        "--lr_c_final",
        type=float,
        default=1e-10,
        help="the finalcritic learning rate (default: 1e-10)",
    )
    parser.add_argument(
        "--lr_decay_type",
        type=str,
        default="linear",
        help="the learning rate decay type (default: linear)",
    )
    parser.add_argument(
        "--lr_decay_ref",
        type=str,
        default="epoch",
        help=(
            "the reference variable that is used for decaying the learning rate "
            "'epoch' or 'step' (default: 'epoch')"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=float,
        default=256,
        help="mini batch size of the SGD (default: 256)",
    )
    parser.add_argument(
        "--replay-size",
        type=int,
        default=int(1e6),
        help="replay buffer size (default: 1e6)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0, help="the random seed (default: 0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="The device the networks are placed on (default: gpu)",
    )
    parser.add_argument(
        "--start_policy",
        type=str,
        default=None,
        help=(
            "The policy which you want to use as the starting point for the training"
            " (default: None)"
        ),
    )
    parser.add_argument(
        "--export",
        type=str,
        default=False,
        help=(
            "Wether you want to export the model in the 'SavedModel' format "
            "such that it can be deployed to hardware (Default: False)"
        ),
    )

    # Parse logger related arguments
    parser.add_argument(
        "--exp_name",
        type=str,
        default="sac",
        help="the name of the experiment (default: sac)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=bool,
        default=True,
        help="log diagnostics to std out (default: True)",
    )
    parser.add_argument(
        "--verbose_fmt",
        type=str,
        default="line",
        help=(
            "log diagnostics std out format (options: 'table' or 'line', default: "
            "line)"
        ),
    )
    parser.add_argument(
        "--verbose_vars",
        nargs="+",
        default=STD_OUT_LOG_VARS_DEFAULT,
        help=("a space seperated list of the values you want to show on the std out."),
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=2,
        help="how often (in epochs) the policy should be saved (default: 2)",
    )
    parser.add_argument(
        "--save_checkpoints",
        type=bool,
        default=False,
        help="use model checkpoints (default: False)",
    )
    parser.add_argument(
        "--use_tensorboard",
        type=bool,
        default=False,
        help="use tensorboard (default: False)",
    )
    parser.add_argument(
        "--tb_log_freq",
        type=str,
        default="low",
        help=(
            "the tensorboard log frequency. Options are 'low' (Recommended: logs at "
            "every epoch) and 'high' (logs at every SGD update batch). Default is 'low'"
            ""
        ),
    )
    args = parser.parse_args()

    # Setup actor critic arguments
    actor_critic = SoftActorCritic
    output_activation = {}
    output_activation["actor"] = safer_eval(args.act_out_a, backend="tf")
    output_activation["critic"] = safer_eval(args.act_out_c, backend="tf")
    ac_kwargs = dict(
        hidden_sizes={
            "actor": [args.hid_a] * args.l_a,
            "critic": [args.hid_c] * args.l_c,
        },
        activation={
            "actor": safer_eval(args.act_a, backend="tf"),
            "critic": safer_eval(args.act_c, backend="tf"),
        },
        output_activation=output_activation,
    )

    # Setup output dir for logger and return output kwargs
    logger_kwargs = setup_logger_kwargs(
        args.exp_name,
        args.seed,
        save_checkpoints=args.save_checkpoints,
        use_tensorboard=args.use_tensorboard,
        tb_log_freq=args.tb_log_freq,
        verbose=args.verbose,
        verbose_fmt=args.verbose_fmt,
        verbose_vars=args.verbose_vars,
    )
    logger_kwargs["output_dir"] = osp.abspath(
        osp.join(
            osp.dirname(osp.realpath(__file__)),
            f"../../../../../data/lac/{args.env.lower()}/runs/run_{int(time.time())}",
        )
    )

    sac(
        lambda: gym.make(args.env, reference_type="periodic"),
        actor_critic=actor_critic,
        ac_kwargs=ac_kwargs,
        use_lyapunov=False,
        opt_type=args.opt_type,
        max_ep_len=args.max_ep_len,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        start_steps=args.start_steps,
        update_every=args.update_every,
        update_after=args.update_after,
        steps_per_update=args.steps_per_update,
        num_test_episodes=args.num_test_episodes,
        alpha=args.alpha,
        alpha3=args.alpha3,
        labda=args.labda,
        gamma=args.gamma,
        polyak=args.polyak,
        target_entropy=args.target_entropy,
        adaptive_temperature=args.adaptive_temperature,
        lr_a=args.lr_a,
        lr_c=args.lr_c,
        lr_a_final=args.lr_a_final,
        lr_c_final=args.lr_c_final,
        lr_decay_type=args.lr_decay_type,
        lr_decay_ref=args.lr_decay_ref,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        seed=args.seed,
        save_freq=args.save_freq,
        device=args.device,
        start_policy=args.start_policy,
        export=args.export,
        logger_kwargs=logger_kwargs,
    )
