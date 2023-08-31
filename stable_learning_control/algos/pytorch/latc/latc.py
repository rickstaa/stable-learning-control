"""Lyapunov (soft) Actor-Twin Critic (LATC) algorithm.

This is a modified version of the Lyapunov Actor-Critic algorithm of
`Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_. Like the original SAC algorithm,
this LAC variant uses two critics instead of one to mitigate a possible underestimation
bias, while the original LAC only uses one critic. For more information, see
`Haarnoja et al. 2018 <https://arxiv.org/pdf/1812.05905.pdf>`_ or the
:ref:`LATC documentation <latc>`.

.. note::
    Code Conventions:
        -   We use a ``_`` suffix to distinguish the next state from the current state.
        -   We use a ``targ`` suffix to distinguish actions/values coming from the target
            network.

.. attention::
    To reduce the amount of code duplication, the code that implements the LATC algorithm
    is found in the :class:`~stable_learning_control.algos.pytorch.lac.lac.LAC` class.
    As a result this module wraps the :func:`~stable_learning_control.algos.pytorch.lac.lac.lac`
    function so that it uses the :class:`~stable_learning_control.algos.pytorch.policies.lyapunov_actor_twin_critic.LyapunovActorTwinCritic`
    as the actor-critic architecture. When this architecture is used, the
    :meth:`~stable_learning_control.algos.pytorch.policies.lyapunov_actor_twin_critic.LyapunovActorTwinCritic.update`
    method is modified such that the two critics.
"""  # noqa: E501
import argparse
import os.path as osp
import time

import gymnasium as gym
import torch

from stable_learning_control.algos.pytorch.lac.lac import lac
from stable_learning_control.algos.pytorch.policies.lyapunov_actor_twin_critic import (
    LyapunovActorTwinCritic,
)
from stable_learning_control.utils.log_utils.helpers import setup_logger_kwargs
from stable_learning_control.utils.safer_eval_util import safer_eval

# Script settings.
STD_OUT_LOG_VARS_DEFAULT = [
    "Epoch",
    "TotalEnvInteracts",
    "AverageEpRet",
    "AverageTestEpRet",
    "AverageTestEpLen",
    "AverageAlpha",
    "AverageLambda",
    "AverageLossAlpha",
    "AverageLossLambda",
    "AverageErrorL",
    "AverageLossPi",
    "AverageEntropy",
]


def latc(env_fn, actor_critic=None, *args, **kwargs):
    """Trains the LATC algorithm in a given environment.

    Args:
        env_fn: A function which creates a copy of the environment. The environment
            must satisfy the gymnasium API.
        actor_critic (torch.nn.Module, optional): The constructor method for a
            Torch Module with an ``act`` method, a ``pi`` module and several
            ``Q`` or ``L`` modules. The ``act`` method and ``pi`` module should
            accept batches of observations as inputs, and the ``Q*`` and ``L``
            modules should accept a batch of observations and a batch of actions as
            inputs. When called, these modules should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)   | Numpy array of actions for each
                                            | observation.
            ``Q*/L``     (batch,)           | Tensor containing one current estimate
                                            | of ``Q*/L`` for the provided
                                            | observations and actions. (Critical:
                                            | make sure to flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)   | Tensor containing actions from policy
                                            | given observations.
            ``logp_pi``  (batch,)           | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly:
                                            | gradients should be able to flow back
                                            | into ``a``.
            ===========  ================  ======================================

            Defaults to
            :class:`~stable_learning_control.algos.pytorch.policies.lyapunov_actor_twin_critic.LyapunovActorTwinCritic`
        *args: The positional arguments to pass to the :meth:`~stable_learning_control.algos.pytorch.lac.lac.lac` method.
        **kwargs: The keyword arguments to pass to the :meth:`~stable_learning_control.algos.pytorch.lac.lac.lac` method.

    .. note::
        Wraps the :func:`~stable_learning_control.algos.pytorch.lac.lac.lac` function so
        that the :class:`~stable_learning_control.algos.pytorch.policies.lyapunov_actor_twin_critic.LyapunovActorTwinCritic`
        architecture is used as the actor critic.
    """  # noqa: E501
    # Get default actor critic if no 'actor_critic' was supplied
    actor_critic = LyapunovActorTwinCritic if actor_critic is None else actor_critic

    # Call lac algorithm.
    return lac(env_fn, actor_critic=actor_critic, *args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a LATC agent in a given environment."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="stable_gym:Oscillator-v1",
        help="the gymnasium env (default: stable_gym:Oscillator-v1)",
    )  # NOTE: Environment found in https://rickstaa.dev/stable-gym.
    parser.add_argument(
        "--hid_a",
        type=int,
        default=64,
        help="hidden layer size of the actor (default: 64)",
    )
    parser.add_argument(
        "--hid_c",
        type=int,
        default=128,
        help="hidden layer size of the lyapunov critic (default: 128)",
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
        default="nn.ReLU",
        help="the hidden layer activation function of the actor (default: nn.ReLU)",
    )
    parser.add_argument(
        "--act_c",
        type=str,
        default="nn.ReLU",
        help="the hidden layer activation function of the critic (default: nn.ReLU)",
    )
    parser.add_argument(
        "--act_out_a",
        type=str,
        default="nn.ReLU",
        help="the output activation function of the actor (default: nn.ReLU)",
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
        default=None,
        help="maximum episode length (default: None)",
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
        help=(
            "the number of env interactions that should elapse between SGD updates "
            "(default: 100)"
        ),
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
        help=(
            "the number of episodes for the performance analysis (default: 10). When "
            "set to zero no test episodes will be performed"
        ),
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
        help="the Lyapunov Lagrance multiplier (default: 0.99)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor (default: 0.99)"
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
            "'epoch' or 'step' (default: epoch)"
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
        "--horizon_length",
        type=int,
        default=0,
        help=(
            "length of the finite-horizon used for the Lyapunov Critic target ( "
            "Default: 0, meaning the infinite-horizon bellman backup is used)."
        ),
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0, help="the random seed (default: 0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "The device the networks are placed on: 'cpu' or 'gpu' (options: "
            "default: cpu)"
        ),
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
            "Whether you want to export the model as a 'TorchScript' such that "
            "it can be deployed on hardware (default: False)"
        ),
    )

    # Parse logger related arguments.
    parser.add_argument(
        "--exp_name",
        type=str,
        default="latc",
        help="the name of the experiment (default: latc)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress logging of diagnostics to stdout (default: False)",
    )
    parser.add_argument(
        "--verbose_fmt",
        type=str,
        default="line",
        help=(
            "log diagnostics stdout format (options: 'table' or 'line', default: "
            "line)"
        ),
    )
    parser.add_argument(
        "--verbose_vars",
        nargs="+",
        default=STD_OUT_LOG_VARS_DEFAULT,
        help=("a space separated list of the values you want to show on the stdout."),
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="how often (in epochs) the policy should be saved (default: 1)",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="use model checkpoints (default: False)",
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="use TensorBoard (default: False)",
    )
    parser.add_argument(
        "--tb_log_freq",
        type=str,
        default="low",
        help=(
            "the TensorBoard log frequency. Options are 'low' (Recommended: logs at "
            "every epoch) and 'high' (logs at every SGD update batch). Default is 'low'"
        ),
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="use Weights & Biases (default: False)",
    )
    parser.add_argument(
        "--wandb_job_type",
        type=str,
        default="train",
        help="the Weights & Biases job type (default: train)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stable-learning-control",
        help="the name of the wandb project (default: stable-learning-control)",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help=(
            "the name of the Weights & Biases group you want to assign the run to "
            "(defaults: None)"
        ),
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help=(
            "the name of the Weights & Biases run (defaults: None, which will be "
            "set to the experiment name)"
        ),
    )
    args = parser.parse_args()

    # Setup actor critic arguments.
    output_activation = {}
    output_activation["actor"] = safer_eval(args.act_out_a, backend="torch")
    ac_kwargs = dict(
        hidden_sizes={
            "actor": [args.hid_a] * args.l_a,
            "critic": [args.hid_c] * args.l_c,
        },
        activation={
            "actor": safer_eval(args.act_a, backend="torch"),
            "critic": safer_eval(args.act_c, backend="torch"),
        },
        output_activation=output_activation,
    )

    # Setup output dir for logger and return output kwargs.
    logger_kwargs = setup_logger_kwargs(
        args.exp_name,
        args.seed,
        save_checkpoints=args.save_checkpoints,
        use_tensorboard=args.use_tensorboard,
        tb_log_freq=args.tb_log_freq,
        use_wandb=args.use_wandb,
        wandb_job_type=args.wandb_job_type,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        quiet=args.quiet,
        verbose_fmt=args.verbose_fmt,
        verbose_vars=args.verbose_vars,
    )
    logger_kwargs["output_dir"] = osp.abspath(
        osp.join(
            osp.dirname(osp.realpath(__file__)),
            f"../../../../../data/latc/{args.env.lower()}/runs/run_{int(time.time())}",
        )
    )
    torch.set_num_threads(torch.get_num_threads())

    lac(
        lambda: gym.make(args.env),
        actor_critic=LyapunovActorTwinCritic,
        ac_kwargs=ac_kwargs,
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
        horizon_length=args.horizon_length,
        seed=args.seed,
        save_freq=args.save_freq,
        device=args.device,
        start_policy=args.start_policy,
        export=args.export,
        logger_kwargs=logger_kwargs,
    )
