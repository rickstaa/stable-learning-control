"""Soft Lyapunov Actor-Critic algorithm

This module contains a pytorch implementation of the LAC algorithm of
`Han et al. 2020 <http://arxiv.org/abs/2004.14288>`_. The original tensorflow
implementation can be
[found here](https://github.com/hithmh/Actor-critic-with-stability-guarantee).
"""

import argparse
import decimal
import itertools

# import json
import os
import time
from copy import deepcopy

# Import os.path as osp
import gym
import machine_learning_control.control.algos.pytorch.lac.core as core
import numpy as np
import torch
from machine_learning_control.control.algos.pytorch.common.buffers import ReplayBuffer
from machine_learning_control.control.utils.gym_utils import (
    is_continuous_space,
    is_discrete_space,
)
from machine_learning_control.control.utils.helpers import (
    calc_gamma_lr_decay,
    calc_linear_lr_decay,
    count_vars,
)
from machine_learning_control.control.utils.logx import EpochLogger
from machine_learning_control.control.utils.run_utils import setup_logger_kwargs
from torch.optim import Adam

SCALE_lambda_MIN_MAX = (0, 20)

global t  # TODO: Make attribute out of this\
# TODO: Add nograd torch.no_grad for some tensors
# TODO: Replace alphas with property!
# TODO: Add detach methods for faster computation
# TODO: Translate to class (Don't forget to update docstring)
# TODO: Add run folder by index creation see panda_openai_sim script
# TODO: Add verbose option to log more data with logger (Remove averages from log std)
# TODO: FIX LOGGER (See fixme in logger)
# FIXME: Make sure the right hyperparameters are here (Do we want the ones for
# oscillator or mujoco)
# Better to set to mujoco because that is more often used
# TODO: Add additional config file that can be loaded from argument
# TODO: Change txt to csv of progress
# TODO: Name adaptive alpha
# TODO: Fix opt_type for lac version
# TODO: Add ability to enable lambda tuning

# QUESTION: Do we need approximate values in lyapunov?
# QUESTION: Does we need to be able to disable lyapunov?
# QUESTION: Why don't we use a double lyapunov trick to correct for overestimation bias?
# Is this because we now use double actor?
# QUESTION: Why don't we use twin network in L update but only the not trained network
# QUESTION: Approximate?

RUN_DB_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "../cfg/_cfg/lac_last_run.json"
    )
)


def lac(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    epochs=50,
    steps_per_epoch=2048,
    max_ep_len=500,
    replay_size=int(1e6),
    gamma=0.9,
    polyak=0.995,
    lr_a=1e-4,
    lr_c=3e-4,
    lr_l=3e-4,
    lr_a_final=1e-10,
    lr_c_final=1e-10,
    lr_l_final=1e-10,
    decaying_lr=True,
    decaying_lr_type="linear",
    alpha=0.99,
    alpha3=0.2,
    labda=0.99,
    target_entropy="auto",
    use_lyapunov=True,
    batch_size=256,
    start_steps=0,
    update_every=100,
    update_after=1000,
    train_per_cycle=80,  # TODO: Give it a clear name or remove it.
    num_test_episodes=10,
    logger_kwargs=dict(),
    save_freq=1,
    opt_type="minimize",
):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided
                                           | observations and actions. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr_a (float): Learning rate (used for both policy learning).

        lr_c (float): Learning rate (used for both value learning).

        decaying_lr (bool): Whether you want to use decaying learning rates.

        decaying_lr_type (str, optional): The type of learning rate decay you want to
            use (options: exponential or linear). Defaults to linear.

        alpha (float): Entropy regularisation coefficient (Equivalent to
            inverse of reward scale in the original SAC paper).

        alpha (float: The Lyapunov lagrance multiplier.

        alpha3 (float): The Lyapunov constraint error boundary.

        target_entropy (str/None/float, optional): Target entropy used while learning
            the entropy temperature (alpha). Set to "auto" if you want to let the
            algorithm define the target_entropy. Set to `None` if you want to disable
            temperature learning. Defaults to "auto".

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        opt_type (str): The optimization type you want to use. Options "maximize" and
            "minimize". Defaults to maximize

        use_lyapunov (bool): Whether the Lyapunov Critic is used (use_lyapunov=True) or
            the regular Q-critic (use_lyapunov=false).
    """

    def heuristic_target_entropy(action_space):
        """Returns a heuristic target entropy for a given action space using the method
        explained in `Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.

        Args:
            action_space (gym.spaces): The action space

        Raises:
            NotImplementedError: If no heuristic target entropy has yet been implemented
                for the given action space.

        Returns:
            [type]: [description]
        """
        if is_continuous_space(action_space):
            heuristic_target_entropy = -np.prod(
                action_space.shape
            )  # Maximum information (bits) contained in action space
        elif is_discrete_space(action_space):
            raise NotImplementedError(
                "The heuristic target entropy is not yet implement for discrete spaces."
            )
        else:
            raise NotImplementedError(
                "The heuristic target entropy is not yet implement for "
                f"{type(action_space)} action spaces."
            )
        return heuristic_target_entropy

    # Setup Epoch logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())  # Write hyperparameters to logger

    # Set random seed for reproducible results
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment and test env
    # NOTE: Done since we want to check the performance of the algorithm during traning
    # TODO: Make sure this works for all shapes
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Get target entropy for automatic temperature tuning
    # TODO: Validate target entropy input argument
    target_entropy = (
        heuristic_target_entropy(env.action_space)
        if (isinstance(target_entropy, str) and target_entropy.lower() == "auto")
        else target_entropy
    )

    # Convert alpha to log_alpha (Used for computational stability)
    log_alpha = torch.tensor(np.log(alpha), requires_grad=True)

    # Convert labda to log_labda (Used for computational stability)
    log_labda = torch.tensor(np.log(labda), requires_grad=True)

    # Ensure that the max env step count is at least equal to the steps in one epoch
    env._max_episode_steps = max_ep_len
    test_env._max_episode_steps = max_ep_len

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Add network graph to tensorboard
    if logger_kwargs["use_tensorboard"]:
        with torch.no_grad():
            logger.add_graph(
                ac,
                (
                    torch.Tensor(env.observation_space.sample()),
                    torch.Tensor(env.action_space.sample()),
                ),
            )

    # Print network information
    print("==Actor==")
    print(ac.pi)
    print("")
    print("==Soft critic 1==")
    print(ac.q1)
    print("")
    print("==Soft critic 2==")
    print(ac.q2)
    print("")

    # Freeze target networks with respect to optimizers (update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    if not use_lyapunov:
        q_params = itertools.chain(
            ac.q1.parameters(), ac.q2.parameters()
        )  # Chain parameter iterators so we can pass them to the optimizer

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

    # Set up optimizers for policy, q-function and alpha temperature regularisation
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr_a)
    if use_lyapunov:
        l_optimizer = Adam(ac.l.parameters(), lr=lr_l)
    else:
        q_optimizer = Adam(q_params, lr=lr_c)  # Pass both SoftQ networks to optimizer
    log_alpha_optimizer = Adam([log_alpha], lr=lr_a)
    log_labda_optimizer = Adam([log_labda], lr=lr_l)

    # Create learning rate decay wrappers
    # TODO: make class method
    # TODO: limit minimum
    # TEST: Whether this works
    if decaying_lr_type.lower() == "exponential":

        # Calculated required gamma
        gamma_a = np.longdouble(
            (calc_gamma_lr_decay(lr_a, lr_a_final, epochs) if decaying_lr else 1.0)
        )  # The decay exponent
        gamma_c = np.longdouble(
            (calc_gamma_lr_decay(lr_c, lr_c_final, epochs) if decaying_lr else 1.0)
        )  # The decay exponent
        gamma_l = np.longdouble(
            (calc_gamma_lr_decay(lr_l, lr_l_final, epochs) if decaying_lr else 1.0)
        )  # The decay exponent

        # Create scheduler
        pi_opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(pi_optimizer, gamma_a)
        if use_lyapunov:
            l_opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                l_optimizer, gamma_l
            )
        else:
            q_opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                q_optimizer, gamma_c
            )
        log_alpha_opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            log_alpha_optimizer, gamma_a
        )
    else:
        # Calculate linear decay rate
        lr_decay_a = (
            (
                lambda epoch: np.longdouble(
                    decimal.Decimal(1.0)
                    - (
                        calc_linear_lr_decay(lr_a, lr_a_final, epochs)
                        * decimal.Decimal(epoch)
                    )
                )
            )
            if decaying_lr
            else lambda epoch: 1.0
        )
        lr_decay_c = (
            (
                lambda epoch: np.longdouble(
                    decimal.Decimal(1.0)
                    - (
                        calc_linear_lr_decay(lr_c, lr_c_final, epochs)
                        * decimal.Decimal(epoch)
                    )
                )
            )
            if decaying_lr
            else lambda epoch: 1.0
        )
        lr_decay_l = (
            (
                lambda epoch: np.longdouble(
                    decimal.Decimal(1.0)
                    - (
                        calc_linear_lr_decay(lr_l, lr_l_final, epochs)
                        * decimal.Decimal(epoch)
                    )
                )
            )
            if decaying_lr
            else lambda epoch: 1.0
        )

        # Create schedulers
        pi_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(
            pi_optimizer, lr_lambda=lr_decay_a
        )
        if use_lyapunov:
            l_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(
                l_optimizer,
                lr_lambda=lr_decay_l,
            )
        else:
            q_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(
                q_optimizer,
                lr_lambda=lr_decay_c,
            )
        log_alpha_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(
            log_alpha_optimizer, lr_lambda=lr_decay_a
        )
        log_labda_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(
            log_labda_optimizer, lr_lambda=lr_decay_a
        )

    if use_lyapunov:
        opt_schedulers = [
            pi_opt_scheduler,
            l_opt_scheduler,
            log_alpha_opt_scheduler,
            log_labda_opt_scheduler,
        ]
    else:
        opt_schedulers = [
            pi_opt_scheduler,
            q_opt_scheduler,
            log_alpha_opt_scheduler,
        ]

    # Store initial learning rates
    if logger_kwargs["use_tensorboard"]:
        # TODO: Add to csv logger
        logger.add_scalar("LearningRates/Lr_a", pi_optimizer.param_groups[0]["lr"], 0)
        if use_lyapunov:
            logger.add_scalar(
                "LearningRates/Lr_l", l_optimizer.param_groups[0]["lr"], 0
            )
            logger.add_scalar(
                "LearningRates/Lr_labda", log_labda_optimizer.param_groups[0]["lr"], 0
            )
        else:
            logger.add_scalar(
                "LearningRates/Lr_c", q_optimizer.param_groups[0]["lr"], 0
            )
        if target_entropy:
            logger.add_scalar(
                "LearningRates/Lr_alpha", log_alpha_optimizer.param_groups[0]["lr"], 0
            )

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        """Update the actor critic network using stochastic gradient descent.

        Args:
            data (dict): Dictionary containing a batch of experiences.
        """
        # FIXME: This is a quick fix so rewrite later!

        # Retrieve state, action and reward from the batch
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        ################################################
        # Optimize (Lyapunov/Q) critic #################
        ################################################
        if use_lyapunov:

            # Zero gradients on the L-critic
            l_optimizer.zero_grad()

            # Get target Lyapunov value (Bellman-backup)
            with torch.no_grad():
                a2_, _ = ac_targ.pi(o2)
                l_pi_targ = ac_targ.l(o2, a2_)
                l_backup = (
                    r + gamma * (1 - d) * l_pi_targ.detach()
                )  # The Lyapunov candidate

            # Calculate current lyapunov value
            l1 = ac.l(o, a)

            # Calculate L_backup
            # NOTE (rickstaa): The 0.5 multiplication factor was added to make the
            # derivation cleaner and can be safely removed without influencing the
            # minimization. We kept it here for consistency.
            # NOTE (rickstaa): I use a manual implementation instead of using
            # F.mse_loss as this is 2 times faster. This can be changed back to
            # F.mse_loss if Torchscript is used.
            l_error = 0.5 * ((l1 - l_backup) ** 2).mean()  # See eq. 7

            # Store l-values
            l_info = dict(LVals=l1.detach().numpy())

            # Record things
            logger.store(
                tb_write=logger_kwargs["use_tensorboard"],
                Lerror=l_error.item(),
                tb_aliases={"LossLya": "Loss/LossLya"},
                **l_info,
            )

            # Perform one gradient descent step for the Lyapunov critic
            l_error.backward()
            l_optimizer.step()
        else:

            # Zero gradients on the Q-critic
            q_optimizer.zero_grad()

            # Retrieve the Q values from the two networks
            q1 = ac.q1(o, a)
            q2 = ac.q2(o, a)

            # Get target Q values (Bellman-backup)
            # NOTE (rickstaa): Here we use max-clipping instead of min-clipping used
            # in the SAC algorithm since we want to minimize the return.
            with torch.no_grad():

                # Target actions come from *current* policy
                a2, logp_a2 = ac.pi(o2)

                # Target Q-values
                q1_pi_targ = ac_targ.q1(o2, a2)
                q2_pi_targ = ac_targ.q2(o2, a2)
                if opt_type.lower() == "minimize":
                    q_pi_targ = torch.max(
                        q1_pi_targ,
                        q2_pi_targ,
                    )  # Use max clipping  to prevent overestimation bias.
                else:
                    q_pi_targ = torch.min(
                        q1_pi_targ, q2_pi_targ
                    )  # Use min clipping to prevent overestimation bias
                # TODO: Replace log_alpha.exp() with alpha --> Make alpha property
                backup = r + gamma * (1 - d) * (q_pi_targ - log_alpha.exp() * logp_a2)

            # MSE loss against Bellman backup
            loss_q1 = ((q1 - backup) ** 2).mean()
            loss_q2 = ((q2 - backup) ** 2).mean()
            loss_q = loss_q1 + loss_q2

            # Store q-values
            q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

            # Record things
            logger.store(
                tb_write=logger_kwargs["use_tensorboard"],
                LossQ=loss_q.item(),
                tb_aliases={"LossQ": "Loss/LossQ"},
                **q_info,
            )

            # Perform one gradient descent step for the Q-Critic
            loss_q.backward()
            q_optimizer.step()

        ################################################
        # Optimize Gaussian actor ######################
        ################################################

        # Zero gradients on the Q-critic
        pi_optimizer.zero_grad()

        # Compute current best action according to policy
        pi, log_pis = ac.pi(o)

        # Entropy-regularised policy loss
        # TODO: Replace log_labda.exp() with labda --> Make labda property
        if use_lyapunov:

            # Get target lyapunov value out of lyapunov actor
            a2, _ = ac.pi(o2)
            lya_l_ = ac.l(o2, a2)

            # Calculate current lyapunov value using the current best action (Prior)
            l1 = ac.l(o, pi)

            # Used when agent has to minimise reward is positive deviation
            l_delta = torch.mean(lya_l_ - l1.detach() + alpha3 * r)

            # Used when agent has to minimise reward is positive deviation
            a_loss = (
                log_labda.exp().detach() * l_delta
                + log_alpha.exp().detach() * log_pis.mean()
            )  # See eq. 12
        else:

            # Retrieve the current Q values for the action given by the current policy
            q1_pi = ac.q1(o, pi)
            q2_pi = ac.q2(o, pi)
            if opt_type.lower() == "minimize":
                q_pi = torch.max(q1_pi, q2_pi)
            else:
                q_pi = torch.min(q1_pi, q2_pi)

            # Calculate Entropy-regularised policy loss
            a_loss = (
                log_alpha.exp().detach() * log_pis - q_pi
            ).mean()  # See Haarnoja eq. 7

        # Store log-likelihood
        pi_info = dict(LogPi=log_pis.detach().numpy())

        # Record things
        logger.store(
            tb_write=logger_kwargs["use_tensorboard"],
            LossPi=a_loss.item(),
            tb_aliases={"LossPi": "Loss/LossPi"},
            **pi_info,
        )

        # Perform one gradient descent step for the Gaussian Actor
        a_loss.backward()
        pi_optimizer.step()

        ################################################
        # Optimize alpha (Entropy temperature) #########
        ################################################
        if target_entropy:

            # Zero gradients on alpha
            log_alpha_optimizer.zero_grad()

            # Calculate alpha loss
            alpha_loss = -(
                log_alpha.exp() * (log_pis + target_entropy).detach()
            ).mean()  # See Haarnoja eq. 17

            # Record things
            logger.store(
                tb_write=logger_kwargs["use_tensorboard"],
                LossLogAlpha=alpha_loss.item(),
                Alpha=log_alpha.exp().detach(),
                tb_aliases={"LossLogAlpha": "Loss/LossLogAlpha"},
            )

            # Perform one gradient descent step for alpha
            alpha_loss.backward()
            log_alpha_optimizer.step()
        else:
            logger.store(
                tb_write=logger_kwargs["use_tensorboard"],
                Alpha=alpha,
            )

        ################################################
        # Optimize labda (Lyapunov temperature) ########
        ################################################
        if use_lyapunov:

            # Zero gradients on labda
            log_labda_optimizer.zero_grad()

            # Calculate labda loss
            # NOTE (rickstaa): Log_labda was used in the lambda_loss function because
            # using lambda caused the gradients to vanish. This is caused since we
            # restrict lambda within a 0-1.0 range using the clamp function (see #38).
            # Using log_lambda also is more numerically stable.
            labda_loss = -(
                log_labda * l_delta.detach()
            ).mean()  # See formulas under eq. 14

            # Record things
            logger.store(
                tb_write=logger_kwargs["use_tensorboard"],
                LossLogLabda=labda_loss.item(),
                Labda=log_labda.exp().detach(),
                tb_aliases={"LossLogLabda": "Loss/LossLogLabda"},
            )

            # Perform one gradient descent step for labda
            labda_loss.backward()
            log_labda_optimizer.step()

        ################################################
        # Update target networks and return ############
        # diagnostics. #################################
        ################################################

        # Finally, update target networks by polyak averaging.
        # TODO: Make function?
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        """Get the action under the current policy.

        Args:
            o (numpy.ndarray): The current observation (state).

            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If false the action is sampled from the stochastic
                policy. Defaults to False.

        Returns:
            numpy.ndarray: The action under the current policy.
        """
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent():
        """Validate the Performance of the AC in a separate test environment."""

        # Perform several steps in the test environment using the current policy
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(
                tb_write=logger_kwargs["use_tensorboard"],
                TestEpRet=ep_ret,
                TestEpLen=ep_len,
                tb_aliases={
                    "TestEpLen": "Performance/TestEpLen",
                    "TestEpRet": "Performance/TestEpRet",
                },
            )

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        # QUESTION: Abbreviations or action ext next_state ect
        o2, r, d, _ = env.step(a)
        ep_ret += r  # Increase episode reward
        ep_len += 1  # Increase episode length

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        # NOTE: state = next_state
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):

            # Log values, reset environment and decay learning rate
            logger.store(
                tb_write=logger_kwargs["use_tensorboard"], EpRet=ep_ret, EpLen=ep_len
            )
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            # for j in range(update_every):
            for j in range(
                train_per_cycle
            ):  # DEBUG: This was changed to be more in line with minghoa
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling (Save model, test performance and log data)
        # FIXME: Quickfix replace with warning if data is not yet available due to the
        # fact that update after is bigger than steps per epoch
        if (t + 1) % steps_per_epoch == 0:  # and (t >= update_after):
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({"env": env}, None, epoch=epoch)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # FIXME: This is how it is done in han but it makes more sense to do this
            # after the optim.step()
            for scheduler in opt_schedulers:
                scheduler.step()  # Decay learning rate

            # Log lr to tensorboard
            if logger_kwargs["use_tensorboard"]:
                logger.add_scalar(
                    "LearningRates/Lr_a", pi_optimizer.param_groups[0]["lr"], t
                )
                if use_lyapunov:
                    logger.add_scalar(
                        "LearningRates/Lr_l", l_optimizer.param_groups[0]["lr"], t
                    )
                    logger.add_scalar(
                        "LearningRates/Lr_labda",
                        log_labda_optimizer.param_groups[0]["lr"],
                        t,
                    )
                else:
                    logger.add_scalar(
                        "LearningRates/Lr_c", q_optimizer.param_groups[0]["lr"], t
                    )
                if target_entropy:
                    logger.add_scalar(
                        "LearningRates/Lr_alpha",
                        log_alpha_optimizer.param_groups[0]["lr"],
                        t,
                    )

            # Log info about epoch
            # TODO: Fails if step per epoch is 50 This is because the replay buffer is
            # FIXME: This needs to be fixed
            # TODO: Add loss alpha
            # TODO: ADd L vals to logger and only display Q if lypunov is disahled
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("Step", t)
            logger.log_tabular("Lr_a", pi_optimizer.param_groups[0]["lr"])
            logger.log_tabular("Lr_alpha", log_alpha_optimizer.param_groups[0]["lr"])
            if use_lyapunov:
                logger.log_tabular("Lr_c", l_optimizer.param_groups[0]["lr"])
                logger.log_tabular(
                    "Lr_labda", log_labda_optimizer.param_groups[0]["lr"]
                )
            else:
                logger.log_tabular("Lr_c", q_optimizer.param_groups[0]["lr"])
            if "Alpha" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["Alpha"]) > 0:
                    logger.log_tabular(
                        "Alpha",
                        with_min_and_max=False,
                        tb_write=logger_kwargs["use_tensorboard"],
                    )
            if "Labda" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["Labda"]) > 0:
                    logger.log_tabular(
                        "Labda",
                        with_min_and_max=False,
                        tb_write=logger_kwargs["use_tensorboard"],
                    )
            if "EpRet" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["EpRet"]) > 0:
                    logger.log_tabular(
                        "EpRet",
                        with_min_and_max=True,
                        tb_write=logger_kwargs["use_tensorboard"],
                    )
            if "TestEpRet" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["TestEpRet"]) > 0:
                    logger.log_tabular(
                        "TestEpRet",
                        with_min_and_max=True,
                        tb_write=logger_kwargs["use_tensorboard"],
                    )
            if "EpLen" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["EpLen"]) > 0:
                    logger.log_tabular(
                        "EpLen",
                        average_only=True,
                        tb_write=logger_kwargs["use_tensorboard"],
                    )
            if "TestEpLen" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["TestEpLen"]) > 0:
                    logger.log_tabular(
                        "TestEpLen",
                        average_only=True,
                        tb_write=logger_kwargs["use_tensorboard"],
                    )
            if "Step" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["Step"]) > 0:
                    logger.log_tabular("Step", t)
            if "Q1Vals" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["Q1Vals"]) > 0:
                    logger.log_tabular("Q1Vals", with_min_and_max=True)
            if "Q2Vals" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["Q2Vals"]) > 0:
                    logger.log_tabular("Q2Vals", with_min_and_max=True)
            if "LogPi" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["LogPi"]) > 0:
                    logger.log_tabular(
                        "LogPi",
                        with_min_and_max=True,
                        tb_write=logger_kwargs["use_tensorboard"],
                    )
            if "LossPi" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["LossPi"]) > 0:
                    logger.log_tabular(
                        "LossPi",
                        average_only=True,
                        tb_write=logger_kwargs["use_tensorboard"],
                    )
            if "LossPi" in logger.epoch_dict.keys():
                if len(logger.epoch_dict["LossPi"]) > 0:
                    logger.log_tabular(
                        "LossPi",
                        average_only=True,
                        tb_write=logger_kwargs["use_tensorboard"],
                    )
            if target_entropy:
                if "LossAlpha" in logger.epoch_dict.keys():
                    if len(logger.epoch_dict["LossAlpha"]) > 0:
                        logger.log_tabular(
                            "LossAlpha",
                            average_only=True,
                            tb_write=logger_kwargs["use_tensorboard"],
                        )
            if use_lyapunov:
                if "LossQ" in logger.epoch_dict.keys():
                    if len(logger.epoch_dict["LossQ"]) > 0:
                        logger.log_tabular(
                            "LossQ",
                            average_only=True,
                            tb_write=logger_kwargs["use_tensorboard"],
                        )
            else:
                if "Lerror" in logger.epoch_dict.keys():
                    if len(logger.epoch_dict["Lerror"]) > 0:
                        logger.log_tabular(
                            "Lerror",
                            average_only=True,
                            tb_write=logger_kwargs["use_tensorboard"],
                        )
                if "LossLabda" in logger.epoch_dict.keys():
                    if len(logger.epoch_dict["LossLabda"]) > 0:
                        logger.log_tabular(
                            "LossLabda",
                            average_only=True,
                            tb_write=logger_kwargs["use_tensorboard"],
                        )
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":

    # Import additional gym environments
    # FIXME: Check where this should be imported
    import machine_learning_control.simzoo.simzoo.envs

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="Trains a SAC agent in a given environment."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Ex3_EKF-v0",
        help="the gym env (default: Oscillator-v1)",
    )  # EX3 env
    parser.add_argument(
        "--hid_a",
        type=int,
        default=64,
        help="hidden layer size of the actor (default: 256)",
    )
    parser.add_argument(
        "--hid_c",
        type=int,
        default=128,
        help="hidden layer size of the lyapunov critic (default: 256)",
    )  # QUESTION: I see in the code minghoa used 64 where in sac they use 256
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
        "--lr_a", type=float, default=1e-4, help="actor learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--lr_c", type=float, default=3e-4, help="critic learning rate (default: 1e-4)"
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
        "--gamma", type=float, default=0.995, help="discount factor (default: 0.995)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0, help="the random seed (default: 0)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="the number of epochs (default: 50)"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lac",
        help="the name of the experiment (default: sac)",
    )
    parser.add_argument(
        "--save-checkpoints",
        type=bool,
        default=False,
        help="use model checkpoints (default: False)",
    )
    parser.add_argument("--opt_type", type=str, default="minimize")
    parser.add_argument(
        "--use-tensorboard",
        type=bool,
        default=False,
        help="use tensorboard (default: False)",
    )
    args = parser.parse_args()

    # Setup output dir for logger and return output kwargs
    logger_kwargs = setup_logger_kwargs(
        args.exp_name,
        args.seed,
        save_checkpoints=args.save_checkpoints,
        use_tensorboard=args.use_tensorboard,
    )
    logger_kwargs["output_dir"] = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"../../../../../data/lac/{args.env.lower()}/runs/run_{int(time.time())}",
        )
    )
    torch.set_num_threads(torch.get_num_threads())

    # # Update last run in json database
    # FIXME: Save last run algorithm for plot and eval commands.
    # run_name = os.path.split(logger_kwargs["output_dir"])[-1]
    # with open(RUN_DB_FILE, "w") as outfile:
    #     json.dump(run_name, outfile)

    # Run SAC algorithm
    # TODO: Reinable arguments
    lac(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(
            hidden_sizes_actor=[args.hid_a] * args.l_a,
            hidden_sizes_critic=[args.hid_c] * args.l_c,
        ),
        replay_size=args.replay_size,
        gamma=args.gamma,
        lr_a=args.lr_a,
        lr_c=args.lr_c,
        seed=args.seed,
        epochs=args.epochs,
        opt_type=args.opt_type,
        logger_kwargs=logger_kwargs,
    )
    print("done")

# TODO: Add all input arguments
