"""Soft Actor-Critic algorithm

This module contains an implementation of the SAC algorithm of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_. This implementation is based
on the one found in the the
`Spinning Up repository <https://github.com/openai/spinningup>`_
"""

import itertools
import time
import os
import argparse
from copy import deepcopy

import gym
import numpy as np
import torch
from torch.optim import Adam

# TODO: RENAME TO

import machine_learning_control.control.algos.sac.core as core
from machine_learning_control.control.algos.common.buffers import ReplayBuffer
from machine_learning_control.control.utils.logx import EpochLogger
from machine_learning_control.control.utils.helpers import count_vars
from machine_learning_control.control.utils.run_utils import setup_logger_kwargs

# QUESTION: Shall I translate this to a class? Where do we place the cost functions?
# TODO: Add run folder by index creation see panda_openai_sim script
# TODO: Add totoal steps argument
# FIXME: The experiments are capped at 200000 steps I think this is due to the max_step
# QUESTION: It looks like the GPU is used is GPU really not faster? To much communication overhead when using GPU right because of small network.

global t  # Make attribute out of this

# TODO: ROLLOUTS NEED TO BE ARG
def sac(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=50,
    epochs=400,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    lr_a=1e-3,
    lr_c=1e-3,
    alpha=0.2,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=400,
    logger_kwargs=dict(),
    save_freq=1,
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

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

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

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Setup Epoch logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())  # Write hyperparameters to logger

    # Set random seed for reproducible results
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment and test env
    # NOTE: Done since we want to check the performance of the algorithm during traning
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Ensure that the max env step count is at least equal to the steps in one epoch
    env._max_episode_steps = steps_per_epoch
    test_env._max_episode_steps = steps_per_epoch

    # Action limit for clamping: critically, assumes all dimensions share the same
    # bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Print network information
    print("==Actor==")
    print(ac.pi)
    print("")
    print("==Soft critic 1==")
    print(ac.q1)
    print("")
    print("==Soft critic 2=")
    print(ac.q2)
    print("")

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(
        ac.q1.parameters(), ac.q2.parameters()
    )  # Chain parameter iterators so we can pass them to the optimizer

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

    # Set up function for computing SAC Q-losses
    # QUESTION: WHY not inside class or one update function? Since it is spinning up
    # I guess redability
    def compute_loss_q(data):
        """Function computes the loss for the soft-Q networks

        Args:
            data (dict): Dictionary containing a batch of experiences.

        Returns:
            (torch.Tensor, dict):
                Tensor containing the q-loss, dictionary with the current q values
                (Usefull for logging).
        """

        # Unpack data from dictionary
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        # Retrieve the Q values from the two networks
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():  # Make sure the gradients are not tracked

            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(
                q1_pi_targ, q2_pi_targ
            )  # Use min clipping to prevent overestimation bias (Replaced V by E(Q-H))
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        # NOTE: WHy would we want to do this togheter?
        # Question: Where did the 1/2 of the paper go? This scaling is because of simplifying the gradient right?
        # Official sac uses 1/2 https://github.com/haarnoja/sac/blob/master/sac/algos/sac.py
        # Can be done since 1/2+1/2?
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(
            Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy()
        )  # Save Q values while detaching graph, gradients and trackers

        # Return Soft actor critic los
        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        """Function computes the loss for the policy network.

        Args:
            data (dict): Dictionary containing a batch of experiences.

        Returns:
            (torch.Tensor, dict):
                Tensor containing the policy-loss, dictionary with the current
                log-likelihood values (Usefull for logging).
        """

        o = data["obs"]
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())  # Detach for computational load

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr_a)
    q_optimizer = Adam(q_params, lr=lr_c)  # Pass both SoftQ networks to optimizer

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        """Update the actor critic network using stochastic gradient descent.

        Args:
            data (dict): Dictionary containing a batch of experiences.
        """

        # First run one gradient descent step for Q1 and Q2 (Both network are in the
        # optimizer)
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(tb_write=True, LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(tb_write=True, LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
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
        """Validate the Performance of the AC in a separate test environment.
        """

        # Perform several steps in the test environment using the current policy
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(tb_write=True, TestEpRet=ep_ret, TestEpLen=ep_len)

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
        # QUESTION: Abreviations or action ext next_state ect
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
            logger.store(tb_write=True, EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({"env": env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpRet", with_min_and_max=True, tb_write=True)
            logger.log_tabular("TestEpRet", with_min_and_max=True, tb_write=True)
            logger.log_tabular("EpLen", average_only=True, tb_write=True)
            logger.log_tabular("TestEpLen", average_only=True, tb_write=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("Q1Vals", with_min_and_max=True)
            logger.log_tabular("Q2Vals", with_min_and_max=True)
            logger.log_tabular("LogPi", with_min_and_max=True, tb_write=True)
            logger.log_tabular("LossPi", average_only=True, tb_write=True)
            logger.log_tabular("LossQ", average_only=True, tb_write=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":

    # Import additional gym environments
    import machine_learning_control.simzoo.simzoo.envs

    # Parse Arguments
    # TODO: Add helper to arguments
    # QUESTION: ADD optimizer argument?
    # QUESTION: ADD activation function argument?
    # TODO: ADD DEFAULTS TO JSON
    # TODO ADD arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "--env",
        # type=str,
        # default="HalfCheeta/h-v2"
        "--env",
        type=str,
        default="Oscillator-v0",
    )  # https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/half_cheetah.xml
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--lra", type=float, default=1e-4)
    parser.add_argument("--lrc", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=float, default=256)
    parser.add_argument("--replay-size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="sac")
    args = parser.parse_args()

    # Setup output dir for logger and return output kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs["output_dir"] = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"../../../../data/sac/{args.env.lower()}/runs/run_{int(time.time())}",
        )
    )
    torch.set_num_threads(torch.get_num_threads())

    # Run SAC algorithm
    sac(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        replay_size=args.replay_size,
        gamma=args.gamma,
        lr_a=args.lra,
        lr_c=args.lrc,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
