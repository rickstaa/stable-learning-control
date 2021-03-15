"""A set of functions that can be used to evaluate the stability and robustness of an
algorithm. This is done by evaluating an algorithm's performance under two types of
disturbances: A disturbance that is applied during the environment step and a
perturbation added to the environmental parameters. For the functions in this
module to work work, these disturbances should be implemented as methods on the
environment. See the
`Robustness Evaluation Documentation <https://rickstaa.github.io/bayesian-learning-control/control/robustness_eval.html>`
on how this is done.
"""  # noqa: E501

import time

from bayesian_learning_control.utils.log_utils import EpochLogger, log_to_std_out

from bayesian_learning_control.control.utils.test_policy import (
    load_policy_and_env,
)


def run_policy(
    env, policy, max_ep_len=None, num_episodes=100, render=True, deterministic=False
):
    """Evaluates a policy inside a given gym environment.

    Args:
        env (:obj:`gym.env`): The gym environment.
        policy (Union[tf.keras.Model, torch.nn.Module]): The policy.
        max_ep_len (int, optional): The maximum episode length. Defaults to None.
        num_episodes (int, optional): Number of episodes you want to perform in the
            environment. Defaults to 100.
        deterministic (bool, optional): Whether you want the action from the policy to
            be deterministic. Defaults to ``False``.
        render (bool, optional): Whether you want to render the episode to the screen.
            Defaults to ``True``.
    """
    assert env is not None, (
        "Environment not found!\n\n It looks like the environment wasn't saved, "
        + "and we can't run the agent in it. :( \n\n Check out the readthedocs "
        + "page on Experiment Outputs for how to handle this situation."
    )

    logger = EpochLogger(verbose_fmt="table")
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    supports_deterministic = True  # Only supported with gaussian algorithms
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

            if supports_deterministic:
                try:
                    a = policy.get_action(o, deterministic=deterministic)
                except TypeError:
                    log_to_std_out(
                        "Input argument 'deterministic' ignored as the algorithm does "
                        "not support deterministic actions. This is only supported for "
                        "gaussian  algorithms.",
                        type="warning",
                    )
                    a = policy.get_action(o)
            else:
                a = policy.get_action(o)
            supports_deterministic = False
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            logger.log("Episode %d \t EpRet %.3f \t EpLen %d" % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    print("")
    logger.log_tabular("EpRet", with_min_and_max=True)
    logger.log_tabular("EpLen", average_only=True)
    logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str)
    parser.add_argument("--len", "-l", type=int, default=0)
    parser.add_argument("--episodes", "-n", type=int, default=100)
    parser.add_argument("--norender", "-nr", action="store_true")
    parser.add_argument("--itr", "-i", type=int, default=-1)
    parser.add_argument("--deterministic", "-d", action="store_true")
    parser.add_argument("--disturbance_type", default=None)  # TODO: Add special flag
    args = parser.parse_args()
    env, policy = load_policy_and_env(args.fpath, args.itr if args.itr >= 0 else "last")
    run_policy(env, policy, args.len, args.episodes, not (args.norender))
