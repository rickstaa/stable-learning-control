"""Helper functions that can be used for evaluating the performance of trained agents.
"""


def test_agent(policy, env, num_episodes):
    """Evaluate the Performance of a agent in a separate test environment.

    Args:
        policy (Union[torch.nn.Module, tf.Module]): The policy you want to test.
        env (:obj:`gym.Env`): The environment in which you want to test the agent.
        num_episodes (int): The number of episodes you want to perform in the test
            environment.

    Returns:
        tuple: tuple containing:

            - ep_ret(:obj:`list`): Episode retentions.
            - ep_len(:obj:`list`): Episode lengths.
    """
    test_ep_ret, test_ep_len = [], []
    for _ in range(num_episodes):
        o, _ = env.reset()
        d, truncated, ep_ret, ep_len = False, False, 0, 0
        while not (d or truncated):
            # Take deterministic actions at test time.
            o, r, d, truncated, _ = env.step(policy.get_action(o, True))
            ep_ret += r
            ep_len += 1
        test_ep_ret.append(ep_ret)
        test_ep_len.append(ep_len)
    return test_ep_ret, test_ep_len
