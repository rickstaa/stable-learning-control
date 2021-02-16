"""File containing some usefull functions used during the robustness evaluation."""


def test_agent(policy, env, num_episodes, max_ep_len=None):
    """Evaluate the Performance of a agent in a separate test environment.

    Args:
        policy (Union[torch.nn.Module, tf.Module]): The policy you want to test.
        env (gym.Env): The environment in which you want to test the agent.
        num_episodes (int): The number of episodes you want to perform in the test
            environment.
        max_episode_len (int): The maximum number of steps in a episode.


    Returns:
        tuple: tuple containing:

            ep_ret(list): Episode retentions.
            ep_len(list): Episode lengths.
    """
    test_ep_ret, test_ep_len = [], []
    for _ in range(num_episodes):
        o, d, ep_ret, ep_len = env.reset(), False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            o, r, d, _ = env.step(policy.get_action(o, True))
            ep_ret += r
            ep_len += 1
        test_ep_ret.append(ep_ret)
        test_ep_len.append(ep_len)
    return test_ep_ret, test_ep_len
