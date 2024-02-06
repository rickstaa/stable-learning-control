"""Script used for performing some quick tests on the FiniteHorizonReplayBuffer class.
"""

import gymnasium as gym

# from stable_learning_control.common.buffers import TrajectoryBuffer
from stable_learning_control.algos.common.buffers import FiniteHorizonReplayBuffer

if __name__ == "__main__":
    env = gym.make("stable_gym:CartPoleCost-v1")

    # Dummy algorithm settings.
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer_size = int(200)
    episodes = 10
    local_steps_per_epoch = 100

    # Create Memory Buffer.
    buffer = FiniteHorizonReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=buffer_size,
        horizon_size=2,
    )

    # Create test dummy data.
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0
    for episode in range(1, episodes + 1):
        print(f"Episode {episode}:")
        d, truncated = False, False
        t = 0
        while not d and not truncated:
            # Retrieve data from the environment.
            a = env.action_space.sample()
            o_, r, d, truncated, _ = env.step(a)
            r = episode + t / 100

            # Store data in buffer.
            buffer.store(o, a, r, o_, d, truncated)

            # Update obs (critical!)
            o = o_
            t += 1

            # Finish path.
            if d or truncated:
                print("Environment terminated or truncated. Resetting.")
                o, _ = env.reset()
                ep_ret, ep_len, t = 0, 0, 0

    print("Done")
    env.close()
