"""Script used for testing the **NEW** trajectory buffer.
"""
import gymnasium as gym

# from stable_learning_control.common.buffers import TrajectoryBuffer
from stable_learning_control.algos.pytorch.common.buffers import TrajectoryBuffer

if __name__ == "__main__":
    # Create dummy environment.
    env = gym.make("stable_gym:CartPoleCost-v1")

    # Dummy algorithm settings.
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer_size = int(1e6)
    epochs = 10
    local_steps_per_epoch = 100

    # Create Memory Buffer.
    buffer = TrajectoryBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=buffer_size,
        preempt=True,
        incomplete=True,
    )

    # Create test dummy data.
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            # Retrieve data from the environment.
            a = env.action_space.sample()
            next_o, r, d, _, _ = env.step(a)

            # Store data in buffer.
            buffer.store(o, a, r, next_o, d)

            # Update obs (critical!)
            o = next_o

            # Finish path.
            if d:
                buffer.finish_path()
                o, _ = env.reset()
                ep_ret, ep_len = 0, 0

        # Retrieve data from buffer.
        buffer_data = buffer.get(flat=False)
