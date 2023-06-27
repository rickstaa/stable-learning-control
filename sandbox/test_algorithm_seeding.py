"""Test the LAC seeding process."""
import gymnasium as gym
import numpy as np
import stable_gym  # noqa: F401
import torch
import torch.nn as nn
from gymnasium.utils import seeding

from stable_learning_control.control.algos.pytorch.common.buffers import ReplayBuffer
from stable_learning_control.control.algos.pytorch.lac.lac import (
    LAC,
    LyapunovActorCritic,
)

if __name__ == "__main__":
    env = gym.make("Oscillator-v1")

    # Seed the environment.
    env.np_random, seed = seeding.np_random(0)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    test_act = env.action_space.sample()
    test_obs = env.observation_space.sample()
    print(f"Action: {test_act}")
    print(f"Observation: {test_obs}")
    obs, info = env.reset()
    print(f"Observation: {obs}")
    print(f"Info: {info}")
    for i in range(3):
        obs, rew, terminal, truncated, info = env.step(env.action_space.sample())
        print(f"Observation: {obs}")
        print(f"Reward: {rew}")
        print(f"Terminal: {terminal}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

    # Seed the numpy random number generator.
    np.random.seed(seed)

    # Create Buffer.
    replay_buffer = ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        rew_dim=1,
        size=10,
        device="cpu",
    )
    replay_buffer.store(
        env.observation_space.sample(),
        env.action_space.sample(),
        99,
        100,
        101,
    )
    replay_buffer.store(
        env.observation_space.sample(),
        env.action_space.sample(),
        1,
        2,
        3,
    )
    replay_buffer.store(
        env.observation_space.sample(),
        env.action_space.sample(),
        4,
        5,
        6,
    )
    replay_buffer.store(
        env.observation_space.sample(),
        env.action_space.sample(),
        7,
        8,
        9,
    )
    replay_buffer.store(
        env.observation_space.sample(),
        env.action_space.sample(),
        10,
        11,
        12,
    )
    for i in range(5):
        batch = replay_buffer.sample_batch(2)
        print(f"Batch observation: {batch['obs']}")
        print(f"Batch action: {batch['act']}")
        print(f"Batch reward: {batch['rew']}")
        print(f"Batch next observation: {batch['obs_next']}")
        print(f"Batch done: {batch['done']}")

    # Create agent and policy.
    torch.manual_seed(seed)
    policy = LAC(
        env,
        actor_critic=LyapunovActorCritic,
        ac_kwargs=dict(
            hidden_sizes={"actor": [64] * 2, "critic": [128] * 2},
            activation=nn.ReLU,
            output_activation=nn.ReLU,
        ),
        opt_type="minimize",
        alpha=0.99,
        alpha3=0.2,
        labda=0.99,
        gamma=0.99,
        polyak=0.995,
        target_entropy=None,
        adaptive_temperature=True,
        lr_a=1e-4,
        lr_c=3e-4,
        device="cpu",
    )
    for i in range(20):
        action = policy.get_action(env.observation_space.sample())
        print(f"Action: {action}")
    print("test")
