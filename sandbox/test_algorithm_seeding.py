"""Test the LAC seeding process."""

from stable_learning_control.control.algos.pytorch.lac.lac import LAC
from stable_learning_control.control.algos.pytorch.lac.lac import LyapunovActorCritic
import stable_gym  # noqa: F401
import gymnasium as gym
from gymnasium.utils import seeding

if __name__ == "__main__":
    # Create environment.
    env = gym.make("Oscillator-v1")

    # Seed the environment.
    generator, seed = seeding.np_random(0)
    env.np_random = generator
    test = generator.random()
    test2 = generator.random()
    test3 = generator.random()
    test4 = generator.random()
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    test_act = env.action_space.sample()
    test_obs = env.observation_space.sample()

    # Check the environment.
    obs, info = env.reset(seed=0)
    test_act = env.action_space.sample()
    test_obs = env.observation_space.sample()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}")
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Info: {info}")
        truncated = info.get("TimeLimit.truncated", False)

    # Create agent and policy.
    agent = LyapunovActorCritic()
    policy = LAC()
