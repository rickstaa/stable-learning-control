"""Script to test out gym environment registration."""

import gym
import machine_learning_control.simzoo.simzoo

env = gym.make("Oscillator-v1")
print("test")
