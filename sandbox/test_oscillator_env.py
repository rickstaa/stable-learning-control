import gym
from machine_learning_control.simzoo.envs.oscillator import Oscillator

gym.make("Oscillator-v0")
print("done")
test = Oscillator()
test.render()
test.render()
