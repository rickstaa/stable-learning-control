import gym
import machine_learning_control.simzoo.simzoo

env = gym.make("Oscillator-v1")
env.reset()
print("Taking 1000 steps in the Oscillator-v1 environment...")
for ii in range(1000):
    env.render()  # Does not work with the Oscillator-v1 environment.
    obs, cost, done, info_doc = env.step(
        env.action_space.sample()
    )  # take a random action
    if ii % 100 == 0:
        print(f"Randoms step {ii}: {obs}")
env.close()
print("All steps were taken!")
