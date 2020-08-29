"""Simple script used to test the performance of a trained model."""

import gym
import torch

import numpy as np
import machine_learning_control.simzoo.simzoo.envs
import matplotlib.pyplot as plt

ENV_NAME = "Oscillator-v0"
# ENV_NAME = "Hopper-v2"
# MAX_EP_LEN = 200
MODEL_PATH = "/home/ricks/Development/machine_learning_control_ws/src/data/lac/oscillator-v0/runs/run_1598723405/pyt_save/model.pt"
# MODEL_PATH = "/home/ricks/Development/machine_learning_control_ws/src/data/sac/hopper-v2/runs/run_1597959914/pyt_save/model.pt"
EP = 1000

# TODO: Change number

# Create environment
env = gym.make(ENV_NAME)

# Load model
LAC = torch.load(MODEL_PATH)

# # Perform several steps in the test environment using the current policy
# for j in range(EP):
#     o, d, ep_ret, ep_len = env.reset(), False, 0, 0
#     env.render()
#     while not (d or (ep_len == MAX_EP_LEN)):
#         # Take deterministic actions at test time
#         o, r, d, _ = env.step(SAC.act(torch.as_tensor(o, dtype=torch.float32), True))
#         ep_ret += r
#         ep_len += 1
#         env.render()

# Take T steps in the environment
T = 600
path = []
t1 = []
s = env.reset()
print(f"Taking {T} steps in the oscillator environment.")

a_lowerbound = env.action_space.low
a_upperbound = env.action_space.high

for i in range(int(T / env.dt)):
    a = LAC.act(torch.as_tensor(s, dtype=torch.float32), True)
    s, r, done, info = env.step(a)
    # s, r, done, info = env.step(np.array([0, 0, 0]))
    path.append(s)
    t1.append(i * env.dt)
print("Finished oscillator environment simulation.")

# Plot results
# observations = (m1, m2, m3, p1, p2, p3, r1, p1 - r1)
print("Plot results.")
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
# ax.plot(t1, np.array(path)[:, 0], color="orange", label="mRNA1")
# ax.plot(t1, np.array(path)[:, 1], color="magenta", label="mRNA2")
# ax.plot(t1, np.array(path)[:, 2], color="sienna", label="mRNA3")
ax.plot(t1, np.array(path)[:, 3], color="blue", label="protein1")
# ax.plot(t1, np.array(path)[:, 4], color="cyan", label="protein2")
# ax.plot(t1, np.array(path)[:, 5], color="green", label="protein3")
# ax.plot(t1, np.array(path)[:, 0:3], color="blue", label="mRNA")
# ax.plot(t1, np.array(path)[:, 3:6], color="blue", label="protein")
ax.plot(t1, np.array(path)[:, 6], color="yellow", label="reference")
ax.plot(t1, np.array(path)[:, 7], color="red", label="error")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
plt.show()
print("Done")