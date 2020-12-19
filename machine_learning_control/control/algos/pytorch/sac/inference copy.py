"""Simple script used to test the performance of a trained model."""


import json
import os

import gym
import machine_learning_control.simzoo.simzoo.envs
import matplotlib.pyplot as plt
import numpy as np
import torch

# ENV_NAME = "Oscillator-v1"
ENV_NAME = "Ex3_EKF-v0"
# ENV_NAME = "Hopper-v2"
# MAX_EP_LEN = 200

# GET LAST RUN PATH
# FIXME: Does not always work
RUN_DB_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "../cfg/_cfg/sac_last_run.json"
    )
)
# print(os.path.dirname(__file__))
# print(__file__)
print(RUN_DB_FILE)
lines = []
with open(RUN_DB_FILE, "r") as f:
    for line in f:
        lines.append(json.loads(line))
LAST_RUN = lines[0]  # FIXME: MAKE READ ONLY FIRST LINE
MODEL_PATH = os.path.abspath(
    os.path.join(
        "/home/ricks/Development/machine_learning_control_ws/src/data/sac/Oscillator-v1/runs/",
        LAST_RUN,
        "pyt_save/model.pt",
    )
)

# HARD MODEL PATH
MODEL_PATH = "/home/ricks/Development/machine_learning_control_ws/src/data/sac/ex3_ekf-v0/runs/run_1598777598/pyt_save/model.pt"
# MODEL_PATH = "/home/ricks/Development/machine_learning_control_ws/src/data/sac/hopper-v2/runs/run_1598608665/pyt_save/model.pt"
EP = 1000

# TODO: Change number

# Create environment
env = gym.make(ENV_NAME)

# Load model
# TODO: Cath error
SAC = torch.load(MODEL_PATH)

# IMPROVE: Clean code and add argument option
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
    a = SAC.act(torch.as_tensor(s, dtype=torch.float32), True)
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
ax.plot(t1, np.array(path)[:, 0], color="yellow", label="x1")
ax.plot(t1, np.array(path)[:, 1], color="green", label="x2")
# ax.plot(t1, np.array(path)[:, 1], color="magenta", label="mRNA2")
# ax.plot(t1, np.array(path)[:, 2], color="sienna", label="mRNA3")
# ax.plot(t1, np.array(path)[:, 3], color="blue", label="protein1")
# ax.plot(t1, np.array(path)[:, 4], color="cyan", label="protein2")
# ax.plot(t1, np.array(path)[:, 5], color="green", label="protein3")
# ax.plot(t1, np.array(path)[:, 0:3], color="blue", label="mRNA")
# ax.plot(t1, np.array(path)[:, 3:6], color="blue", label="protein")
# ax.plot(t1, np.array(path)[:, 6], color="yellow", label="reference")
# ax.plot(t1, np.array(path)[:, 7], color="red", label="error")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
plt.show()
print("Done")
