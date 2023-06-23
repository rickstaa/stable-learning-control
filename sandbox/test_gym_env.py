"""Small script to see if a gymnasium environment can be imported.
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# import ros_gazebo_gym  # noqa: F401
import stable_gym  # NOTE: Found at https://github.com/rickstaa/stable-gym # noqa: F401

RANDOM_STEP = True
ENV_NAME = "Oscillator-v1"
# ENV_NAME = "Ex3EKF-v1"
# ENV_NAME = "CartPoleCost-v0"
# ENV_NAME = "PandaReach-v1"

if __name__ == "__main__":
    env = gym.make(ENV_NAME)

    # Take T steps in the environment.
    T = 1000
    tau = 0.1
    path = []
    t1 = []
    s = env.reset()
    print(f"Taking {T} steps in the Cartpole environment.")
    for i in range(int(T / tau)):
        action = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, done, info = env.step(action)
        try:
            env.render()
        except NotImplementedError:
            pass
        path.append(s)
        t1.append(i * tau)
    print("Finished Cartpole environment simulation.")

    # Plot results.
    print("Plot results.")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(t1, np.array(path)[:, 0], color="orange", label="x")
    ax.plot(t1, np.array(path)[:, 1], color="magenta", label="x_dot")
    ax.plot(t1, np.array(path)[:, 2], color="sienna", label="theta")
    ax.plot(t1, np.array(path)[:, 3], color="blue", label=" theat_dot1")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print("Done")
