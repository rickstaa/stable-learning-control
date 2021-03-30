"""Small script to see if a gym environment can be imported.
"""

import gym
import bayesian_learning_control.simzoo.simzoo  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

RANDOM_STEP = True
# ENV_NAME = "Oscillator-v1"
# ENV_NAME = "Ex3EKF-v1"
ENV_NAME = "CartPoleCost-v0"

if __name__ == "__main__":

    env = gym.make(ENV_NAME)

    # Take T steps in the environment
    T = 1000
    path = []
    t1 = []
    s = env.reset()
    print(f"Taking {T} steps in the Cartpole environment.")
    for i in range(int(T / env.dt)):
        action = (
            env.np_random.uniform(
                env.action_space.low, env.action_space.high, env.action_space.shape
            )
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, done, info = env.step(action)
        try:
            env.render()
        except NotImplementedError:
            pass
        path.append(s)
        t1.append(i * env.dt)
    print("Finished Cartpole environment simulation.")

    # Plot results
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