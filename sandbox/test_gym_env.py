"""Small example script to see if a gymnasium environment can be imported. In this
example, we use the ``Oscillator-v1`` environment in the :stable_gym:`stable_gym <>`
package.
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

RANDOM_STEP = True
# ENV_NAME = "stable_gym:Oscillator-v1"
# ENV_NAME = "stable_gym:Ex3EKF-v1"
ENV_NAME = "stable_gym:CartPoleCost-v1"
# ENV_NAME = "PandaReach-v1"

if __name__ == "__main__":
    env = gym.make(ENV_NAME, render_mode="human")

    # Retrieve time step.
    tau = env.dt if hasattr(env, "dt") else env.tau if hasattr(env, "tau") else 0.01

    # Take one episode in the environment.
    d, truncated, t = False, False, 0
    path = []
    time = []
    o, _ = env.reset()
    print(f"Performing 1 epsisode in the '{ENV_NAME}' environment.")
    while not d and not truncated:
        a = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        o, r, d, truncated, _ = env.step(a)
        t += tau
        path.append(o)
        time.append(t)
    print(f"Finished '{ENV_NAME}' environment simulation.")

    # Plot results.
    print("Plot results.")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(time, np.array(path)[:, 0], color="orange", label="x")
    ax.plot(time, np.array(path)[:, 1], color="magenta", label="x_dot")
    ax.plot(time, np.array(path)[:, 2], color="sienna", label="theta")
    ax.plot(time, np.array(path)[:, 3], color="blue", label="theta_dot1")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()

    print("Done")
    env.close()
