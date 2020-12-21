"""This module contains several disturbers that can be used to disturb the agent during inference.
"""

import numpy as np


def cartpole_disturber(
    time, s, action, env, eval_params, form_of_eval, disturber=None, initial_pos=0.0
):
    """Disturber used for disturbing the cartpole environment.
    """
    if form_of_eval == "impulse":
        if time == eval_params["impulse_instant"]:
            d = eval_params["magnitude"] * np.sign(s[0])
        else:
            d = 0
        s_, r, done, info = env.step(action, impulse=d)

    elif form_of_eval == "constant_impulse":
        if time % eval_params["impulse_instant"] == 0:
            d = eval_params["magnitude"] * np.sign(s[0])
        else:
            d = 0
        s_, r, done, info = env.step(action, impulse=d)
    elif form_of_eval == "various_disturbance":
        if eval_params["form"] == "sin":
            d = (
                np.sin(2 * np.pi / eval_params["period"] * time + initial_pos)
                * eval_params["magnitude"]
            )
        s_, r, done, info = env.step(action, impulse=d)

    elif form_of_eval == "trained_disturber":
        d, _ = disturber.choose_action(s, time)
        s_, r, done, info = env.step(action, process_noise=d)
    else:
        s_, r, done, info = env.step(action)
        done = False
    # done = False
    return s_, r, done, info


def oscillator_disturber(
    time, s, action, env, eval_params, form_of_eval, disturber=None, initial_pos=0.0
):
    if form_of_eval == "impulse":
        if time == eval_params["impulse_instant"]:

            d = eval_params["magnitude"] * (-np.sign(action))
        else:
            d = np.zeros_like(action)
    elif form_of_eval == "constant_impulse":
        if time % eval_params["impulse_instant"] == 0:
            d = eval_params["magnitude"] * (-np.sign(action))
        else:
            d = np.zeros_like(action)
    elif form_of_eval == "various_disturbance":
        if eval_params["form"] == "sin":
            d = (
                np.sin(2 * np.pi / eval_params["period"] * time + initial_pos)
                * eval_params["magnitude"]
                * np.ones_like(action)
            )

    else:
        d = np.zeros_like(action)
        # action = 0*action
    s_, r, done, info = env.step(action + d)
    done = False
    return s_, r, done, info
