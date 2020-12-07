"""File containing the algorithm parameters.
"""

import os
import sys
import time

# Environment parameters
ENV_NAME = "oscillator"  # The gym environment you want to train in
ENV_SEED = None  # The environment seed
RANDOM_SEED = 0  # The numpy random seed

# Setup log path and time string
dirname = os.path.dirname(__file__)
LOG_PATH = os.path.abspath(
    os.path.join(
        dirname,
        "../../../../data/lac/" + ENV_NAME,
        "LAC" + time.strftime("%Y%m%d_%H%M"),
    )
)
timestr = time.strftime("%Y%m%d_%H%M")
LOG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        f"../../../../data/lac/{ENV_NAME}/runs/run_{int(timestr)}",
    )
)

# Main training loop parameters
TRAIN_PARAMS = {
    "episodes": int(6e4),  # The number of episodes you want to perform
    "episodes": int(6e4),  # The number of episodes you want to perform
    # "episodes": int(2e4),  # The number of episodes you want to perform
    "num_of_training_paths": 10,  # Number of training rollouts stored for analysis
    "evaluation_frequency": 2048,  # After how many steps the performance is evaluated
    "num_of_evaluation_paths": 20,  # number of rollouts for evaluation
    "num_of_trials": 1,  # number of randomly seeded trained agents
    "start_of_trial": 0,  # The start number of the rollouts (used during model save)
}

# Main evaluation parameters
EVAL_PARAMS = {
    # "eval_list": ["run_1600694865"],
    "eval_list": ["run_1600703794"],
    "additional_description": timestr,
    "trials_for_eval": [str(i) for i in range(0, 3)],
    "num_of_paths": 10,  # number of path for evaluation
    "plot_average": True,
    "directly_show": True,
}

# Learning algorithm parameters
ALG_PARAMS = {
    "memory_capacity": int(1e6),  # The max replay buffer size
    "min_memory_size": 1000,  # The minimum replay buffer size before STG starts
    "batch_size": 256,  # The SGD batch size
    # "labda": 1.0,  # Initial value for the lyapunov constraint lagrance multiplier
    "labda": 0.99,  # Initial value for the lyapunov constraint lagrance multiplier
    # "alpha": 1.0,  # The initial value for the entropy lagrance multiplier
    "alpha": 0.99,  # The initial value for the entropy lagrance multiplier
    "alpha3": 0.2,  # The value of the stability condition multiplier
    "tau": 5e-3,  # Decay rate used in the polyak averaging
    "lr_a": 1e-4,  # The actor learning rate
    "lr_l": 3e-4,  # The lyapunov critic
    "gamma": 0.9,  # Discount factor
    "steps_per_cycle": 100,  # The number of steps after which the model is trained
    "train_per_cycle": 80,  # How many times SGD is called during the training
    "adaptive_alpha": True,  # Enables automatic entropy temperature tuning
    "target_entropy": None,  # Set alpha target entropy, when None == -(action_dim)
    "network_structure": {
        "critic": [128, 128],
        "actor": [64, 64],
    },  # The network structure of the agent.
}

# Environment parameters
ENVS_PARAMS = {
    "oscillator": {
        "max_ep_steps": 800,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "Ex3_EKF-v0": {
        "max_ep_steps": 500,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "Ex4_EKF": {
        "max_ep_steps": 100,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
}

# Check if specified environment is valid
if ENV_NAME in ENVS_PARAMS.keys():
    ENV_PARAMS = ENVS_PARAMS[ENV_NAME]
else:
    print(
        f"Environmen {ENV_NAME} does not exist yet. Please specify a valid environment "
        "and try again."
    )
    sys.exit(0)

# Other paramters
LOG_SIGMA_MIN_MAX = (-20, 2)  # Range of log std coming out of the GA network
SCALE_lambda_MIN_MAX = (0, 1)  # Range of lambda lagrance multiplier
