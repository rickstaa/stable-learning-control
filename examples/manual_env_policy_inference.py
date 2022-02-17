"""A small script which shows how to manually load a saved environment and policy when
the CLI fails.
"""

import gym
import ros_gazebo_gym  # noqa: F401
from bayesian_learning_control.control.utils.test_policy import (
    load_policy_and_env,
    load_pytorch_policy,
    load_tf_policy,
    run_policy,
)

AGENT_TYPE = "torch"  # The type of agent that was trained. Options: 'tf2' and 'torch'.
AGENT_FOLDER = "/home/ricks/Development/work/bayesian-learning-control/data/2022-02-17_staa_lac_panda_reach/2022-02-17_09-35-31-staa_lac_panda_reach_s25"  # noqa: E501

if __name__ == "__main__":
    # NOTE: STEP 1a: Try to load the policy and environment
    try:
        env, policy = load_policy_and_env(AGENT_FOLDER)
    except Exception:
        # NOTE: STEP: 1b: If step 1 fails recreate the environment and load the
        #  Pytorch/TF2 agent separately.

        # Create the environment
        # NOTE: Here the 'FlattenObservation' wrapper is used to make sure the alg works
        # with dictionary based observation spaces.
        env = gym.make("PandaReach-v1")
        env = gym.wrappers.FlattenObservation(env)

        # Load the policy
        if AGENT_TYPE.lower() == "tf2":
            policy = load_tf_policy(AGENT_FOLDER, itr="last", env=env)  # Load TF2 agent
        else:
            policy = load_pytorch_policy(
                AGENT_FOLDER, itr="last", env=env
            )  # Load Pytorch agent

    # Step 2: Try to run the policy on the environment
    try:
        run_policy(env, policy)
    except Exception:
        raise Exception(
            "Something went wrong while trying to run the inference. Please check the "
            "'AGENT_FOLDER' and try again. If the problem persists please open a issue "
            "on https://github.com/rickstaa/bayesian-learning-control/issues."
        )
