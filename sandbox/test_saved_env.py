"""Small test script to test why the saved agent that was trained on the ros_gazebo_gym
Panda environment can not be loaded.
"""

import ros_gazebo_gym  # noqa: F401
from bayesian_learning_control.control.utils.test_policy import (
    load_policy_and_env,
    run_policy,
    load_pytorch_policy,
    # load_tf_policy,
)
import gym

AGENT_FOLDER = "/home/ricks/Development/bayesian-learning-control/data/2022-02-09_staa_lac_panda_reach/2022-02-09_23-21-17-staa_lac_panda_reach_s33453459/torch_save"  # noqa: E501

if __name__ == "__main__":
    # STEP 1a: Try to load the policy and environment
    # NOTE: This might fail if the environment was not
    try:
        _, policy = load_policy_and_env(AGENT_FOLDER)
    except Exception:
        # STEP: 1b: If step 1 fails recreate the environment and load the Pytorch/TF2
        # agent
        env = gym.make("PandaReach-v1")
        env = gym.wrappers.FlattenObservation(
            env
        )  # NOTE: Done to make sure the alg works with dict observation spaces
        policy = load_pytorch_policy(
            AGENT_FOLDER, itr="last", env=env
        )  # Load Pytorch agent
        # load_tf_policy(AGENT_FOLDER, itr="last", env=env)  # Load TF2 agent

    # Step 2: Try to run the policy on the environment
    run_policy(env, policy)
