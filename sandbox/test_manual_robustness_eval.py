import gym
from bayesian_learning_control.control.utils.test_policy import load_policy_and_env
from bayesian_learning_control.control.utils.eval_robustness import (
    run_disturbed_policy,
    plot_robustness_results,
)
import bayesian_learning_control.simzoo.simzoo

_, policy = load_policy_and_env("./data/lac/oscillator-v1/runs/run_1616504319")
env = gym.make("Oscillator-v1")
run_results_df = run_disturbed_policy("test", policy, disturbance_type="step")
plot_robustness_results(run_results_df)
