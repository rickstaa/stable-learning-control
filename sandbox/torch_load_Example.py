import torch
import os.path as osp

from machine_learning_control.control.utils.log_utils.logx import EpochLogger

from machine_learning_control.control.algos.pytorch import LAC  # Import Algorithm
import machine_learning_control.simzoo.simzoo  # Import the mlc gym envs

MODEL_LOAD_FOLDER = "./data/lac/oscillator-v1/runs/run_1614680001"
MODEL_PATH = osp.join(MODEL_LOAD_FOLDER, "torch_save/model_state.pt")

# Restore the model
config = EpochLogger.load_config(
    MODEL_LOAD_FOLDER
)  # Retrieve the experiment configuration
env = EpochLogger.load_env(MODEL_LOAD_FOLDER)  # Retrieve the used environment
model = LAC(env=env, ac_kwargs=config["ac_kwargs"])
restored_model_state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(
    restored_model_state_dict,
)

# Create dummy observations and retrieve the best action
obs = torch.rand(env.observation_space.shape)
a = model.get_action(obs)
L_value = model.ac.L(obs, torch.from_numpy(a))

# Print results
print(f"The LAC agent thinks it is a good idea to take action {a}.")
print(f"It assigns a Lyapunov Value of {L_value} to this action.")
