import torch

from machine_learning_control.control.algos.pytorch import SAC  # Import Algorithm
from machine_learning_control.control.utils.log_utils.logx import EpochLogger
from machine_learning_control.control.utils import safer_eval

# from machine_learning_control.control.algos.pytorch import LAC  # Import Algorithm
import machine_learning_control.simzoo.simzoo  # Import the mlc gym envs

MODEL_LOAD_PATH = "/home/ricks/Development/work/machine-learning-control/data/lac/oscillator-v1/runs/run_1614675772/torch_save/model_state.pt"
ENV_LOAD_PATH = "/home/ricks/Development/work/machine-learning-control/data/lac/oscillator-v1/runs/run_1614675772"

# Restore the model
config = EpochLogger.load_config(ENV_LOAD_PATH)
env = EpochLogger.load_env(ENV_LOAD_PATH)
test = safer_eval("torch.nn.ReLU")
model = SAC(env=env, ac_kwargs=config["ac_kwargs"])
# model = LAC(env=env)
restored_model_state_dict = torch.load(MODEL_LOAD_PATH, map_location="cpu")
model.load_state_dict(
    restored_model_state_dict,
)
# model.restore(MODEL_LOAD_PATH)

# Take action
obs = torch.rand(env.observation_space.shape)
act = torch.rand(env.action_space.shape)
print(model.get_action(obs))
value = model.ac.Q1(obs, act)
# LAC.get_action(obs)
