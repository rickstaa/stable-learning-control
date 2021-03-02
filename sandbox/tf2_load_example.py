import tensorflow as tf
import os.path as osp

from machine_learning_control.control.utils.log_utils.logx import EpochLogger

from machine_learning_control.control.algos.tf2 import LAC  # Import Algorithm

MODEL_LOAD_FOLDER = "./data/lac/oscillator-v1/runs/run_1614673367"
MODEL_PATH = osp.join(MODEL_LOAD_FOLDER, "tf2_save")

# Restore the model
config = EpochLogger.load_config(
    MODEL_LOAD_FOLDER
)  # Retrieve the experiment configuration
env = EpochLogger.load_env(MODEL_LOAD_FOLDER)  # Retrieve the used environment
model = LAC(env=env, ac_kwargs=config["ac_kwargs"])
weights_checkpoint = tf.train.latest_checkpoint(MODEL_PATH)  # Restore latest checkpoint
model.load_weights(
    weights_checkpoint,
)

# Create dummy observations and retrieve the best action
obs = tf.random.uniform((1, env.observation_space.shape[0]))
a = model.get_action(obs)
L_value = model.ac.L([obs, tf.expand_dims(a, axis=0)])

# Print results
print(f"The LAC agent thinks it is a good idea to take action {a}.")
print(f"It assigns a Lyapunov Value of {L_value} to this action.")
