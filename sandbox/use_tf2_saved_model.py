import os
import tensorflow as tf
from machine_learning_control.control.utils.log_utils.logx import EpochLogger

model_path = "./data/lac/oscillator-v1/runs/run_1614673367/tf2_save"

# Load model and environment
loaded_model = tf.saved_model.load(model_path)
loaded_env = EpochLogger.load_env(os.path.dirname(model_path))

# Get action for dummy observation
obs = tf.random.uniform((1, loaded_env.observation_space.shape[0]))
a = loaded_model.get_action(obs)
print(f"\nThe model thinks it is a good idea to take action: {a.numpy()}")