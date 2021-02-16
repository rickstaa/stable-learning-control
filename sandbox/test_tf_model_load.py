import tensorflow as tf

new_model = tf.keras.models.load_model("data/lac/oscillator-v1/runs/run_1613392946")
new_model.summary()

print("jan")
