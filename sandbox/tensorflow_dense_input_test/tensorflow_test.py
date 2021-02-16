import tensorflow as tf

# Create simple one layer network
layers = []
layers.append(
    tf.keras.layers.Dense(64, input_shape=(8,), activation=None, name="l{}".format(1))
)
net = tf.keras.Sequential(layers)

# Create dummy input
test_input_correct = tf.random.uniform((2, 8))  # False size
test_input_false = tf.random.uniform((8,))  # Correct size
output = net(test_input_correct)
output = net(test_input_false)
print("jan")
