import tensorflow as tf

x = tf.constant(5.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 3

print(tape.gradient(y, x).numpy())  # -> 75.0
