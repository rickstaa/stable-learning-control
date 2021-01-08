# Control

The following algorithms are implemented in the Machine Learning Control package:

    - [Soft Actor-Critic (SAC)](https://rickstaa.github.io/machine-learning-control/control/algorithms/sac.html)
    - [Lyapunov Actor-Critic (LAC)](https://rickstaa.github.io/machine-learning-control/control/algorithms/lac.html)

They are all implemented with [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) (non-recurrent) actor-critics, making them suitable for fully-observed, non-image-based RL environments, e.g. the [Gym Mujoco](https://gym.openai.com/envs/#mujoco) environments.

Machine Learning Control has two implementations for each algorithm: one that uses [PyTorch](https://pytorch.org/) as the neural network library, and one that uses [Tensorflow v2](https://www.tensorflow.org/) as the neural network library.

// TODO: Add Algorithms description.
// TODO: Adds documentation link.
