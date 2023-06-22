# Control

The following algorithms are implemented in the Stable Learning Control package:

*   [Soft Actor-Critic (SAC)](https://rickstaa.github.io/stable-learning-control/control/algorithms/sac.html)
*   [Lyapunov Actor-Critic (LAC)](https://rickstaa.github.io/stable-learning-control/control/algorithms/lac.html)

They are all implemented with [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) (non-recurrent) actor-critics, making them suitable for fully-observed, non-image-based RL environments, e.g. the [gymnasium Mujoco](https://gymnasium.farama.org/environments/mujoco/) environments.

Stable Learning Control has two implementations for each algorithm: one that uses [PyTorch](https://pytorch.org/) as the neural network library, and one that uses [Tensorflow v2](https://www.tensorflow.org/) as the neural network library. The default backend is [Pytorch](https://pytorch.org). Please run the `pip install .[tf]` command if you want to use
the [Tensorflow v2](https://www.tensorflow.org/) implementations.
