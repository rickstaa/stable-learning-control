# Stable Learning Control

[![Baysian Learning Control CI](https://github.com/rickstaa/stable-learning-control/actions/workflows/stable_learning_control.yml/badge.svg)](https://github.com/rickstaa/stable-learning-control/actions/workflows/stable_learning_control.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/rickstaa/stable-learning-control)](https://github.com/rickstaa/stable-learning-control/releases)
[![Python 3](https://img.shields.io/badge/Python->=3.8-brightgreen)](https://www.python.org/)
[![codecov](https://codecov.io/gh/rickstaa/stable-learning-control/branch/main/graph/badge.svg?token=4SAME74CJ7)](https://codecov.io/gh/rickstaa/stable-learning-control)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Package Overview

The Stable Learning Control (SLC) framework is a collection of robust Reinforcement Learning control algorithms designed to ensure stability. These algorithms are built upon the Lyapunov actor-critic architecture introduced by [Han et al. 2020](http://arxiv.org/abs/2004.14288). They guarantee stability and robustness by leveraging [Lyapunov stability theory](https://en.wikipedia.org/wiki/Lyapunov_stability). These algorithms are specifically tailored for use with [gymnasium environments](https://gymnasium.farama.org/) that feature a positive definite cost function. Several ready-to-use compatible environments can be found in the [stable-gym](https://github.com/rickstaa/stable-gym) package.

> **Note**
> This framework follows a code structure similar to the [Spinningup](https://spinningup.openai.com/en/latest/) educational package. By doing this, we aim to make it easier for new researchers to start with our Algorithms. If you are new to RL, you are therefore highly encouraged to check out the SpinningUp documentation and play with it before diving into our codebase. Our implementation sometimes deviates from the [Spinningup](https://spinningup.openai.com/en/latest/) version to increase code maintainability, extensibility and readability.

## Installation and Usage

Please see the [docs](https://rickstaa.github.io/stable-learning-control/) for installation and usage instructions.

## Contributing

We use [husky](https://github.com/typicode/husky) pre-commit hooks and github actions to enforce high code quality. Please check the [contributing guidelines](CONTRIBUTING.md) before contributing to this repository.
