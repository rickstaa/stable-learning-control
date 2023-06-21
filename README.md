# Bayesian Learning Control

[![Baysian Learning Control CI](https://github.com/rickstaa/bayesian-learning-control/actions/workflows/bayesian_learning_control.yml/badge.svg)](https://github.com/rickstaa/bayesian-learning-control/actions/workflows/bayesian_learning_control.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/rickstaa/bayesian-learning-control)](https://github.com/rickstaa/bayesian-learning-control/releases)
[![Python 3](https://img.shields.io/badge/Python->=3.8-brightgreen)](https://www.python.org/)
[![codecov](https://codecov.io/gh/rickstaa/bayesian-learning-control/branch/main/graph/badge.svg?token=RFM3OELQ3L)](https://codecov.io/gh/rickstaa/bayesian-learning-control)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Package Overview

Welcome to the Bayesian Learning Control (BLC) framework! The Bayesian Learning Control framework enables you to automatically create, train and deploy various safe (stable and robust) Reinforcement Learning (RL) and Imitation learning (IL) control algorithms directly from real-world data. This framework is made up of four main modules:

*   [Modeling](./bayesian_learning_control/modeling): Module that uses state of the art System Identification and State Estimation techniques to create an Openai gym environment out of real data.
*   [Control](./bayesian_learning_control/control): Module used to train several [Bayesian Learning Control](https://rickstaa.github.io/bayesian-learning-control/control/control.html) RL/IL agents on the built gym environments.
*   [Hardware](./bayesian_learning_control/hardware): Module that can be used to deploy the trained RL/IL agents onto the hardware of your choice.

This framework follows a code structure similar to the [Spinningup](https://spinningup.openai.com/en/latest/) educational package. By doing this, we hope to make it easier for new researchers to get started with our Algorithms. If you are new to RL, you are therefore highly encouraged first to check out the SpinningUp documentation and play with before diving into our codebase. Our implementation sometimes deviates from the [Spinningup](https://spinningup.openai.com/en/latest/) version to increase code maintainability, extensibility and readability.

## Clone the repository

Since the repository contains several git submodules to use all the features, it needs to be cloned using the `--recurse-submodules` argument:

```bash
git clone --recurse-submodules https://github.com/rickstaa/bayesian-learning-control.git
```

If you already cloned the repository and forgot the `--recurse-submodule` argument you can pull the submodules using the following git command:

```bash
git submodule update --init --recursive
```

## Installation and Usage

Please see the [docs](https://rickstaa.github.io/bayesian-learning-control/) for installation and usage instructions.

## Contributing

We use [husky](https://github.com/typicode/husky) pre-commit hooks and github actions to enforce high code quality. Please check the [contributing guidelines](CONTRIBUTING.md) before contributing to this repository.
