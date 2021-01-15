# Machine Learning Control

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/rickstaa/machine-learning-control)](https://github.com/rickstaa/machine-learning-control/releases)
[![Python 3](https://img.shields.io/badge/Python-3.9%20%7C%203.8%20%7C%203.7%20%7C%203.6-green)](https://www.python.org/)
[![Linux CI](https://github.com/rickstaa/machine-learning-control/workflows/Linux%20CI/badge.svg)](https://github.com/rickstaa/machine-learning-control/actions?query=workflow%3A%22Linux+CI%22)
[![Mac CI](https://github.com/rickstaa/machine-learning-control/workflows/Mac%20CI/badge.svg)](https://github.com/rickstaa/machine-learning-control/actions?query=workflow%3A%22Mac+CI%22)
[![Win CI](https://github.com/rickstaa/machine-learning-control/workflows/Win%20CI/badge.svg)](https://github.com/rickstaa/machine-learning-control/actions?query=workflow%3A%22Win+CI%22)
[![Docs CI](https://github.com/rickstaa/machine-learning-control/workflows/Docs%20CI/badge.svg)](https://github.com/rickstaa/machine-learning-control/actions?query=workflow%3A%22Docs+CI%22)

## Package Overview

Welcome to the `Machine Learning Control` (MLC) framework! The Machine Learning Control framework enables
you to automatically create, train and deploy various Reinforcement Learning (RL) and
Imitation learning (IL) control algorithms directly from real-world data. This framework
is made up of four main modules:

-   [Modeling](./machine_learning_control/modeling): Module that uses state of the art System Identification and State Estimation techniques to create an Openai gym environment out of real data.
-   [Simzoo](https://github.com/rickstaa/simzoo): Module that contains several already created [Machine Learning Control](https://rickstaa.github.io/machine-learning-control/simzoo/simzoo.html) Openai gym environments.
-   [Control](./machine_learning_control/control): Module used to train several [Machine Learning Control](https://rickstaa.github.io/machine-learning-control/control/control.html) RL/IL agents on the built gym environments.
-   [Hardware](./machine_learning_control/hardware): Module that can be used to deploy the trained RL/IL agents onto the hardware of your choice.

This framework follows a code structure similar to the [Spinningup](https://spinningup.openai.com/en/latest/) educational package. By doing this, we hope to make it easier for new researchers to get started with our Algorithms. If you are new to RL, you are therefore highly encouraged first to check out the SpinningUp documentation and play with before diving into our codebase. Our implementation sometimes deviates from the [Spinningup](https://spinningup.openai.com/en/latest/) version to increase code maintainability, extensibility and readability. You can find a list of the main differences in the documentations.

## Clone the repository

Since the repository contains several git submodules to use all the features, it needs
to be cloned using the `--recurse-submodules` argument:

```bash
git clone --recurse-submodules https://github.com/rickstaa/machine-learning-control.git
```

If you already cloned the repository and forgot the `--recurse-submodule` argument you
can pull the submodules using the following git command:

```bash
git submodule update --init --recursive
```

## Installation and Usage

Please see the [docs](https://rickstaa.github.io/machine-learning-control/) for installation and usage instructions.
