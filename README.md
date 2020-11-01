# Machine Learning Control

Repository for the machine learning control framework.

## Clone the repository

Since the repository contains several git submodules to use all the features, it needs
to be cloned using the `--recurse-submodules` argument:

```bash
git clone --recurse-submodules https://github.com/rickstaa/machine_learning_control.git
```

If you already cloned the repository and forgot the `--recurse-submodule` argument you
can pull the submodules using the following git command:

```bash
git submodule update --init --recursive
```

## Create conda environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n mlc python=3.6
```

To activate the env:

```bash
conda activate mlc
```

## Installation Environment

```bash
pip install .
```

## Run the experiments

After the environment has been successfully set up you can run the algorithms as follows:

### SAC

```bash
python "./machine_learning_control/control/algos/sac/sac.py" --env="Oscillator-v0" --lr_a="1e-4" --lr_c="3e-4" --gamma="0.995" --batch-size="256" --replay-size="1000000" --l_a="2" --l_c="2" --hid_c="256" --hid_a="256"
```

### LAC
- - - -
-    [ test]