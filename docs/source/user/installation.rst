============
Installation
============

.. contents:: Table of Contents

:mlc:`Machine Learning Control <>` requires `Python3`_ and `OpenAI Gym`_ to work. It is
currently only supported on Linux and OSX. Although it should work on Windows it has not
been thoroughly tested.

.. admonition:: You Should Know

    Many examples and benchmarks in :mlc:`Machine Learning Control <>` refer to RL environments that use the `MuJoCo`_ physics engine.
    MuJoCo is proprietary software that requires a license, which is free to trial and free for students, but otherwise is not free.
    As a result, installing it is optional, but because of its importance to the research community---it is the de facto standard for
    benchmarking deep RL algorithms in continuous control---it is preferred.

    Don't worry if you decide not to install MuJoCo, though. You can definitely get started using the Machine Learning Framework by running RL
    algorithms with the included :mlc:`Machine Learning Control <>` `simzoo`_ gym environments. Additionally, you can also provide your own gym
    environments or use the `MuJoCo`_ the `Classic Control`_ and `Box2d`_ environments in Gym, which are totally free to use.


.. _`Python3`: https://www.python.org/
.. _`OpenAi gym`: https://gym.openai.com/
.. _`Classic Control`: https://gym.openai.com/envs/#classic_control
.. _`Box2d`: https://gym.openai.com/envs/#box2d
.. _`MuJoCo`: http://www.mujoco.org/index.html
.. _`simzoo`: ../simzoo/simzoo.html

Installing Python
=================

We recommend installing Python through Anaconda. Anaconda is a library that includes Python and many useful packages for
Python, as well as an environment manager called Conda that makes package management simple.

Follow `the installation instructions`_ for Anaconda here. Download and install Anaconda3 (at time of writing, `Anaconda3-5.3.1`_).
Then create a Conda Python 3.7 env for organizing packages used in the Machine Learning Control package:

.. code-block:: bash

    conda create -n mlc python=3.7

To use Python from the environment you just created, activate the environment with:

.. code-block:: bash

    conda activate mlc

.. admonition:: You Should Know

    If you're new to python environments and package management, this stuff can quickly get confusing or overwhelming,
    and you'll probably hit some snags along the way. (Especially, you should expect problems like, "I just installed
    this thing, but it says it's not found when I try to use it!") You may want to read through some clear explanations
    about what package management is, why it's a good idea, and what commands you'll typically have to execute to
    correctly use it.

    `FreeCodeCamp`_ has a good explanation worth reading. There's a shorter description on `Towards Data Science`_ which
    is also helpful and informative. Finally, if you're an extremely patient person, you may want to read the (dry,
    but very informative) `documentation page from Conda`_.

.. _`the installation instructions`: https://docs.continuum.io/anaconda/install/
.. _`Anaconda3-5.3.1`: https://repo.anaconda.com/archive/
.. _`FreeCodeCamp`: https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c
.. _`Towards Data Science`: https://towardsdatascience.com/environment-management-with-conda-python-2-3-b9961a8a5097
.. _`documentation page from Conda`: https://conda.io/docs/user-guide/tasks/manage-environments.html
.. _`this Github issue for Tensorflow`: https://github.com/tensorflow/tensorflow/issues/20444

Installing the MLC package
==========================

After you successfully setup your python environment, you can install the :mlc:`Machine Learning Control <>` package and its dependencies using the
`pip package manager`_. This is done by running the following command in your terminal:

.. code-block:: bash

    pip install -e .

.. _`pip package manager`: https://pip.pypa.io/en/stable/installing/


Installing MuJoCo (Optional)
============================

First, go to the `mujoco-py`_ github page. Follow the installation instructions in the README, which describe how to install
the MuJoCo physics engine and the mujoco-py package (which allows the use of MuJoCo from Python).

.. admonition:: You Should Know

    To use the MuJoCo simulator, you will need to get a `MuJoCo license`_. Free 30-day licenses are available to
    anyone and free 1-year licenses are available to full-time students.

Once you have installed MuJoCo, install the corresponding Gym environments with

.. parsed-literal::

    pip install gym[mujoco, robotics]

And then check that things are working by running PPO in the Walker2d-v2 environment with

.. parsed-literal::

    python -m machine_learning_control.run sac --hid "[32, 32]" --env Walker2d-v2 --exp_name mujocotest

.. _`mujoco-py`: https://github.com/openai/mujoco-py
.. _`MuJoCo license`: https://www.roboti.us/license.html
