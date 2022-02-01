============
Installation
============

.. contents:: Table of Contents

BLC requires `Python3`_ and `OpenAI Gym`_ to work. It is
currently only supported on Linux and OSX.

.. warning::

    Although it could work on Windows, it has not been thoroughly tested.

.. _`Python3`: https://www.python.org/
.. _`OpenAi gym`: https://gym.openai.com/

Install system dependencies
===========================

The BLC package uses `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ to distribute training over multiple CPU's. In order for this to work
a Message Passing Library should be present on your system. For mac and Linux systems, the `Open MPI <https://www.open-mpi.org/>`_ library can be used.
On Linux, this library can be
installed using the following command:

.. code-block:: bash

    sudo apt install libopenmpi-dev

For mac this command can be used:

.. code-block:: bash

    brew install openmpi

.. attention::

    For Windows, the `Microsoft MPI <https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_ package can be used. As we currently do not offer support for our package with Windows, we did not test the BLC package's compatibility.

.. _install:

Installing Python
=================

We recommend installing Python through Anaconda. Anaconda is a library that includes Python and many useful packages for
Python, as well as an environment manager called Conda that makes package management simple.

Follow `the installation instructions`_ for Anaconda here. Download and install Anaconda3 (at time of writing, `Anaconda3-5.3.1`_).
Then create a Conda Python 3.7 env for organizing packages used in the BLC package:

.. code-block:: bash

    conda create -n blc python=3.7

To use Python from the environment, you just created, activate the environment with:

.. code-block:: bash

    conda activate blc

Alternatively, you can also use Python its `venv <https://docs.python.org/3/library/venv.html>`_ package to create a virtual environment. When
using this option, make sure you set the `--system-site-packages` flag when creating the environment when you need access to the system python packages
(e.g. when you use `ros_gazebo_gym <https://rickstaa.dev/ros-gazebo-gym>`_ environments).

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

Installing the BLC package
==========================

After you successfully setup your python environment, you can install the BLC package and its dependencies in
this environment. The BLC has two versions you can install:

    - A version which uses `Pytorch`_ as the backend of the RL/IL algorithms.
    - A version which uses `Tensorflow 2.0`_ as the RL/IL algorithms.

.. attention::

    We choose PyTorch as the default backend as it, in our opinion, is easier to work with than Tensorflow. However, at the time of writing, it
    is slightly slower than the Tensorflow backend. This is caused because the agents used in the BLC package use components that are
    not yet supported by `TorchScript`_ (responsible for creating a fast compiled version of PyTorch script). As PyTorch has shown to be faster
    in most implementations, this will likely change in the future. You can track the status of this speed problem
    `here <https://github.com/pytorch/pytorch/issues/29843>`_.

.. _`TorchScript`: https://pytorch.org/docs/stable/jit.html

Install the Pytorch version
---------------------------

By default, the version with the Pytorch backend is installed you can install this version using the following bash command:

.. code-block:: bash

    pip install -e .


Install the Tensorflow version
------------------------------

If you want to use the `Tensorflow 2.0`_ version please use the following command inside your Conda environment:

.. code-block:: bash

    pip install -e .[tf]

.. _`pip package manager`: https://pip.pypa.io/en/stable/installing/
.. _`Pytorch`: https://pytorch.org/
.. _`Tensorflow 2.0`: https://www.tensorflow.org/guide/effective_tf2

Installing MuJoCo (Optional)
============================


The BLC package comes bundled with several gym environments. Out of the box it includes the following environments:

* The Openai gym `Algorithmic`_ environments.
* The Openai gym `ToyText`_.
* The Openai gym `Classic Control`_ environments.
* The :ref:`BLC Simzoo <simzoo_module>` environments.

Out of the box, the BLC package does not include the Openai gym `MuJoCo`_ and `Robotics`_ environments, often used in RL benchmarks. If you want to
use the BLC package with these environments first go to the `mujoco-py`_ github page. Follow the README installation instructions, which describe how to install the `MuJoCo physics engine`_ and the `mujoco-py` package (which allows the use of MuJoCo from Python).

.. admonition:: You Should Know

    To use the MuJoCo simulator, you will need to get a `MuJoCo license`_. Free 30-day licenses are available to
    anyone and free 1-year licenses are available to full-time students.

Once you have installed MuJoCo, install the corresponding Gym environments with

.. parsed-literal::

    pip install gym[mujoco, robotics]

And then check that things are working by running SAC in the Walker2d-v2 environment with

.. parsed-literal::

    python -m bayesian_learning_control.run sac --hid "[32, 32]" --env Walker2d-v2 --exp_name mujocotest

.. _`MuJoCo`: https://gym.openai.com/envs/#mujoco
.. _`Robotics`: https://gym.openai.com/envs/#robotics
.. _`Algorithmic`: https://gym.openai.com/envs/#algorithmic
.. _`ToyText`: https://gym.openai.com/envs/#toy_text
.. _`Classic Control`: https://gym.openai.com/envs/#classic_control
.. _`mujoco-py`: https://github.com/openai/mujoco-py
.. _`MuJoCo license`: https://www.roboti.us/license.html
.. _`Box2d`: https://gym.openai.com/envs/#box2d
.. _`MuJoCo physics engine`: http://www.mujoco.org/index.html
