============
Installation
============

.. contents:: Table of Contents

The SLC framework requires several system and :python:`Python <>` dependencies to work.
In this section, we will explain how to install these dependencies. The package was only
tested on Ubuntu but should also work on other Linux distributions and Macs. Installing
the package on Windows may be possible, but this still needs to be tested.

Install system dependencies
===========================

The SLC package uses `mpi4py`_ to distribute training over multiple CPUs. To work, a
Message Passing Library should be on your system. The `Open MPI`_ library can be used
for Mac and Linux systems. On Linux, this library can be installed using the following
command:

.. code-block:: bash

    sudo apt install libopenmpi-dev

For mac this command can be used:

.. code-block:: bash

    brew install openmpi

.. note::
    The `Microsoft MPI`_ package can be used for Windows. 

.. attention::
    The MPI functionality is not yet fully implemented for the SLC algorithms. As a result,
    the MPI library is not yet required to run the SLC package. However, it is still
    recommended to install the MPI library as it will be required in the future.
    
.. _`mpi4py`: https://mpi4py.readthedocs.io/en/stable/
.. _`Open MPI`: https://www.open-mpi.org/
.. _`Microsoft MPI`: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi

.. _install:

Installing Python
=================

We recommend installing :python:`Python <>` through Anaconda. Anaconda is a library that
includes Python and many valuable packages for Python, as well as an environment manager
called Conda that simplifies package management. Download and install Anaconda3 (at the
time of writing, `Anaconda3-5.3.1`_) according to the `Anaconda documentation`_. Then 
create a Conda Python env for organizing packages used in the SLC package:

.. code-block:: bash

    conda create -n slc python=3.10

To use Python from the environment you just created, activate the environment with:

.. code-block:: bash

    conda activate slc

Alternatively, you can use Python's `venv`_ package to create a virtual environment.

.. admonition:: You Should Know

    If you're new to Python environments and package management, this stuff can quickly
    get confusing or overwhelming, and you'll probably hit some snags along the way. Especially,
    you should expect problems like, ``I just installed this thing, but it says it's not found 
    when I try to use it!``. You can read through some clear explanations about package
    management, why it's a good idea, and what commands you'll typically have to execute
    to use it correctly. `FreeCodeCamp`_ has a good explanation worth reading. There's a
    shorter description of `Towards Data Science`_ which is also helpful and informative.
    Finally, if you're incredibly patient, you may want to read the 
    (dry but very informative) `documentation page from Conda`_.

.. _`venv`: https://docs.python.org/3/library/venv.html
.. _`Anaconda3-5.3.1`: https://repo.anaconda.com/archive/
.. _`Anaconda documentation`: https://docs.continuum.io/free/anaconda/install/
.. _`FreeCodeCamp`: https://www.freecodecamp.org/news/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c
.. _`Towards Data Science`: https://towardsdatascience.com/environment-management-with-conda-python-2-3-b9961a8a5097
.. _`documentation page from Conda`: https://conda.io/docs/user-guide/tasks/manage-environments.html

.. _install_slc:

Installing Stable Learning Control
==================================

After you successfully setup your python environment, you can install the SLC package and its dependencies in
this environment. The SLC has two versions you can install:

    - ``torch``: A version which uses :torch:`Pytorch <>` as the backend of the RL algorithms.
    - ``tf2``: A **experimental** version which uses :tensorflow:`TensorFlow 2.0 <>` as the backend
      of the RL algorithms.

.. note::
    We choose PyTorch as the default backend as it is easier to work with than TensorFlow. However, at the
    time of writing, it is slightly slower than the TensorFlow backend. This is caused because the agents
    used in the SLC package use components not yet supported by :torch:`TorchScript <docs/stable/jit.html>`
    (responsible for creating a fast compiled version of PyTorch script). As PyTorch has shown to be faster
    in most implementations, this will likely change in the future. You can track the status of this speed
    problem `here <https://github.com/pytorch/pytorch/issues/29843>`_.

Install the Pytorch version
---------------------------

We use the `pip package manager`_ to install the SLC package and its dependencies. After you installed pip 
you can install the SLC package using the following bash command:

.. code-block:: bash

    pip install -e .

This command will install the SLC package with the :torch:`Pytorch <>` backend in your Conda environment.

.. important::
    If you are using Conda, you may come across issues while installing or utilizing the SLC package,
    such as installation errors or script freezing. To effectively resolve these problems, it is
    recommended to install the mpi4py_ package from within Conda instead of using pip. This can
    be accomplished by executing the following command:

    .. code-block:: bash

        conda install mpi4py

.. _`pip package manager`: https://pip.pypa.io/en/stable/getting-started/

Install the TensorFlow version
------------------------------

.. attention::
    As stated above, the Pytorch version was used during our experiments. As a result, the
    TensorFlow version is less well-tested than the Pytorch version and has limited support.
    It should therefore be considered experimental, as no guarantees can be given about the
    correctness of these algorithms.

If you still want to use the :tensorflow:`TensorFlow 2.0 <>` version, you can install the SLC
package with the the following command:

.. code-block:: bash

    pip install -e .[tf2]

.. warning::
    If you want to use the GPU version of TensorFlow, you must ensure you performed all
    the steps described in the `TensorFlow installation guide`_. It is also essential to
    know that depending on the version of TensorFlow and PyTorch you use, you might have
    to install different versions of `CUDA`_ and `cuDNN`_ (see `the TensorFlow`_ and 
    `Pytorch` documentation). As a result, some combinations of TensorFlow and Pytorch
    are not compatible with each other. You are therefore advised to create two separate
    Conda environments, one for Pytorch and one for TensorFlow. Additionally, if you did
    choose to use `venv`_ instead of Conda, you must ensure the correct version of `CUDA`_
    and `cuDNN`_ are installed on your system.

.. _`TensorFlow installation guide`: https://www.tensorflow.org/install/pip
.. _`the TensorFlow`: https://www.tensorflow.org/install/source#gpu
.. _`Pytorch`: https://pytorch.org/get-started/locally/
.. _`CUDA`: https://developer.nvidia.com/cuda-toolkit
.. _`cuDNN`: https://developer.nvidia.com/cudnn

.. _gym_envs_install:

Installing gymnasium environments
=================================

The algorithms in the SLC package are designed to work with any :gymnasium:`gymnasium based environment <api/env>` 
with a continuous action space. However, stability and performance of stable RL algorithms like :ref:`LAC <lac>` are
only guaranteed for environments with a positive definite cost function (i.e., environments where a cost is minimised).
As a result, even though the ``opt_type`` algorithm variable can be used to train on standard 
:gymnasium:`gymnasium <environments/classic_control>` and :gymnasium:`Mujoco <environments/mujoco>` environments
in which the reward is maximised stability guarantees no longer hold. We, however, provide a set of custom
environments which are compatible with the stable algorithms:

* :stable-gym:`stable-gym <>`: Several gymnasium environments with cost functions compatible with (stable) 
  RL agents (i.e. positive definite). 
* :ros-gazebo-gym:`ros-gazebo-gym packages <>`: A framework for training RL algorithms on ROS Gazebo 
  robots that can return positive definite cost functions (i.e. when ``positive_reward`` is set to ``True``).

Please refer to the documentation of these packages for more information on installing
these environments. After you install these environments or any custom environment, you
can use them in the SLC package by specifying the module name of the environment in the
:ref:`--env_name <env_flags>` argument of the SLC command line interface. For example, if you want to train
the :ref:`LAC <lac>` algorithm on the `CartPoleCost-v1`_ environment of the
:stable-gym:`stable-gym <>` package, you can use the following command:

.. code-block:: bash

    python -m stable_learning_control.run lac --env_name stable_gym:CartPole-v1

.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
.. _`CartPoleCost-v1`: https://rickstaa.dev/stable-gym/envs/classic_control/cartpole_cost.html
