.. _algos:

================
Available Agents
================

The SLC package contains several stable RL algorithms together with their unstable baselines.
All these algorithms are implemented with `MLP`_ (non-recurrent) actor-critics, making them
suitable for fully-observed, non-image-based RL environments, e.g., the `gymnasium Mujoco`_
environments. They are implemented in a modular way, allowing for easy extension to other
types of environments and/or neural network architectures.

.. _`MLP`: https://en.wikipedia.org/wiki/Multilayer_perceptron
.. _`gymnasium Mujoco`: https://gymnasium.farama.org/environments/mujoco/

Stable Agents
-------------

.. important::

   As explained in the :ref:`installation section <gym_envs_install>` of the documentation,
   although the ``opt_type`` algorithm variable can be used to train on standard
   :gymnasium:`gymnasium <>` environments, the stable RL agents require a positive definite
   cost function to guarantee stability (and robustness). Several custom environments with
   positive definite cost functions can be found in the :stable-gym:`stable-gym <>` and
   :ros-gazebo-gym:`ros-gazebo-gym <>` packages.

The SLC package currently contains the following theoretically stable RL algorithms:

.. toctree::
   :maxdepth: 1

   algorithms/lac

Unstable Agents
---------------

The SLC package currently contains the following (unstable) baseline RL algorithms:

.. toctree::
   :maxdepth: 1

   algorithms/sac
