.. _algos:

================
Available Agents
================

The SLC package includes a collection of robust RL algorithms accompanied by their less stable baselines. These algorithms are designed with non-recurrent `MLP`_ actor-critic models, making them well-suited for fully observable RL environments that do not rely on image data, such as the `gymnasium Mujoco`_ and `stable-gym`_ environments. The implementation follows a modular approach, allowing for seamless adaptation to different types of environments and neural network architectures.

.. _`MLP`: https://en.wikipedia.org/wiki/Multilayer_perceptron
.. _`gymnasium Mujoco`: https://gymnasium.farama.org/environments/mujoco/
.. _`stable-gym`: https://rickstaa.dev/stable-gym/

Stable Agents
-------------

.. important::
   As explained in the :ref:`installation section <gym_envs_install>` of the documentation,
   although the ``opt_type`` algorithm variable can be used to train on standard
   :gymnasium:`gymnasium <>` environments, the stable RL agents require a positive definite
   cost function to guarantee stability (and robustness). Several custom environments with
   positive definite cost functions can be found in the :stable-gym:`stable-gym <>` and
   :ros-gazebo-gym:`ros-gazebo-gym <>` packages. When using the latter, make sure to set
   ``positive_reward`` to ``True``.

The SLC package currently contains the following theoretically stable RL algorithms:

.. toctree::
   :maxdepth: 1

   algorithms/lac
   algorithms/latc

Unstable Agents
---------------

The SLC package currently contains the following (unstable) baseline RL algorithms:

.. toctree::
   :maxdepth: 1

   algorithms/sac
