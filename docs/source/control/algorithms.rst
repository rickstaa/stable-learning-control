.. _algorithms:

================
Available Agents
================

Reinforcement Learning Algorithms
=================================

The Reinforcement learning algorithms are all implemented with `MLP`_ (non-recurrent) actor-critics,
making them suitable for fully-observed, non-image-based RL environments, e.g.,
the `gymnasium Mujoco`_ environments. The SLC currently contains
the following RL agents:

.. _`MLP`: https://en.wikipedia.org/wiki/Multilayer_perceptron
.. _`gymnasium Mujoco`: https://gymnasium.farama.org/environments/mujoco/

.. toctree::
   :maxdepth: 1

   algorithms/sac
   algorithms/lac

Imitation Learning Agents
=========================

.. todo::
   No Imitation Learning agents have been implemented. We are planning to add Imitation Learning
   agents in the future.
