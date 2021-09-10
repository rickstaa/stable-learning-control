.. _simzoo_module:

======
Simzoo
======

.. contents:: Table of Contents

The `Simzoo <simzoo_repo>`_ module contains several `OpenAI gym environments`_ that can be used with the BLC package. It is automatically
installed when you install the BLC package but can also be installed
as a stand-alone package.

.. _`simzoo_repo`: https://github.com/rickstaa/simzoo
.. _`OpenAI gym environments`: https://gym.openai.com/

Environments
============

Biological environments
-----------------------

Gym environments that are based on Biological systems.

.. toctree::
    :maxdepth: 1

    envs/biological/oscillator

Classic control environments
----------------------------

Environments for classical control theory problems.

.. toctree::
    :maxdepth: 1

    envs/classic_control/ex3_ekf
    envs/classic_control/cart_pole_cost

Robotics environment
--------------------

Robotics control problems.

.. toctree::
    :maxdepth: 1

    envs/robotics/gazebo_panda_gym

.. important::

    The Robotics environments use third-party packages like `Gazebo`_ simulator for simulating the Robots. Please check the documentation of the environment in question on how to set up these dependencies.

.. _`Gazebo`: http://gazebosim.org/

How to use
==========

Here's a bare minimum example of using one of our environments. This will run an instance of the
:ref:`Oscillator-v1 <oscillator>` environment for 800 timesteps. You should see the observations
being printed to the console.

.. code-block:: python

    import gym
    import bayesian_learning_control.simzoo.simzoo

    env = gym.make('Oscillator-v1')
    env.reset()
    print("Taking 1000 steps in the Oscillator-v1 environment...")
    for ii in range(1000):
        env.render()  # Does not work with the Oscillator-v1 environment.
        obs, cost, done, info_doc = env.step(env.action_space.sample())  # take a random action
        if ii % 100 == 0:
            print(f"Randoms step {ii}: {obs}")
    env.close()
    print("All steps were taken!")


.. important::

    The Environments that are currently in the Simzoo package don't have a render method.

Add new Environments
====================

Please follow the steps provided in the `Openai Gym documentation`_ when creating a new environment. After
you created your environment, you can use the `simzoo/simzoo/__init__.py`_ file to register them to the
BLC package. Additionally, you can also add your environment into the ``__init__.py`` files in the subfolders to
shorten the environment namespace (see `simzoo/simzoo/envs/biological/__init__.py`_ for an example).

.. _`simzoo/simzoo/__init__.py`: https://github.com/rickstaa/simzoo/blob/main/simzoo/__init__.py
.. _`simzoo/simzoo/envs/biological/__init__.py`: https://github.com/rickstaa/simzoo/blob/main/simzoo/envs/biological/__init__.py

Simzoo/simzoo/__init__.py
-------------------------

.. literalinclude:: ../../../bayesian_learning_control/simzoo/simzoo/__init__.py
   :language: python
   :linenos:


.. _`OpenAi gym documentation`: https://github.com/openai/gym/blob/master/docs/creating-environments.md
