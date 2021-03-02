.. _`Simzoo`: https://github.com/rickstaa/simzoo

======
Simzoo
======

.. contents:: Table of Contents


The `Simzoo`_ module contains the machine_learning_control `OpenAI gym environments`_. It is automatically
installed when you install the :mlc:`Machine Learning Control <>` package but can also be installed
as a stand-alone package.

.. _`OpenAI gym environments`: https://gym.openai.com/

Environments
------------

It contains several easier data-only (no-render method) environments as well as more difficult Robotics environments.

Data-only environments
~~~~~~~~~~~~~~~~~~~~~~

The data-only environments are included in this module:

.. toctree::
   :maxdepth: 1

   envs/data_only/oscillator
   envs/data_only/ex3_ekf

Robotics environment
~~~~~~~~~~~~~~~~~~~~

The following Robotics environments are included in this module:

.. toctree::
   :maxdepth: 1

   envs/robotics/gazebo_panda_gym

.. important::

    The Robotics environments use third-party packages like `Gazebo`_ simulator for simulating the Robots. Please check the documentation of the environment in question on how to set up these dependencies.

.. _`Gazebo`: http://gazebosim.org/

How to use
----------

Here's a bare minimum example of using one of our environments. This will run an instance of the
:ref:`Oscillator-v1 <oscillator>` environment for 800 timesteps. You should see the observations
being printed to the console.

.. code-block:: python

    import gym
    import machine_learning_control.simzoo.simzoo

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
--------------------

Please follow the steps provided in the `Openai Gym documentation`_ when creating a new environment. After
you created your environment, you can use the `simzoo/simzoo/__init__.py`_ file to register them to the
:mlc:`Machine Learning Control <>` package.

.. _`simzoo/simzoo/__init__.py`: https://github.com/rickstaa/simzoo/blob/main/simzoo/__init__.py

Simzoo/simzoo/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../machine_learning_control/simzoo/simzoo/__init__.py
   :language: python
   :linenos:


.. _`OpenAi gym documentation`: https://github.com/openai/gym/blob/master/docs/creating-environments.md
