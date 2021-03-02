Gazebo Panda Gym Environment
============================

A gym environment that can be used for training the `Franka Emika Panda Robot`_ on several robot
tasks. It was implemented using the Robot Operating System (ROS) and uses the Gazebo simulator to
simulate the robot.

.. _`Franka Emika Panda Robot`: https://www.franka.de/

Taks environments
-----------------

The `panda_openai_sim` package currently contains the following task environments:

-   **PandaPickAndPlace-v0:** Lift a block into the air.
-   **PandaPush-v0:** Push a block to a goal position.
-   **PandaReach-v0:** Move fetch to a goal position.
-   **PandaSlide-v0:** Slide a puck to a goal position.

These environments were based on the original `openai_gym robotics environments <https://gym.openai.com/envs/#robotics>`_.

Installation and Usage
----------------------

Please see the `gazebo panda gym environment documentation <https://rickstaa.github.io/gazebo-panda-gym/>`_ for installation
and usage instructions.
