ros_gazebo_gym environments
===========================

The BLC package also works with the task environments contained in the `ros_gazebo_gym`_
framework. This framework provides a way to train RL algorithms on ROS-based robots simulated in Gazebo. Only the panda environment has been extensively
tested with the BLC package.

Task environments
-----------------

The `ros_gazebo_gym`_ package currently contains the following task environments:

-   **PandaPickAndPlace-v1:** Lift a block into the air.
-   **PandaPush-v1:** Push a block to a goal position.
-   **PandaReach-v1:** Move fetch to a goal position.
-   **PandaSlide-v1:** Slide a puck to a goal position.

These environments were based on the original `openai_gym robotics environments <https://gym.openai.com/envs/#robotics>`_.

Installation and Usage
----------------------

The official `ros_gazebo_gym documentation`_ provides more information about the `ros_gazebo_gym`_ framework and its task environments. Below we added the
installation and usage steps for using the BLC algorithms with the Panda task environment.

Installation
~~~~~~~~~~~~

To use the Panda task environment, you first have to install ROS noetic (see the `ROS documentation`_).
After you install ROS you can then `create a catkin workspace <http://wiki.ros.org/catkin/Tutorials/create_a_workspace>`_
and install the ros dependencies using the following command:

.. code-block:: bash

    rosdep install --from-path src --ignore-src -r -y --skip-keys libfranka

After the ROS dependencies have been installed, you have to compile the libfranka library (see the `frankaemika documentation`_). If this is finished
you can build the `ros_gazebo_gym` package using the following command:

.. code-block: bash

    'catkin build -DCMAKE_BUILD_TYPE=Debug -DFranka_DIR:PATH=/home/<USER_NAME>/libfranka/build

Virtual environment
~~~~~~~~~~~~~~~~~~~

.. warning::

    When using ROS inside an anaconda environment, you might run into problems (see `this issue`_). As a result, the
    python environment setup given on the :ref:`Installation page <install>` does not work for the `ros_gazebo_gym`_
    task environments. If you want to use the BLC algorithms with any of the `ros_gazebo_gym`_  environments, you are
    advised to use the BLC package inside a virtual env. This virtual env can be created using
    the following command:

        .. code:: bash

            python -m venv ./blc --system-site-packages

    You can then source this environment using the ``. ./blc/bin/activate`` command. The  ``--system-site-packages`` flag makes sure that
    the virtual environment has access to the system site-packages. Alternatively, you can also use the
    `RoboStack ros-noetic <https://github.com/RoboStack/ros-noetic>`_ `conda-forge <https://conda-forge.org/>`_ packages
    (see this `blog post <https://medium.com/robostack/cross-platform-conda-packages-for-ros-fa1974fd1de3>`_ for more
    information.

Usage
~~~~~

The `open_ros` environments can be imported like any other environment (see the `gym documentation`_). You, however, have to make sure
that you first build and source the catkin workspace (i.e. ``. ./develop/setup.bash``) before your import. You can find usage examples
for each task environment in the `ros_gazebo_gym_examples`_ repository.

.. _`this issue`: https://answers.ros.org/question/256886/conflict-anaconda-vs-ros-catking_pkg-not-found/
.. _`ros documentation`: http://wiki.ros.org/noetic
.. _`ros_gazebo_gym`: https://rickstaa.github.io/ros-gazebo-gym
.. _`Franka Emika Panda Robot`: https://www.franka.de/
.. _`gym documentation`: https://gym.openai.com/docs/
.. _`frankaemika documentation`: https://frankaemika.github.io/docs/installation_linux.html
.. _`ros_gazebo_gym documentation`: https://rickstaa.github.io/ros-gazebo-gym
.. _`ros_gazebo_gym_examples`: https://github.com/rickstaa/ros-gazebo-gym-examples
