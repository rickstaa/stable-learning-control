===============
Use with Docker
===============

The SLC package is designed to be fully compatible with `Docker`_, offering an improved experience in terms of reproducibility and ease of use. This document outlines the steps required to build the Docker image and run containers within the SLC package for various experiments and utilities.

.. _Docker: https://docs.docker.com/get-docker/

Prerequisites
-------------

Before you can run the SLC package in `Docker`_, you must have Docker installed and running on your machine. Visit the `Docker website <Docker_>`_ to download and install Docker for your specific operating system.

Building the Docker Image
-------------------------

First, ensure the Dockerfile is located at the root of the repository. The Dockerfile should contain the instructions to set up the environment for the SLC package. To build the Docker image, execute the following command in your terminal:

.. code-block:: bash

    docker build -t slc .

This command builds a Docker image tagged as `slc` based on the instructions in the Dockerfile.

Running Experiments
-------------------

Once the image is successfully built, you can run experiments using the following command:

.. code-block:: bash

    docker run -t -v /path/to/experiments:/stable-learning-control/experiments -v /path/to/data:/stable-learning-control/data slc sac --env Walker2d-v4 --exp_name walker

Ensure to substitute ``/path/to/experiments`` and ``/path/to/data`` with the precise paths on your host machine where the experiment configuration files are situated and where you intend to store the data. This process mounts the designated directories inside the container, enabling the experiment to execute seamlessly as if it were on the host machine. For more information on running experiments, refer to the :ref:`running_experiments` section.

.. note::
    You can also use relative paths to mount directories. However, please ensure to prefix them with the relative path prefix ``./``. If this prefix is not used, Docker will interpret the path as a volume name and create a `Docker volume`_ with that name, which will then be mounted to the container.

.. tip:: 
    By default, the' root' user owns the ``data`` folder created within the Docker container. Suppose you want to access the data from the host machine. In that case, you can change the ownership of the ``data`` folder to your user by running the following command:

    .. code-block:: bash

        sudo chown -R $USER:$USER /path/to/data

.. _Docker volume: https://docs.docker.com/storage/volumes/

Visualizing Experiments
-----------------------

For experiments requiring visualization, you can forward the display to your host machine using:

.. code-block:: bash

    docker run -t -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /path/to/experiments:/stable-learning-control/experiments -v /path/to/data:/stable-learning-control/data slc sac --env Walker2d-v4 --exp_name walker

This command configures the container to use the host machine's display for graphical output. It's essential on some systems to allow Docker containers to access the X server with:

.. code-block:: bash

    xhost +local:docker

Please note that this command lowers the security of your system by allowing local connections from any Docker container to the X server. To revert this change, run:

.. code-block:: bash

    xhost -local: Docker

Running Other Utilities
-----------------------

The SLC package includes additional utilities such as :ref:`plotting <plotting>` and :ref:`robustness evaluations <robustness_eval>` tools. These can be run in a Docker container similar to the experiments. For example:

.. code-block:: bash

    docker run -v /path/to/utility_files:/utility_files slc <utility_command>

Replace ``<utility_command>`` with the actual command you would use to run the utility, ensuring the paths are correctly mounted to provide the necessary files to the container.

VSCode Dev Container
--------------------

The SLC package also includes a `VSCode Dev Container <vscode_dev_>`_ configuration to simplify the development process. This configuration sets up a Docker container with all the necessary dependencies and extensions to develop the SLC package. For more information on how to use the VSCode Dev Container see the `VSCode Dev Container documentation <vscode_dev_>`_.

.. _`vscode_dev`: https://code.visualstudio.com/docs/devcontainers/containers
