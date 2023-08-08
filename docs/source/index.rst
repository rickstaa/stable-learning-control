.. Stable Learning Control master file, created by
   sphinx-quickstart on Wed Aug 15 04:21:07 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================================
Welcome to Stable Learning Control documentation
================================================

.. figure:: ./images/safe_panda.jpg

Welcome to the :slc:`Stable Learning Control <>` (SLC) framework! This framework contains
a collection of robust Reinforcement Learning (RL) control algorithms designed to ensure stability.
These algorithms are built upon the Lyapunov actor-critic architecture introduced by
`Han et al. 2020`_. They guarantee stability and robustness by leveraging `Lyapunov stability theory`_.
These algorithms are specifically tailored for use with :gymnasium:`gymnasium environments <>`
that feature a positive definite cost function (i.e. environments in which the cost is minimized). Several 
ready-to-use compatible environments can be found in the  :stable-gym:`stable-gym <>` and 
:ros-gazebo-gym:`Ros Gazebo Gym <>` packages.

.. note::

   This framework was built upon the `SpinningUp`_ educational resource. By doing this, we 
   hope to make it easier for new researchers to start with our Algorithms. If you are new
   to RL, check out the SpinningUp documentation and play with it before diving into our
   codebase. Our implementation deviates from the `SpinningUp`_ version to increase code
   maintainability, extensibility, and readability.

.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
.. _`Lyapunov stability theory`: https://en.wikipedia.org/wiki/Lyapunov_stability
.. _`SpinningUp`: https://spinningup.openai.com/en/latest/

.. toctree::
   :maxdepth: 2
   :caption: Usage

   usage/installation
   usage/algorithms
   usage/running
   usage/hyperparameter_tuning
   usage/saving_and_loading
   usage/plotting
   usage/eval_robustness
   usage/benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   utils/loggers
   utils/mpi
   utils/run_utils
   utils/plotter
   utils/testers

.. toctree::
   :maxdepth: 2
   :caption: Development

   dev/contributing.rst
   dev/doc_dev.rst
   dev/license.rst

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   autoapi/index.rst

.. toctree::
   :maxdepth: 3
   :caption: Etc.

   etc/acknowledgements

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
