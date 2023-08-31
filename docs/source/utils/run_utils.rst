=========
Run Utils
=========

.. _runner:

Command Line Interface
======================

The SLC package contains a command line interface (CLI) that can invoke several run utilities directly from the command line. You
can use the following command to check which utilities are available quickly:

.. code-block:: bash

    python -m stable_learning_control.run --help

The SLC package currently contains the following utilities:

+-----------------------+------------------------------------------------------------------------------------------------------------------------------+
| Utility               | description                                                                                                                  |
+=======================+==============================================================================================================================+
| :ref:`test_policy`    | Utility used to evaluate the performance of the trained policy in a given environment.                                       |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------+
| :ref:`plot`           | Utility used to plot diagnostics from experiments.                                                                           |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------+
| :ref:`eval_robustness`| Utility used to evaluate the robustness of a trained policy against external disturbances and changes in initial conditions. |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------+

See the corresponding module documentation for more information on a given utility.

.. autofunction:: stable_learning_control.run.run

.. _exp_grid_utility:

ExperimentGrid utility
======================

SLC ships with a tool called ExperimentGrid for making hyperparameter ablations easier. This is based on (but simpler than) `the rllab tool`_ called VariantGenerator.

.. _`the rllab tool`: https://github.com/rll/rllab/tree/master/rllab/misc/instrument.py#L173

.. autoclass:: stable_learning_control.utils.run_utils.ExperimentGrid
    :members:

.. _exp_call_utility:

Calling Experiments utility
===========================

.. autofunction:: stable_learning_control.utils.run_utils.call_experiment
