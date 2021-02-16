=========
Run Utils
=========

.. contents:: Table of Contents

ExperimentGrid
==============

Spinning Up ships with a tool called ExperimentGrid for making hyperparameter ablations easier. This is based on (but simpler than) `the rllab tool`_ called VariantGenerator.

.. _`the rllab tool`: https://github.com/rll/rllab/blob/master/rllab/misc/instrument.py#L173

.. autoclass:: machine_learning_control.control.utils.run_utils.ExperimentGrid
    :members:


Calling Experiments
===================

.. autofunction:: machine_learning_control.control.utils.run_utils.call_experiment

.. autofunction:: machine_learning_control.control.utils.log.setup_logger_kwargs
