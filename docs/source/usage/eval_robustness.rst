.. _robustness_eval:

=====================
Evaluating Robustness
=====================

SLC ships with a handy utility for evaluating the policy's robustness. This is done by assessing the policy performance for several episodes inside a given environment and applying several disturbances. You can run it with:

.. parsed-literal::

    python -m stable_learning_control.run eval_robustness [path/to/output_directory] [disturber] [-h] [--list_disturbers] [--disturber_config DISTURBER_CONFIG] [--data_dir DATA_DIR] [--itr ITR] [--len LEN] [--episodes EPISODES] [--render] [--deterministic]
        [--disable_baseline] [--observations [OBSERVATIONS [OBSERVATIONS ...]]] [--references [REFERENCES [REFERENCES ...]]]
        [--reference_errors [REFERENCE_ERRORS [REFERENCE_ERRORS ...]]] [--absolute_reference_errors] [--merge_reference_errors] [--use_subplots] [--use_time] [--save_result]
        [--save_plots] [--figs_fmt FIGS_FMT] [--font_scale FONT_SCALE] [--use_wandb] [--wandb_job_type WANDB_JOB_TYPE] [--wandb_project WANDB_PROJECT] [--wandb_group WANDB_GROUP]
        [--wandb_run_name WANDB_RUN_NAME]

The most important input arguments are:

.. option:: output_dir

    :obj:`str`. The path to the output directory where the agent and environment were saved. 

.. option:: disturber

    :obj:`str`. The name of the disturber you want to evaluate. Can include an unloaded module in 'module:disturber_name' style.

.. option:: --cfg, --disturber_config DISTURBER_CONFIG, default=None

    :obj:`str`. The configuration you want to pass to the disturber. It sets up the range of disturbances you wish to evaluate. Expects a
    dictionary that depends on the specified disturber (e.g. ``"{'mean': [0.25, 0.25], 'std': [0.05, 0.05]}"`` for
    :class:`~stable_learning_control.disturbers.ObservationRandomNoiseDisturber` disturber).

.. note::

    For more information about all the input arguments available for the ``eval_robustness`` tool you can use the ``--help`` flag or check the :ref:`robustness evaluation utility <eval_robustness>`
    documentation or :ref:`the API reference <autoapi>`.

Robustness eval configuration file (yaml)
-----------------------------------------

The SLC CLI comes with a handy configuration file loader that can be used to load `YAML`_ configuration files.
These configuration files provide a convenient way to store your robustness evaluation parameters such that results
can be reproduced. You can supply the CLI with an experiment configuration file using the ``--eval_cfg`` flag. The
configuration file format equals the format expected by the :ref:`--exp_cfg <exp_cfg>` flag of the :ref:`run experiments <running_experiments>` utility.

.. option:: --eval_cfg

    :obj:`path str`. Sets the path to the ``yml`` config file used for loading experiment hyperparameter.

Available disturbers
====================

The disturbers contained in the SLC package can be listed with the ``--list_disturbers`` flag. The following disturbers are currently available:

.. autosummary::

    ~stable_learning_control.disturbers.action_impulse_disturber.ActionImpulseDisturber
    ~stable_learning_control.disturbers.action_random_noise_disturber.ActionRandomNoiseDisturber
    ~stable_learning_control.disturbers.env_attributes_disturber.EnvAttributesDisturber
    ~stable_learning_control.disturbers.observation_random_noise_disturber.ObservationRandomNoiseDisturber

To get more information about the configuration values a given disturber expects, add the ``--help`` flag after a given disturber name. For example:

.. parsed-literal::

    python -m stable_learning_control.run eval_robustness [path/to/output_directory] ObservationRandomNoiseDisturber --help

Results
=======

Saved files
-----------

The robustness evaluation tool can save several files to disk that contain information about the robustness evaluation:

+--------------------------------------------------------------------------------------------+
| **Output Directory Structure**                                                             |
+-----------------------+--------------------------------------------------------------------+
|``figures/``           | | A directory containing the robustness evaluation plots when the  |
|                       | | ``--save_plots`` flag was used.                                  |
+-----------------------+--------------------------------------------------------------------+
|``eval_statistics.csv``| | File with general performance diagnostics for the episodes and   |
|                       | | disturbances used during the robustness evaluation.              |
+-----------------------+--------------------------------------------------------------------+
|``eval_results.csv``   | | Pandas data frame containing all the data that was collected for |
|                       | | the episodes and disturbances used during the robustness         |
|                       | | evaluation. This file is only present when the ``--save_results``|
|                       | | flag is set and can be used to create custom plots.              |
+-----------------------+--------------------------------------------------------------------+

These files will be saved inside the ``eval`` directory inside the output directory.

.. tip:: 
    
    You can also log these results to Weights & Biases by adding the and ``--use_wandb`` flag to the
    CLI command (see :ref:`eval_robustness` for more information).

Plots
-----

Default plots
^^^^^^^^^^^^^

By default, the following plots are displayed when running the robustness evaluation:

.. figure:: ../images/plots/lac/example_lac_robustness_eval_obs_plot.svg
    :align: center

    This plot shows how the mean observation (states) paths change under different disturbances. This
    reference value is also shown if the environment dictionary contains a reference key.

.. figure:: ../images/plots/lac/example_lac_robustness_eval_costs_plot.svg
    :align: center

    This plot shows how the mean reward changes under different disturbances.

.. figure:: ../images/plots/lac/example_lac_robustness_eval_reference_errors_plot.svg
    :align: center

    This plot shows how the mean reference error changes under different disturbances. This plot is
    only shown if the environment dictionary contains a reference key.

.. todo::
    
    Update reference_errors plot.

.. _`robust_custom_plots`:

Create custom plots
^^^^^^^^^^^^^^^^^^^

You can also create any plots you like using the ``eval_results.csv`` data frame saved during the robustness evaluation. An example
of how this is done can be found in ``stable_learning_control/examples/manual_robustness_eval_plots.py``. This example loads the ``eval_results.csv``
data frame and uses it to plot how observation one changes under different disturbances. It also shows reference one.

.. literalinclude:: /../../examples/manual_robustness_eval_plots.py
   :language: python
   :linenos:
   :lines: 5-

Running this code will give you the following figure:

.. figure:: ../images/plots/lac/example_lac_robustness_eval_custom_plot.svg
    :align: center

    This plot shows how the mean of observation one changed under different disturbances. The first reference value is also shown.

.. todo:: 
    
    Update figure.

.. _env_add:

Use with custom environments
============================

The :ref:`robustness evaluation utility <eval_robustness>` can be used with any :gymnasium:`gymnasium environment <>`. If you
want to show the reference and reference error plots, add the ``reference`` and ``reference_error`` keys to the environment
dictionary that is returned by the environments ``step`` and ``reset`` methods.

How to add new disturber
========================

The disturbers in the SLC package are implemented as gymnasium:`gymnasium wrappers <api/wrappers/>`. Because of this, they can
be used with any :gymnasium:`gymnasium environment <>`. If you want to add aor new disturber, you only have to ensure that it is
a Python class that inherits from the :class:`gym.Wrapper` class. For more infmation about
:gymnasium:`gymnasium wrappers <api/wrappers/>` please checkout the :gymnasium:`gymnasium documentation <api/wrappers/>`. After
implementing your disturber, you can create a pull request to add it to the SLC package or use it through the ``--disturber``
by specifying the module containing your disturber and the disturber class name. For example:

.. parsed-literal::

    python -m stable_learning_control.run eval_robustness [path/to/output_directory] --disturber "my_module.MyDisturber"

Special attributes
------------------

The SLC package looks for several attributes in the disturber class to get information about the disturber that can be used during the robustness evaluation. These attributes are:

.. describe:: disturbance_label

    :obj:`str`. Can be used to set the label of the disturber in the plots. If not present the :ref:`robustness evaluation utility <eval_robustness>` will generate a label based on the disturber configuration.

Manual robustness evaluation
============================

A script version of the eval robustness tool can be found in the ``examples`` folder (i.e. :slc:`eval_robustness.py <blob/main/examples/eval_robustness.py>`). This script can be used
when you want to perform some quick tests without implementing a disturber class.
