.. _tuning:

=====================
Hyperparameter tuning
=====================

Using ExperimentGrid
--------------------

An easy way to find good hyperparameters is to run the same algorithm with many possible hyperparameters. LC ships with a simple tool for facilitating
this, called `ExperimentGrid`_.


Consider the example in ``machine_learning_control/examples/pytorch/sac_exp_grid_search.py``:

.. literalinclude:: /../../examples/pytorch/sac_exp_grid_search.py
   :language: python
   :linenos:
   :lines: 17-
   :emphasize-lines: 21-27, 30

(An equivalent Tensorflow example is available in ``machine_learning_control/examples/tf2/sac_exp_grid_search.py``.)

After making the ExperimentGrid object, parameters are added to it with

.. parsed-literal::

    eg.add(param_name, values, shorthand, in_name)

where ``in_name`` forces a parameter to appear in the experiment name, even if it has the same value across all experiments.

After all parameters have been added,

.. parsed-literal::

    eg.run(thunk, \*\*run_kwargs)

runs all experiments in the grid (one experiment per valid configuration), by providing the configurations as kwargs to the function ``thunk``. ``ExperimentGrid.run`` uses a
function named `call_experiment`_ to launch ``thunk``, and ``**run_kwargs`` specify behaviors for ``call_experiment``.
See `the documentation page`_ for details.

Except for the absence of shortcut kwargs (you can't use ``hid`` for ``ac_kwargs:hidden_sizes`` in ``ExperimentGrid``), the basic behaviour of ``ExperimentGrid`` is the same as running things from the command line. (In fact, ``machine_learning_control.run`` uses an ``ExperimentGrid`` under the hood.)

.. _`ExperimentGrid`: ./control_utils.html#experimentgrid-utility
.. _`the documentation page`: ./control_utils.html#experimentgrid-utility
.. _`call_experiment`: ./control_utils.html#calling-experiments-utility

Using the Ray tuning package
-----------------------------

The MLC package can also be used with more advanced tuning algorithms. An example of how to use MLC with
the Ray Tuning package can be found in ``machine_learning_control/examples/torch/sac_ray_hyper_parameter_tuning.py`` and
``machine_learning_control/examples/tf2/sac_ray_hyper_parameter_tuning.py``. The requirements for this example can be installed using
the following command:

.. code-block:: bash

    pip install .[tuning]

Consider the example in ``machine_learning_control/examples/pytorch/sac_ray_hyper_parameter_tuning.py``:

.. literalinclude:: /../../examples/pytorch/sac_ray_hyper_parameter_tuning.py
   :language: python
   :linenos:
   :lines: 18-
   :emphasize-lines: 23-38, 52-58, 65-77, 83-97

(An equivalent Tensorflow example is available in ``machine_learning_control/examples/tf2/sac_ray_hyper_parameter_tuning.py``.)

In this example, on line ``23-38`` we first create a small wrapper function that makes sure that the Ray Tuner serves the hyperparameters in the MLC algorithm's
format. Following in line ``52-58`` we set the starting point for several hyperparameters used in the hyperparameter search. Next, on
line ``65-77``, we define the hyperparameter search space. Lastly, we start the hyperparameter search using the :meth:`tune.run` method online ``83-97``.

When we now run the script, the Ray tuner will search for the best hyperparameter combination. While doing so, it will print the results both to the ``std_out`` and a
Tensorboard file. You can check these Tensorboard logs using the ``tensorboard --logdir ./data/ray_results`` command. For more information on how the ray tuning
package works, see the `Ray tuning documentation`_.

.. _`Ray tuning documentation`: https://docs.ray.io/en/latest/tune/index.html
