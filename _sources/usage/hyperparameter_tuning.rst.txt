=====================
Hyperparameter Tuning
=====================

Hyperparameter tuning is crucial in RL as it directly impacts the agent's performance and stability. Properly selected
hyperparameters can lead to faster convergence, improved overall task performance and generalizability. Because of this,
the SLC package provides several tools to help with hyperparameter tuning.

Use the ExperimentGrid utility
------------------------------

As outlined in the :ref:`Running Experiments <running_multiple_experiments>` section, the SLC package includes a utility 
class called :ref:`ExperimentGrid <exp_grid_utility>`, which enables the execution of multiple experiments **sequentially**. 
You can utilize this utility in two ways: by supplying the :ref:`CLI <runner>` with more than one value for a specific argument 
(refer to :ref:`Running Experiments <running_experiments>`), or by directly employing the 
:class:`~stable_learning_control.utils.run_utils.ExperimentGrid` class (see :ref:`running_multiple_experiments`). These 
methods facilitate running numerous experiments with distinct hyperparameter combinations, enabling a hyperparameter grid search
to identify the optimal parameter setting for your task. For instance, to execute the LAC algorithm on the `CartPoleCost-v1`_
environment with various values for actor and critic learning rates using the :ref:`CLI <runner>`, employ the following command:

.. code-block:: bash

    python -m stable_learning_control.run lac --env CartPoleCost-v1 --lr_a 0.001 0.01 0.1 --lr_c 0.001 0.01 0.1

.. _`CartPoleCost-v1`: https://rickstaa.dev/stable-gym/envs/classic_control/cartpole_cost.html

.. tip:: 
    You can enable logging of TensorBoard and Weights & Biases by adding the ``--use_tensorboard`` and ``--use_wandb`` flags to the
    above command. These tools will allow you to track the performance of your experiments and compare the results of
    different hyperparameter combinations. For more information on how to use these logging utilities, see :ref:`loggers`.

Use the Ray tuning package
--------------------------

The SLC package can also be used with more advanced tuning libraries like `Ray Tune`_, which uses cutting-edge optimization algorithms to 
find the best hyperparameters for your model faster. An example of how to use SLC with the Ray Tuning package can be found in
``stable_learning_control/examples/torch/sac_ray_hyper_parameter_tuning.py`` and 
``stable_learning_control/examples/tf2/sac_ray_hyper_parameter_tuning.py``. The requirements for this example can be
installed using the following command:

.. code-block:: bash

    pip install .[tuning]

Consider the example in ``stable_learning_control/examples/pytorch/sac_ray_hyper_parameter_tuning.py``:

.. literalinclude:: /../../examples/pytorch/sac_ray_hyper_parameter_tuning.py
   :language: python
   :linenos:
   :lines: 32-
   :emphasize-lines: 12, 15-29, 38-48, 55, 58-65, 69-94, 97

In this example, a boolean on line ``12`` can enable Weights & Biases logging. On lines ``15-29``, we first create a small wrapper
function that ensures that the Ray Tuner serves the hyperparameters in the SLC algorithm's format. Following lines ``38-48`` setup
a Weights & Biases callback if the ``USE_WANDB`` constant is set to ``True``. On line ``55``, we then set the starting point for
several hyperparameters used in the hyperparameter search. Next, we define the hyperparameter search space on lines ``58-65`` 
while we initialise the Ray Tuner instance on lines ``69-94``. Lastly, we start the hyperparameter search by calling the
Tuners ``fit`` method on line ``97``.

When running the script, the Ray tuner will search for the best hyperparameter combination. While doing so will print
the results to the ``stdout``, a TensorBoard logging file and the Weights & Biases portal. You can check the TensorBoard logs using the
``tensorboard --logdir ./data/ray_results`` command and the Weights & Biases results on `the Weights & Biases website`_. For more information on how the `Ray Tune`_ tuning package works, see
the `Ray Tune documentation`_.

.. _`Ray Tune`: https://docs.ray.io/en/latest/tune/index.html
.. _`the Weights & Biases website`: https://wandb.ai
.. _`Ray Tune documentation`: https://docs.ray.io/en/latest/tune/index.html

..  note::
    
    An equivalent TensorFlow example is available in ``stable_learning_control/examples/tf2/sac_ray_hyper_parameter_tuning.py``.
