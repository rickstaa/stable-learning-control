.. _saving_and_loading:

=====================================
Saving and Loading Experiment Outputs
=====================================

.. contents:: Table of Contents

In this section, we'll cover

- what outputs come from Machine Learning Control algorithm implementations,
- what formats they're stored in and how they're organised,
- where they are stored and how you can change that,
- and how to load and run trained policies.

Algorithm Outputs
=================

Each algorithm is set up to save a training run's hyperparameter configuration, learning progress, trained agent
and value functions, and a copy of the environment if possible (to make it easy to load up the agent and environment
simultaneously). The output directory contains the following:

+---------------------------------------------------------------------------------------+
| **Output Directory Structure**                                                        |
+-----------------------+---------------------------------------------------------------+
|``torch_save/``        | | **PyTorch implementations only.** A directory containing    |
|                       | | everything needed to restore the trained agent and value    |
|                       | | functions (`Details for PyTorch saves below`_).             |
+-----------------------+---------------------------------------------------------------+
|``tf2_save/``          | | **Tensorflow implementations only.** A directory containing |
|                       | | everything needed to restore the trained agent and value    |
|                       | | functions (`Details for Tensorflow saves below`_).          |
+-----------------------+---------------------------------------------------------------+
|``config.json``        | | A dict containing an as-complete-as-possible description    |
|                       | | of the args and kwargs you used to launch the training      |
|                       | | function. If you passed in something which can't be         |
|                       | | serialised to JSON, it should get handled gracefully by the |
|                       | | logger, and the config file will represent it with a string.|
|                       | | Note: this is meant for record-keeping only. Launching an   |
|                       | | experiment from a config file is not currently supported.   |
+-----------------------+---------------------------------------------------------------+
|``progress.(csv/txt)`` | | A (comma/tab) separated value file containing records of the|
|                       | | metrics recorded by the logger throughout training. eg,     |
|                       | | ``Epoch``,   ``AverageEpRet``, etc.                         |
+-----------------------+---------------------------------------------------------------+
|``vars.pkl``           | | A pickle file containing anything about the algorithm state |
|                       | | which should get stored. Currently, all algorithms only use |
|                       | | this to save a copy of the environment.                     |
+-----------------------+---------------------------------------------------------------+

.. admonition:: You Should Know

    Sometimes environment-saving fails because the environment can't be pickled, and ``vars.pkl`` is empty. This is known
    to be a problem for Gym Box2D environments in older versions of Gym, which can't be saved in this manner.

.. admonition:: You Should Know

    The only file in here that you should ever have to use "by hand" is the ``config.json`` file. Our agent testing utility
    will load things from the ``tf2_save/`` or ``torch_save/`` directory, and our plotter interprets the contents of ``progress.txt``,
    and those are the correct tools for interfacing with these outputs. But there is no tooling for ``config.json``--it's just
    there as a reference for the hyperparameters used when you ran the experiment.

PyTorch Save Directory Info
---------------------------
.. _`Details for PyTorch saves below`:

The ``torch_save`` directory contains:

+----------------------------------------------------------------------------------+
| **Pyt_Save Directory Structure**                                                 |
+-------------------+--------------------------------------------------------------+
|``checkpoints/``   | | Folder that when the ``save_checkpoints`` cmd line argument|
|                   | | is set to ``True`` contains the state of the model at      |
|                   | | multiple ``checkpoints`` during training.                  |
+-------------------+--------------------------------------------------------------+
|``model_state.pt`` | | This file contains the 'state_dict' that contains the      |
|                   | | saved model weights. These weights can be used to restore  |
|                   | | the trained agent's state on an initiated instance of the  |
|                   | | respective Algorithm Class.                                |
+-------------------+--------------------------------------------------------------+
|``save_info.json`` | | A file used by the :mlc:`MLC <>` package to ease model     |
|                   | | loading. This file is not meant for the user.              |
+-------------------+--------------------------------------------------------------+

Tensorflow Save Directory Info
------------------------------
.. _`Details for Tensorflow saves below`:

The ``tf2_save`` directory contains:

+-------------------------------------------------------------------------------------------+
| **TF2_Save Directory Structure**                                                          |
+---------------------------+---------------------------------------------------------------+
|``checkpoints/``           | | Folder that when the ``save_checkpoints`` cmd line argument |
|                           | | is set to ``True`` contains the state of the model at       |
|                           | | multiple ``checkpoints`` during training.                   |
+---------------------------+---------------------------------------------------------------+
|``variables/``             | | A directory containing outputs from the Tensorflow Saver.   |
|                           | | See the `Tensorflow save and load documentation`_ for more  |
|                           | | info.                                                       |
+---------------------------+---------------------------------------------------------------+
|``checkpoint``             | | A checkpoint summary file that stores information about the |
|                           | | saved checkpoints.                                          |
+---------------------------+---------------------------------------------------------------+
|``weights_checkpoint.*``   | | Two checkpoint data files ending with the ``.data*`` and    |
|                           | | ``.index`` file extensions. These are the actual files that |
|                           | | are used by the :obj:`tf.train.Checkpoint` method to        |
|                           | | restore the model.                                          |
+---------------------------+---------------------------------------------------------------+
|``save_info.json``         | | A file used by the :mlc:`MLC <>` package to ease model      |
|                           | | loading  this file is not meant for the user.               |
+---------------------------+---------------------------------------------------------------+
|``saved_model.json``       | | The full TensorFlow program saved in the `SavedModel`       |
|                           | | format. This file can be used to deploy your model to       |
|                           | | hardware. See the `hardware deployment documentation`_ for  |
|                           | | more info.                                                  |
+---------------------------+---------------------------------------------------------------+

.. _`hardware deployment documentation`: ../hardware/hardware.html
.. _`SavedModel`: https://www.tensorflow.org/guide/saved_model
.. _`Tensorflow save and load documentation`: https://www.tensorflow.org/tutorials/keras/save_and_load

Save Directory Location
=======================

Experiment results will, by default, be saved in the same directory as the Machine Learning Control package,
in a folder called ``data``:

.. parsed-literal::

    machine_learning_control/
        **data/**
            ...
        docs/
            ...
        machine_learning_control/
            ...
        LICENSE
        setup.py

You can change the default results directory by modifying ``DEFAULT_DATA_DIR`` in ``machine_learning_control/user_config.py``.

Loading and Running Trained Policies
====================================

Test Policy utility
-------------------

:mlc:`Machine Learning Control <>` ships with an evaluation utility that can be used to check a trained policy's performance. For cases where the environment
is successfully saved alongside the agent, it's a cinch to watch the trained agent act in the environment using:


.. parsed-literal::

    python -m machine_learning_control.run test_policy path/to/output_directory

For more information on how to use this utility see the :ref:`test_policy <test_policy>` documentation or the code :ref:`api`.

.. _manual_policy_testing:

Manual policy testing
---------------------

Load Pytorch Policy
~~~~~~~~~~~~~~~~~~~

Pytorch Policies can be loaded using the :obj:`torch.load` method. For more information on how to load PyTorch models see
the `PyTorch documentation`_.

.. code-block:: python
    :linenos:
    :emphasize-lines: 6, 12-14, 15, 17, 18-19

    import torch
    import os.path as osp

    from machine_learning_control.utils.log_utils.logx import EpochLogger

    from machine_learning_control.control.algos.pytorch import LAC

    MODEL_LOAD_FOLDER = "./data/lac/oscillator-v1/runs/run_1614680001"
    MODEL_PATH = osp.join(MODEL_LOAD_FOLDER, "torch_save/model_state.pt")

    # Restore the model
    config = EpochLogger.load_config(
        MODEL_LOAD_FOLDER
    )  # Retrieve the experiment configuration
    env = EpochLogger.load_env(MODEL_LOAD_FOLDER)
    model = LAC(env=env, ac_kwargs=config["ac_kwargs"])
    restored_model_state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(
        restored_model_state_dict,
    )

    # Create dummy observations and retrieve the best action
    obs = torch.rand(env.observation_space.shape)
    a = model.get_action(obs)
    L_value = model.ac.L(obs, torch.from_numpy(a))

    # Print results
    print(f"The LAC agent thinks it is a good idea to take action {a}.")
    print(f"It assigns a Lyapunov Value of {L_value} to this action.")

In this example, observe that

* On line 6, we import the algorithm we want to load.
* On line 12-14, we use the :meth:`~machine_learning_control.utils.log_utils.logx.EpochLogger.load_config` method to restore the hyperparameters that were used during the experiment. This saves us time in setting up the right hyperparameters.
* on line 15, we use the :meth:`~machine_learning_control.utils.log_utils.logx.EpochLogger.load_config` method to restore the environment that was used during the experiment. This saves us time in setting up the environment.
* on line 17, we import the model weights.
* on line 18-19, we load the saved weights onto the algorithm.

Additionally, each algorithm also contains a :obj:`~machine_learning_control.control.algos.pytorch.lac.LAC.restore` method which serves as a
wrapper around the :obj:`torch.load` and  :obj:`torch.nn.Module.load_state_dict` methods.

.. _`Pytorch Documentation`: https://pytorch.org/tutorials/beginner/saving_loading_models.html

Load Tensorflow Policy
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    :linenos:
    :emphasize-lines: 6, 12-14, 15, 17, 18-19

    import tensorflow as tf
    import os.path as osp

    from machine_learning_control.utils.log_utils.logx import EpochLogger

    from machine_learning_control.control.algos.tf2 import LAC

    MODEL_LOAD_FOLDER = "./data/lac/oscillator-v1/runs/run_1614673367"
    MODEL_PATH = osp.join(MODEL_LOAD_FOLDER, "tf2_save")

    # Restore the model
    config = EpochLogger.load_config(
        MODEL_LOAD_FOLDER
    )  # Retrieve the experiment configuration
    env = EpochLogger.load_env(MODEL_LOAD_FOLDER)
    model = LAC(env=env, ac_kwargs=config["ac_kwargs"])
    weights_checkpoint = tf.train.latest_checkpoint(MODEL_PATH)
    model.load_weights(
        weights_checkpoint,
    )

    # Create dummy observations and retrieve the best action
    obs = tf.random.uniform((1, env.observation_space.shape[0]))
    a = model.get_action(obs)
    L_value = model.ac.L([obs, tf.expand_dims(a, axis=0)])

    # Print results
    print(f"The LAC agent thinks it is a good idea to take action {a}.")
    print(f"It assigns a Lyapunov Value of {L_value} to this action.")

In this example, observe that

* On line 6, we import the algorithm we want to load.
* On line 12-14, we use the :meth:`~machine_learning_control.utils.log_utils.logx.EpochLogger.load_config` method to restore the hyperparameters that were used during the experiment. This saves us time in setting up the right hyperparameters.
* on line 15, we use the :meth:`~machine_learning_control.utils.log_utils.logx.EpochLogger.load_config` method to restore the environment that was used during the experiment. This saves us time in setting up the environment.
* on line 17, we import the model weights.
* on line 18-19, we load the saved weights onto the algorithm.

Additionally, each algorithm also contains a :obj:`~machine_learning_control.control.algos.tf2.lac.LAC.restore` method which serves as a
wrapper around the :obj:`tf.train.latest_checkpoint` and  :obj:`tf.keras.Model.load_weights` methods.

Deploy the saved result onto hardware
=====================================

As stated above, the Tensorflow version of the algorithm also saves the full model in the `SavedModel format`_ this format is very useful for sharing or deploying
with `TFLite`_, `TensorFlow.js`_, `TensorFlow Serving`_, or `TensorFlow Hub`_. For more information, see :ref:`the hardware deployment documentation <hardware>`.

.. important::
    TensorFlow also PyTorch multiple ways to deploy trained models to hardware (see the `PyTorch serving documentation`_). However, at the time of writing,
    these methods currently do not support the agents used in the :mlc:`MLC <>` package. For more information, see
    `this issue <https://github.com/pytorch/pytorch/issues/29843>`_.


.. _`TFLITE`: https://www.tensorflow.org/lite
.. _`Tensorflow.js`: https://js.tensorflow.org
.. _`TensorFlow Serving`: https://www.tensorflow.org/tfx/tutorials/serving/rest_simple
.. _`TensorFlow Hub`: https://www.tensorflow.org/hub
.. _`SavedModel format`: https://www.tensorflow.org/guide/saved_model
.. _`PyTorch serving documentation`: https://pytorch.org/blog/model-serving-in-pyorch/
