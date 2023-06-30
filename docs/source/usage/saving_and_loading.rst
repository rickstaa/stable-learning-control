.. _saving_and_loading:

==================
Experiment Outputs
==================

.. contents:: Table of Contents

In this section, we'll cover

- what outputs come from SLC algorithm implementations,
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
|``tf2_save/``          | | **TensorFlow implementations only.** A directory containing |
|                       | | everything needed to restore the trained agent and value    |
|                       | | functions (`Details for TensorFlow saves below`_).          |
+-----------------------+---------------------------------------------------------------+
|``config.json``        | | A :obj:`dict` containing an as-complete-as-possible         |
|                       | | description of the args and kwargs you used to launch the   |
|                       | | training function. If you passed in something which can't   |
|                       | | be serialised to JSON, it should get handled gracefully by  |
|                       | | the logger, and the config file will represent it with a    |
|                       | | string. Note: this is meant for record-keeping only.        |
|                       | | Launching an experiment from a config file is not currently |
|                       | | supported.                                                  |
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
    to be a problem for gymnasium Box2D environments in older versions of gymnasium, which can't be saved in this manner.

.. admonition:: You Should Know

    The only file in here that you should ever have to use "by hand" is the ``config.json`` file. Our agent testing utility
    will load things from the ``tf2_save/`` or ``torch_save/`` directory, and our plotter interprets the contents of ``progress.txt``,
    which are the correct tools for interfacing with these outputs. But there is no tooling for ``config.json``--it's just
    there as a reference for the hyperparameters used when you ran the experiment.

.. _checkpoints:

PyTorch Save Directory Info
---------------------------
.. _`Details for PyTorch saves below`:

The ``torch_save`` directory contains:

+-----------------------------------------------------------------------------------+
| **Pyt_Save Directory Structure**                                                  |
+-------------------+---------------------------------------------------------------+
|``checkpoints/``   | | Folder that when the ``save_checkpoints`` cmd line argument |
|                   | | is set to ``True`` contains the state of both the env and   |
|                   | | model at multiple ``checkpoints`` during training.          |
+-------------------+---------------------------------------------------------------+
|``model_state.pt`` | | This file contains the :obj:`state_dict` that contains the  |
|                   | | saved model weights. These weights can be used to restore   |
|                   | | the trained agent's state on an initiated instance of the   |
|                   | | respective Algorithm Class.                                 |
+-------------------+---------------------------------------------------------------+
|``save_info.json`` | | A file used by the SLC package to ease model                |
|                   | | loading. This file is not meant for the user.               |
+-------------------+---------------------------------------------------------------+

TensorFlow Save Directory Info
------------------------------
.. _`Details for TensorFlow saves below`:

The ``tf2_save`` directory contains:

+-------------------------------------------------------------------------------------------------+
| **TF2_Save Directory Structure**                                                                |
+---------------------------+---------------------------------------------------------------------+
|``checkpoints/``           | | Folder that when the ``save_checkpoints`` cmd line argument is set|
|                           | | to ``True`` contains the state of both the env and model at       |
|                           | | multiple ``checkpoints`` during training.                         |
+---------------------------+---------------------------------------------------------------------+
|``variables/``             | | A directory containing outputs from the TensorFlow Saver. See the |
|                           | | `TensorFlow save and load documentation`_ for more info.          |
+---------------------------+---------------------------------------------------------------------+
|``checkpoint``             | | A checkpoint summary file that stores information about the saved |
|                           | | checkpoints.                                                      |
+---------------------------+---------------------------------------------------------------------+
|``weights_checkpoint.*``   | | Two checkpoint data files ending with the ``.data*`` and          |
|                           | | ``.index`` file extensions. These are the actual files that are   |
|                           | | used by the :obj:`tf.train.Checkpoint` method to restore the      |
|                           | | model.                                                            |
+---------------------------+---------------------------------------------------------------------+
|``save_info.json``         | | A file used by the SLC package to ease model loading this file is |
|                           | | not meant for the user.                                           |
+---------------------------+---------------------------------------------------------------------+
|``saved_model.json``       | | The full TensorFlow program saved in the `SavedModel`_ format.    |
|                           | | this file can be used to deploy your model to hardware. See the   |
|                           | | :ref:`hardware deployment documentation<hardware>` for more info. |
+---------------------------+---------------------------------------------------------------------+

.. _`SavedModel`: https://www.tensorflow.org/guide/saved_model
.. _`TensorFlow save and load documentation`: https://www.tensorflow.org/tutorials/keras/save_and_load

Save Directory Location
=======================

Experiment results will, by default, be saved in the same directory as the SLC package,
in a folder called ``data``:

.. parsed-literal::

    stable_learning_control/
        **data/**
            ...
        docs/
            ...
        stable_learning_control/
            ...
        LICENSE
        setup.py

You can change the default results directory by modifying ``DEFAULT_DATA_DIR`` in ``stable_learning_control/user_config.py``.

Loading and Running Trained Policies
====================================

If Environment Saves Successfully
---------------------------------

SLC ships with an evaluation utility that can be used to check a trained policy's performance. In cases where the environment
is successfully saved alongside the agent, you can watch the trained agent act in the environment using:

.. parsed-literal::

    python -m stable_learning_control.run test_policy path/to/output_directory

.. seealso::

    For more information on using this utility, see the :ref:`test_policy` documentation or the code :ref:`the API reference <autoapi>`.

.. _manual_policy_testing:

Environment Not Found Error
---------------------------

If the environment wasn't saved successfully, you could expect ``test_policy.py`` to crash with something that looks like

.. parsed-literal::

    Traceback (most recent call last):
      File "stable_learning_control/utils/test_policy.py", line 153, in <module>
        run_policy(env, get_action, args.len, args.episodes, not(args.norender))
      File "stable_learning_control/utils/test_policy.py", line 114, in run_policy
        "and we can't run the agent in it. :( \n\n Check out the documentation " +
    AssertionError: Environment not found!

    It looks like the environment wasn't saved, and we can't run the agent in it. :(

    Check out the documentation page on the Test Policy utility for how to handle this situation.

In this case, watching your agent perform is slightly more painful but possible if you can
recreate your environment easily. You can try the code below in IPython or use the steps in the :ref:`load_pytorch_policy`
or :ref:`load_tf2_policy` documentation below to load the policy in a Python script.

.. code-block::

    >>> import gym
    >>> from stable_learning_control.utils.test_policy import load_pytorch_policy, run_policy
    >>> import your_env
    >>> env = gym.make('<YOUR_ENV_NAME>')
    >>> policy = load_pytorch_policy("/path/to/output_directory", env=env)
    >>> run_policy(env, policy)
    Logging data to /tmp/experiments/1536150702/progress.txt
    Episode 0    EpRet -163.830      EpLen 93
    Episode 1    EpRet -346.164      EpLen 99
    ...

If you want to load a Tensorflow agent, please replace the :meth:`~stable_learning_control.utils.test_policy.load_pytorch_policy` with
:meth:`~stable_learning_control.utils.test_policy.load_tf_policy`. An example script for manually loading policies can be found in the
``examples`` folder (i.e. :slc:`manual_env_policy_inference.py <blob/main/examples/manual_env_policy_inference.py>`).

.. _load_pytorch_policy:

Load Pytorch Policy
~~~~~~~~~~~~~~~~~~~

Pytorch Policies can be loaded using the :obj:`torch.load` method. For more information on how to load PyTorch models, see
the :torch:`PyTorch documentation <tutorials/beginner/saving_loading_models.html>`.

.. code-block:: python
    :linenos:
    :emphasize-lines: 6, 12-14, 15, 17, 18-20

    import torch
    import os.path as osp

    from stable_learning_control.utils.log_utils.logx import EpochLogger

    from stable_learning_control.algos.pytorch import LAC

    MODEL_LOAD_FOLDER = "./data/lac/oscillator-v1/runs/run_1614680001"
    MODEL_PATH = osp.join(MODEL_LOAD_FOLDER, "torch_save/model_state.pt")

    # Restore the model.
    config = EpochLogger.load_config(
        MODEL_LOAD_FOLDER
    )  # Retrieve the experiment configuration.
    env = EpochLogger.load_env(MODEL_LOAD_FOLDER)
    model = LAC(env=env, ac_kwargs=config["ac_kwargs"])
    restored_model_state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(
        restored_model_state_dict,
    )

    # Create dummy observations and retrieve the best action.
    obs = torch.rand(env.observation_space.shape)
    a = model.get_action(obs)
    L_value = model.ac.L(obs, torch.from_numpy(a))

    # Print results.
    print(f"The LAC agent thinks it is a good idea to take action {a}.")
    print(f"It assigns a Lyapunov Value of {L_value} to this action.")

In this example, observe that

* On line 6, we import the algorithm we want to load.
* On line 12-14, we use the :meth:`~stable_learning_control.utils.log_utils.logx.EpochLogger.load_config` method to restore the hyperparameters that were used during the experiment. This saves us time in setting up the correct hyperparameters.
* on line 15, we use the :meth:`~stable_learning_control.utils.log_utils.logx.EpochLogger.load_config` method to restore the environment used during the experiment. This saves us time in setting up the environment.
* on line 17, we import the model weights.
* on line 18-19, we load the saved weights onto the algorithm.

Additionally, each algorithm also contains a :obj:`~stable_learning_control.algos.pytorch.lac.LAC.restore` method, which serves as a
wrapper around the :obj:`torch.load` and  :obj:`torch.nn.Module.load_state_dict` methods.

.. _load_tf2_policy:

Load TensorFlow Policy
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    :linenos:
    :emphasize-lines: 6, 12-14, 15, 17, 18-20

    import tensorflow as tf
    import os.path as osp

    from stable_learning_control.utils.log_utils.logx import EpochLogger

    from stable_learning_control.algos.tf2 import LAC

    MODEL_LOAD_FOLDER = "./data/lac/oscillator-v1/runs/run_1614673367"
    MODEL_PATH = osp.join(MODEL_LOAD_FOLDER, "tf2_save")

    # Restore the model.
    config = EpochLogger.load_config(
        MODEL_LOAD_FOLDER
    )  # Retrieve the experiment configuration.
    env = EpochLogger.load_env(MODEL_LOAD_FOLDER)
    model = LAC(env=env, ac_kwargs=config["ac_kwargs"])
    weights_checkpoint = tf.train.latest_checkpoint(MODEL_PATH)
    model.load_weights(
        weights_checkpoint,
    )

    # Create dummy observations and retrieve the best action.
    obs = tf.random.uniform((1, env.observation_space.shape[0]))
    a = model.get_action(obs)
    L_value = model.ac.L([obs, tf.expand_dims(a, axis=0)])

    # Print results.
    print(f"The LAC agent thinks it is a good idea to take action {a}.")
    print(f"It assigns a Lyapunov Value of {L_value} to this action.")

In this example, observe that

* On line 6, we import the algorithm we want to load.
* On line 12-14, we use the :meth:`~stable_learning_control.utils.log_utils.logx.EpochLogger.load_config` method
  to restore the hyperparameters that were used during the experiment. This saves us time in setting up the correct
  hyperparameters.
* on line 15, we use the :meth:`~stable_learning_control.utils.log_utils.logx.EpochLogger.load_config` method to
  restore the environment used during the experiment. This saves us time in setting up the environment.
* on line 17, we import the model weights.
* on line 18-19, we load the saved weights onto the algorithm.

Additionally, each algorithm also contains a :obj:`~stable_learning_control.algos.tf2.lac.LAC.restore` method
which serves as a wrapper around the :obj:`tf.train.latest_checkpoint` and  :obj:`tf.keras.Model.load_weights` methods.

Using Trained Value Functions
-----------------------------

The ``test_policy.py`` tool doesn't help you look at trained value functions; if you want to use those, you must load the policy manually. Please see the :ref:`manual_policy_testing` documentation for an example of how to do this.

..  _hardware:

Deploy the saved result onto hardware
=====================================

Deploy PyTorch Algorithms
-------------------------

.. attention::

    PyTorch provides multiple ways to deploy trained models to hardware (see the :torch:`PyTorch serving documentation <blog/model-serving-in-pyorch>`). 
    Unfortunately, at the time of writing, these methods currently do not support the agents used in the SLC package. For more information, see
    `this issue`_.

.. _`this issue`: https://github.com/pytorch/pytorch/issues/29843

Deploy TensorFlow Algorithms
----------------------------

As stated above, the TensorFlow version of the algorithm also saves the entire model in the `SavedModel format`_ this format is handy for sharing or deploying
with `TFLite`_, `TensorFlow.js`_, `TensorFlow Serving`_, or `TensorFlow Hub`_. If you want to deploy your trained model onto hardware, you first have to make sure
you set the ``--export`` cmd-line argument to ``True`` when training the algorithm. This will cause the complete TensorFlow program, including trained parameters
(i.e, tf.Variables) and computation, to be saved in the ``tf2_save/saved_model.pb`` file. This `SavedModel`_ can be loaded onto the hardware using
the :obj:`tf.saved_model.load` method.

.. code-block:: python

    import os
    import tensorflow as tf
    from stable_learning_control.utils.log_utils.logx import EpochLogger

    model_path = "./data/lac/oscillator-v1/runs/run_1614673367/tf2_save"

    # Load model and environment.
    loaded_model = tf.saved_model.load(model_path)
    loaded_env = EpochLogger.load_env(os.path.dirname(model_path))

    # Get action for dummy observation.
    obs = tf.random.uniform((1, loaded_env.observation_space.shape[0]))
    a = loaded_model.get_action(obs)
    print(f"\nThe model thinks it is a good idea to take action: {a.numpy()}")

For more information on deploying TensorFlow models, see `the TensorFlow documentation`_.

.. _`TFLITE`: https://www.tensorflow.org/lite
.. _`TensorFlow.js`: https://js.tensorflow.org
.. _`TensorFlow Serving`: https://www.tensorflow.org/tfx/tutorials/serving/rest_simple
.. _`TensorFlow Hub`: https://www.tensorflow.org/hub
.. _`SavedModel format`: https://www.tensorflow.org/guide/saved_model
.. _`the TensorFlow documentation`: https://www.tensorflow.org/guide/saved_model
.. _`SavedModel`: https://www.tensorflow.org/guide/saved_model
