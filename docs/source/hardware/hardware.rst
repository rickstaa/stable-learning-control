.. _hardware:

========
Hardware
========

This module can be used to deploy algorithms that ware trained using the `control module`_
onto the hardware of your choice. At the moment, we did not yet implement tools to ease the
deploy of our algorithms onto hardware. As a result, you have to use the methods provided by
the backend to deploy our algorithms onto hardware.

Deploy PyTorch Algorithms
=========================

.. attention::
    PyTorch also provides multiple ways to deploy trained models to hardware (see the :torch:`PyTorch serving documentation <blog/model-serving-in-pyorch>`).
    However, at the time of writing, these methods currently do not support the agents used in the SLC package.
    For more information, see `this issue <https://github.com/pytorch/pytorch/issues/29843>`_.

Deploy Tensorflow Algorithms
============================

.. _tf_deploy:

To deploy a TensorFlow algorithm onto hardware, you first have to make sure you set the ``--export`` cmd-line argument
to ``True`` when training the algorithm. This will cause the complete TensorFlow program, including trained parameters
(i.e, tf.Variables) and computation, to be saved in the ``tf2_save/saved_model.pb`` file. This `SavedModel`_ can
then be loaded onto the hardware using the :obj:`tf.saved_model.load` method.

.. code-block:: python
    :caption: Tensorflow model deploy example

    import os
    import tensorflow as tf
    from stable_learning_control.utils.log_utils.logx import EpochLogger

    model_path = "./data/lac/oscillator-v1/runs/run_1614673367/tf2_save"

    # Load model and environment
    loaded_model = tf.saved_model.load(model_path)
    loaded_env = EpochLogger.load_env(os.path.dirname(model_path))

    # Get action for dummy observation
    obs = tf.random.uniform((1, loaded_env.observation_space.shape[0]))
    a = loaded_model.get_action(obs)
    print(f"\nThe model thinks it is a good idea to take action: {a.numpy()}")

For more information on deploying TensorFlow models, see `the TensorFlow documentation`_.

.. _`PyTorch serving documentation`: https://pytorch.org/blog/model-serving-in-pyorch/
.. _`the tensorflow documentation`: https://www.tensorflow.org/guide/saved_model
.. _`SavedModel`: https://www.tensorflow.org/guide/saved_model
.. _`control module`: ../control/control.html
