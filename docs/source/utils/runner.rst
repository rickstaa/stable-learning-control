.. _runner:

Command Line Interface
======================

This CLI can be used to invoke the utilities of each of the four BLC modules directly. You
can use the following command to check which utilities are available quickly:

.. code-block:: bash

    python -m bayesian_learning_control.run --help

The BLC package currently contains the following utilities:

+---------+--------------------+------------------------------------------------------------------------------------------------------------------------------+
| Module  | Utility            | description                                                                                                                  |
+=========+====================+==============================================================================================================================+
| Control | `test_policy`_     | Utility used to evaluate the performance of the trained policy in a given environment.                                       |
+---------+--------------------+------------------------------------------------------------------------------------------------------------------------------+
|         | `plot`_            | Utility used to to plot diagnostics from experiments.                                                                        |
+---------+--------------------+------------------------------------------------------------------------------------------------------------------------------+
|         | `eval_robustness`_ | Utility used to evaluate the robustness of a trained policy against external disturbances and changes in initial conditions. |
+---------+--------------------+------------------------------------------------------------------------------------------------------------------------------+

.. _`test_policy`: ../control/control_utils.html#test-policy-utility
.. _`plot`: ../control/control_utils.html#plot-utility
.. _`eval_robustness`: ../control/control_utils.html#robustness-eval-utility

See the corresponding module documentation for more information on a given utility.

.. autofunction:: bayesian_learning_control.run.run
