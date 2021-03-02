.. _ex3_ekf:

Ex3_EKF gym environment
=======================

A gym environment for a noisy master-slave system. This environment can be used to train a
RL based stationary Kalman filter.

Observation space
-----------------

-   **hat_x_1:** The estimated angle.
-   **hat_x_2:** The estimated frequency.
-   **x_1:** Actual angle.
-   **x_2:** Actual frequency.

Action space
---------------

-   **u1:** First action coming from the RL Kalman filter.
-   **u2:** Second action coming from the RL Kalman filter.

Environment goal
----------------
The agent's goal in the Ex3_EKF environment is to act so that
the estimator correctly estimated the original noisy system. By doing this, it serves
as an RL based stationary Kalman filter.

Environment step return
-----------------------

In addition to the observations, the environment also returns the current reference and
the error when a step is taken. This results in returning the following array:

.. code-block:: python

    [hat_x_1, hat_x_2, x_1, x_2, reference, error]


How to use
----------

This environment is part of the `Simzoo package <https://github.com/rickstaa/simzoo>`_.
It is therefore registered as a gym environment when you import the Simzoo package.
If you want to use the environment in stand-alone mode, you can register it yourself.
