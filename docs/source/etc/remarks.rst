=======
Remarks
=======

Implementation differences with SpinningUp
------------------------------------------

While developing this framework we tried to stay close to the structure of the `SpinningUp`_
package. Below are the main changes we made to this structure:

* We translated the function based algorithms to Class based equivalents.

Additionally the functionalities of several components was extended.

EpochLogger
~~~~~~~~~~~

* Our version also supports `Tensorboard`_ logging through the ``use_tensorboard`` argument.
* Our version allows you to set the logger output format.

.. _`Tensorboard`: https://www.tensorflow.org/tensorboard/

Run (CLI)
~~~~~~~~~

* Our version also contains a ``robustness_evaluation`` module.
* Our version also allows you to specify your algorithm input arguments using a configuration file. This can be done by supplying the `--exp_cfg` argument.

.. _`SpinningUp`: ./hardware/hardware.html

.. todo::
    Add module references.
    Clean up text
