.. _sac:

=================
Soft Actor-Critic
=================

.. contents:: Table of Contents

.. important::
    The SAC algorithm has no stability guarantees. Please use the :ref:`LAC <lac>` algorithm if
    you require stability guarantees.

Background
==========

Soft Actor-Critic (SAC) is an algorithm that optimises a stochastic policy in an off-policy way,
forming a bridge between stochastic policy optimisation and `DDPG-style`_ approaches. It isn't a
direct successor to `TD3`_ (having been published roughly concurrently). Still, it incorporates
the clipped double-Q trick, and due to the inherent stochasticity of the policy in SAC, it also
winds up benefiting from something like target policy smoothing.

A central feature of SAC is **entropy regularisation.** The policy is trained to maximise a
trade-off between expected return and `entropy`_, a measure of randomness in the policy. This
is closely connected to the exploration-exploitation trade-off: increasing entropy results in
more exploration, which can accelerate learning later on. It can also prevent the policy from
prematurely converging to a bad local optimum.

.. _`DDPG-style`: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
.. _`TD3`: https://spinningup.openai.com/en/latest/algorithms/td3.html?highlight=TD3
.. _`Soft Actor-Critic`: https://arxiv.org/abs/1801.01290
.. _`entropy`: https://en.wikipedia.org/wiki/Entropy_(information_theory)

Quick Facts
-----------

* SAC is an off-policy algorithm.
* The version of SAC implemented here can only be used for environments with continuous action spaces.
* An alternate version of SAC, which slightly changes the policy update rule, can be implemented to
  handle discrete action spaces.
* The SLC implementation of SAC does not support parallelisation.

Further Reading
---------------

The version implemented here was based on the version implemented in the `SpinningUp repository`_. For more
information on the SAC algorithm, you are referred to the `SpinningUp documentation`_ or the original
paper of `Haarnoja et al., 2019`_. Our implementation slightly differs from the SpinningUp version in
that we also added the Automatic Entropy Tuning scheme introduced by `Haarnoja et al., 2019`_. As a
result, during training, the entropy Lagrange Multiplier :math:`\alpha` is updated by

.. math::

    \alpha \leftarrow \max(0, \alpha + \delta \bigtriangledown_{\alpha}J(\alpha)))

where :math:`\delta` is the learning rate. As explained in `Haarnoja et al., 2019`_, this constrains
the policy's average entropy.

.. _`Haarnoja et al., 2019`: https://arxiv.org/pdf/1812.05905.pdf
.. _`SpinningUp repository`: https://spinningup.openai.com/en/latest/algorithms/sac.html
.. _`SpinningUp documentation`: https://spinningup.openai.com/en/latest/algorithms/sac.html

Implementation
==============

.. admonition:: You Should Know

    In what follows, we give documentation for the PyTorch and TensorFlow implementations of SAC
    in SLC. They have nearly identical function calls and docstrings, except for details relating
    to model construction. However, we include both full docstrings for completeness.

Algorithm: PyTorch Version
------------------------------

.. autofunction:: stable_learning_control.algos.pytorch.sac.sac

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch version of the SAC algorithm is implemented by subclassing the :class:`torch.nn.Module` class. As a result,
the model weights are saved using the ``model_state`` dictionary (
:attr:`~stable_learning_control.algos.pytorch.sac.SAC.state_dict`). These saved weights
can be found in the ``torch_save/model_state.pt`` file. For an example of how to load a model
using this file, see :ref:`saving_and_loading` or the 
:torch:`PyTorch documentation <tutorials/beginner/saving_loading_models.html>`.

Algorithm: TensorFlow Version
---------------------------------

.. attention::
    The TensorFlow version is still experimental. It is not guaranteed to work, and it is not
    guaranteed to be up-to-date with the PyTorch version.

.. autofunction:: stable_learning_control.algos.tf2.sac.sac

Saved Model Contents: TensorFlow Version
----------------------------------------

The TensorFlow version of the SAC algorithm is implemented by subclassing the :class:`tf.nn.Model`
class. As a result, both the full model and the current model weights are saved. The full model
can be found in the ``saved_model.pb`` file, while the current weights checkpoints are in the
``tf_safe/weights_checkpoint*`` file. For an example of using these two methods, see
:ref:`saving_and_loading` or the `TensorFlow documentation`_.

.. _`TensorFlow documentation`: https://www.tensorflow.org/tutorials/keras/save_and_load

References
==========

Relevant Papers
---------------

- `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`_, Haarnoja et al, 2018
- `Soft Actor-Critic: Algorithms and Applications`_, Haarnoja et al, 2019
- `Learning to Walk via Deep Reinforcement Learning`_, Haarnoja et al, 2018

.. _`Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`: https://arxiv.org/abs/1801.01290
.. _`Soft Actor-Critic: Algorithms and Applications`: https://arxiv.org/abs/1812.05905
.. _`Learning to Walk via Deep Reinforcement Learning`: https://arxiv.org/abs/1812.11103

Other Public Implementations
----------------------------

- `SAC release repo`_ (original "official" codebase).
- `Softlearning repo`_ (current "official" codebase).
- `Yarats and Kostrikov repo`_
- `SpinningUp repo`_ (The version our version was based on).

.. _`SAC release repo`: https://github.com/haarnoja/sac
.. _`Softlearning repo`: https://github.com/rail-berkeley/softlearning
.. _`Yarats and Kostrikov repo`: https://github.com/denisyarats/pytorch_sac
.. _`SpinningUp repo`: https://github.com/openai/spinningup/tree/master/spinup
