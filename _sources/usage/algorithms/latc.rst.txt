.. _latc:

=================================
Lyapunov Actor-Twin Critic (LATC) 
=================================

.. contents:: Table of Contents

.. seealso::
    This document assumes you are familiar with the :ref:`Lyapunov Actor-Critic (LAC) <lac>` algorithm.
    It is not a comprehensive guide but mainly depicts the difference between the
    :ref:`Lyapunov Actor-Twin Critic <latc>` and :ref:`Lyapunov Actor-Critic (LAC) <lac>` algorithms. It
    is therefore meant to complement the :ref:`LAC <lac>` algorithm documentation.

.. important::
    Like the LAC algorithm, this LATC algorithm only guarantees stability in **mean cost** when trained on
    environments with a positive definite cost function (i.e. environments in which the cost is minimised).
    The ``opt_type`` argument can be set to ``maximise`` when training in environments where the reward is
    maximised. However, because the `Lyapunov's stability conditions`_ are not satisfied, the LAC algorithm
    no longer guarantees stability in **mean** cost.

.. _`Lyapunov's stability conditions`: https://www.cds.caltech.edu/~murray/courses/cds101/fa02/caltech/mls93-lyap.pdf

Background
==========

The Laypunov Actor-Twin Critic (LATC) algorithm is a successor to the :ref:`LAC <lac>` algorithm. In contrast
to its predecessor, the LATC algorithm employs a dual-critic approach, aligning it more closely with the
:ref:`SAC <sac>` algorithm upon which LAC was built initially. In the SAC framework, these dual critics
served to counteract overestimation bias by selecting the minimum value from both critics for the actor updates. 
In our case, we employ the maximum to minimise the cost, thus addressing potential underestimation bias in
Lyapunov values. For a deeper exploration of this concept, refer to the research paper by `Haarnoja et. al 2019`_.
For more information on the inner workings of the LAC algorithm, refer to the :ref:`LAC <lac>` algorithm
documentation. Below only the differences between the LAC and LATC algorithms are discussed.

.. _`Haarnoja et. al 2019`: https://arxiv.org/abs/1801.01290

Differences with the LAC algorithm
------------------------------------

Like its direct predecessor, the LATC algorithm also uses **entropy regularisation** to increase exploration and a
Gaussian actor and value-critic to develop the best action. The main difference lies in the fact that the
:class:`~stable_learning_control.algos.pytorch.policies.lyapunov_actor_twin_critic.LyapunovActorTwinCritic`
contains two critic instead of one. These critics are identical to the critic used in the :ref:`LAC <lac>`
algorithm but trained separately. Following their maximum is used to update the actor. Because of this the policy issues
optimised according to

.. math::
    :label: latc_policy_update

    \min_{\theta} \underE{s \sim \mathcal{D} \\ \xi \sim \mathcal{N}}{\lambda(\bm{L_{c_{max}}(s^{'}, f_{\theta}(\epsilon, s^{'})})-L_{c}(s, a) + \alpha_{3}c) + \mathcal{\alpha}\log \pi_{\theta}(f_{\theta}(\epsilon, s)|s) + \mathcal{H}_{t}}

Where :math:`L_{c_{max}}` now represents the maximum of the two critics. The rest of the algorithm remains the same.

.. important:: 
    Because the LATC and LAC algorithms are so similar, the :meth:`~stable_learning_control.algos.pytorch.latc.latc` is
    implemented as a wrapper around the :meth:`~stable_learning_control.algos.pytorch.lac.lac` function. This wrapper
    only changes the actor-critic architecture to :class:`~stable_learning_control.algos.pytorch.policies.lyapunov_actor_twin_critic.LyapunovActorTwinCritic`.
    To prevent code duplication, the :class:`stable_learning_control.algos.pytorch.policies.lyapunov_actor_critic.LyapunovActorCritic` class
    was modified to use the maximum of the two critics when the :class:`~stable_learning_control.algos.pytorch.policies.lyapunov_actor_twin_critic.LyapunovActorTwinCritic`
    class is set as the actor-critic architecture.

Quick Fact
----------

* LATC is an off-policy algorithm.
* It is guaranteed to be stable in mean cost.
* The version of LATC implemented here can only be used for environments with continuous action spaces.
* An alternate version of LATC, which slightly changes the policy update rule, can be implemented to handle discrete action spaces.
* The SLC implementation of LATC does not support parallelisation.


Further Reading
---------------

For more information on the LATC algorithm, please check out the :ref:`LAC <lac>` documentation and the original paper of `Han et al., 2020`_.

.. _`Han et al., 2020`: https://arxiv.org/abs/2004.14288

Pseudocode
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Lyapunov-based Actor-Twin Critic (LATC)}
        \label{alg1}
    \begin{algorithmic}[1]
        \REQUIRE Maximum episode length $N$ and maximum update steps $M$
        \REPEAT
            \STATE Samples $s_{0}$ according to $\rho$
            \FOR{$t=1$ to $N$}
                \STATE Sample $a$ from $\pi(a|s)$ and step forward
                \STATE Observe $s'$, $c$ and store ($s$, $a$, $c$, $s'$) in $\mathcal{D}$
            \ENDFOR
            \FOR{$i=1$ to $M$}
                \STATE Sample mini-batches of transitions from $D$ and update $L_{c}$, $L2_{c}$, $\pi$, Lagrance multipliers with eq. (7) and (14) of Han et al., 2020 and the new actor update rule described above
            \ENDFOR
        \UNTIL{eq. 11 of Han et al., 2020 is satisfied}
    \end{algorithmic}
    \end{algorithm}

Implementation
==============

.. admonition:: You Should Know

    In what follows, we give documentation for the PyTorch and TensorFlow implementations of LATC in SLC.
    They have nearly identical function calls and docstrings, except for details relating to model construction.
    However, we include both full docstrings for completeness.

Algorithm: PyTorch Version
--------------------------

.. autofunction:: stable_learning_control.algos.pytorch.latc.latc

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch version of the LATC algorithm is implemented by subclassing the :class:`torch.nn.Module` class. Because of this and because
the LATC algorithm is implemented as a wrapper around the LAC algorithm; the model weights are saved using the ``model_state`` dictionary ( :attr:`~stable_learning_control.algos.pytorch.lac.LAC.state_dict`). 
These saved weights can be found in the ``torch_save/model_state.pt`` file. For an example of how to load a model using
this file, see :ref:`saving_and_loading` or the :torch:`PyTorch documentation <tutorials/beginner/saving_loading_models.html>`.

Algorithm: TensorFlow Version
-----------------------------

.. attention::
    The TensorFlow version is still experimental. It is not guaranteed to work, and it is not
    guaranteed to be up-to-date with the PyTorch version.

.. autofunction:: stable_learning_control.algos.tf2.latc.latc

Saved Model Contents: TensorFlow Version
----------------------------------------

The TensorFlow version of the LATC algorithm is implemented by subclassing the :class:`tf.nn.Model` class. As a result, both the
full model and the current model weights are saved. The complete model can be found in the ``saved_model.pb`` file, while the current
weights checkpoints are found in the ``tf_safe/weights_checkpoint*`` file. For an example of using these two methods, see :ref:`saving_and_loading`
or the :tensorflow:`TensorFlow documentation <tutorials/keras/save_and_load>`.
