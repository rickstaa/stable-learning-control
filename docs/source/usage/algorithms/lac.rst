.. _lac:

=====================
Lyapunov Actor-Critic
=====================

.. contents:: Table of Contents

.. seealso::

    This document assumes you are familiar with the `Soft Actor-Critic (SAC)`_ algorithm.
    It is not meant to be a comprehensive guide but mainly depicts the difference between
    the :ref:`SAC <sac>` and `Lyapunov Actor-Critic (LAC)`_ algorithms. For more information,
    readers are referred to the original papers of `Haarnoja et al., 2019`_ (SAC) and
    `Han et al., 2020`_ (LAC).

.. _`Soft Actor-Critic (SAC)`: https://arxiv.org/abs/1801.01290
.. _`Haarnoja et al., 2019`: https://arxiv.org/abs/1801.01290
.. _`Lyapunov Actor-Critic (LAC)`: http://arxiv.org/abs/2004.14288
.. _`Han et al., 2020`: http://arxiv.org/abs/2004.14288


.. important::

    The LAC algorithm only guarantees stability in **mean cost** when trained on environments 
    with a positive definite cost function (i.e. environments in which the cost is minimized).
    The ``opt_type`` argument can be set to ``maximize`` when training in environments where
    the reward is maximized. However, because the `Lyapunov's stability conditions`_ are not satisfied,
    the LAC algorithm no longer guarantees stability in **mean** cost.

Background
==========

The Lyapunov Actor-Critic (LAC) algorithm can be seen as a direct successor of the :ref:`SAC <sac>`
algorithm. Although the :ref:`SAC <sac>` algorithm achieved impressive performance in various robotic
control tasks, it does not guarantee its actions are stable. From a control-theoretic perspective,
stability is the most critical property for any control system since it is closely related to robotic
systems' safety, robustness, and reliability. Using Lyapunov's method, the LAC algorithm solves the
aforementioned issues by proposing a data-based stability theorem that guarantees the system stays
stable in **mean cost**.

Lyapunov critic function
------------------------

.. _lyap_conditions:

The concept of `Lyapunov stability`_ is a useful and general approach for studying robotics Systems
stability. In Lyapunov's (direct) method, a scalar **"energy-like"** function, called a Lyapunov function,
is constructed to analyse a system's stability. According to `Lyapunov's stability conditions`_ a
dynamic autonomous system

.. math::

    \dot{x} = X(x), \quad \textrm{where} \quad X(0) = 0

    \textrm{with} \quad x^{*}(t) = 0, t \geq t_0;

is said to be asymptotically stable if such an **"energy"** function :math:`V(x)` exist such that in some
neighbourhood :math:`\mathcal{V}^{*}` around an equilibrium point :math:`x = 0 (\left \| x < k \right \|)`

.. _lyap_condition_2:

#. :math:`V(x)` and its partial derivatives are continuous.
#. :math:`V(x)` is positive definite
#. :math:`\dot{V}(x)` is negative semi-definite.

In classical control theory, this concept is often used to design controllers that ensure that the difference of a
Lyapunov function along the state trajectory is always negative definite. This results in stable closed-loop
system dynamics as the state is guaranteed to decrease the Lyapunov function's value and eventually converge
to the equilibrium. The biggest challenge with this approach is that finding such a function is difficult and
quickly becomes impractical. In learning-based methods, for example, since we do not have complete information
about the system, finding such a Lyapunov Function would result in trying out all possible consecutive data pairs
in the state space, i.e., to verify infinite inequalities :math:`L_{t+1}-L_{t} < 0`. The LAC algorithm solves
this by taking a data-based approach in which the controller/policy and a `Lyapunov critic function`_, both
parameterised by deep neural networks, are jointly learned. In this way, the actor learns to control the
system while only choosing actions guaranteed to be stable in mean cost. This inherent stability makes
the LAC algorithm incredibly useful for stabilising and tracking robotic systems tasks.

.. _`Han et. al 2019`: http://arxiv.org/abs/2004.14288
.. _`Lyapunov stability`: https://en.wikipedia.org/wiki/Lyapunov_stability
.. _`Lyapunov critic function`: https://en.wikipedia.org/wiki/Lyapunov_function
.. _`Lyapunov's stability conditions`: https://www.cds.caltech.edu/~murray/courses/cds101/fa02/caltech/mls93-lyap.pdf

Differences with the SAC algorithm
------------------------------------

Like its predecessor, the LAC algorithm also uses **entropy regularisation** to increase exploration and a
Gaussian actor and value-critic to develop the best action. The main difference lies in how the critic network
and the actor policy function are defined.

Critic network definition
~~~~~~~~~~~~~~~~~~~~~~~~~

The LAC algorithm uses a single Lyapunov Critic instead of the double Q-Critic used in the :ref:`SAC <sac>` algorithm. This
new Lyapunov critic is similar to the Q-Critic, but a square output activation function is used instead of an
Identity output activation function. This is done to ensure that the network output is positive, such that
:ref:`condition (2) <lyap_condition_2>` of the :ref:`Lyapunov's stability conditions <lyap_conditions>`
holds.

.. math::
    L_{c}(s,a) = f_{\phi}(s,a)^{T}f_{\phi}(s,a)

Similar to :ref:`SAC <sac>` during training, :math:`L_{c}` is updated by `mean-squared Bellman error (MSBE) minimisation`_ using the following objective function

.. math::
    J(L_{c}) = E_{\mathcal{D}}\left[\frac{1}{2}(L_{c}(s,a)-L_{target}(s,a))^2\right]


Where :math:`L_{target}` is the approximation target received from the `infinite-horizon discounted return value function`_

.. math::
   :nowrap:

    \begin{gather*}
    L(s) = E_{a\sim \pi}L_{target}(s,a) \\
    \textrm{with} \\
    L_{target}(s,a) = c + \max_{a'}\gamma L_{c}^{'}(s^{'}, a^{'})
    \end{gather*}

and :math:`\mathcal{D}` the set of collected transition pairs.

.. note::

    As explained by `Han et al., 2020`_ the sum of cost over a finite time horizon can also be used as the
    approximation target. This version still needs to be implemented in the SLC framework.

.. _`mean-squared Bellman error (MSBE) minimisation`: https://spinningup.openai.com/en/latest/algorithms/ddpg.html?highlight=msbe#the-q-learning-side-of-ddpg
.. _`infinite-horizon discounted return value function`: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#value-functions
.. _`Belman equation`: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations

Policy function definition
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the LAC algorithm, the policy is optimised according to

.. math::

    \min_{\theta} \underE{s \sim \mathcal{D} \\ \xi \sim \mathcal{N}}{\lambda(L_{c}(s^{'}, f_{\theta}(\epsilon, s^{'}))-L_{c}(s, a) + \alpha_{3}c) + \mathcal{\alpha}\log \pi_{\theta}(f_{\theta}(\epsilon, s)|s) + \mathcal{H}_{t}}

In this :math:`f_{\theta}(\epsilon, s)` represents the quashed Gaussian policy

.. math::

    \tilde{a}_{\theta}(s, \xi) = \tanh\left( \mu_{\theta}(s) + \sigma_{\theta}(s) \odot \xi \right), \;\;\;\;\; \xi \sim \mathcal{N}(0, I).

and :math:`\mathcal{H}_{t}` the desired minimum expected entropy. When comparing this function with policy loss used in the :ref:`SAC <sac>` algorithm,

.. math::

    \max_{\theta} \underE{s \sim \mathcal{D} \\ \xi \sim \mathcal{N}}{Q_{\phi_1}(s,\tilde{a}_{\theta}(s,\xi)) - \alpha \log \pi_{\theta}(\tilde{a}_{\theta}(s,\xi)|s) + \mathcal{H}_{t}},

several differences stand out. First, the policy is minimised instead of maximised in the LAC algorithm. With the LAC algorithm, the
objective is to stabilise a system or track a given reference. In these cases, instead of achieving a high return, we want to reduce the
difference between the current position and a reference or equilibrium position. This leads us to the second difference that can be
observed: the term in the :ref:`SAC <sac>` algorithm that represents the Q-values.

.. math::

    Q_{\phi_1}(s, f_{\theta}(\epsilon, s))

is in the LAC algorithm replaced by

.. math::

    \lambda(L_{c}(s^{'}, f_{\theta}(\epsilon, s^{'})) - L_{c}(s, a)  + \alpha_{3}c)

As a result, in the LAC algorithm, the loss function now increases the probability of actions that cause the system to be close to the
equilibrium or reference value while decreasing the likelihood of actions that drive the system away from these values.
The :math:`a_{3}c` `quadratic regularisation`_ term ensures that the mean cost is positive. The :math:`\lambda` term represents the
Lyapunov Lagrange multiplier and is responsible for weighting the relative importance of the stability condition. Similar to the
entropy Lagrange multiplier :math:`\alpha` used in the :ref:`SAC <sac>` algorithm this term is updated by

.. math::

    \lambda \leftarrow \max(0, \lambda + \delta \bigtriangledown_{\lambda}J(\lambda)))

where :math:`\delta` is the learning rate. This is done to constrain the average Lyapunov value during training.

.. _`quadratic regularisation`: https://proceedings.neurips.cc/paper/2019/file/0a4bbceda17a6253386bc9eb45240e25-Paper.pdf
.. _`Lyapunov stable system`: https://en.wikipedia.org/wiki/Lyapunov_stability
.. _`Lyapunov critic function`: https://en.wikipedia.org/wiki/Lyapunov_function
.. _`entropy`: https://en.wikipedia.org/wiki/Entropy_(information_theory)

Quick Fact
----------

* LAC is an off-policy algorithm.
* It is guaranteed to be stable in mean cost.
* The version of LAC implemented here can only be used for environments with continuous action spaces.
* An alternate version of LAC, which slightly changes the policy update rule, can be implemented to handle discrete action spaces.
* The SLC implementation of LAC does not support parallelisation.

Further Reading
---------------

For more information on the LAC algorithm, please check out the original paper of `Han et al., 2020`_.

.. _`Han et al., 2020`: http://arxiv.org/abs/2004.14288


Pseudocode
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Lyapunov-based Actor-Critic (LAC)}
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
                \STATE Sample mini-batches of transitions from $D$ and update $L_{c}$, $\pi$, Lagrance multipliers with eq. (7) and (14) of Han et al., 2020
            \ENDFOR
        \UNTIL{eq. 11 of Han et al., 2020 is satisfied}
    \end{algorithmic}
    \end{algorithm}

.. _`11 of Han et al., 2020`: https://arxiv.org/pdf/2004.14288.pdf
.. _`eq. (7) and (14) from Han et al., 2020`: https://arxiv.org/pdf/2004.14288.pdf

Implementation
==============

.. admonition:: You Should Know

    In what follows, we give documentation for the PyTorch and TensorFlow implementations of LAC in SLC.
    They have nearly identical function calls and docstrings, except for details relating to model construction.
    However, we include both full docstrings for completeness.

Algorithm: PyTorch Version
--------------------------

.. autofunction:: stable_learning_control.control.algos.pytorch.lac.lac

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch version of the LAC algorithm is implemented by subclassing the :class:`torch.nn.Module` class. As a
result the model weights are saved using the ``model_state`` dictionary ( :attr:`~stable_learning_control.control.algos.pytorch.lac.LAC.state_dict`). 
These saved weights can be found in the ``torch_save/model_state.pt`` file. For an example of how to load a model using
this file, see :ref:`saving_and_loading` or the :torch:`PyTorch documentation <tutorials/beginner/saving_loading_models.html>`.

Algorithm: TensorFlow Version
-----------------------------

.. attention::

    The TensorFlow version is still experimental. It is not guaranteed to work, and it is not
    guaranteed to be up-to-date with the PyTorch version.

.. autofunction:: stable_learning_control.control.algos.tf2.lac.lac

Saved Model Contents: TensorFlow Version
----------------------------------------

The TensorFlow version of the SAC algorithm is implemented by subclassing the :class:`tf.nn.Model` class. As a result, both the
full model and the current model weights are saved. The complete model can be found in the ``saved_model.pb`` file, while the current
weights checkpoints are found in the ``tf_safe/weights_checkpoint*`` file. For an example of using these two methods, see :ref:`saving_and_loading`
or the :tensorflow:`TensorFlow documentation <tutorials/keras/save_and_load>`.

References
==========

Relevant Papers
---------------

- `Actor-Critic Reinforcement Learning for Control with Stability Guarantee`_, Han et al, 2020
- `The general problem of the stability of motion`_, J. Mawhin, 2005
- `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`_, Haarnoja et al, 2018
- `Soft Actor-Critic: Algorithms and Applications`_, Haarnoja et al, 2019

.. _`Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`: https://arxiv.org/abs/1801.01290
.. _`Soft Actor-Critic: Algorithms and Applications`: https://arxiv.org/abs/1812.05905
.. _`Actor-Critic Reinforcement Learning for Control with Stability Guarantee`: http://arxiv.org/abs/2004.14288
.. _`The general problem of the stability of motion`: https://www.researchgate.net/publication/242019659_Alexandr_Mikhailovich_Liapunov_The_general_problem_of_the_stability_of_motion_1892

Acknowledgements
----------------

* Parts of this documentation are directly copied, with the author's consent, from the original paper of `Han et. al 2019`_.
