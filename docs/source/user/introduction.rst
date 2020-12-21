.. _`Machine Learning Control`: https://github.com/rickstaa/machine-learning-control

============
Introduction
============

.. contents:: Table of Contents

What This Is
============

Welcome to the :mlc:`Machine Learning Control <>` (MLC) framework! The Machine Learning Control framework enables
you to automatically create, train and deploy various Reinforcement Learning (RL) and
Imitation learning (IL) control algorithms directly from real-world data. This framework
is made up of four main modules:

* `Modeling`_: Module that uses state of the art System Identification and State Estimation techniques to create an Openai gym environment out of real data.
* `Simzoo`_: Module that contains several already created :mlc:`Machine Learning Control <>` `Openai gym`_ environments.
* `Control`_: Module used to train several :mlc:`Machine Learning Control <>` RL/IL agents on the built gym environments.
* `Hardware`_: Module that can be used to deploy the trained RL/IL agents onto the hardware of your choice.

This framework follows a code structure similar to the `SpinningUp`_ educational resource. By doing this, we hope to make
it easier for new researchers to get started with our Algorithms. If you are new to RL, you are therefore highly
encouraged first to check out the SpinningUp documentation and play with before diving into our codebase. Our
implementation sometimes deviates from the `SpinningUp`_ version to increase code maintainability, extensibility
and readability. You can find a list of the main differences in `remarks section of this document`_.

.. _`Modeling`: ./modeling/modeling.html
.. _`Simzoo`: ./simzoo/simzoo.html
.. _`Control`: ./control/control.html
.. _`Hardware`: ./hardware/hardware.html
.. _`SpinningUp`: ./hardware/hardware.html
.. _`Openai gym`: https://gym.openai.com/
.. _`remarks section of this document`: ../etc/remarks.html

.. warning::

    The Machine Learning Control framework is still in its development state. We can therefore not guarantee that it is bug free.
    Please open :issue:`an issue<>` if you experience problems or something is unclear.

Why We Built This
=================

.. todo::
    Add short explanation.

How This Serves Our Mission
===========================

.. todo::
    Add short explanation.

Code Design Philosophy
======================

.. todo::
    Add short explanation.

Long-Term Support and Support History
=====================================

.. todo::
    Add short explanation.
