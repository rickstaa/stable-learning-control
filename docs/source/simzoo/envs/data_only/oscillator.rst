.. _oscillator:

Oscillator gym environment
==========================

A gym environment for a synthetic oscillatory network of transcriptional regulators
called a repressilator. A repressilator is a three-gene regulatory network where the
dynamics of mRNA and proteins follow an oscillatory behaviour
(`see Elowitch et al. 2000 <https://www-nature-com.tudelft.idm.oclc.org/articles/35002125>`_
).

Observation space
-----------------

-   **m1:** Lacl mRNA concentration.
-   **m2:** tetR mRNA concentration.
-   **m3:** CI mRNA concentration.
-   **p1:** lacI (repressor) protein concentration (Inhibits transcription tetR gene).
-   **p2:** tetR (repressor) protein concentration (Inhibits transcription CI).
-   **p3:** CI (repressor) protein concentration (Inhibits transcription of lacI).

Action space
------------

-   **u1:** Number of Lacl proteins produced during continuous growth under repressor saturation (Leakiness).
-   **u2:** Number of tetR proteins produced during continuous growth under repressor saturation (Leakiness).
-   **u3:** Number of CI proteins produced during continuous growth under repressor saturation (Leakiness).

Environment goal
----------------

The goal of the agent in the oscillator environment is to act in such a way that one
of the proteins of the synthetic oscillatory network follows a supplied reference
signal.

Environment step return
-----------------------

In addition to the observations, the environment also returns the current reference and
the error when a step is taken. This results in returning the following array:

.. code-block:: python

    [m1, m2, m3, p1, p2, p3, reference, error]

How to use
----------

This environment is part of the `Simzoo package <https://github.com/rickstaa/simzoo>`_.
It is therefore registered as a gym environment when you import the Simzoo package. If
you want to use the environment in stand-alone mode, you can register it yourself.
