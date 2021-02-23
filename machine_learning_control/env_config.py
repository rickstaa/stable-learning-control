"""Module used for adding extra Gym environments.

Example:
    This module allows you to add your own custom gym environments to the
    :mlc:`Machine Learning Control <>` package. These environments should inherit from
    the :class:`gym.env` class. See
    `this issue on the openai github <https://github.com/openai/gym/blob/master/docs/creating-environments.md>`_
    for more information on how to create custom environments. Environments that are
    added to this file can be called directly without using the module prefix (eg.
    ``Oscillator-v1`` instead of
    ``machine_learning_control.simzoo.simzoo:Oscillator-v1``).

    .. code-block:: python

        import custom environment
"""  # noqa: E501
