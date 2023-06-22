"""File used for adding extra gymnasium environments.

Example:
    This module allows you to add your own custom gymnasium environments to the
    SLC package. These environments should inherit from
    the :class:`gym.env` class. See
    `this issue on the Openai github <https://www.gymlibrary.dev/content/environment_creation/>`_
    for more information on how to create custom environments. Environments that are
    added to this file can be called directly without using the module prefix (eg.
    ``Oscillator-v1`` instead of ``stable_gym:Oscillator-v1``).

    .. code-block:: python

        # Import environments you want to use
        import custom_environment_1
        import custom_environment_2
"""  # noqa: E501
