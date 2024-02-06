"""Contains a small gymnasium wrapper that can be used to disturb a physics parameter of
a gymnasium environment.
"""

import gymnasium as gym


class EnvAttributesDisturber(gym.Wrapper):
    """A gymnasium wrapper that can be used to disturb a physics parameter of a
    gymnasium environment.

    Attributes:
        disturbance_label (str): A label for the disturbance that can be used for
            logging or plotting. Used in the
            :mod:`~stable_learning_control.utils.eval_robustness` utility.
    """

    def __init__(self, env, attributes, values):
        """Initialise the EnvAttributesDisturber object.

        Args:
            env (gym.Env): The gymnasium environment.
            attributes (list): A list of attributes to disturb.
            values (list): A list of values to set the parameters to.

        Raises:
            ValueError: The number of parameters and values must be the same.
            AttributeError: The parameter does not exist in the environment.
        """
        attributes = attributes if isinstance(attributes, list) else [attributes]
        values = values if isinstance(values, list) else [values]
        super().__init__(env)

        # Throw a warning if params and values are not of the same length.
        if len(attributes) != len(values):
            raise ValueError(
                "The number of parameters and values must be the same. "
                "Got {} parameters and {} values.".format(len(attributes), len(values))
            )

        # Try to update the parameter value.
        for attribute, value in zip(attributes, values):
            try:
                setattr(self.env, attribute, value)
            except AttributeError:
                raise AttributeError(
                    "The parameter '{}' does not exist in the environment.".format(
                        attribute
                    )
                )

        # Disturbance label.
        if len(attributes) == 1:
            self.disturbance_label = "{}_{}".format(attributes[0], values[0])
        else:
            disturbance_label = []
            for idx, (attribute, value) in enumerate(zip(attributes, values)):
                disturbance_label.append("attr{}_{}".format(idx + 1, round(value, 2)))
            self.disturbance_label = "_".join(disturbance_label)
