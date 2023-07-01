"""Module used for creating TensorFlow learning rate schedulers."""
import numpy as np

from stable_learning_control.utils.import_utils import import_tf

tf = import_tf()  # Throw custom warning if tf is not installed.


def get_lr_scheduler(decaying_lr_type, lr_start, lr_final, steps):
    """Creates a learning rate scheduler.

    Args:
        decaying_lr_type (str): The learning rate decay type that is used (
        options are: ``linear`` and ``exponential`` and ``constant``).
        lr_start (float): Initial learning rate.
        lr_end (float): Final learning rate.
        steps (int, optional): Number of steps/epochs used in the training. This
            includes the starting step.

    Returns:
        tensorflow.keras.optimizers.schedules.LearningRateSchedule: A learning rate
            scheduler object.

    .. seealso::
        See the :tf2:`TensorFlow <keras/optimizers/schedules>` documentation on how to
        implement other decay options.
    """  # noqa: E501
    if decaying_lr_type.lower() != "constant" and lr_start != lr_final:
        if decaying_lr_type.lower() == "exponential":
            exponential_decay_rate = np.float64(lr_final) / np.float64(lr_start)
            lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                lr_start,
                (steps - 1),
                np.float64(exponential_decay_rate),
            )
        else:
            lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
                lr_start,
                (steps - 1),
                lr_final,
                power=1.0,
            )

        return lr_scheduler
    else:
        return lambda step: lr_start  # Return a constant learning rate.
