"""Module that can be used to make the eval calls inside the utility scripts more safe.

.. important::
    The :meth:`safer_eval` method can only evaluate expression that use modules which
    are imported in this file.
"""
# Import modules to which you want users to have access.
import torch  # noqa: F401

import stable_learning_control as stable_learning_control  # noqa: F401
import stable_learning_control as slc  # noqa: F401
from stable_learning_control.utils.import_utils import import_tf

tf = import_tf(frail=False)
tensorflow = tf


def safer_eval(*args, backend=None):
    """Function used to make sure that only the in this module imported packages can be
    used inside the eval method. This was done in a attempt to make eval a little bit
    more save.

    Args:
        backend (str): The machine learning backend you want to use. Options are ``tf2``
            or ``torch``. By default ``None``, meaning no backend is assumed.
    Returns:
        args: The eval return values.
    """
    # Import the nn module based on the backend type.
    # NOTE: This was done to enable users to specify `nn.relu` instead of
    # `torch.nn.ReLu`.
    if backend is not None and backend.lower() == "torch":
        from torch import nn
    elif backend is not None and backend.lower() == "tf2":
        nn = import_tf(module_name="tensorflow.nn")
    else:
        nn = None

    eval_safe_globals = {k: v for k, v in globals().items() if k not in {"os", "sys"}}
    eval_safe_globals["__builtins__"] = {}
    return eval(
        *args, {**globals(), "os": None, "sys": None, "__import__": None}, {"nn": nn}
    )
