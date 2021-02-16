"""Module used to make the eval calls inside the run.py script more save. For more
info see https://python-reference.readthedocs.io/en/latest/docs/functions/eval.html.
The `:meth:save_eval` method can only evaluate expresions that use modules which are
imported in this file.
"""

# Import modules to which you want users to have access
import machine_learning_control.simzoo.simzoo  # noqa: F401
import torch  # noqa: F401
from machine_learning_control.control.utils.import_tf import import_tf

tf = import_tf(frail=False)
tensorflow = tf


def safe_eval(*args, backend=None):
    """Function used to make sure that only the in this module imported packages can be
    used inside the eval method. This was done in a attempt to make eval a little bit
    more save.

    Args:
        backend (str): The machine_learning backend you want to use. By default
            ``None``, meaning no backend is assumed.

    .. note::
        Please open a issue on https://github.com/rickstaa/machine-learning-control if
        something is not working.

    Returns:
        args: The eval return values.
    """
    # Import the nn module based on the backend type
    # NOTE: This was done to enable users to specify `nn.relu` instead of
    # `torch.nn.ReLu`.
    if backend is not None and backend.lower() in ["torch", "pytorch"]:
        from torch import nn
    elif backend is not None and backend.lower() in ["tensorflow", "tf"]:
        nn = import_tf(module="tensorflow.nn")

    eval_safe_globals = {k: v for k, v in globals().items() if k not in {"os", "sys"}}
    eval_safe_globals["__builtins__"] = {}
    return eval(
        *args, {**globals(), "os": None, "sys": None, "__import__": None}, {"nn": nn}
    )
