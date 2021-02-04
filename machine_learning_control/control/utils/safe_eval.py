# flake8: noqa E401
"""Module used to make the eval calls inside the run.py script more save. For more
info see https://python-reference.readthedocs.io/en/latest/docs/functions/eval.html.
The `:meth:save_eval` method can only evaluate expresions that use modules which are
imported in this file.
"""

# Import modules to which you want users to have access
import machine_learning_control.control
import machine_learning_control.simzoo.simzoo
import torch
import torch.nn as nn
from machine_learning_control.control.utils.import_tf import import_tf

tf = import_tf(frail=False)


def safe_eval(*args):
    """Function used to make sure that only the in this module imported packages can be
    used inside the eval method. This was done in a attempt to make eval a little bit
    more save.

    .. note::
        Please open a issue on https://github.com/rickstaa/machine-learning-control if
        something is not working.

    Returns:
        args: The eval return values.
    """
    eval_safe_globals = {k: v for k, v in globals().items() if k not in {"os", "sys"}}
    eval_safe_globals["__builtins__"] = {}
    return eval(*args, {**globals(), "os": None, "sys": None, "__import__": None})
