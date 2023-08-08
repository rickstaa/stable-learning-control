"""A module that makes eval calls inside the utility scripts safer.

This is done by setting the globals to a dictionary containing only modules we want the
end user to access. This way, users can not use the eval method to (accidentally)
execute arbitrary code on the system through the CLI.
"""
# Import modules to which you want users to have access.
import torch  # noqa: F401

import stable_learning_control as stable_learning_control  # noqa: F401
from stable_learning_control.utils.import_utils import import_tf

tf = import_tf(frail=False)  # Suppress import warning.
tf_nn = import_tf(module_name="tensorflow.nn", frail=False)  # suppress import warning
torch_nn = torch.nn

AVAILABLE_GLOBAL_MODULES = {
    "torch": torch,
    "tensorflow": tf,
    "tf": tf,  # Create alias.
    "stable_learning_control": stable_learning_control,
    "slc": stable_learning_control,  # Create alias.
}


def safer_eval(*args, backend=None):
    """Function that executes the eval function with a safer set of globals.

    .. note::
        This is done by setting the globals to a dictionary containing only modules we
        want the end user to access. This way, users can not use the eval method to
        (accidentally) execute arbitrary code on the system through the CLI.

    Args:
        backend (str): The machine learning backend you want to use. Options are ``tf2``
            or ``torch``. Defaults to ``None`` meaning no backend is assumed and both
            backends are tried.

    Returns:
        args: The eval return values.
    """
    if backend is not None:
        return eval(
            *args,
            {
                **AVAILABLE_GLOBAL_MODULES,
                "tf": tf if backend.lower() == "tf2" else None,
                "torch": torch if backend.lower() == "torch" else None,
            },
            {
                "nn": tf_nn if backend.lower() == "tf2" else torch_nn
            },  # Add the `nn` shorthand based on the backend.
        )

    # If no backend is specified, try both backends.
    try:
        return eval(
            *args,
            AVAILABLE_GLOBAL_MODULES,
            {"nn": torch_nn},
        )
    except Exception:
        return eval(
            *args,
            AVAILABLE_GLOBAL_MODULES,
            {"nn": tf_nn},
        )
