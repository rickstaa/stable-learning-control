"""Small wrapper function tries to install tensorflow and throws a custom warning if it
is not installed.
"""

import importlib
import sys

TF_WARN_MESSAGE = (
    "No module named '{}'. Did you run the `pip install .[tf]` " "command?"
)


def import_tf(module=None, frail=True, dry_run=False):
    """Tries to import tensorflow and throws custom warning if tensorflow is not
    installed.

    Args:
        module (str): The python module you want to import (eg. tensorflow.nn). By
            default ``None``, meaning the Tensorflow package is imported.
        frail (bool): Throw ImportError when tensorflow can not be imported.
        dry_run (bool): Do not actually tag next version if it's true, by Default False/

    Raises:
        ImportError: A custom import error if tensorflow is not installed.

    Returns:
        union[tf, None]: Tensorflow module or None if tensorflow could not be loaded.
    """
    module = "tensorflow" if module is None else module
    if module in sys.modules:
        if not dry_run:
            return sys.modules[module]
    elif importlib.util.find_spec("tensorflow") is not None:
        if not dry_run:
            return importlib.import_module(module)
    else:
        if frail:
            raise ImportError(TF_WARN_MESSAGE.format(module))

    return None
