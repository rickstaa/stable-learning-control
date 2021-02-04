"""Small wrapper function tries to install tensorflow and throws a custom warning if it
is not installed.
"""

import importlib
import sys

TF_WARN_MESSAGE = (
    "No module named 'tensorflow'. Did you run the `pip install .[tf]` " "command?"
)


def import_tf(frail=True):
    """Tries to import tensorflow and throws custom warning if tensorflow is not
    installed.

    Args:
        frail: Throw ImportError when tensorflow can not be imported.

    Raises:
        ImportError: A custom import error if tensorflow is not installed.

    Returns:
        union[tf, None]: Tensorflow module or None if tensorflow could not be loaded.
    """

    if "tensorflow" in sys.modules:
        return sys.modules["tensorflow"]
    elif importlib.util.find_spec("tensorflow") is not None:
        return importlib.import_module("tensorflow")
    else:
        if frail:
            raise ImportError(TF_WARN_MESSAGE)
        return None
