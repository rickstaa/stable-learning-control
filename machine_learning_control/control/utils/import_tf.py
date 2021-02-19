"""Small wrapper function tries to install tensorflow and throws a custom warning if it
is not installed.
"""

import importlib
import sys

TF_WARN_MESSAGE = (
    "No module named '{}'. Did you run the `pip install .[tf]` " "command?"
)


def import_tf(module_name=None, class_name=None, frail=True, dry_run=False):
    """Tries to import tensorflow and throws custom warning if tensorflow is not
    installed.

    Args:
        module_name (str): The python module you want to import (eg. tensorflow.nn). By
            default ``None``, meaning the Tensorflow package is imported.
        class_name (str): The python class you want to import (eg. Adam
            from :module:`tensorflow.keras.optimizers`). By default ``None``.
        frail (bool): Throw ImportError when tensorflow can not be imported.
        dry_run (bool): Do not actually import tensorflow if available, by Default
            ``False``.

    Raises:
        ImportError: A custom import error if tensorflow is not installed.

    Returns:
        union[tf, bool]: Tensorflow module or class if ``dry_run`` is set to ``False``.
            Returns a success bool if ``dry_run`` is set to ``True``.
    """
    module_name = "tensorflow" if module_name is None else module_name
    try:
        if module_name in sys.modules:
            if not dry_run:
                if class_name is None:
                    return sys.modules[module_name]
                else:
                    return getattr(sys.modules[module_name], class_name)
            else:
                return True
        elif importlib.util.find_spec("tensorflow") is not None:
            if not dry_run:
                if class_name is None:
                    return importlib.import_module(module_name)
                else:
                    return getattr(importlib.import_module(module_name), class_name)
            else:
                return True
        else:
            if frail:
                raise ImportError(TF_WARN_MESSAGE.format(module_name))
            return False
    except (ImportError, KeyError, AttributeError) as e:
        raise ImportError(TF_WARN_MESSAGE.format(module_name)) from e
