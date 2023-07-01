"""Contains functions that can be used to import modules and classes while repressing
the :class`ImportError` when a module is not found.
"""
import importlib
import sys

from stable_learning_control.utils.log_utils.helpers import friendly_err


def lazy_importer(module_name, class_name=None, frail=False):
    """A simple lazy importer tries to import a module/class but is too lazy to complain
    when it is not found. This function can be used to (lazily) load modules and
    classes, meaning only loading them if available.

    Args:
        module_name (str): The python module you want to import (eg. tensorflow.nn).
        class_name (str): The python class you want to import from a given python
            module by default ``None``.
        frail (bool, optional): Throw ImportError when module can not be imported.
            Defaults to ``False``.

    Raises:
        ImportError: A custom import error that is raised when the module is not
            installed and ``frail`` is ``True``.

    Returns:
        module: The imported (class) module. Returns ``None`` if the module is not
            found.
    """
    # Return module if it is already imported.
    if module_name in sys.modules:
        if class_name is None:
            return sys.modules[module_name]
        else:
            return getattr(sys.modules[module_name], class_name)

    # Try to import module or class.
    try:
        if class_name is None:
            return importlib.import_module(module_name)
        else:
            return getattr(importlib.import_module(module_name), class_name)
    except ImportError:
        if frail:
            import_msg = f"No module named '{module_name}'."
            if "tensorflow" in module_name:
                import_msg += " Did you run the `pip install .[tf2]` command?"
            raise ImportError(friendly_err(import_msg))


def tf_installed():
    """Checks if TensorFlow is installed.

    Returns:
        bool: Returns ``True`` if TensorFlow is installed.
    """
    if "tensorflow" in sys.modules:
        return True
    elif importlib.util.find_spec("tensorflow") is not None:
        return True
    else:
        return False


def import_tf(module_name=None, class_name=None, frail=True):
    """Tries to import TensorFlow and throws custom warning if TensorFlow is not
    installed.

    Args:
        module_name (str, optional): The tensorflow python module you want to import
            (eg. tensorflow.nn). By default ``None``, meaning the TensorFlow package is
            imported.
        class_name (str): The python class you want to import from the tensorflow python
            module (eg. Adam from :mod:`tensorflow.keras.optimizers`). By default
            ``None``.
        frail (bool, optional): Throw :class:`ImportError` when TensorFlow can not be
            imported. Defaults to ``True``.

    Raises:
        ImportError: A custom import error if TensorFlow is not installed.

    Returns:
        module: The imported (class) module.
    """
    module_name = "tensorflow" if module_name is None else module_name
    if "tensorflow" not in module_name:
        raise ValueError(
            f"Expected module name to contain 'tensorflow' but got '{module_name}'."
        )

    return lazy_importer(module_name, class_name, frail)
