"""Contains functions that can be used to import modules and classes while repressing
the ImportError when a module is not found.
"""
import importlib
import sys

IMPORT_WARNING = "No module named '{}'. Did you run the `pip install .{}` command?"


def lazy_importer(module_name, class_name=None, frail=False, dry_run=False):
    """A simple lazy importer tries to import a module but is too lazy to complain when
    a module is not installed. This function can be used to (lazily) load modules
    meaning only load modules if they are available.

    Args:
        module_name (str): The python module you want to import
            (eg. tensorflow.nn).
        class_name (str): The python class you want to import. By default ``None``.
        frail (bool, optional): Throw ImportError when module can not be imported.
            Defaults to ``False``.
        dry_run (bool, optional): Do not actually import TensorFlow if available.
            Defaults to ``False``.

    Raises:
        ImportError: Raised when the module is not installed.

    Returns:
        Union[:obj:`tf`, :obj:`bool`]:
            - The imported module or class if ``dry_run`` is set to ``False``.
            - Returns a success bool if ``dry_run`` is set to ``True``.
    """
    try:
        if module_name in sys.modules:
            if not dry_run:
                if class_name is None:
                    return sys.modules[module_name]
                else:
                    return getattr(sys.modules[module_name], class_name)
            else:
                return True
        elif importlib.util.find_spec(module_name) is not None:
            if not dry_run:
                if class_name is None:
                    return importlib.import_module(module_name)
                else:
                    return getattr(importlib.import_module(module_name), class_name)
            else:
                return True
        else:
            if frail:
                raise ImportError(
                    IMPORT_WARNING.format(
                        module_name,
                        "[tf2]"
                        if module_name.lower() == "tensorflow"
                        else ("[tuning]" if module_name == "ray" else ""),
                    )
                )
            return False
    except (ImportError, KeyError, AttributeError) as e:
        if ImportError:
            if not frail:
                return False
        raise ImportError(
            IMPORT_WARNING.format(
                module_name,
                "[tf2]"
                if module_name.lower() == "tensorflow"
                else ("[tuning]" if module_name == "ray" else ""),
            )
        ) from e


def import_tf(module_name=None, class_name=None, frail=True, dry_run=False):
    """Tries to import TensorFlow and throws custom warning if TensorFlow is not
    installed.

    Args:
        module_name (str, optional): The python module you want to import
            (eg. tensorflow.nn). By default ``None``, meaning the TensorFlow package is
            imported.
        class_name (str): The python class you want to import (eg. Adam
            from :mod:`tensorflow.keras.optimizers`). By default ``None``.
        frail (bool, optional): Throw ImportError when TensorFlow can not be imported.
            Defaults to ``true``.
        dry_run (bool, optional): Do not actually import TensorFlow if available.
            Defaults to ``False``.

    Raises:
        ImportError: A custom import error if TensorFlow is not installed.

    Returns:
        Union[:obj:`tf`, :obj:`bool`]:
            - TensorFlow module or class if ``dry_run`` is set to ``False``.
            - Returns a success bool if ``dry_run`` is set to ``True``.
    """
    module_name = "tensorflow" if module_name is None else module_name
    return lazy_importer(module_name, class_name, frail, dry_run)
