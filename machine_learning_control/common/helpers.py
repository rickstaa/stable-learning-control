"""Contains several helper functions that are used across all the MLC modules.
"""
import itertools
import string
from collections.abc import Iterable

import numpy as np
import torch


def convert_to_tuple(input_var):
    """Converts input into a tuple.

    Args:
        input_arg (Union[int, float, list]): A input variable.

    Returns:
        tuple: A tuple.
    """

    return (
        input_var
        if isinstance(input_var, tuple)
        else tuple(input_var)
        if isinstance(input_var, list)
        else (input_var,)
    )


def flatten(items):
    """Flatten a list with any nested iterable.

    Args:
        items (list): A nested list.

    Returns:
        list: A flattened version of the list.
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def sum_tuples(*args):
    """Returns the elementwise sum of a several tuples/lists.

    Args:
        *args (Union[tuple, list]): Input arguments for which you want to calculate the
            elementwise sum.

    Returns:
        tuple: A tuple containing the elementwise sum of the input tuples/lists.
    """
    elem_sum = [sum(x) for x in zip(*args)]
    return elem_sum[0] if len(elem_sum) == 1 else tuple(elem_sum)


def combine_shapes(*args):
    """Combines multiple tuples/ints/floats into one tuple.

    Args:
        *args (Union[tuple,int,float]): Input arguments to combine

    Returns:
        Tuple: A tuple in which al the input arguments are combined.
    """
    return tuple(
        itertools.chain(
            *[[item] if isinstance(item, (int, float)) else list(item) for item in args]
        )
    )


def strict_dict_update(input_dict, update_obj):
    """Updates a dictionary with values supplied in another dictionary or list. If a
    list is supplied this list will be applied to all keys in the dictionary. This
    function performs strict update, meaning that it does not add new keys to the
    dictionary.

    Args:
        input_dict (dict): The input dictionary.
        update_dict (Union[dict, list]): Dictionary or list containing the update
            values.

    Returns:
        (tuple): tuple containing:

            - input_dict(:obj:`list`): The new updated dictionary.
            - ignored (:obj:`str`): The ignored keys.
    """
    ignored = []
    if isinstance(update_obj, dict):
        for key, val in update_obj.items():
            if key in input_dict.keys():
                input_dict[key] = val
            else:
                ignored.append(key)
    else:
        for key in input_dict:
            input_dict[key] = update_obj
    return input_dict, ignored


def valid_str(v):
    """Convert a value or values to a string which could go in a filepath.

    .. note::
        Partly based on `this gist`_.

        .. _`this gist`: https://gist.github.com/seanh/93666

    Args:
        v (list): List with values.
    """
    if hasattr(v, "__name__"):
        return valid_str(v.__name__)

    if isinstance(v, tuple) or isinstance(v, list):
        return "-".join([valid_str(x) for x in v])

    # Valid characters are '-', '_', and alphanumeric. Replace invalid chars
    # with '-'.
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = "".join(c if c in valid_chars else "-" for c in str_v)
    return str_v


def all_bools(vals):
    """Check if list contains only strings.

    Args:
        vals (list): List with values.

    Returns:
        bool: Boolean specifying the result.
    """
    return all([isinstance(v, bool) for v in vals])


def is_scalar(obj):  # noqa: C901
    """Recursive function that checks whether a input

    Args:
        obj (object): Object for which you want to check if it is a scalar.

    Returns:
        boole: Boolean specifying whether the object is a scalar.
    """
    if type(obj) in [int, float]:
        return True
    elif np.isscalar(obj):
        if type(obj) == np.str_:
            try:
                float(obj)
                return True
            except ValueError:
                return False
        else:
            return True
    elif type(obj) == np.ndarray:
        if obj.shape == (1,):
            return is_scalar(obj[0])
        else:
            return False
    elif type(obj) == torch.Tensor:
        if len(obj.shape) == 0:
            try:
                float(obj)
                return True
            except ValueError:
                return False
        elif len(obj.shape) <= 1:
            if obj.shape[0] <= 1:
                return is_scalar(obj[0])
            else:
                return False
        else:
            return False
    elif type(obj) == str:
        try:
            float(obj)
            return True
        except ValueError:
            return False
    else:
        return False
