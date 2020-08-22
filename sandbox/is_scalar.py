import torch
import numpy as np


def is_scalar(obj):
    """Recursive function that checks whether a input

    Args:
        obj (object): Object for which you want to check if it is a scalar.

    Returns:
        boole: Boolean specifying whether the object is a scalar.
    """

    # Check if obj is scalar
    if type(obj) in [int, float]:
        return True
    elif type(obj) == np.ndarray:
        if obj.size > 1:
            return is_scalar(obj)
        else:
            return False
    elif type(obj) == torch.Tensor:
        if obj.size > 1:
            return is_scalar(obj)
        else:
            return False
    elif type(obj) == str:
        return obj.isnumeric()
    else:
        return False
