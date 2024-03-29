"""Contains several Pytorch helper functions."""

import numpy as np
import torch
import torch.nn as nn

from stable_learning_control.algos.common.helpers import get_activation_function
from stable_learning_control.utils.log_utils.helpers import log_to_std_out


def retrieve_device(device_type="cpu"):
    """Retrieves the available computational device given a device type.

    Args:
        device_type (str): The device type (options: ``cpu``, ``gpu``, ``gpu:0``,
            ``gpu:1``, etc.). Defaults to ``cpu``.

    Returns:
        :obj:`torch.device`: The Pytorch device object.
    """
    device_type = device_type.lower()
    if "gpu" in device_type:
        if not torch.cuda.is_available():
            log_to_std_out(
                "GPU computing was enabled but the GPU can not be reached. "
                "Reverting back to using CPU.",
                "yellow",
                type="warning",
            )
            device = torch.device("cpu")
        else:
            device_id = int(device_type.split(":")[1]) if ":" in device_type else 0
            if device_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{device_id}")
            else:
                log_to_std_out(
                    f"GPU with ID {device_id} not found. Reverting back to the first "
                    "available GPU.",
                    "yellow",
                    type="warning",
                )
                device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    log_to_std_out(
        f"Torch is using the {device}.",
        type="info",
    )
    return device


def mlp(sizes, activation, output_activation=nn.Identity):
    """Create a multi-layered perceptron using pytorch.

    Args:
        sizes (list): The size of each of the layers.
        activation (union[:obj:`torch.nn.modules.activation`, :obj:`str`]): The
            activation function used for the hidden layers.
        output_activation (union[:obj:`torch.nn.modules.activation`, :obj:`str`], optional):
            The activation function used for the output layers. Defaults to
            :class:`torch.nn.Identity`.

    Returns:
        torch.nn.Sequential: The multi-layered perceptron.
    """  # noqa: E501
    # Try to retrieve the activation function if a string was supplied.
    if isinstance(activation, str):
        activation = get_activation_function(activation, backend="torch")
    if isinstance(output_activation, str):
        output_activation = get_activation_function(output_activation, backend="torch")

    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    """Returns the total number of parameters of a pytorch module.

    Args:
        module (torch.nn.Module): The module.

    Returns:
        :obj:`numpy.int64`: The total number of parameters inside the module.
    """
    return sum([np.prod(p.shape) for p in module.parameters()])


def compare_models(model_1, model_2):
    """Compares two models to see if the weights are equal.

    Args:
        model_1 (torch.nn.Module): The first Pytorch model.
        model_2 (torch.nn.Module): The second Pytorch model.

    Raises:
        Exception: Raises Key error if the graph of the two models is different.

    Returns:
        bool: Bool specifying whether the weights of two models are equal.
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismatch found at", key_item_1[0])
            else:
                raise KeyError(
                    "Model weights could not be compared between the two models as "
                    f"the two models appear to be different. Parameter {key_item_1[0]} "
                    f"which is found in model 1, does not exist in the model 2."
                )

    if models_differ == 0:
        print("Models match perfectly! :)")
        return True
    else:
        return False


def rescale(data, min_bound, max_bound):
    """Rescale normalized data (i.e. between ``-1`` and ``1``) to a desired range.

    Args:
        data (Union[torch.Tensor, numpy.ndarray, list]): Normalized input data.
        min_bound (Union[numpy.ndarray, list]): Array containing the minimum value of
            the desired range.
        max_bound (Union[numpy.ndarray, list]): Array containing the maximum value of
            the desired range.

    Returns:
        Union[Torch.Tensor, numpy.ndarray]: Array which has it values scaled between
            the min and max boundaries.
    """
    was_numpy = isinstance(data, np.ndarray)
    data = torch.as_tensor(data)
    min_bound = torch.as_tensor(min_bound, device=data.device)
    max_bound = torch.as_tensor(max_bound, device=data.device)

    # Return rescaled data in the same format as the input data.
    data_rescaled = (data + 1.0) * (max_bound - min_bound) / 2 + min_bound
    return data_rescaled.cpu().numpy() if was_numpy else data_rescaled


def np_to_torch(input_object, dtype=None, device=None):
    """Converts all numpy arrays in a python object to Torch Tensors.

    Args:
        input_item (obj): The python object.
        dtype (type, optional): The type you want to use for storing the data in the
            tensor. Defaults to ``None`` (i.e. torch default will be used).
        device (str, optional): The computational device on which the tensors should be
            stored. (options: ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1``, etc.). Defaults
            to ``None`` (i.e. torch default device will be used).

    Returns:
        object: The output python object in which numpy arrays have been converted to
            torch tensors.
    """
    if isinstance(input_object, dict):
        return {
            k: np_to_torch(v, dtype=dtype, device=device)
            for k, v in input_object.items()
        }
    elif isinstance(input_object, list):
        return [np_to_torch(v, dtype=dtype, device=device) for v in input_object]
    elif isinstance(input_object, np.ndarray):
        try:
            return torch.as_tensor(input_object, dtype=dtype, device=device)
        except TypeError:
            return input_object
    else:
        return input_object
