"""Several Pytorch helper functions.
"""

import torch
import torch.nn as nn
from machine_learning_control.control.utils.logx import colorize


def retrieve_device(device_type="cpu"):
    """Retrieves the available computational device given a device type.

    Args:
        device_type (string): The device type (options: ``cpu`` and
            ``gpu``). Defaults to ``cpu``.
    Returns:
        torch.device: The Pytorch device object.
    """
    device_type = (
        "cpu" if device_type.lower() not in ["gpu", "cpu"] else device_type.lower()
    )
    if torch.cuda.is_available() and device_type == "gpu":
        device = torch.device("cuda")
    elif not torch.cuda.is_available() and device_type == "gpu":
        print(
            colorize(
                "WARN: GPU computing was enabled but the GPU can not be reached. "
                "Reverting back to using CPU.",
                "yellow",
                bold=True,
            ),
        )
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(colorize(f"INFO: Torch is using the {device_type}.", "cyan", bold=True))
    return device


def parse_network_structure(hidden_sizes, activation, output_activation):
    """Function that parses the network related input arguments to split them into
    'actor' and 'critic' related arguments.

    Args:
        hidden_sizes (Union[tuple, list]): [description]
        activation (object): [description]
        output_activation ([type]): [description]

    Returns:
        tuple: tuple containing:

            hidden_sizes_parsed(dict): Episode retentions.
            activation_parsed(dict): Episode lengths
            output_activation_parsed(dict): Episode lengths
    """
    hidden_sizes_parsed, activation_parsed, output_activation_parsed = {}, {}, {}
    if isinstance(hidden_sizes, dict):
        hidden_sizes_parsed["actor"] = (
            hidden_sizes["actor"] if "actor" in hidden_sizes.keys() else (256, 256)
        )
        hidden_sizes_parsed["critic"] = (
            hidden_sizes["critic"] if "critic" in hidden_sizes.keys() else (256, 256)
        )
    else:
        hidden_sizes_parsed["actor"] = hidden_sizes
        hidden_sizes_parsed["critic"] = hidden_sizes
    if isinstance(activation, dict):
        activation_parsed["actor"] = (
            hidden_sizes["actor"] if "actor" in activation.keys() else nn.ReLU
        )
        activation_parsed["critic"] = (
            activation["critic"] if "critic" in activation.keys() else nn.ReLU
        )
    else:
        activation_parsed["actor"] = activation
        activation_parsed["critic"] = activation
    if isinstance(output_activation, dict):
        output_activation_parsed["actor"] = (
            output_activation["actor"]
            if "actor" in output_activation.keys()
            else nn.ReLU
        )
        output_activation_parsed["critic"] = (
            output_activation["critic"]
            if "critic" in output_activation.keys()
            else nn.Identity
        )
    else:
        output_activation_parsed["actor"] = output_activation
        output_activation_parsed["critic"] = output_activation

    return (
        hidden_sizes_parsed,
        activation_parsed,
        output_activation_parsed,
    )
