"""Module used for serializing json objects.

.. note::
    This module was based on
    `spinningup repository <https://github.com/openai/spinningup/blob/master/spinup/utils/serialization_utils.py>`_.
"""  # noqa

import json
import os.path as osp


def convert_json(obj):
    """Convert obj to a version which can be serialised with JSON.

    Args:
        obj (object): Object which you want to convert to json.

    Returns:
        object: Serialised json object.
    """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    """Check if object can be serialised with JSON.

    Args:
        v (object): object you want to check.

    Returns:
        bool: Boolean specifying whether the object can be serialised by json.
    """
    try:
        json.dumps(v)
        return True
    except Exception:
        return False


def save_to_json(input_object, output_filename, output_path):
    """Save python object to Json file. This method will serialize the object to
    JSON, while handling anything which can't be serialized in a graceful way
    (writing as informative a string as possible).

    Args:
        input_object (object): The input object you want to save.
        output_filename (str): The output filename.
        output_path (str): The output path.
    """
    input_object_json = convert_json(input_object)
    output = json.dumps(
        input_object_json, separators=(",", ":\t"), indent=4, sort_keys=True
    )
    output_filename = (
        output_filename + ".json"
        if osp.splitext(output_filename)[1] != ".json"
        else output_filename
    )
    with open(osp.join(output_path, output_filename), "w") as out:
        out.write(output)


def load_from_json(path):
    """Load data from json file.

    Args:
        path (str): The path of the json file you want to load.
    Returns:
        (object): The Json load object.
    """
    path = path + ".json" if osp.splitext(path)[1] != ".json" else path
    content = open(path)
    return json.load(content)
