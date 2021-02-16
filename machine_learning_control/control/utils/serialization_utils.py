"""Module used for serializing json objects.

This module was cloned from the
`spinningup repository <https://github.com/openai/spinningup/blob/master/spinup/utils/serialization_utils.py>`_.
"""  # noqa

import json


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
