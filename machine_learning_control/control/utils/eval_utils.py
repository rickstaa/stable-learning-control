"""File containing some usefull functions used during the robustness evaluation."""

from collections.abc import Iterable


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
