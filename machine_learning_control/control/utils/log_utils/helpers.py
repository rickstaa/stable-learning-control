"""Several logging related helper functions.
"""

import os.path as osp

import joblib
from machine_learning_control.control.utils import import_tf

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def friendly_err(err_msg):
    """Add whitespace line to error message to make it more readable.

    Args:
        err_msg (str): Error message.

    Returns:
        str: Error message with extra whitespace line.
    """
    return "\n\n" + err_msg + "\n\n"


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.

    This function was originally written by John Schulman.

    Args:
        string (str): The string you want to colorize.
        color (str): The color you want to use.
        bold (bool, optional): Whether you want the text to be bold text has to be bold.
        highlight (bool, optional):  Whether you want to highlight the text. Defaults to
            False.

    Returns:
        str: Colorized string.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


def restore_tf_graph(sess, fpath):
    """Loads graphs saved by Logger.

    Will output a dictionary whose keys and values are from the 'inputs'
    and 'outputs' dict you specified with logger.setup_tf_saver().

    Args:
        sess: A Tensorflow session.
        fpath: Filepath to save directory.

    Returns:
        A dictionary mapping from keys to tensors in the computation graph
        loaded from ``fpath``.

    Raises:
        ImportError: Raised when this method is called while tensorflow is not
        installed.
    """
    tf = import_tf()  # Import tf if installed otherwise throw warning

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], fpath)
    model_info = joblib.load(osp.join(fpath, "model_info.pkl"))
    graph = tf.get_default_graph()
    model = dict()
    model.update(
        {k: graph.get_tensor_by_name(v) for k, v in model_info["inputs"].items()}
    )
    model.update(
        {k: graph.get_tensor_by_name(v) for k, v in model_info["outputs"].items()}
    )
    return model
