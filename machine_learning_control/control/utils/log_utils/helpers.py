"""Several logging related helper functions.
"""

import os.path as osp
import time
import joblib
from machine_learning_control.control.utils import import_tf
from machine_learning_control.user_config import (
    DEFAULT_DATA_DIR,
    DEFAULT_STD_OUT_TYPE,
    FORCE_DATESTAMP,
)

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


def setup_logger_kwargs(
    exp_name,
    seed=None,
    save_checkpoints=False,
    use_tensorboard=False,
    tb_log_freq="low",
    verbose=True,
    verbose_fmt=DEFAULT_STD_OUT_TYPE,
    verbose_vars=[],
    data_dir=None,
    datestamp=False,
):
    """Sets up the output_dir for a logger and returns a dict for logger kwargs.

    If no seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name

    If a seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name/exp_name_s[seed]

    If datestamp is true, amend to

    ::

        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in
    ``machine_learning_control/user_config.py``.

    Args:
        exp_name (str): Name for experiment.
        seed (int, optional): Seed for random number generators used by experiment.
        save_checkpoints (bool, optional): Save checkpoints during training.
            Defaults to ``False``.
        use_tensorboard (bool, optional): Whether you want to use tensorboard. Defaults
            to True.
        tb_log_freq (str, optional): The tensorboard log frequency. Options are 'low'
            (Recommended: logs at every epoch) and 'high' (logs at every SGD update "
            batch). Defaults to 'low' since this is less resource intensive.
        verbose (bool, optional): Whether you want to log to the std_out. Defaults
            to ``True``.
        verbose_fmt (str, optional): The format in which the statistics are
            displayed to the terminal. Options are "table" which supplies them as a
            table and "line" which prints them in one line. Defaults to "line".
        verbose_vars (list, optional): A list of variables you want to log to the
            std_out. By default all variables are logged.
        data_dir (str, optional): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in
            ``machine_learning_control/user_config.py``. Defaults to None.
        datestamp (bool, optional): Whether to include a date and timestamp in the
            name of the save directory. Defaults to ``False``.

    Returns:
        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ""
    relpath = "".join([ymd_time, exp_name])

    # Make a seed-specific subfolder in the experiment directory.
    if seed is not None:
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = "".join([hms_time, "-", exp_name, "_s", str(seed)])
        else:
            subfolder = "".join([exp_name, "_s", str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(
        output_dir=osp.join(data_dir, relpath),
        exp_name=exp_name,
        save_checkpoints=save_checkpoints,
        use_tensorboard=use_tensorboard,
        tb_log_freq=tb_log_freq,
        verbose=verbose,
        verbose_fmt=verbose_fmt,
        verbose_vars=verbose_vars,
    )
    return logger_kwargs
