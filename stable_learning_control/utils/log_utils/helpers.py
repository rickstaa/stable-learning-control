"""Several logging related helper functions."""

import os.path as osp
import time

from gymnasium.utils import colorize as gym_colorize

from stable_learning_control.user_config import (
    DEFAULT_DATA_DIR,
    DEFAULT_STD_OUT_TYPE,
    DEFAULT_WANDB_JOB_TYPE,
    DEFAULT_WANDB_PROJECT,
    FORCE_DATESTAMP,
)
from stable_learning_control.utils.mpi_utils.mpi_tools import proc_id

LOG_TYPES = {
    "info": {"color": "green", "bold": True, "highlight": False, "prefix": "INFO: "},
    "warning": {
        "color": "yellow",
        "bold": True,
        "highlight": False,
        "prefix": "WARNING: ",
    },
    "error": {"color": "red", "bold": True, "highlight": False, "prefix": "ERROR: "},
}


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.

    .. seealso::
        This function wraps the :meth:`gym.utils.colorize` function to make sure that it
        also works with empty color strings.

    Args:
        string (str): The string you want to colorize.
        color (str): The color you want to use.
        bold (bool, optional): Whether you want the text to be bold. Defaults to
            ``False``.
        highlight (bool, optional):  Whether you want to highlight the text. Defaults to
            ``False``.

    Returns:
        str: Colorized string.
    """
    if color:  # If not empty.
        return gym_colorize(string, color, bold, highlight)
    else:
        return string


def log_to_std_out(
    msg, color="", bold=False, highlight=False, type=None, *args, **kwargs
):
    """Print a colorized message to stdout.

    Args:
        msg (str): Message you want to log.
        color (str, optional): Color you want the message to have. Defaults to
            ``""``.
        bold (bool, optional): Whether you want the text to be bold text has to be
            bold.
        highlight (bool, optional):  Whether you want to highlight the text.
            Defaults to ``False``.
        type (str, optional): The log message type. Options are: ``info``, ``warning``
            and ``error``. Defaults to ``None``.
        *args: All args to pass to the print function.
        **kwargs: All kwargs to pass to the print function.
    """
    if proc_id() == 0:
        color = (
            LOG_TYPES[type.lower()]["color"]
            if (type is not None and type.lower() in LOG_TYPES.keys())
            else color
        )
        bold = (
            LOG_TYPES[type.lower()]["bold"]
            if (type is not None and type.lower() in LOG_TYPES.keys())
            else bold
        )
        highlight = (
            LOG_TYPES[type.lower()]["highlight"]
            if (type is not None and type.lower() in LOG_TYPES.keys())
            else highlight
        )
        prefix = (
            LOG_TYPES[type.lower()]["prefix"]
            if (type is not None and type.lower() in LOG_TYPES.keys())
            else ""
        )
        print(
            colorize((str(prefix) + str(msg)), color, bold=bold, highlight=highlight),
            *args,
            **kwargs,
        )


def setup_logger_kwargs(
    exp_name,
    seed=None,
    save_checkpoints=False,
    use_tensorboard=False,
    tb_log_freq="low",
    use_wandb=False,
    wandb_job_type=DEFAULT_WANDB_JOB_TYPE,
    wandb_project=DEFAULT_WANDB_PROJECT,
    wandb_group=None,
    wandb_run_name=None,
    quiet=False,
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
    ``stable_learning_control/user_config.py``.

    Args:
        exp_name (str): Name for experiment.
        seed (int, optional): Seed for random number generators used by experiment.
        save_checkpoints (bool, optional): Save checkpoints during training.
            Defaults to ``False``.
        use_tensorboard (bool, optional): Whether you want to use TensorBoard. Defaults
            to ``True``.
        tb_log_freq (str, optional): The TensorBoard log frequency. Options are ``low``
            (Recommended: logs at every epoch) and ``high`` (logs at every SGD update "
            batch). Defaults to ``low`` since this is less resource intensive.
        use_wandb (bool, optional): Whether you want to use Weights & Biases. Defaults
            to ``False``.
        wandb_job_type (str, optional): The Weights & Biases job type. Defaults to
            ``None``.
        wandb_project (str, optional): The name of the Weights & Biases project you
            want to log to. Defaults to ``None``.
        wandb_group (str, optional): The name of the Weights & Biases group you want to
            assign the run to. Defaults to ``None``.
        wandb_run_name (str, optional): The name of the Weights & Biases run. Defaults
            to ``None`` which means that the run name is automatically generated.
        quiet (bool, optional): Whether you want to suppress logging of the diagnostics
            to the stdout. Defaults to ``False``.
        verbose_fmt (str, optional): The format in which the diagnostics are
            displayed to the terminal. Options are ``table`` which supplies them as a
            table and ``line`` which prints them in one line. Defaults to ``line``.
        verbose_vars (list, optional): A list of variables you want to log to the
            stdout. By default all variables are logged.
        data_dir (str, optional): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in
            ``stable_learning_control/user_config.py``. Defaults to ``None``.
        datestamp (bool, optional): Whether to include a date and timestamp in the
            name of the save directory. Defaults to ``False``.

    Returns:
        dict: logger_kwargs
            A dict containing output_dir and exp_name.
    """
    # Datestamp forcing.
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path.
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
        use_wandb=use_wandb,
        wandb_job_type=wandb_job_type,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
        tb_log_freq=tb_log_freq,
        quiet=quiet,
        verbose_fmt=verbose_fmt,
        verbose_vars=verbose_vars,
    )
    return logger_kwargs


def dict_to_mdtable(d, key="Name", val="Value"):
    """Convert a dictionary to a markdown table.

    Args:
        d (dict): The dictionary you want to convert.
        key (str, optional): The name of the key column. Defaults to ``"Name"``.
        val (str, optional): The name of the value column. Defaults to ``"Value"``.

    Returns:
        str: The markdown table.
    """
    rows = [f"| {key} | {val} |"]
    rows += ["|--|--|"]
    rows += [f"| {k} | {v} |" for k, v in d.items()]
    return "  \n".join(rows)
