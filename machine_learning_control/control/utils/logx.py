"""Module containing some simple logging functionality.

This module extends the logx module of
`the SpinningUp repository <https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py>`_
so that besides logging to tab-separated-values file
(path/to/output_directory/progress.txt)
it also logs the data to a tensorboard file.
"""  # noqa

import atexit
import importlib
import json
import os
import os.path as osp
import shutil
import sys
import time
import warnings

import joblib
import numpy as np
import torch
from machine_learning_control.control.utils.helpers import is_scalar
from machine_learning_control.control.utils.mpi_tools import (
    mpi_statistics_scalar,
    proc_id,
)
from machine_learning_control.control.utils.serialization_utils import convert_json
from torch.utils.tensorboard import SummaryWriter

# Import tensorflow is it is installed
if "tensorflow" in sys.modules:
    TF_AVAILABLE = True
elif importlib.util.find_spec("tensorflow") is not None:
    tf = importlib.import_module("tensorflow")
    TF_AVAILABLE = True
else:
    TF_AVAILABLE = False


# IMPORT LOGGER FORMAT
# FIXME: Replace with input argument
from machine_learning_control.user_config import LOG_FMT, LOG_IGNORE

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

# TODO: Remove tf dependency if tf is not installed


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

    if not TF_AVAILABLE:
        print(
            colorize(
                "Warning: restore_tf_graph method could not be used as tensorflow is "
                "not installed in the current environment. Please install tensorflow "
                "if you want to use tensorflow related logging methods.",
                color="yellow",
                bold=True,
            )
        )
        raise ImportError("No module named 'tensorflow'")

    # Load tensorflow graph
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


class Logger:
    """A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(
        self,
        output_dir=None,
        output_fname="progress.txt",
        exp_name=None,
        log_ignore=LOG_IGNORE,
        log_fmt=LOG_FMT,
        use_tensorboard=False,
        save_checkpoints=False,
    ):
        """Initialise a Logger.

        Args:
            output_dir (string, optional): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.
            output_fname (string, optional): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.
            exp_name (string.optional): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
            log_fmt (string): The format in which the statistics are displayed to the
                terminal. Options are "tab" which supplies them as a table and "line"
                which prints them in one line. Defaults to "tab".
            save_checkpoints (bool, optional): Save checkpoints during training.
                Defaults to False.
            log_ignore (string): Dictionary containing the keys for which you don't want
                the statistics to be printed to the terminal.
            use_tensorboard (bool): Specifies whether you want also log to tensorboard.

        Attributes:
            tb_writer (torch.utils.tensorboard.writer.SummaryWriter): A tensorboard
                writer. Is only created when the :py:attr:`.use_tensorboard` variable
                is set to `True`, if not it returns `None`.
            output_dir (str): The directory in which the log data and models are saved.
            output_file (str): The name of the file in which the progress data is saved.
            exp_name (str): Experiment name.
        """
        if proc_id() == 0:

            # Create log directory
            self.output_dir = output_dir or "/tmp/experiments/%i" % int(time.time())
            if osp.exists(self.output_dir):
                print(
                    colorize(
                        (
                            "WARN: Log dir %s already exists! Storing info there "
                            "anyway." % self.output_dir
                        ),
                        "yellow",
                        bold=True,
                    )
                )
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), "w")
            atexit.register(self.output_file.close)
            print(
                colorize(
                    "Logging data to %s" % self.output_file.name, "green", bold=True
                )
            )
            self.log_fmt = log_fmt
            self.log_ignore = log_ignore
        else:
            self.output_dir = None
            self.output_file = None
            self.log_fmt = None
            self.log_ignore = None
        self.exp_name = exp_name
        self._first_row = True
        self._log_headers = []
        self._log_current_row = {}
        self._save_checkpoints = save_checkpoints
        self._checkpoint = 0

        self.tb_writer = None
        self._use_tensorboard = use_tensorboard
        self._tabular_to_tb_dict = (
            dict()
        )  # Stores whether tabular is logged to tensorboard when dump_tabular is called
        self._step_count_dict = (
            dict()
        )  # Used for keeping count of the current global step

    def log(self, msg, color="green"):
        """Print a colorized message to stdout.

        Args:
            msg (string): Message you want to log.
            color (str, optional): Color you want the message to have. Defaults to
                "green".
        """
        if proc_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(
        self,
        key,
        val,
        tb_write=False,
        tb_prefix=None,
        tb_alias=None,
    ):
        """Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file, tensorboard and
        stdout (otherwise they will not get saved anywhere).

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            tb_write (bool, optional): Boolean specifying whether you also want to write
                the value to the tensorboard logfile. Defaults to False.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty dict(). If not supplied the variable name is
                used.
        """
        if self._first_row:
            self._log_headers.append(key)
        else:
            # DEBUG: CHANGED THIS!
            pass
            # assert key in self._log_headers, (
            #     "Trying to introduce a new key %s that you didn't include in the "
            #     "first iteration" % key
            # )
        assert key not in self._log_current_row, (
            "You already set %s this iteration. Maybe you forgot to call dump_tabular()"
            % key
        )

        self._log_current_row[key] = val
        self._tabular_to_tb_dict[key] = {
            "tb_write": tb_write,
            "tb_prefix": tb_prefix,
            "tb_alias": tb_alias,
        }

    def save_config(self, config):
        """Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialise the config to JSON, while
        handling anything which can't be serialised in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json["exp_name"] = self.exp_name
        if proc_id() == 0:
            output = json.dumps(
                config_json, separators=(",", ":\t"), indent=4, sort_keys=True
            )
            print(colorize("Saving config:\n", color="cyan", bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), "w") as out:
                out.write(output)

    def save_state(self, state_dict, itr=None, epoch=None):
        """Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you
        previously set up saving for with ``setup_tf_saver``.

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.
            itr (int/None): Current iteration of training. Defaults to None
            epoch (int/None): Current epoch of the SGD. Defaults to None
        """
        if proc_id() == 0:
            fname = "vars.pkl" if itr is None else "vars%d.pkl" % itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except:
                self.log("Warning: could not pickle state_dict.", color="red")
            if hasattr(self, "tf_saver_elements"):
                self._tf_simple_save(itr)
            if hasattr(self, "pytorch_saver_elements"):
                self._pytorch_simple_save(itr, epoch)

    def setup_tf_saver(self, sess, inputs, outputs):
        """Set up easy model saving for tensorflow.

        Call once, after defining your computation graph but before training.

        Args:
            sess: The Tensorflow session in which you train your computation
                graph.
            inputs (dict): A dictionary that maps from keys of your choice
                to the tensorflow placeholders that serve as inputs to the
                computation graph. Make sure that *all* of the placeholders
                needed for your outputs are included!
            outputs (dict): A dictionary that maps from keys of your choice
                to the outputs from your computation graph.
        """
        if not TF_AVAILABLE:
            print(
                colorize(
                    "Warning: The Logger.setup_tf_saver could not be set up as "
                    "tensorflow is not installed in the current environment. Please "
                    "install tensorflow if you want to use tensorflow related logging "
                    "methods.",
                    color="yellow",
                    bold=True,
                )
            )
        else:
            self.tf_saver_elements = dict(session=sess, inputs=inputs, outputs=outputs)
            self.tf_saver_info = {
                "inputs": {k: v.name for k, v in inputs.items()},
                "outputs": {k: v.name for k, v in outputs.items()},
            }

    def _tf_simple_save(self, itr=None):
        """Saves tf model.

        Uses simple_save to save a trained model, plus info to make it easy
        to associated tensors to variables after restore.

        Args:
            itr (int/None): Current iteration of training. Defaults to None.
        """
        if proc_id() == 0:
            assert hasattr(
                self, "tf_saver_elements"
            ), "First have to setup saving with self.setup_tf_saver"
            fpath = "tf1_save" + ("%d" % itr if itr is not None else "")
            fpath = osp.join(self.output_dir, fpath)
            if osp.exists(fpath):
                # simple_save refuses to be useful if fpath already exists,
                # so just delete fpath if it's there.
                shutil.rmtree(fpath)
            tf.saved_model.simple_save(export_dir=fpath, **self.tf_saver_elements)
            joblib.dump(self.tf_saver_info, osp.join(fpath, "model_info.pkl"))

    def setup_pytorch_saver(self, what_to_save):
        """Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None, epoch=None):
        """Saves the PyTorch model (or models).

        Args:
            itr(int/None): Current iteration of training. Defaults to None
        """
        if proc_id() == 0:
            assert hasattr(
                self, "pytorch_saver_elements"
            ), "First have to setup saving with self.setup_pytorch_saver"

            # Create filename
            fpath = "pyt_save"
            fpath = osp.join(self.output_dir, fpath)
            fname = "model" + ("%d" % itr if itr is not None else "") + ".pt"
            fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)

            # Create Checkpoints Name
            if self._save_checkpoints:
                epoch_name = (
                    "epoch%d" % epoch
                    if epoch is not None
                    else "epoch" + str(self._checkpoint)
                )
                model_name = "model%d" % itr if itr is not None else ""
                cpath = osp.join(fpath, "checkpoints", model_name, str(epoch_name))
                cname = "model" + ("%d" % itr if itr is not None else "") + ".pt"
                cname = osp.join(cpath, cname)
                os.makedirs(cpath, exist_ok=True)

            # Save model
            with warnings.catch_warnings():
                # FIXME: Improve saving!
                warnings.simplefilter("ignore")
                # We are using a non-recommended way of saving PyTorch models,
                # by pickling whole objects (which are dependent on the exact
                # directory structure at the time of saving) as opposed to
                # just saving network weights. This works sufficiently well
                # for the purposes of Spinning Up, but you may want to do
                # something different for your personal PyTorch project.
                # We use a catch_warnings() context to avoid the warnings about
                # not being able to save the source code.

                # Save model
                torch.save(self.pytorch_saver_elements, fname)

                # Save checkpoint
                if self._save_checkpoints:
                    torch.save(self.pytorch_saver_elements, cname)
                    self._checkpoint += 1  # Increase epoch

    def dump_tabular(self, global_step=None):
        """Write all of the diagnostics from the current iteration.

        Writes both to stdout, tensorboard and to the output file.

        Args:
            global_step (int, optional): Global step value to record. Uses internal step
            counter if global step is not supplied.
        """
        if proc_id() == 0:
            global_step = global_step if global_step else self._global_step
            vals = []
            key_lens = [len(key) for key in self._log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = "%" + "%d" % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len

            # Print results to terminal based on the set format
            if self.log_fmt.lower() == "line":  # Use row format
                valstr = [
                    key
                    + (
                        ":" + "%.3g" % self._log_current_row[key]
                        if hasattr(self._log_current_row[key], "__float__")
                        else self._log_current_row[key]
                    )
                    + "|"
                    for key in self._log_current_row.keys()
                    if key not in self.log_ignore
                ]
                print("".join(valstr)[:-1])
                vals.extend(self._log_current_row.values())
            else:
                print("-" * n_slashes)
            for key in self._log_headers:

                # Print tabular to stdout
                if self.log_fmt.lower() != "line":  # Use tabular format
                    val = self._log_current_row.get(key, "")
                    valstr = (
                        ("%8.3g" % val if hasattr(val, "__float__") else val)
                        if key not in self.log_ignore
                        else ""
                    )
                    print(fmt % (key, valstr))
                    vals.append(val)

                # Write tabular to tensorboard log
                if self._tabular_to_tb_dict[key]["tb_write"]:
                    var_name = (
                        self._tabular_to_tb_dict[key]["tb_alias"]
                        if self._tabular_to_tb_dict[key]["tb_alias"]
                        else key
                    )
                    var_name = (
                        self._tabular_to_tb_dict[key]["tb_prefix"] + "/" + var_name
                        if self._tabular_to_tb_dict[key]["tb_prefix"]
                        else var_name
                    )
                    self._write_to_tb(var_name, val, global_step=global_step)
            if self.log_fmt.lower() != "line":
                print("-" * n_slashes, flush=True)

            # Write tabular to output file
            if self.output_file is not None:
                if self._first_row:
                    self.output_file.write("\t".join(self._log_headers) + "\n")
                self.output_file.write("\t".join(map(str, vals)) + "\n")
                self.output_file.flush()
        self._log_current_row.clear()
        self._first_row = False

    def _write_to_tb(self, var_name, data, global_step=None):
        """Writes data to tensorboard log file.

        It currently works with scalars, histograms and images. For other data types
        please use :py:attr:`tb_writer`. directly.

        Args:
            var_name (string): Data identifier.

            data (int, float, numpy.ndarray, torch.Tensor): Data you want to write.

            global_step (int, optional): Global step value to record. Uses internal step
                counter if global step is not supplied.
        """

        # Try to write data to tb as as historgram
        if not self.tb_writer:
            self.use_tensorboard = (
                True  # Property that creates tf writer if set to True
            )
        if is_scalar(data):  # Extra protection since trying to write a list freezes tb
            try:  # Try to write as scalar
                self.tb_writer.add_scalar(var_name, data, global_step=global_step)
            except (
                AssertionError,
                NotImplementedError,
                NameError,
                ValueError,
                TypeError,
            ):
                pass
        else:

            # Try to write data to tb as as historgram
            try:
                self.tb_writer.add_histogram(var_name, data, global_step=global_step)
            except (
                AssertionError,
                NotImplementedError,
                NameError,
                ValueError,
                TypeError,
            ):
                pass

            # Try to write data as image
            try:
                self.tb_writer.add_image(var_name, data, global_step=global_step)
            except (
                AssertionError,
                NotImplementedError,
                NameError,
                ValueError,
                TypeError,
            ):
                pass

    @property
    def use_tensorboard(self):
        """Get or set the current value of use_tensorboard

        Setting the use_tensorboard variable to True will create a tensorboard writer.
        Setting it to False will delete existing tensorboard writers.
        """
        return self._use_tensorboard

    @use_tensorboard.setter
    def use_tensorboard(self, value):

        # Change tensorboard boolean
        self._use_tensorboard = value

        # Create tensorboard writer if use_tensorboard == True else delete
        if self._use_tensorboard and not self.tb_writer:  # Create writer object
            exp_name = "-" + self.exp_name if self.exp_name else ""
            self.tb_writer = SummaryWriter(
                log_dir=self.output_dir,
                comment=f"{exp_name.upper()}-data_" + time.strftime("%Y%m%d-%H%M%S"),
            )
            atexit.register(self.tb_writer.close)  # Make sure the writer is closed
        elif not self._use_tensorboard and self.tb_writer:  # Delete tensorboard writer
            self.tb_writer.close()  # Close writer
            atexit.unregister(self.tb_writer.close)  # Make sure the writer is closed
            self.tb_writer = None

    @property
    def _global_step(self):
        """Retrieve the current estimated global step count."""
        return max(self._step_count_dict.values())

    def get_logdir(self, *args, **kwargs):
        """Get summary writer logdir.

        Wrapper that calls the `SummaryWriter.get_logdir` method while first making
        sure a SummaryWriter object exits. The `get_logdir` method can be used to
        retrieve the directory where event files will be written.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.get_logdir(*args, **kwargs)

    def add_hparams(self, *args, **kwargs):
        """Adds_hparams to tb summary.

        Wrapper that calls the `SummaryWriter.add_hparams` method while first making
        sure a SummaryWriter object exits. The `add_hparams` method adds a set of
        hyperparameters to be compared in TensorBoard.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_hparams(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        """Add scalar to tb summary.

        Wrapper that calls the `SummaryWriter.add_scalar` method while first making
        sure a SummaryWriter object exits. The `add_scalar` method is used to add scalar
        data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_scalar(*args, **kwargs)

    def add_scalars(self, *args, **kwargs):
        """Add scalars to tb summary.

        Wrapper that calls the `SummaryWriter.add_scalars` method while first making
        sure a SummaryWriter object exits. The `add_scalars` method is used to add many
        scalar data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_scalars(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        """Add historgram to tb summary.
        Wrapper that calls the `SummaryWriter.add_histogram` method while first making
        sure a SummaryWriter object exits. The `add_histogram` method is used to add
        a histogram to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_histogram(*args, **kwargs)

    def add_histogram_raw(self, *args, **kwargs):
        """Adds raw histogram to tb summary.

        Wrapper that calls the `SummaryWriter.add_histogram_raw` method while first
        making sure a SummaryWriter object exits. The `add_histogram_raw` method is used
        to add histograms with raw data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_histogram_raw(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        """Add image to tb summary.

        Wrapper that calls the `SummaryWriter.add_image` method while first making
        sure a SummaryWriter object exits. The `add_image` method is used to add image
        data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_image(*args, **kwargs)

    def add_images(self, *args, **kwargs):
        """Add images to tb summary.

        Wrapper that calls the `SummaryWriter.add_images` method while first making
        sure a SummaryWriter object exits. The `add_images` method is used to add
        batched image data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_images(*args, **kwargs)

    def add_image_with_boxes(self, *args, **kwargs):
        """Add a image with boxes to summary.

        Wrapper that calls the `SummaryWriter.add_image_with_boxes` method while
        first making sure a SummaryWriter object exits. The `add_image_with_boxes`
        method is used to add image and draw bounding boxes on the image.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_image_with_boxes(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        """Add figure to tb summary.

        Wrapper that calls the `SummaryWriter.add_figure` method while
        first making sure a SummaryWriter object exits. The `add_figure`
        method is used to render a matplotlib figure into an image and add it to
        summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_figure(*args, **kwargs)

    def add_video(self, *args, **kwargs):
        """Add video to tb summary.

        Wrapper that calls the `SummaryWriter.add_video` method while
        first making sure a SummaryWriter object exits. The `add_video`
        method is used to add video data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_video(*args, **kwargs)

    def add_audio(self, *args, **kwargs):
        """Add audio to tb summary.

        Wrapper that calls the `SummaryWriter.add_audio` method while
        first making sure a SummaryWriter object exits. The `add_audio`
        method is used to add audio data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_audio(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        """Add text to tb summary.

        Wrapper that calls the `SummaryWriter.add_text` method while
        first making sure a SummaryWriter object exits. The `add_text`
        method is used to add text data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_text(*args, **kwargs)

    def add_onnx_graph(self, *args, **kwargs):
        """Add onnyx graph to tb summary.

        Wrapper that calls the `SummaryWriter.add_onnx_graph` method while
        first making sure a SummaryWriter object exits. The `add_onnx_graph`
        method is used to add a onnx graph data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_onnx_graph(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        """Add graph to tb summary.

        Wrapper that calls the `SummaryWriter.add_graph` method while
        first making sure a SummaryWriter object exits. The `add_graph`
        method is used to add a graph data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_graph(*args, **kwargs)

    def add_embedding(self, *args, **kwargs):
        """Add embedding to tb summary.

        Wrapper that calls the `SummaryWriter.add_embedding` method while
        first making sure a SummaryWriter object exits. The `add_embedding`
        method is used to add embedding projector data to summary.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_embedding(*args, **kwargs)

    def add_pr_curve(self, *args, **kwargs):
        """Add pr curve to tb summary.

        Wrapper that calls the `SummaryWriter.add_pr_curve` method while
        first making sure a SummaryWriter object exits. The `add_pr_curve`
        method is used to add a precision recall curve.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_pr_curve(*args, **kwargs)

    def add_pr_curve_raw(self, *args, **kwargs):
        """Add raw pr curve to tb summary.

        Wrapper that calls the `SummaryWriter.add_pr_curve_raw` method while
        first making sure a SummaryWriter object exits. The `add_pr_curve_raw`
        method is used to add a precision recall curve with raw data.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_pr_curve_raw(*args, **kwargs)

    def add_custom_scalars_multilinechart(self, *args, **kwargs):
        """Add custom scalars multilinechart to tb summary.

        Wrapper that calls the `SummaryWriter.add_custom_scalars_multilinechart`
        method while first making sure a SummaryWriter object exits. The
        `add_custom_scalars_multilinechart`, which is shorthand for creating
        `multilinechart` is similar to `add_custom_scalars`, but the only necessary
        argument is *tags*.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_custom_scalars_multilinechart(*args, **kwargs)

    def add_custom_scalars_marginchart(self, *args, **kwargs):
        """Adds custom scalars marginchart to tb summary.

        Wrapper that calls the `SummaryWriter.add_custom_scalars_marginchart`
        method while first making sure a SummaryWriter object exits. The
        `add_custom_scalars_marginchart`, which is shorthand for creating marginchart
        is similar the `add_custom_scalars`, but the only necessary argument is *tags*,
        which should have exactly 3 elements.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_custom_scalars_marginchart(*args, **kwargs)

    def add_custom_scalars(self, *args, **kwargs):
        """Add custom scalar to tb summary.

        Wrapper that calls the `SummaryWriter.add_custom_scalars` method while
        first making sure a SummaryWriter object exits. The `add_custom_scalars`
        method is used to create special charts by collecting charts tags in 'scalars'.
        Note that this function can only be called once for each `SummaryWriter` object.
        Because it only provides metadata to tensorboard, the function can be called
        before or after the training loop.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_custom_scalars(*args, **kwargs)

    def add_mesh(self, *args, **kwargs):
        """Add mesh to tb summary.

        Wrapper that calls the `SummaryWriter.add_mesh` method while
        first making sure a SummaryWriter object exits. The `add_mesh`
        method is used to add meshes or 3D point clouds to TensorBoard.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.add_mesh(*args, **kwargs)

    def flush(self, *args, **kwargs):
        """Flush tb event file to disk.

        Wrapper that calls the `SummaryWriter.flush` method while
        first making sure a SummaryWriter object exits. The `flush`
        method is used to flush the event file to disk.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        return self.tb_writer.flush(*args, **kwargs)


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.

    Attributes:
        epoch_dict (dict): Dictionary used to store variables you want to log into the
            epoch_loggers current state.
    """

    def __init__(self, *args, **kwargs):
        """Constructs all the necessary attributes for the EpochLogger object."""
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, tb_write=False, global_step=None, tb_aliases=dict(), **kwargs):
        """Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.

        Args:
            tb_write (bool, optional): Boolean specifying whether you also want to write
                the value to the tensorboard logfile. Defaults to False.
            global_step (int, optional): Global step value to record. Uses internal step
                counter if global step is not supplied.
            tb_aliases (dict, optional): Dictionary that can be used to set aliases for
                the variables you want to store. Defaults to empty dict().
        """
        for k, v in kwargs.items():

            # Store variable values in epoch_dict and increase global step count
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
                self._step_count_dict[k] = 0
            self.epoch_dict[k].append(v)

            # Increase the step count for all the keys
            # NOTE: This is done in such a way that two values of a given key do not
            # get the same global step value assigned to them
            self._step_count_dict[k] = (
                self._step_count_dict[k] + 1
                if self._step_count_dict[k] + 1 >= self._global_step
                else self._global_step
            )

            # Check if global step was supplied else use internal counter
            global_step = global_step if global_step else self._global_step

            # Check if a alias was given for the current parameter
            var_name = k if k not in tb_aliases.keys() else tb_aliases[k]

            # Write variable value to tensorboard
            if tb_write:
                self._write_to_tb(var_name, v, global_step=global_step)

    def log_tabular(
        self,
        key,
        val=None,
        with_min_and_max=False,
        average_only=False,
        tb_write=False,
        tb_prefix=None,
        tb_alias=None,
    ):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
            tb_write (bool, optional): Boolean specifying whether you also want to write
                the value to the tensorboard logfile. Defaults to False.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty dict(). If not supplied the variable name is
                used.
        """
        if val is not None:
            super().log_tabular(
                key, val, tb_write=tb_write, tb_prefix=tb_prefix, tb_alias=tb_alias
            )
        else:
            v = self.epoch_dict[key]
            vals = (
                np.concatenate(v)
                if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                else v
            )
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(
                "Average" + key,
                stats[0],
                tb_write=tb_write,
                tb_prefix=tb_prefix + "/Average" if tb_prefix else "Average",
                tb_alias=tb_alias,
            )
            if not (average_only):
                super().log_tabular(
                    "Std" + key,
                    stats[1],
                    tb_write=tb_write,
                    tb_prefix=tb_prefix + "/Std" if tb_prefix else "Std",
                    tb_alias=tb_alias,
                )
            if with_min_and_max:
                super().log_tabular(
                    "Max" + key,
                    stats[3],
                    tb_write=tb_write,
                    tb_prefix=tb_prefix + "/Max" if tb_prefix else "Max",
                    tb_alias=tb_alias,
                )
                super().log_tabular(
                    "Min" + key,
                    stats[2],
                    tb_write=tb_write,
                    tb_prefix=tb_prefix + "/Min" if tb_prefix else "Min",
                    tb_alias=tb_alias,
                )
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = (
            np.concatenate(v)
            if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
            else v
        )
        return mpi_statistics_scalar(vals)
