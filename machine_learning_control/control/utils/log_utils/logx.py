"""Module containing some simple multi-purpose logger.

This module extends the logx module of
`the SpinningUp repository <https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py>`_
so that besides logging to tab-separated-values file
(path/to/output_directory/progress.txt)
it also logs the data to a tensorboard file.
"""  # noqa

# TODO: Add WARN INFO methods -> Replace with proper logger!

import atexit
import json
import os
import os.path as osp
import pickle
import time

import joblib
import numpy as np
import torch
from machine_learning_control.control.common.helpers import is_scalar
from machine_learning_control.control.utils import import_tf
from machine_learning_control.control.utils.log_utils.helpers import colorize
from machine_learning_control.control.utils.mpi_tools import (
    mpi_statistics_scalar,
    proc_id,
)
from machine_learning_control.control.utils.serialization_utils import convert_json
from machine_learning_control.user_config import DEFAULT_STD_OUT_TYPE
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(
        self,
        output_dir=None,
        output_fname="progress.csv",
        exp_name=None,
        verbose=True,
        verbose_fmt=DEFAULT_STD_OUT_TYPE,
        verbose_vars=[],
        use_tensorboard=False,
        save_checkpoints=False,
    ):
        """Initialise a Logger.

        Args:
            output_dir (str, optional): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.
            output_fname (str, optional): Name for the (tab/comma) separated-value
                file containing metrics logged throughout a training run. Defaults to
                to ``progress.csv``.
            exp_name (str, optional): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
            verbose (bool, optional): Whether you want to log to the std_out. Defaults
                to ``True``.
            verbose_fmt (str, optional): The format in which the statistics are
                displayed to the terminal. Options are "tab" which supplies them as a
                table and "line" which prints them in one line. Default is set in the
                :py:mod:`user_config` file.
            verbose_vars (list, optional): A list of variables you want to log to the
                std_out. By default all variables are logged.
            use_tensorboard (bool): Specifies whether you want also log to Tensorboard.
                This variable is set to True if you log a variable to Tensorboard.
            save_checkpoints (bool, optional): Save checkpoints during training.
                Defaults to ``False``.

        Attributes:
            tb_writer (torch.utils.tensorboard.writer.SummaryWriter): A tensorboard
                writer. Is only created when the :py:attr:`.use_tensorboard` variable
                is set to `True`, if not it returns `None`.
            output_dir (str): The directory in which the log data and models are saved.
            output_file (str): The name of the file in which the progress data is saved.
            exp_name (str): Experiment name.
        """
        if proc_id() == 0:

            # Parse output_fname to see if csv was requested
            extension = os.path.splitext(output_fname)[1]
            self._output_csv = True if extension.lower() == ".csv" else False

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
            self.verbose = verbose
            self.verbose_table = verbose_fmt.lower() != "line"
            self.verbose_vars = [
                item.replace("Avg", "Average") for item in verbose_vars
            ]
        else:
            self.output_dir = None
            self.output_file = None
            self.verbose = None
            self.verbose_table = None
            self.verbose_vars = None
        self.exp_name = exp_name
        self._first_row = True
        self._log_headers = []
        self._log_current_row = {}
        self._save_checkpoints = save_checkpoints
        self._checkpoint = 0

        self.tb_writer = None
        self.use_tensorboard = use_tensorboard  # Create tensorboard writer
        self._tabular_to_tb_dict = (
            dict()
        )  # Stores whether tabular is logged to tensorboard when dump_tabular is called
        self._step_count_dict = (
            dict()
        )  # Used for keeping count of the current global step

    def log(self, msg, color="green"):
        # TODO: Use this instead of print(colorize()) inside script
        """Print a colorized message to stdout.

        Args:
            msg (str): Message you want to log.
            color (str, optional): Color you want the message to have. Defaults to
                "green".
        """
        if proc_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_to_tb(self, key, val, tb_prefix=None, tb_alias=None, global_step=None):
        """Log a value to Tensorboard.

        Args:
            key (str):  The name of the diagnostic.
            val: A value for the diagnostic.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty dict(). If not supplied the variable name is
                used.
            global_step (int, optional): Global step value to record. Uses internal step
                counter if global step is not supplied.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        var_name = tb_alias if tb_alias is not None else key
        var_name = tb_prefix + "/" + var_name if tb_prefix is not None else var_name
        self._write_to_tb(var_name, val, global_step=global_step)

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
            key (str): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            tb_write (bool, optional): Boolean specifying whether you also want to write
                the value to the tensorboard logfile. Defaults to ``False``.
            tb_metrics (union[list[str], str], optional): List containing the metrics
                you want to write to tensorboard. Options are ``['avg', 'std', 'min',
                'max']. Defaults to ``avg``.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty dict(). If not supplied the variable name is
                used.
        """
        if self._first_row:
            self._log_headers.append(key)
        else:
            assert key in self._log_headers, (
                "Trying to introduce a new key %s that you didn't include in the "
                "first iteration" % key
            )
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

    def dump_tabular(self, global_step=None):
        """Write all of the diagnostics from the current iteration.

        Writes both to stdout, tensorboard and to the output file.

        Args:
            global_step (int, optional): Global step value to record. Uses internal step
            counter if global step is not supplied.
        """
        if proc_id() == 0:
            vals = []
            print_dict = {}
            print_keys = []
            print_vals = []

            # Retrieve data from current row
            for key in self._log_headers:
                val = self._log_current_row.get(key, "")
                valstr = (
                    ("%8.3g" if self.verbose_table else "%.3g") % val
                    if hasattr(val, "__float__")
                    else val
                )
                print_keys.append(key)
                print_vals.append(valstr)
                print_dict[key] = valstr
                vals.append(val)

            # Log to stdout
            if self.verbose:
                key_filter = self.verbose_vars if self.verbose_vars else print_keys
                if self.verbose_table:
                    key_lens = [len(key) for key in self._log_headers]
                    max_key_len = max(15, max(key_lens))
                    keystr = "%" + "%d" % max_key_len
                    fmt = "| " + keystr + "s | %15s |"
                    n_slashes = 22 + max_key_len
                    print("-" * n_slashes)
                    print_str = "\n".join(
                        [
                            fmt % (key, val)
                            for key, val in zip(print_keys, print_vals)
                            if key in key_filter
                        ]
                    )
                    print(print_str)
                    print("-" * n_slashes, flush=True)
                else:
                    print_str = "|".join(
                        [
                            "%s:%s"
                            % (
                                "Steps"
                                if key == "TotalEnvInteracts"
                                else key.replace("Average", "Avg"),
                                val,
                            )
                            for key, val in zip(print_keys, print_vals)
                            if key in key_filter
                        ]
                    )
                    print(print_str)
            else:  # Increase epoch steps and time on the same line
                print(
                    colorize(
                        "\r{}: {:8.3G}, {}: {:8.3g}, {}: {:8.3G} s".format(
                            "Epoch",
                            float(print_dict["Epoch"]),
                            "Step",
                            float(print_dict["TotalEnvInteracts"]),
                            "Time",
                            float(print_dict["Time"]),
                        ),
                        "green",
                    ),
                    end="",
                )

            # Log to file
            if self.output_file is not None:
                if self._first_row:
                    self.output_file.write("\t".join(self._log_headers) + "\n")
                self.output_file.write("\t".join(map(str, vals)) + "\n")
                self.output_file.flush()

            # Write tabular to tensorboard log
            for key in self._log_headers:
                if self._tabular_to_tb_dict[key]["tb_write"]:
                    val = self._log_current_row.get(key, "")
                    # Use internal counter if global_step is None
                    if global_step is None:
                        if key in self._log_headers:
                            global_step = self._global_step
                    var_name = (
                        self._tabular_to_tb_dict[key]["tb_alias"]
                        if self._tabular_to_tb_dict[key]["tb_alias"] is not None
                        else key
                    )
                    var_name = (
                        self._tabular_to_tb_dict[key]["tb_prefix"] + "/" + var_name
                        if self._tabular_to_tb_dict[key]["tb_prefix"] is not None
                        else var_name
                    )
                    self._write_to_tb(var_name, val, global_step=global_step)

        self._log_current_row.clear()
        self._first_row = False

    def get_logdir(self, *args, **kwargs):
        """Get Logger and Tensorboard Summary writer logdirs.

        Args:
            *args: All args to pass to thunk.
            **kwargs: All kwargs to pass to thunk.

        Returns(dict): dict containing:
            output_dir(str): Logger output directory.
            tb_output_dir(str): Tensorboard writer output directory.
        """
        if self._use_tensorboard:
            return {
                "output_dir": self.output_dir,
                "tb_output_dir": self.tb_writer.get_logdir(*args, **kwargs),
            }
        else:
            return {
                "output_dir": self.output_dir,
                "tb_output_dir": "",
            }

    def save_config(self, config):
        """Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
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
            except (ValueError, pickle.PicklingError):
                self.log("Warning: could not pickle state_dict.", color="red")
            if hasattr(self, "tf_saver_elements"):
                self._tf_save(itr)
            if hasattr(self, "pytorch_saver_elements"):
                self._pytorch_save(itr, epoch)

    def setup_tf_saver(self, what_to_save):
        """Set up easy model saving for a single Tensorlow model.

        Because Tensorlow saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save (object): Any PyTorch model or serializable object containing
                Tensorflow models.
        """
        global tf
        tf = import_tf()  # Import tf if installed otherwise throw warning
        self.tf_saver_elements = what_to_save
        print(
            colorize(
                "INFO: Policy will be saved to '{}'.\n".format(self.output_dir),
                "cyan",
                bold=True,
            )
        )

    def setup_pytorch_saver(self, what_to_save):
        """Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save (object): Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save
        print(
            colorize(
                "INFO: Policy will be saved to '{}'.\n".format(self.output_dir),
                "cyan",
                bold=True,
            )
        )

    def _tf_save(self, itr=None, epoch=None):
        """Saves the PyTorch model/models using their 'state_dict'.

        Args:
            itr(int/None): Current iteration of training. Defaults to None
        """
        if proc_id() == 0:

            save_fail_warning = colorize(
                "WARN: The object you tried to save doesn't have a 'save_weights' we "
                "can use to retrieve the model weights. Please make sure you supplied "
                "the 'setup_pytorch_saver' method with a valid 'tf.keras.Model' object "
                "or implemented a 'save_weights' method on your object.",
                "yellow",
                bold=True,
            )

            assert hasattr(
                self, "tf_saver_elements"
            ), "First have to setup saving with self.setup_tf_saver"

            # Create filename
            fpath = "tf_save" + ("%d" % itr if itr is not None else "")
            fpath = osp.join(self.output_dir, fpath)
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
                os.makedirs(cpath, exist_ok=True)

            # Save model
            if isinstance(self.tf_saver_elements, tf.keras.Model) or hasattr(
                self.tf_saver_elements, "save_weights"
            ):
                self.tf_saver_elements.save_weights(
                    osp.join(fpath, "weights_checkpoint")
                )
            else:
                print(save_fail_warning)

            # Save checkpoint
            if self._save_checkpoints:
                if isinstance(self.tf_saver_elements, tf.keras.Model) or hasattr(
                    self.tf_saver_elements, "save_weights"
                ):
                    self.tf_saver_elements.save_weights(
                        osp.join(cpath, "weights_checkpoint")
                    )
                else:
                    print(save_fail_warning)

                self._checkpoint += 1  # Increase epoch

    def _pytorch_save(self, itr=None, epoch=None):
        """Saves the PyTorch model/models using their 'state_dict'.

        Args:
            itr(int/None): Current iteration of training. Defaults to None
        """
        if proc_id() == 0:

            save_fail_warning = colorize(
                "WARN: The object you tried to save doesn't have a 'state_dict' we can"
                "use to retrieve the model weights. Please make sure you supplied the "
                "'setup_pytorch_saver' method with a valid 'torch.nn.module' object or "
                "implemented a'state_dict' method on your object.",
                "yellow",
                bold=True,
            )

            assert hasattr(
                self, "pytorch_saver_elements"
            ), "First have to setup saving with self.setup_pytorch_saver"

            # Create filename
            fpath = "torch_save"
            fpath = osp.join(self.output_dir, fpath)
            fname = "model_state" + ("%d" % itr if itr is not None else "") + ".pt"
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
                cname = "model_state" + ("%d" % itr if itr is not None else "") + ".pt"
                cname = osp.join(cpath, cname)
                os.makedirs(cpath, exist_ok=True)

            # Save model
            if isinstance(self.pytorch_saver_elements, torch.nn.Module) or hasattr(
                self.pytorch_saver_elements, "state_dict"
            ):
                torch.save(self.pytorch_saver_elements.state_dict(), fname)
            else:
                print(save_fail_warning)

            # Save checkpoint
            if self._save_checkpoints:
                if isinstance(self.pytorch_saver_elements, torch.nn.Module) or hasattr(
                    self.pytorch_saver_elements, "state_dict"
                ):
                    torch.save(self.pytorch_saver_elements.state_dict(), cname)
                else:
                    print(save_fail_warning)

                self._checkpoint += 1  # Increase epoch

    def _write_to_tb(self, var_name, data, global_step=None):
        """Writes data to tensorboard log file.

        It currently works with scalars, histograms and images. For other data types
        please use :py:attr:`tb_writer`. directly.

        Args:
            var_name (str): Data identifier.
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
        return self._use_tensorboard

    @use_tensorboard.setter
    def use_tensorboard(self, value):
        """Custom setter that makes sure a Tensorboard writer is present on the
        :attr:`.MyClass.tb_writer` attribute when ``use_tensorboard`` is set to ``True``
        . This tensorboard writer can be used to write to the Tensorboard.

        Args:
            value (bool): Whether you want to use tensorboard logging.
        """
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
        return max(list(self._step_count_dict.values()) + [0.0])

    """Tensorboard related methods
    Below are several wrapper methods which make the Tensorboard writer methods
    available directly on the :class:`.EpochLogger` class.
    """

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
        self._tb_index_dict = dict()
        self._n_table_dumps = 0

    def store(
        self,
        tb_write=False,
        tb_aliases=dict(),
        extend=False,
        global_step=None,
        **kwargs,
    ):
        """Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.

        Args:
            tb_write (union[bool, dict], optional): Boolean or dict of key boolean pairs
                specifying whether you also want to write the value to the tensorboard
                logfile. Defaults to ``False``.
            tb_aliases (dict, optional): Dictionary that can be used to set aliases for
                the variables you want to store. Defaults to empty dict().
            extend (bool, optional): Boolean specifying whether you want to extend the
                values to the log buffer. By default ``false`` meaning the values are
                appended to the buffer.
            global_step (int, optional): Global step value to record. Uses internal step
                counter if global step is not supplied.
        """
        for k, v in kwargs.items():

            # Store variable values in epoch_dict and increase global step count
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
                self._step_count_dict[k] = 0
            if extend:
                self.epoch_dict[k].extend(v)
            else:
                self.epoch_dict[k].append(v)

            # Increase the step count for all the keys
            # NOTE: This is done in such a way that two values of a given key do not
            # get the same global step value assigned to them
            self._step_count_dict[k] = (
                self._step_count_dict[k] + 1
                if self._step_count_dict[k] + 1 >= self._global_step
                else self._global_step
            )

            # Check if a alias was given for the current parameter
            var_name = k if k not in tb_aliases.keys() else tb_aliases[k]

            # Write variable value to tensorboard
            tb_write_key = (
                (tb_write[k] if k in tb_write.keys() else False)
                if isinstance(tb_write, dict)
                else tb_write
            )
            if tb_write_key:
                global_step = (
                    global_step if global_step is not None else self._global_step
                )  # Use internal counter if global_step is None
                self._write_to_tb(var_name, v, global_step=global_step)

    def log_to_tb(
        self,
        keys,
        val=None,
        with_min_and_max=False,
        average_only=False,
        tb_prefix=None,
        tb_alias=None,
        global_step=None,
    ):
        """Log a diagnostic to Tensorboard. This function takes or a list of keys or a
        key-value pair. If only keys are supplied, averages will be calculated using
        the new data found in the Loggers internal storage. If a key-value pair is
        supplied, this pair will be directly logged to Tensorboard.

        Args:
            keys (union[list[str], str]): The name(s) of the diagnostic.
            val: A value for the diagnostic.
            with_min_and_max (bool): If true, log min and max values of the
                diagnostic.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty dict(). If not supplied the variable name is
                used.
            global_step (int, optional): Global step value to record. Uses internal step
                counter if global step is not supplied.
        """
        if val is not None:  # When key and value are supplied use direct write
            super().log_to_tb(
                keys,
                val,
                tb_prefix=tb_prefix,
                tb_alias=tb_alias,
                global_step=global_step,
            )
        else:  # When only keys are supplied use internal storage
            keys = [keys] if not isinstance(keys, list) else keys
            for key in keys:
                if global_step is None:  # Retrieve global step if not supplied
                    if self._n_table_dumps >= 1:
                        global_step_tmp = self._global_step
                    elif key in self.epoch_dict.keys():
                        global_step_tmp = len(self.epoch_dict[key])
                else:
                    global_step_tmp = global_step

                self._log_tb_statistics(
                    key,
                    with_min_and_max=with_min_and_max,
                    average_only=average_only,
                    tb_prefix=tb_prefix,
                    tb_alias=tb_alias,
                    global_step=global_step_tmp,
                )

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
            key (str): The name of the diagnostic. If you are logging a
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
                key,
                val,
                tb_write=tb_write,
                tb_prefix=tb_prefix,
                tb_alias=tb_alias,
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

    def dump_tabular(self, *args, **kwargs):
        """Small wrapper around the :class:`.Logger` parent class which makes sure that
        the tensorboard index track dictionary is reset after the table is dumped.

        Args:
            *args: All args to pass to parent method.
            **kwargs: All kwargs to pass to parent method.
        """
        super().dump_tabular(*args, **kwargs)
        self._n_table_dumps += 1
        self._tb_index_dict = {
            key: 0 for key in self._tb_index_dict.keys()
        }  # Reset tensorboard logging index storage dict

    def get_stats(self, key):
        """Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.

        Args:
            key (str): The key for which you want to get the stats.

        Returns:
            (tuple): tuple containing:
                mean(float): The current mean value.
                std(float): The current  mean standard deviation.
                min(float): The current mean value.
                max(float): The current mean value.
        """
        v = self.epoch_dict[key]
        vals = (
            np.concatenate(v)
            if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
            else v
        )
        return mpi_statistics_scalar(vals)

    def _log_tb_statistics(
        self,
        key,
        with_min_and_max=False,
        average_only=False,
        tb_prefix=None,
        tb_alias=None,
        global_step=None,
    ):
        """Calculates the statistics of a given key from all the new data found in the
        Loggers internal storage.

        Args:
            key (union[list[str], str]): The name of the diagnostic.
            with_min_and_max (bool): If true, log min and max values of the
                diagnostic.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty dict(). If not supplied the variable name is
                used.
            global_step (int, optional): Global step value to record. Uses internal step
                counter if global step is not supplied.
        """
        if key not in self._tb_index_dict.keys():
            self._tb_index_dict[key] = 0

        v = self.epoch_dict[key]
        vals = (
            np.concatenate(v[self._tb_index_dict[key] :])
            if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
            else v[self._tb_index_dict[key] :]
        )
        stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
        super().log_to_tb(
            "Average" + key,
            stats[0],
            tb_prefix=tb_prefix + "/Average" if tb_prefix else "Average",
            tb_alias=tb_alias,
            global_step=global_step,
        )
        if not (average_only):
            super().log_to_tb(
                "Std" + key,
                stats[1],
                tb_prefix=tb_prefix + "/Std" if tb_prefix else "Std",
                tb_alias=tb_alias,
                global_step=global_step,
            )
        if with_min_and_max:
            super().log_to_tb(
                "Max" + key,
                stats[3],
                tb_prefix=tb_prefix + "/Max" if tb_prefix else "Max",
                tb_alias=tb_alias,
                global_step=global_step,
            )
            super().log_to_tb(
                "Min" + key,
                stats[2],
                tb_prefix=tb_prefix + "/Min" if tb_prefix else "Min",
                tb_alias=tb_alias,
                global_step=global_step,
            )
        self._tb_index_dict[key] += len(vals)
