"""Contains a multi-purpose logger that can be used to log data and save trained models.

.. note::
    This module extends the logx module of
    `the SpinningUp repository <https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py>`_
    so that besides logging to tab-separated-values file (path/to/output_directory/progress.txt)
    it also logs the data to a tensorboard file.
"""  # noqa

import atexit
import glob
import json
import os
import os.path as osp
import pickle
import time

import joblib
import numpy as np
import torch
from bayesian_learning_control.common.helpers import is_scalar
from bayesian_learning_control.utils.import_utils import import_tf
from bayesian_learning_control.user_config import DEFAULT_STD_OUT_TYPE
from bayesian_learning_control.utils.log_utils import log_to_std_out
from bayesian_learning_control.utils.mpi_utils.mpi_tools import (
    mpi_statistics_scalar,
    proc_id,
)
from bayesian_learning_control.utils.serialization_utils import (
    convert_json,
    load_from_json,
    save_to_json,
)
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
        save_checkpoints=False,
        backend="torch",
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
                displayed to the terminal. Options are ``tab`` which supplies them as a
                table and ``line`` which prints them in one line. Default is set in the
                :mod:`~bayesian_learning_control.user_config` file.
            verbose_vars (list, optional): A list of variables you want to log to the
                std_out. By default all variables are logged.
            save_checkpoints (bool, optional): Save checkpoints during training.
                Defaults to ``False``.
            backend (str, optional): The backend you want to use for writing to
                tensorboard. Options are: ``tf`` or ``torch``. Defaults to ``torch``.

        Attributes:
            tb_writer (torch.utils.tensorboard.writer.SummaryWriter): A tensorboard
                writer. This is only created when you log a variable to tensorboard or
                set the :py:attr:`.Logger.use_tensorboard` variable to ``True``.
            output_dir (str): The directory in which the log data and models are saved.
            output_file (str): The name of the file in which the progress data is saved.
            exp_name (str): Experiment name.
        """
        if proc_id() == 0:

            # Parse output_fname to see if csv was requested
            extension = osp.splitext(output_fname)[1]
            self._output_csv = True if extension.lower() == ".csv" else False

            self.output_dir = output_dir or "/tmp/experiments/%i" % int(time.time())
            if osp.exists(self.output_dir):
                self.log(
                    (
                        "Log dir %s already exists! Storing info there anyway."
                        % self.output_dir
                    ),
                    type="warning",
                )
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), "w")
            atexit.register(self.output_file.close)
            self.log("Logging data to %s" % self.output_file.name, type="info")
            self.verbose = verbose
            self.verbose_table = verbose_fmt.lower() != "line"
            self.verbose_vars = ["Epoch", "TotalEnvInteracts"] + [
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
        self._save_info_saved = False

        self._use_tf_backend = backend.lower() in ["tf", "tensorflow"]
        self.tb_writer = None
        self._tabular_to_tb_dict = (
            dict()
        )  # Stores whether tabular is logged to tensorboard when dump_tabular is called
        self._step_count_dict = (
            dict()
        )  # Used for keeping count of the current global step

    def log(
        self,
        msg,
        color="",
        bold=False,
        highlight=False,
        type=None,
        *args,
        **kwargs,
    ):
        """Print a colorized message to ``stdout``.

        Args:
            msg (str): Message you want to log.
            color (str, optional): Color you want the message to have. Defaults to
                ``""``.
            bold (bool, optional): Whether you want the text to be bold text has to be
                bold.
            highlight (bool, optional):  Whether you want to highlight the text.
                Defaults to ``False``.
            type (str, optional): The log message type. Options are: ``info``,
                ``warning`` and ``error``. Defaults to ``None``.
            *args: All args to pass to the print function.
            **kwargs: All kwargs to pass to the print function.
        """
        if proc_id() == 0:
            log_to_std_out(
                msg,
                color=color,
                bold=bold,
                highlight=highlight,
                type=type,
                *args,
                **kwargs,
            )

    def log_to_tb(self, key, val, tb_prefix=None, tb_alias=None, global_step=None):
        """Log a value to Tensorboard.

        Args:
            key (str):  The name of the diagnostic.
            val: A value for the diagnostic.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty :obj:`dict`. If not supplied the variable name
                is used.
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
        After using :meth:`log_tabular` to store values for each diagnostic,
        make sure to call :meth:`dump_tabular` to write them out to file,
        tensorboard and ``stdout`` (otherwise they will not get saved anywhere).

        Args:
            key (str): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                :meth:`.EpochLogger.store`, the key here has to match the key you used
                there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via :meth:`.EpochLogger.store`, do *not* provide a
                ``val`` here.
            tb_write (bool, optional): Boolean specifying whether you also want to write
                the value to the tensorboard logfile. Defaults to ``False``.
            tb_metrics (Union[list[str], str], optional): List containing the metrics
                you want to write to tensorboard. Options are [``avg``, ``std``,
                ``min``, ``max``].`` Defaults to ``avg``.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty :obj:`dict`. If not supplied the variable name
                is used.
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

    def dump_tabular(self, global_step=None):  # noqa: C901
        """Write all of the diagnostics from the current iteration.

        Writes both to ``stdout``, tensorboard and to the output file.

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
                    self.log("-" * n_slashes)
                    print_str = "\n".join(
                        [
                            fmt % (key, val)
                            for key, val in zip(print_keys, print_vals)
                            if key in key_filter
                        ]
                    )
                    self.log(print_str)
                    self.log("-" * n_slashes, flush=True)
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
                    self.log(print_str)
            else:  # Increase epoch steps and time on the same line
                self.log(
                    "\r{}: {:8.3G}, {}: {:8.3g}, {}: {:8.3G} s".format(
                        "Epoch",
                        float(print_dict["Epoch"]),
                        "Step",
                        float(print_dict["TotalEnvInteracts"]),
                        "Time",
                        float(print_dict["Time"]),
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
        """Get Logger and Tensorboard SummaryWriter logdirs.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Returns:
            (dict): dict containing:

                - output_dir(:obj:`str`): Logger output directory.
                - tb_output_dir(:obj:`str`): Tensorboard writer output directory.
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

        Example:
            .. code-block:: python

                logger = EpochLogger(**logger_kwargs)
                logger.save_config(locals())

        Args:
           config (object): Configuration Python object you want to save.
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json["exp_name"] = self.exp_name
        if proc_id() == 0:
            output = json.dumps(
                config_json, separators=(",", ":\t"), indent=4, sort_keys=True
            )
            self.log("Saving config:\n", type="info")
            self.log(output)
            with open(osp.join(self.output_dir, "config.json"), "w") as out:
                out.write(output)

    @classmethod
    def load_config(cls, config_path):
        """Loads an experiment configuration.

        Args:
           config_path (str): Folder that contains the config files you want to load.

        Returns:
            (object): Object containing the loaded configuration.
        """
        if proc_id() == 0:
            if not osp.basename(config_path).endswith(".json"):
                load_path = glob.glob(
                    osp.join(config_path, "**", "config.json"), recursive=True
                )
                if len(load_path) == 0:
                    raise FileNotFoundError(
                        f"No 'config.json' file found in '{config_path}'. Please check "
                        "your `config_path` and try again."
                    )
                load_path = load_path[0]
            else:
                load_path = config_path
            return load_from_json(load_path)

    @classmethod
    def load_env(cls, env_path):
        """Loads a pickled environment.

        Args:
           config_path (str): Folder that contains the pickled environment.

        Returns:
            (:obj:`gym.env`): The gym environment.
        """
        if proc_id() == 0:
            if not osp.basename(env_path).endswith(".pkl"):
                load_path = glob.glob(
                    osp.join(env_path, "**", "vars.pkl"), recursive=True
                )
                if len(load_path) == 0:
                    raise FileNotFoundError(
                        f"No 'vars.pkl' file found in '{env_path}'. Please check  "
                        "your 'env_path' and try again."
                    )
                load_path = load_path[0]
            else:
                load_path = env_path
        # try to load environment from save
        # NOTE: Sometimes this will fail because the environment could not be pickled.
        try:
            state = joblib.load(load_path)
            return state["env"]
        except Exception:
            raise ValueError(
                "Something went wrong while trying to load the pickled environment "
                "please check the environment and try again."
            )

    def save_to_json(self, input_object, output_filename, output_path=None):
        """Save python object to Json file. This method will serialize the object to
        JSON, while handling anything which can't be serialized in a graceful way
        (writing as informative a string as possible).

        Args:
            input_object (object): The input object you want to save.
            output_filename (str): The output filename.
            output_path (str): The output path. By default the
                :attr:`.Logger.output_dir`  is used.
        """
        if proc_id() == 0:
            input_object_json = convert_json(input_object)
            if self.exp_name is not None:
                input_object_json["exp_name"] = self.exp_name
            output_path = self.output_dir if output_path is None else output_path
            save_to_json(
                input_object=input_object_json,
                output_filename=output_filename,
                output_path=output_path,
            )

    def load_from_json(self, path):
        """Load data from json file.

        Args:
            path (str): The path of the json file you want to load.
        Returns:
            (object): The Json load object.
        """
        if proc_id() == 0:
            return load_from_json(path)

    def save_state(self, state_dict, itr=None):
        """Saves the state of an experiment.

        .. important::
            To be clear: this is about saving *state*, not logging diagnostics.
            All diagnostic logging is separate from this function. This function
            will save whatever is in ``state_dict``---usually just a copy of the
            environment---and the most recent parameters for the model you
            previously set up saving for with :meth:`setup_tf_saver` or
            :meth:`setup_pytorch_saver`.

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.
            itr (Union[int, None]): Current iteration of training (e.g. epoch). Defaults
                to None
        """
        if proc_id() == 0:

            # Save training state (environment, ...)
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, "vars.pkl"))
            except (ValueError, pickle.PicklingError):
                self.log("Warning: could not pickle state_dict.", color="red")

            # Save model state
            if hasattr(self, "tf_saver_elements"):
                backend_folder_name = "tf2_save"
                self._tf_save(itr)
            if hasattr(self, "pytorch_saver_elements"):
                backend_folder_name = "torch_save"
                self._pytorch_save(itr)

            # Save checkpoint state
            if self._save_checkpoints and itr is not None:
                itr_name = (
                    "iter%d" % itr
                    if itr is not None
                    else "iter" + str(self._checkpoint)
                )
                fpath = osp.join(
                    self.output_dir, backend_folder_name, "checkpoints", str(itr_name)
                )
                fname = osp.join(fpath, "vars.pkl")
                os.makedirs(fpath, exist_ok=True)
                try:
                    joblib.dump(state_dict, fname)
                except (ValueError, pickle.PicklingError):
                    self.log("Warning: could not pickle state_dict.", color="red")

    def setup_tf_saver(self, what_to_save):
        """Set up easy model saving for a single Tensorlow model.

        Args:
            what_to_save (object): Any PyTorch model or serializable object containing
                Tensorflow models.
        """
        global tf
        tf = import_tf()  # Import tf if installed otherwise throw warning
        self.tf_saver_elements = what_to_save
        self.log("Policy will be saved to '{}'.\n".format(self.output_dir), type="info")

    def setup_pytorch_saver(self, what_to_save):
        """Set up easy model saving for a single PyTorch model.

        Args:
            what_to_save (object): Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save
        self.log("Policy will be saved to '{}'.\n".format(self.output_dir), type="info")

    def _tf_save(self, itr=None):
        """Saves the PyTorch model/models using their ``state_dict``.

        Args:
            itr (Union[int, None]): Current iteration of training (e.g. epoch). Defaults
                to None
        """
        if proc_id() == 0:

            save_fail_warning = (
                "The object you tried to save doesn't have a 'save_weights' we "
                "can use to retrieve the model weights. Please make sure you supplied "
                "the 'setup_tf_saver' method with a valid 'tf.keras.Model' object "
                "or implemented a 'save_weights' method on your object.",
            )

            assert hasattr(
                self, "tf_saver_elements"
            ), "First have to setup saving with self.setup_tf_saver"

            # Create filename
            fpath = osp.join(self.output_dir, "tf2_save")
            fname = osp.join(fpath, "weights_checkpoint")
            os.makedirs(fpath, exist_ok=True)

            # Create Checkpoints name
            if self._save_checkpoints and itr is not None:
                itr_name = (
                    "iter%d" % itr
                    if itr is not None
                    else "iter" + str(self._checkpoint)
                )
                cpath = osp.join(fpath, "checkpoints", str(itr_name))
                cname = osp.join(cpath, "weights_checkpoint")
                os.makedirs(cpath, exist_ok=True)

            # Save additional algorithm information
            if not self._save_info_saved:
                save_info = {"alg_name": self.tf_saver_elements.__class__.__name__}
                self.save_to_json(
                    save_info,
                    output_filename="save_info.json",
                    output_path=fpath,
                )

            # Save model
            if isinstance(self.tf_saver_elements, tf.keras.Model) or hasattr(
                self.tf_saver_elements, "save_weights"
            ):
                self.tf_saver_elements.save_weights(fname)
            else:
                self.log(save_fail_warning, type="warning")

            # Save checkpoint
            if self._save_checkpoints and itr is not None:
                if isinstance(self.tf_saver_elements, tf.keras.Model) or hasattr(
                    self.tf_saver_elements, "save_weights"
                ):
                    self.tf_saver_elements.save_weights(cname)
                else:
                    self.log(save_fail_warning, type="warning")

                self._checkpoint += 1  # Increase epoch

    def _pytorch_save(self, itr=None):
        """Saves the PyTorch model/models using their ``state_dict``.

        Args:
            itr (Union[int, None]): Current iteration of training (e.g. epoch). Defaults
                to None
        """
        if proc_id() == 0:

            save_fail_warning = (
                "The object you tried to save doesn't have a 'state_dict' we can"
                "use to retrieve the model weights. Please make sure you supplied the "
                "'setup_pytorch_saver' method with a valid 'torch.nn.Module' object or "
                "implemented a'state_dict' method on your object."
            )

            assert hasattr(
                self, "pytorch_saver_elements"
            ), "First have to setup saving with self.setup_pytorch_saver"

            # Create filename
            fpath = osp.join(self.output_dir, "torch_save")
            fname = osp.join(fpath, "model_state.pt")
            os.makedirs(fpath, exist_ok=True)

            # Create Checkpoints Name
            if self._save_checkpoints and itr is not None:
                itr_name = (
                    "iter%d" % itr
                    if itr is not None
                    else "iter" + str(self._checkpoint)
                )
                cpath = osp.join(fpath, "checkpoints", str(itr_name))
                cname = osp.join(cpath, "model_state.pt")
                os.makedirs(cpath, exist_ok=True)

            # Save additional algorithm information
            if not self._save_info_saved:
                save_info = {"alg_name": self.pytorch_saver_elements.__class__.__name__}
                self.save_to_json(
                    save_info,
                    output_filename="save_info.json",
                    output_path=fpath,
                )

            # Save model
            if isinstance(self.pytorch_saver_elements, torch.nn.Module) or hasattr(
                self.pytorch_saver_elements, "state_dict"
            ):
                torch.save(self.pytorch_saver_elements.state_dict(), fname)
            else:
                self.log(save_fail_warning, type="warning")

            # Save checkpoint
            if self._save_checkpoints:
                if isinstance(self.pytorch_saver_elements, torch.nn.Module) or hasattr(
                    self.pytorch_saver_elements, "state_dict"
                ):
                    torch.save(self.pytorch_saver_elements.state_dict(), cname)
                else:
                    self.log(save_fail_warning, type="warning")

                self._checkpoint += 1  # Increase epoch

    def _write_to_tb(self, var_name, data, global_step=None):
        """Writes data to tensorboard log file.

        It currently works with scalars, histograms and images. For other data types
        please use :py:attr:`.Logger.tb_writer`. directly.

        Args:
            var_name (str): Data identifier.
            data (Union[int, float, numpy.ndarray, torch.Tensor]): Data you want to
                write.
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
                self.add_scalar(var_name, data, global_step=global_step)
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
                self.add_histogram(var_name, data, global_step=global_step)
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
                self.add_image(var_name, data, global_step=global_step)
            except (
                AssertionError,
                NotImplementedError,
                NameError,
                ValueError,
                TypeError,
            ):
                self.log(
                    (
                        "WARNING: Variable {} could ".format(var_name.capitalize())
                        + "not be written to tensorboard as the '{}' ".format(
                            self.__class__.__name__
                        )
                        + "class does not yet support objects of type {}.".format(
                            type(data)
                        )
                    ),
                    type="warning",
                )
                pass

    @property
    def use_tensorboard(self):
        """Variable specifying whether the logger uses Tensorboard. A Tensorboard writer
        is created on the :attr:`.Logger.tb_writer` attribute when
        :attr:`~.Logger.use_tensorboard` is set to ``True``
        """
        return self._use_tensorboard

    @use_tensorboard.setter
    def use_tensorboard(self, value):
        """Custom setter that makes sure a Tensorboard writer is present on the
        :attr:`.Logger.tb_writer` attribute when ``use_tensorboard`` is set to ``True``
        . This tensorboard writer can be used to write to the Tensorboard.

        Args:
            value (bool): Whether you want to use tensorboard logging.
        """
        self._use_tensorboard = value

        # Create tensorboard writer if use_tensorboard == True else delete
        if self._use_tensorboard and not self.tb_writer:  # Create writer object
            if self._use_tf_backend:
                self.log("Using Tensorflow as the Tensorboard backend.", type="info")
                tf = import_tf()  # Import tf if installed otherwise throw warning
                self.tb_writer = tf.summary.create_file_writer(self.output_dir)
            else:
                self.log(
                    "Using Torch.utils.tensorboard as the Tensorboard backend.",
                    type="info",
                )
                exp_name = "-" + self.exp_name if self.exp_name else ""
                self.tb_writer = SummaryWriter(
                    log_dir=self.output_dir,
                    comment=f"{exp_name.upper()}-data_"
                    + time.strftime("%Y%m%d-%H%M%S"),
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

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_hparams`
        method while first making sure a SummaryWriter object exits. The ``add_hparams``
        method adds a set of hyperparameters to be compared in TensorBoard.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_hparams' method is not available when using the 'tensorflow' "
                "backend."
            )
        return self.tb_writer.add_hparams(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        """Add scalar to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_scalar`
        or :obj:`tf.summary.scalar` (depending on the backend) method while first making
        sure a SummaryWriter object exits. These methods can be used to add a scalar
        to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step")
            global tf
            with self.tb_writer.as_default():
                return tf.summary.scalar(*args, **kwargs)
        else:
            return self.tb_writer.add_scalar(*args, **kwargs)

    def scalar(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_scalar` method available directly under
        the ``scalar`` alias.

        Args:
            *args: All args to pass to the :meth:`add_scalar` method.
            **kwargs: All kwargs to pass to the :meth:`add_scalar` method.
        """
        self.add_scalar(*args, **kwargs)

    def add_scalars(self, *args, **kwargs):
        """Add scalars to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_scalars`
        method while first making sure a SummaryWriter object exits. The ``add_scalars``
        method can be used to add many scalar data to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_scalars' method is not available when using the 'tensorflow' "
                "backend. Please use the 'add_scalar' method instead."
            )
        return self.tb_writer.add_scalars(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        """Add historgram to tb summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_histogram` or
        :obj:`tf.summary.histogram` (depending on the backend) method while first making
        sure a SummaryWriter object exits. These methods can be used to add a histogram
        to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step")
            global tf
            with self.tb_writer.as_default():
                return tf.summary.histogram(*args, **kwargs)
        else:
            return self.tb_writer.add_histogram(*args, **kwargs)

    def histogram(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_histogram` method available directly under
        the ``histogram`` alias.

        Args:
            *args: All args to pass to the :meth:`add_histogram` method.
            **kwargs: All kwargs to pass to the :meth:`add_histogram` method.
        """
        self.add_histogram(*args, **kwargs)

    def add_histogram_raw(self, *args, **kwargs):
        """Adds raw histogram to tb summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_histogram_raw` method while
        first making sure a SummaryWriter object exits. The ``add_histogram_raw`` method
        can be used to add histograms with raw data to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_histogram_raw' method is not available when using the "
                "'tensorflow' backend."
            )
        return self.tb_writer.add_histogram_raw(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        """Add image to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_image`
        or :obj:`tf.summary.image` (depending on the backend) method while first making
        sure a SummaryWriter object exits. These methods can be used to add a image
        to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step")
            global tf
            with self.tb_writer.as_default():
                return tf.summary.image(*args, **kwargs)
        else:
            return self.tb_writer.add_image(*args, **kwargs)

    def image(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_image` method available directly under
        the ``image`` alias.

        Args:
            *args: All args to pass to the :meth:`add_image` method.
            **kwargs: All kwargs to pass to the :meth:`add_image` method.
        """
        self.add_image(*args, **kwargs)

    def add_images(self, *args, **kwargs):
        """Add images to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_images`
        method while first making sure a SummaryWriter object exits. The ``add_images``
        method is used to add batched image data to summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_images' method is not available when using the 'tensorflow' "
                "backend. Please use the 'add_image' method instead."
            )
        return self.tb_writer.add_images(*args, **kwargs)

    def add_image_with_boxes(self, *args, **kwargs):
        """Add a image with boxes to summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_image_with_boxes` method while
        first making sure a SummaryWriter object exits. The ``add_image_with_boxes``
        method is used to add image and draw bounding boxes on the image.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_image_with_boxes' method is not available when using the "
                "'tensorflow' backend."
            )
        return self.tb_writer.add_image_with_boxes(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        """Add figure to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_figure`
        method while first making sure a SummaryWriter object exits. The ``add_figure``
        method is used to render a matplotlib figure into an image and add it to
        summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_figure' method is not available when using the 'tensorflow' "
                "backend."
            )
        return self.tb_writer.add_figure(*args, **kwargs)

    def add_video(self, *args, **kwargs):
        """Add video to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_video`
        method while first making sure a SummaryWriter object exits. The ``add_video``
        method is used to add video data to summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_video' method is not available when using the 'tensorflow' "
                "backend."
            )
        return self.tb_writer.add_video(*args, **kwargs)

    def add_audio(self, *args, **kwargs):
        """Add audio to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_audio`
        or :obj:`tf.summary.audio` (depending on the backend) method while first making
        sure a SummaryWriter object exits. These methods can be used to add audio data
        to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step")
            global tf
            with self.tb_writer.as_default():
                return tf.summary.audio(*args, **kwargs)
        else:
            return self.tb_writer.add_audio(*args, **kwargs)

    def audio(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_audio` method available directly under
        the ``audio`` alias.

        Args:
            *args: All args to pass to the :meth:`add_audio` method.
            **kwargs: All kwargs to pass to the :meth:`add_audio` method.
        """
        self.add_audio(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        """Add text to tb summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_text` or
        :obj:`tf.summary.add_text` (depending on the backend) method while first making
        sure a SummaryWriter object exits. These methods can be used to add text
        to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step")
            global tf
            with self.tb_writer.as_default():
                return tf.summary.text(*args, **kwargs)
        else:
            return self.tb_writer.add_text(*args, **kwargs)

    def text(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_text` method available directly under
        the ``text`` alias.

        Args:
            *args: All args to pass to the :meth:`add_text` method.
            **kwargs: All kwargs to pass to the :meth:`add_text` method.
        """
        self.add_text(*args, **kwargs)

    def add_onnx_graph(self, *args, **kwargs):
        """Add onnyx graph to tb summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_onnx_graph` method while first
        making sure a SummaryWriter object exits. The ``add_onnx_graph``
        method is used to add a onnx graph data to summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_onnx_graph' method is not available when using the "
                "'tensorflow' backend."
            )
        return self.tb_writer.add_onnx_graph(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        """Add graph to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_graph`
        method while first making sure a SummaryWriter object exits. The ``add_graph``
        method is used to add a graph data to summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_graph' method is not available when using the 'tensorflow' "
                "backend. Please use the 'trace_export' method instead."
            )
        return self.tb_writer.add_graph(*args, **kwargs)

    def add_embedding(self, *args, **kwargs):
        """Add embedding to tb summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_embedding` while first making
        sure a SummaryWriter object exits. The ``add_embedding`` method is used to add
        embedding projector data to summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_embedding' method is not available when using the "
                "'tensorflow' backend."
            )
        return self.tb_writer.add_embedding(*args, **kwargs)

    def add_pr_curve(self, *args, **kwargs):
        """Add pr curve to tb summary.

        :obj:`torch.utils.tensorboard.SummaryWriter.add_pr_curve` while first making
        while first making sure a SummaryWriter object exits. The ``add_pr_curve``
        method is used to add a precision recall curve.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_pr_curve' method is not available when using the "
                "'tensorflow' backend."
            )
        return self.tb_writer.add_pr_curve(*args, **kwargs)

    def add_pr_curve_raw(self, *args, **kwargs):
        """Add raw pr curve to tb summary.

        :obj:`torch.utils.tensorboard.SummaryWriter.add_pr_curve_raw` while first making
        first making sure a SummaryWriter object exits. The ``add_pr_curve_raw``
        method is used to add a precision recall curve with raw data.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_pr_curve_raw' method is not available when using the "
                "'tensorflow' backend."
            )
        return self.tb_writer.add_pr_curve_raw(*args, **kwargs)

    def add_custom_scalars_multilinechart(self, *args, **kwargs):
        """Add custom scalars multilinechart to tb summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_custom_scalars_multilinechart`
        while first making method while first making sure a SummaryWriter object exits.
        The ``add_custom_scalars_multilinechart``, which is shorthand for creating
        ``multilinechart`` is similar to ``add_custom_scalars``, but the only necessary
        argument is *tags*.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_custom_scalars_multilinechart' method is not available "
                "when using the 'tensorflow' backend."
            )
        return self.tb_writer.add_custom_scalars_multilinechart(*args, **kwargs)

    def add_custom_scalars_marginchart(self, *args, **kwargs):
        """Adds custom scalars marginchart to tb summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_custom_scalars_marginchart`
        method while first making sure a SummaryWriter object exits. The
        ``add_custom_scalars_marginchart``, which is shorthand for creating marginchart
        is similar the ``add_custom_scalars``, but the only necessary argument is
        *tags*, which should have exactly 3 elements.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_custom_scalars_marginchart' method is not available when "
                "using the 'tensorflow' backend."
            )
        return self.tb_writer.add_custom_scalars_marginchart(*args, **kwargs)

    def add_custom_scalars(self, *args, **kwargs):
        """Add custom scalar to tb summary.


        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_custom_scalars`
        while first making sure a SummaryWriter object exits. The ``add_custom_scalars``
        method is used to create special charts by collecting charts tags in 'scalars'.

        .. note::
            Note that this function can only be called once for each SummaryWriter
            object. Because it only provides metadata to tensorboard, the function can
            be called before or after the training loop.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_custom_scalars' method is not available when using the "
                "'tensorflow' backend."
            )
        return self.tb_writer.add_custom_scalars(*args, **kwargs)

    def add_mesh(self, *args, **kwargs):
        """Add mesh to tb summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_mesh` while first making sure a
        SummaryWriter object exits. The ``add_mesh`` method is used to add meshes or
        3D point clouds to TensorBoard.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Tensorflow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_mesh' method is not available when using the 'tensorflow' "
                "backend."
            )
        return self.tb_writer.add_mesh(*args, **kwargs)

    def flush(self, *args, **kwargs):
        """Flush tb event file to disk.


        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.flush` or
        :obj:`tf.summary.flush` (depending on the backend) method while first making
        sure a SummaryWriter object exits. These methods can be used to flush the event
        file to disk.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists
        if self._use_tf_backend:
            global tf
            with self.tb_writer.as_default():
                return tf.summary.flush(*args, **kwargs)
        else:
            return self.tb_writer.flush(*args, **kwargs)

    def record_if(self, *args, **kwargs):
        """Sets summary recording on or off per the provided boolean value.

        Wrapper that calls the :obj:`tf.summary.record_if` method while first making
        sure a SummaryWriter object exits.

        Args:
            *args: All args to pass to the Summary object.
            **kwargs: All kwargs to pass to the Summary object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Torch backend.
        """
        if not self._use_tf_backend:
            raise NotImplementedError(
                "The 'record_if' method is not available when using the 'torch' "
                "backend."
            )
        global tf
        return tf.summary.record_if(*args, **kwargs)

    def should_record_summaries(self):
        """Returns boolean Tensor which is true if summaries should be recorded.

        Wrapper that calls the :obj:`tf.summary.should_record_summaries` method while
        first making sure a SummaryWriter object exits.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                Torch backend.
        """
        if not self._use_tf_backend:
            raise NotImplementedError(
                "The 'should_record_summaries' method is not available when using the "
                "'torch' backend."
            )
        global tf
        return tf.summary.should_record_summaries()

    def trace_export(self, *args, **kwargs):
        """Stops and exports the active trace as a Summary and/or profile file.

        Wrapper that calls the :obj:`tf.summary.trace_export` method while first making
        sure a SummaryWriter object exits.

        Args:
            *args: All args to pass to the Summary object.
            **kwargs: All kwargs to pass to the Summary object.

        Raises:
            NotImplementedError:
                Raised if you try to call this method when using the Torch backend.
        """
        if not self._use_tf_backend:
            raise NotImplementedError(
                "The 'trace_export' method is not available when using the 'torch' "
                "backend."
            )
        global tf
        return tf.summary.trace_export(*args, **kwargs)

    def trace_off(self):
        """Stops the current trace and discards any collected information.

        Wrapper that calls the :obj:`tf.summary.trace_off` method while first making
        sure a SummaryWriter object exits.

        Raises:
            NotImplementedError:
                Raised if you try to call this method when using the Torch backend.
        """
        if not self._use_tf_backend:
            raise NotImplementedError(
                "The 'trace_off' method is not available when using the 'torch' "
                "backend."
            )
        global tf
        return tf.summary.trace_off()

    def trace_on(self):
        """Starts a trace to record computation graphs and profiling information.

        Wrapper that calls the :obj:`tf.summary.trace_on` method while first making
        sure a SummaryWriter object exits.

        Raises:
            NotImplementedError:
                Raised if you try to call this method when using the Torch backend.
        """
        if not self._use_tf_backend:
            raise NotImplementedError(
                "The 'trace_on' method is not available when using the 'torch' backend."
            )
        global tf
        return tf.summary.trace_on()

    def write(self, *args, **kwargs):
        """Starts a trace to record computation graphs and profiling information.

        Wrapper that calls the :obj:`tf.summary.write` method while first making sure a
        SummaryWriter object exits.

        Args:
            *args: All args to pass to the Summary object.
            **kwargs: All kwargs to pass to the Summary object.

        Raises:
            NotImplementedError:
                Raised if you try to call this method when using the Torch backend.
        """
        if not self._use_tf_backend:
            raise NotImplementedError(
                "The 'write' method is not available when using the 'torch' backend."
            )
        global tf
        return tf.summary.write()


class EpochLogger(Logger):
    """
    A variant of :class:`.Logger` tailored for tracking average values over epochs.

    **Typical use case:** there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average/std/min/max value of that quantity.

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
           :class:`.EpochLogger` current state.
    """

    def __init__(self, *args, **kwargs):
        """Constructs all the necessary attributes for the :class:`.EpochLogger`
        object.
        """
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
        """Save something into the :class:`.EpochLogger`'s current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.

        Args:
            tb_write (Union[bool, dict], optional): Boolean or dict of key boolean pairs
                specifying whether you also want to write the value to the tensorboard
                logfile. Defaults to ``False``.
            tb_aliases (dict, optional): Dictionary that can be used to set aliases for
                the variables you want to store. Defaults to empty :obj:`dict`.
            extend (bool, optional): Boolean specifying whether you want to extend the
                values to the log buffer. By default ``False`` meaning the values are
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
            keys (Union[list[str], str]): The name(s) of the diagnostic.
            val: A value for the diagnostic.
            with_min_and_max (bool): If ``True``, log min and max values of the
                diagnostic.
            average_only (bool): If ``True``, do not log the standard deviation
                of the diagnostic.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty :obj:`dict`. If not supplied the variable name
                is used.
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
                :meth:`store`, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via :meth:`store`, do *not* provide a ``val``
                here.
            with_min_and_max (bool): If ``True``, log min and max values of the
                diagnostic over the epoch.
            average_only (bool): If ``True``, do not log the standard deviation
                of the diagnostic over the epoch.
            tb_write (bool, optional): Boolean specifying whether you also want to write
                the value to the tensorboard logfile. Defaults to False.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty :obj:`dict`. If not supplied the variable name
                is used.
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
        """Small wrapper around the :meth:`.Logger.dump_tabular` method which
        makes sure that the tensorboard index track dictionary is reset after the table
        is dumped.

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

                - mean(:obj:`float`): The current mean value.
                - std(:obj:`float`): The current  mean standard deviation.
                - min(:obj:`float`): The current mean value.
                - max(:obj:`float`): The current mean value.
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
            key (Union[list[str], str]): The name of the diagnostic.
            with_min_and_max (bool): If ``True``, log min and max values of the
                diagnostic.
            average_only (bool): If ``True``, do not log the standard deviation
                of the diagnostic.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty :obj:`dict`. If not supplied the variable name
                is used.
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
