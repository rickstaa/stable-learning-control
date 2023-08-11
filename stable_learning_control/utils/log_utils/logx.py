"""Contains a multi-purpose logger that can be used to log data and save trained models.

.. note::
    This module extends the logx module of
    `the SpinningUp repository <https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py>`_
    so that it:
    
    - Also logs in line format (besides tabular format).
    - Logs to a file with a .csv extension (besides .txt).
    - Logs to TensorBoard (besides logging to a file).
    - Logs to Weights & Biases (besides logging to a file).
"""  # noqa
import atexit
import copy
import glob
import json
import os
import os.path as osp
import pickle
import re
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from stable_learning_control.common.helpers import (
    convert_to_tb_config,
    convert_to_wandb_config,
    is_scalar,
    parse_config_env_key,
)
from stable_learning_control.user_config import (
    DEFAULT_STD_OUT_TYPE,
    PRINT_CONFIG,
    TB_HPARAMS_FILTER,
    TB_HPARAMS_METRICS,
)
from stable_learning_control.utils.import_utils import import_tf, lazy_importer
from stable_learning_control.utils.log_utils.helpers import (
    dict_to_mdtable,
    log_to_std_out,
)
from stable_learning_control.utils.mpi_utils.mpi_tools import (
    mpi_statistics_scalar,
    proc_id,
)
from stable_learning_control.utils.serialization_utils import (
    convert_json,
    load_from_json,
    save_to_json,
)

# Import ray tuner if installed.
ray = lazy_importer(module_name="ray")


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
        quiet=False,
        verbose_fmt=DEFAULT_STD_OUT_TYPE,
        verbose_vars=[],
        save_checkpoints=False,
        backend="torch",
        output_dir_exists_warning=True,
        use_wandb=False,
        wandb_job_type=None,
        wandb_project=None,
        wandb_group=None,
        wandb_run_name=None,
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
            quiet (bool, optional): Whether you want to suppress logging of the
                diagnostics to the stdout. Defaults to ``False``.
            verbose_fmt (str, optional): The format in which the diagnostics are
                displayed to the terminal. Options are ``tab`` which supplies them as a
                table and ``line`` which prints them in one line. Default is set in the
                :mod:`~stable_learning_control.user_config` file.
            verbose_vars (list, optional): A list of variables you want to log to the
                stdout. By default all variables are logged.
            save_checkpoints (bool, optional): Save checkpoints during training.
                Defaults to ``False``.
            backend (str, optional): The backend you want to use for writing to
                TensorBoard. Options are: ``tf2`` or ``torch``. Defaults to ``torch``.
            output_dir_exists_warning (bool, optional): Whether to print a warning
                when the output directory already exists. Defaults to ``True``.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging.
                Defaults to ``False``.
            wandb_job_type (str, optional): The Weights & Biases job type. Defaults to
                ``None``.
            wandb_project (str, optional): The name of the Weights & Biases
                project. Defaults to ``None`` which means that the project name is
                automatically generated.
            wandb_group (str, optional): The name of the Weights & Biases group you want
                to assign the run to. Defaults to ``None``.
            wandb_run_name (str, optional): The name of the Weights & Biases run.
                Defaults to ``None`` which means that the run name is automatically
                generated.

        Attributes:
            tb_writer (torch.utils.tensorboard.writer.SummaryWriter): A TensorBoard
                writer. This is only created when you log a variable to TensorBoard or
                set the :attr:`Logger.use_tensorboard` variable to ``True``.
            output_dir (str): The directory in which the log data and models are saved.
            output_file (str): The name of the file in which the progress data is saved.
            exp_name (str): Experiment name.
            wandb (wandb): A Weights & Biases object. This is only created when you set
                the :attr:`Logger.use_wandb` variable to ``True``.
        """
        if proc_id() == 0:
            # Parse output_fname to see if csv was requested.
            self._output_file_extension = osp.splitext(output_fname)[1]
            self._output_file_sep = (
                "," if self._output_file_extension.lower() == ".csv" else "\t"
            )

            self.output_dir = str(
                Path(output_dir or "/tmp/experiments/%i" % int(time.time())).resolve()
            )
            if osp.exists(self.output_dir):
                if output_dir_exists_warning:
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
            self.log("Logging data to '%s'." % self.output_file.name, type="info")
            self.quiet = quiet
            self.verbose_table = verbose_fmt.lower() != "line"
            self.verbose_vars = [
                re.sub(r"avg|Avg|AVG", "Average", var) for var in verbose_vars
            ]
        else:
            self.output_dir = None
            self.output_file = None
            self.quiet = None
            self.verbose_table = None
            self.verbose_vars = None
        self.exp_name = exp_name
        self._first_row = True
        self._log_headers = []
        self._log_current_row = {}
        self._last_metrics = None
        self._save_checkpoints = save_checkpoints
        self._checkpoint = 0
        self._save_info_saved = False
        self._use_tensorboard = None
        self._tf = None
        self.wandb = None
        self._config = None

        # Training output files paths storage variables.
        self._config_file_path = None
        self._output_file_path = None
        self._state_path = None
        self._state_checkpoints_dir_path = None
        self._save_info_path = None
        self._model_path = None
        self._model_checkpoints_dir_path = None

        self._use_tf_backend = backend.lower() == "tf2"
        self.tb_writer = None
        self._tabular_to_tb_dict = (
            dict()
        )  # Stores if tabular is logged to TensorBoard when dump_tabular is called.
        self._step_count_dict = (
            dict()
        )  # Used for keeping count of the current global step.

        # Setup Weights & Biases if requested.
        if use_wandb:
            import wandb

            self.wandb = wandb
            wandb.init(
                job_type=wandb_job_type,
                dir=self.output_dir,
                project=wandb_project,
                group=wandb_group,
                name=wandb_run_name,
            )

            # Initialize artifacts.
            self._wandb_artifacts = dict()
            self._wandb_artifacts["config"] = self.wandb.Artifact(
                name="config",
                type="metadata",
                description="Contains training configuration data.",
            )
            self._wandb_artifacts["progress"] = self.wandb.Artifact(
                name="progress",
                type="diagnostics",
                description="Contains training diagnostics.",
            )
            self._wandb_artifacts["state"] = self.wandb.Artifact(
                name="state",
                type="state",
                description="Contains training state data.",
            )
            self._wandb_artifacts["model"] = self.wandb.Artifact(
                name="model", type="model", description="Contains model related data."
            )

            # Ensure training artifacts are uploaded to WandB at the end of the run.
            atexit.register(self._log_wandb_artifacts)

            # Ensure hparams and final metrics are logged if backend is PyTorch.
            if not self._use_tf_backend:
                atexit.register(self._log_tb_hparams)

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
        """Log a value to TensorBoard.

        Args:
            key (str):  The name of the diagnostic.
            val (object): A value for the diagnostic.
            tb_prefix(str, optional): A prefix which can be added to group the
                variables.
            tb_alias (str, optional): A tb alias for the variable you want to store
                store. Defaults to empty :obj:`dict`. If not supplied the variable name
                is used.
            global_step (int, optional): Global step value to record. Uses internal step
                counter if global step is not supplied.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists.
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
        TensorBoard and ``stdout`` (otherwise they will not get saved anywhere).

        Args:
            key (str): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                :meth:`EpochLogger.store`, the key here has to match the key you used
                there.
            val (object): A value for the diagnostic. If you have previously saved
                values for this key via :meth:`EpochLogger.store`, do *not* provide a
                ``val`` here.
            tb_write (bool, optional): Boolean specifying whether you also want to write
                the value to the TensorBoard logfile. Defaults to ``False``.
            tb_metrics (Union[list[str], str], optional): List containing the metrics
                you want to write to TensorBoard. Options are [``avg``, ``std``,
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

    def dump_tabular(
        self,
        global_step=None,
    ):
        """Write all of the diagnostics from the current iteration.

        Writes both to ``stdout``, TensorBoard and to the output file.

        Args:
            global_step (int, optional): Global step value to record. Uses internal step
                counter if global step is not supplied.
        """
        if proc_id() == 0:
            vals = []
            print_dict = {}
            print_keys = []
            print_vals = []

            # Retrieve data and max value length from current row.
            val_lens = [len(str(val)) for val in self._log_current_row.values()]
            max_val_len = max(8, max(val_lens))
            for key in self._log_headers:
                val = self._log_current_row.get(key, "")
                if isinstance(val, str):
                    valstr = str(val).rjust(max_val_len)
                else:
                    valstr = (
                        f"%{max_val_len}.3g" if self.verbose_table else "%.3g"
                    ) % val
                print_keys.append(key)
                print_vals.append(valstr)
                print_dict[key] = valstr
                vals.append(val)

            # Log to stdout.
            if not self.quiet:
                if self.verbose_vars:
                    filtered_keys = self.verbose_vars

                    # Make sure Epoch and EnvInteract are always shown if present.
                    for item in reversed(["Epoch", "TotalEnvInteracts"]):
                        if item not in filtered_keys and item in print_keys:
                            filtered_keys.insert(0, item)
                else:
                    filtered_keys = print_keys
                if self.verbose_table:
                    key_lens = [len(key) for key in self._log_headers]
                    max_key_len = max(15, max(key_lens))
                    keystr = "%" + "%d" % max_key_len
                    fmt = "| " + keystr + "s | %s |"
                    n_slashes = max_val_len + max_key_len + 7
                    self.log("-" * n_slashes)
                    print_str = "\n".join(
                        [
                            fmt % (key, val)
                            for key, val in zip(print_keys, print_vals)
                            if key in filtered_keys
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
                            if key in filtered_keys
                        ]
                    )
                    self.log(print_str)
            else:  # Increase epoch steps and time on the same line.
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

            # Log to file.
            if self.output_file is not None:
                if self._first_row:
                    self.output_file.write(
                        f"{self._output_file_sep}".join(self._log_headers) + "\n"
                    )
                self.output_file.write(
                    f"{self._output_file_sep}".join(map(str, vals)) + "\n"
                )
                self.output_file.flush()
                self._output_file_path = self.output_file.name

            # Write tabular to TensorBoard log.
            for key in self._log_headers:
                if self._tabular_to_tb_dict[key]["tb_write"]:
                    val = self._log_current_row.get(key, "")
                    # Use internal counter if global_step is None.
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

            # Log to Ray tune if available.
            if (
                ray is not None
                and ray.is_initialized()
                and ray.tune.is_session_enabled()
            ):
                ray.tune.report(**self._log_current_row)

            # Log to Weights & Biases if available.
            if self.wandb is not None:
                filtered_log_current_row = {
                    k: v
                    for k, v in self._log_current_row.items()
                    if not isinstance(v, str)
                }  # Weights & Biases doesn't support string values.
                self.wandb.log(filtered_log_current_row)

        # Store last metrics.
        self._last_metrics = {
            k: v for k, v in self._log_current_row.items() if k in TB_HPARAMS_METRICS
        }

        # Clear the current row.
        self._log_current_row.clear()
        self._first_row = False

    def get_logdir(self, *args, **kwargs):
        """Get Logger and TensorBoard SummaryWriter logdirs.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Returns:
            (dict): dict containing:

                - output_dir(:obj:`str`): Logger output directory.
                - tb_output_dir(:obj:`str`): TensorBoard writer output directory.
        """
        log_dir = {"output_dir": self.output_dir}
        if self._use_tensorboard:
            log_dir["tb_output_dir"] = self.tb_writer.get_logdir(*args, **kwargs)
        if self.wandb is not None:
            log_dir["wandb_output_dir"] = self.wandb.run.dir
        return log_dir

    def save_config(self, config):
        """Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). The resulting JSON will be saved
        to the output directory, made available to TensorBoard and Weights & Biases and
        printed to stdout.

        Example:
            .. code-block:: python

                logger = EpochLogger(**logger_kwargs)
                logger.save_config(locals())

        Args:
            config (object): Configuration Python object you want to save.
        """
        if proc_id() == 0:
            config = parse_config_env_key(config)
            self._config = config
            config_json = convert_json(config)
            if self.exp_name is not None:
                config_json["exp_name"] = self.exp_name
            output = json.dumps(
                config_json, separators=(",", ":\t"), indent=4, sort_keys=True
            )
            config_file = osp.join(self.output_dir, "config.json")
            self.log("Saving config to '%s'." % config_file, type="info")
            if PRINT_CONFIG:
                self.log("\nconfig:\n")
                self.log(output)
            with open(config_file, "w") as out:
                out.write(output)
            self._config_file_path = config_file

            # Update Weights & Biases config.
            if self.wandb is not None:
                self.wandb.config.update(self._wandb_config)

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
            (:obj:`gym.env`): The gymnasium environment.
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
        # try to load environment from save.
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
                :attr:`Logger.output_dir`  is used.
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
                to ``None``.
        """
        if proc_id() == 0:
            # Save training state (environment, ...)
            try:
                state_path = osp.join(self.output_dir, "vars.pkl")
                joblib.dump(state_dict, state_path)
                self._state_path = state_path
            except (ValueError, pickle.PicklingError):
                self.log("Warning: could not pickle state_dict.", color="red")

            # Save model state.
            if hasattr(self, "_tf_saver_elements"):
                backend_folder_name = "tf2_save"
                self._tf_save(itr)
            if hasattr(self, "_pytorch_saver_elements"):
                backend_folder_name = "torch_save"
                self._pytorch_save(itr)

            # Save checkpoint state.
            if self._save_checkpoints and itr is not None:
                itr_name = (
                    "iter%d" % itr
                    if itr is not None
                    else "iter" + str(self._checkpoint)
                )
                fdir = osp.join(self.output_dir, backend_folder_name, "checkpoints")
                fpath = osp.join(fdir, str(itr_name))
                fname = osp.join(fpath, "vars.pkl")
                os.makedirs(fpath, exist_ok=True)
                try:
                    joblib.dump(state_dict, fname)
                    self._state_checkpoints_dir_path = fdir
                except (ValueError, pickle.PicklingError):
                    self.log("Warning: could not pickle state_dict.", color="red")

    def setup_tf_saver(self, what_to_save):
        """Set up easy model saving for a single Tensorlow model.

        Args:
            what_to_save (object): Any Tensorflow model or serializable object
                containing TensorFlow models.
        """
        self._tf = import_tf()  # Throw custom warning if tf is not installed.
        self._tf_saver_elements = what_to_save
        self.log("Policy will be saved to '{}'.\n".format(self.output_dir), type="info")

    def setup_pytorch_saver(self, what_to_save):
        """Set up easy model saving for a single PyTorch model.

        Args:
            what_to_save (object): Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self._pytorch_saver_elements = what_to_save
        self.log("Policy will be saved to '{}'.\n".format(self.output_dir), type="info")

    def _tf_save(self, itr=None):
        """Saves the PyTorch model/models using their ``state_dict``.

        Args:
            itr (Union[int, None]): Current iteration of training (e.g. epoch). Defaults
                to ``None``.
        """
        if proc_id() == 0:
            save_fail_warning = (
                "The object you tried to save doesn't have a 'save_weights' we "
                "can use to retrieve the model weights. Please make sure you supplied "
                "the 'setup_tf_saver' method with a valid 'tf.keras.Model' object "
                "or implemented a 'save_weights' method on your object.",
            )

            assert hasattr(
                self, "_tf_saver_elements"
            ), "First have to setup saving with self.setup_tf_saver"

            # Create filename.
            fpath = osp.join(self.output_dir, "tf2_save")
            fname = osp.join(fpath, "weights_checkpoint")
            os.makedirs(fpath, exist_ok=True)

            # Create Checkpoints name.
            if self._save_checkpoints and itr is not None:
                itr_name = (
                    "iter%d" % itr
                    if itr is not None
                    else "iter" + str(self._checkpoint)
                )
                cdir = osp.join(fpath, "checkpoints")
                cpath = osp.join(cdir, str(itr_name))
                cname = osp.join(cpath, "weights_checkpoint")
                os.makedirs(cpath, exist_ok=True)

            # Save additional algorithm information.
            if not self._save_info_saved:
                save_info = {
                    "alg_name": self._tf_saver_elements.__class__.__name__,
                }
                if hasattr(self._tf_saver_elements, "_setup_kwargs"):
                    save_info["setup_kwargs"] = self._tf_saver_elements._setup_kwargs
                self.save_to_json(
                    save_info,
                    output_filename="save_info.json",
                    output_path=fpath,
                )
                self._save_info_saved = True
                self._save_info_path = osp.join(fpath, "save_info.json")

            # Save model.
            if isinstance(self._tf_saver_elements, self._tf.keras.Model) or hasattr(
                self._tf_saver_elements, "save_weights"
            ):
                self._tf_saver_elements.save_weights(fname)
                self._model_path = fname
            else:
                self.log(save_fail_warning, type="warning")

            # Save checkpoint.
            if self._save_checkpoints and itr is not None:
                if isinstance(self._tf_saver_elements, self._tf.keras.Model) or hasattr(
                    self._tf_saver_elements, "save_weights"
                ):
                    self._tf_saver_elements.save_weights(cname)
                    self._model_checkpoints_dir_path = cdir
                else:
                    self.log(save_fail_warning, type="warning")

                self._checkpoint += 1  # Increase epoch.

    def _pytorch_save(self, itr=None):
        """Saves the PyTorch model/models using their ``state_dict``.

        Args:
            itr (Union[int, None]): Current iteration of training (e.g. epoch). Defaults
                to ``None``.
        """
        if proc_id() == 0:
            save_fail_warning = (
                "The object you tried to save doesn't have a 'state_dict' we can"
                "use to retrieve the model weights. Please make sure you supplied the "
                "'setup_pytorch_saver' method with a valid 'torch.nn.Module' object or "
                "implemented a'state_dict' method on your object."
            )

            assert hasattr(
                self, "_pytorch_saver_elements"
            ), "First have to setup saving with self.setup_pytorch_saver"

            # Create filename.
            fpath = osp.join(self.output_dir, "torch_save")
            fname = osp.join(fpath, "model_state.pt")
            os.makedirs(fpath, exist_ok=True)

            # Create Checkpoints Name.
            if self._save_checkpoints and itr is not None:
                itr_name = (
                    "iter%d" % itr
                    if itr is not None
                    else "iter" + str(self._checkpoint)
                )
                cdir = osp.join(fpath, "checkpoints")
                cpath = osp.join(cdir, str(itr_name))
                cname = osp.join(cpath, "model_state.pt")
                os.makedirs(cpath, exist_ok=True)

            # Save additional algorithm information.
            if not self._save_info_saved:
                save_info = {
                    "alg_name": self._pytorch_saver_elements.__class__.__name__,
                }
                if hasattr(self._pytorch_saver_elements, "_setup_kwargs"):
                    save_info[
                        "setup_kwargs"
                    ] = self._pytorch_saver_elements._setup_kwargs
                self.save_to_json(
                    save_info,
                    output_filename="save_info.json",
                    output_path=fpath,
                )
                self._save_info_saved = True
                self._save_info_path = osp.join(fpath, "save_info.json")

            # Save model.
            if isinstance(self._pytorch_saver_elements, torch.nn.Module) or hasattr(
                self._pytorch_saver_elements, "state_dict"
            ):
                torch.save(self._pytorch_saver_elements.state_dict(), fname)
                self._model_path = fname
            else:
                self.log(save_fail_warning, type="warning")

            # Save checkpoint.
            if self._save_checkpoints:
                if isinstance(self._pytorch_saver_elements, torch.nn.Module) or hasattr(
                    self._pytorch_saver_elements, "state_dict"
                ):
                    torch.save(self._pytorch_saver_elements.state_dict(), cname)
                    self._model_checkpoints_dir_path = cdir
                else:
                    self.log(save_fail_warning, type="warning")

                self._checkpoint += 1  # Increase epoch.

    def _write_to_tb(self, var_name, data, global_step=None):
        """Writes data to TensorBoard log file.

        It currently works with scalars, histograms and images. For other data types
        please use :attr:`Logger.tb_writer`. directly.

        Args:
            var_name (str): Data identifier.
            data (Union[int, float, numpy.ndarray, torch.Tensor, tf.Tensor]): Data you
                want to write.
            global_step (int, optional): Global step value to record. Uses internal step
                counter if global step is not supplied.
        """
        # Try to write data to tb as as histogram.
        if not self.tb_writer:
            self.use_tensorboard = (
                True  # Property that creates tf writer if set to True.
            )
        if is_scalar(data):  # Extra protection since trying to write a list freezes tb.
            try:  # Try to write as scalar.
                self.add_tb_scalar(var_name, data, global_step=global_step)
            except (
                AssertionError,
                NotImplementedError,
                NameError,
                ValueError,
                TypeError,
            ):
                pass
        # NOTE: The options below are not yet used in the SLC but were added to show how
        # the logger can be extended to write other data types to TensorBoard.
        else:
            # Try to write data to tb as as histogram.
            try:
                self.add_tb_histogram(var_name, data, global_step=global_step)
            except (
                AssertionError,
                NotImplementedError,
                NameError,
                ValueError,
                TypeError,
                ImportError,
            ):
                pass

            # Try to write data as image.
            try:
                self.add_tb_image(var_name, data, global_step=global_step)
            except (
                AssertionError,
                NotImplementedError,
                NameError,
                ValueError,
                TypeError,
                ImportError,
            ):
                pass

            # Try to write data as text.
            # NOTE: This should be the last option as the other objects can also contain
            # caffe2 identifier strings.
            try:
                self.add_tb_text(var_name, data, global_step=global_step)
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
                        + "not be written to TensorBoard as the '{}' ".format(
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
        """Variable specifying whether the logger uses TensorBoard. A TensorBoard writer
        is created on the :attr:`Logger.tb_writer` attribute when
        :attr:`~Logger.use_tensorboard` is set to ``True``
        """
        return self._use_tensorboard

    @use_tensorboard.setter
    def use_tensorboard(self, value):
        """Custom setter that makes sure a TensorBoard writer is present on the
        :attr:`Logger.tb_writer` attribute when ``use_tensorboard`` is set to ``True``
        . This TensorBoard writer can be used to write to the TensorBoard.

        Args:
            value (bool): Whether you want to use TensorBoard logging.
        """
        self._use_tensorboard = value

        # Create TensorBoard writer if use_tensorboard == True else delete.
        if self._use_tensorboard and not self.tb_writer:  # Create writer object.
            if self._use_tf_backend:
                self.log("Using TensorFlow as the TensorBoard backend.", type="info")
                self._tf = import_tf()  # Throw custom warning if tf is not installed.
                self.tb_writer = self._tf.summary.create_file_writer(self.output_dir)
            else:
                self.log(
                    "Using Torch.utils.tensorboard as the TensorBoard backend.",
                    type="info",
                )
                exp_name = "-" + self.exp_name if self.exp_name else ""
                self.tb_writer = SummaryWriter(
                    log_dir=self.output_dir,
                    comment=f"{exp_name.upper()}-data_"
                    + time.strftime("%Y%m%d-%H%M%S"),
                )
                atexit.register(self.flush_tb_writer)  # Make sure all data is written.
                atexit.register(self.tb_writer.close)  # Make sure the writer is closed.

            # Add config TensorBoard.
            if self._config:
                self.add_tb_text(
                    "Config", dict_to_mdtable(self._tb_config), global_step=0
                )
            else:
                self.log(
                    "No hyper parameters found. Please ensure that you have called "
                    "the 'save_config' method of the 'ExperimentLogger' class "
                    "if you want to have the used hyper parameters written to ",
                    "TensorBoard.",
                    type="warning",
                )
        elif not self._use_tensorboard and self.tb_writer:  # Delete TensorBoard writer.
            self.flush_tb_writer()  # Make sure all data is written.
            self.tb_writer.close()  # Close writer.
            atexit.unregister(self.flush_tb_writer)  # Make sure the writer is closed.
            atexit.unregister(self.tb_writer.close)  # Make sure the writer is closed.
            self.tb_writer = None

    @property
    def log_current_row(self):
        """Return the current row of the logger."""
        log_current_row = copy.deepcopy(self._log_current_row)
        return log_current_row

    @property
    def _global_step(self):
        """Retrieve the current estimated global step count."""
        return max(list(self._step_count_dict.values()) + [0])

    """
    Weights & Biases related methods.
    """

    @property
    def _wandb_config(self):
        """Transform the config to a format that looks better on Weights & Biases."""
        if self.wandb and self._config:
            return convert_to_wandb_config(self._config)
        return None

    def watch_model_in_wandb(self, model):
        """Watch model parameters in Weights & Biases.

        Args:
            model (torch.nn.Module): Model to watch on Weights & Biases.
        """
        if self.wandb:
            try:
                self.wandb.watch(model)
            except ValueError:
                self.log(
                    (
                        "WARNING: Model could not be watched on Weights & Biases as "
                        f"the '{type(model)}' class is not yet supported."
                    ),
                    type="warning",
                )

    def _log_wandb_artifacts(self):
        """Log all stored artifacts to Weights & Biases."""
        if self.wandb:
            # Log configuration artifacts
            if self._config_file_path and os.path.isfile(self._config_file_path):
                self._wandb_artifacts["config"].add_file(self._config_file_path)

            # Log progress artifacts.
            if self._output_file_path and os.path.isfile(self._output_file_path):
                self._wandb_artifacts["progress"].add_file(self._output_file_path)

            # Log training state artifacts.
            if self._state_path and os.path.isfile(self._state_path):
                self._wandb_artifacts["state"].add_file(self._state_path)

            # Log training state checkpoints artifacts.
            if self._state_checkpoints_dir_path and os.path.isdir(
                self._state_checkpoints_dir_path
            ):
                self._wandb_artifacts["state"].add_dir(self._state_checkpoints_dir_path)

            # Log model artifacts.
            if self._model_path and os.path.isfile(self._model_path):
                self._wandb_artifacts["model"].add_file(self._model_path)

            # Log model checkpoints artifacts.
            if self._model_checkpoints_dir_path and os.path.isdir(
                self._model_checkpoints_dir_path
            ):
                self._wandb_artifacts["model"].add_dir(self._model_checkpoints_dir_path)

            # Log all artifacts to Weights & Biases.
            for artifact in self._wandb_artifacts.values():
                self.wandb.log_artifact(artifact)

    """TensorBoard related methods
    Below are several wrapper methods which make the TensorBoard writer methods
    available directly on the :class:`EpochLogger` class.
    """

    @property
    def _tb_config(self):
        """Modify the config to a format that looks better on Tensorboard."""
        if self.use_tensorboard and self._config:
            return convert_to_tb_config(self._config)
        return None

    @property
    def _tb_hparams(self):
        """Transform the config to a format that is accepted as hyper parameters by
        TensorBoard.
        """
        if self.use_tensorboard and self._config:
            tb_config = {}
            for key, value in self._tb_config.items():
                if key not in TB_HPARAMS_FILTER and value is not None:
                    if isinstance(value, type):
                        value = value.__module__ + "." + value.__name__
                    elif not isinstance(value, (bool, str, float, int, torch.Tensor)):
                        value = str(value)
                    tb_config[key] = value
            return tb_config
        return None

    def _log_tb_hparams(self):
        """Log hyper parameters together with final metrics to TensorBoard."""
        if self.use_tensorboard and self._tb_hparams and self._last_metrics:
            # Log hyper parameters and metrics.
            self.add_tb_hparams(self._tb_hparams, self._last_metrics)

    def log_model_to_tb(self, model, input_to_model=None, *args, **kwargs):
        """Add model to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_graph`
        or :obj:`tf.summary.graph` (depending on the backend) method while first making
        sure a SummaryWriter object exits.

        Args:
            model (union[torch.nn.Module, tf.keras.Model]): Model to add to the summary.
            input_to_model (union[torch.Tensor, tf.Tensor]): Input tensor to the model.
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        if self._use_tf_backend:
            self.use_tensorboard = True  # Make sure SummaryWriter exists.
            with self.tb_writer.as_default():
                kwargs["step"] = kwargs.pop("global_step", self._global_step)
                self._tf.summary.trace_on(graph=True)
                model(input_to_model)
                self._tf.summary.trace_export(
                    "model",
                    **kwargs,
                )
        else:
            self.add_tb_graph(model, input_to_model=input_to_model, *args, **kwargs)

    def add_tb_scalar(self, *args, **kwargs):
        """Add scalar to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_scalar`
        or :obj:`tf.summary.scalar` (depending on the backend) method while first making
        sure a SummaryWriter object exits. These methods can be used to add a scalar
        to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists.
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step", self._global_step)
            with self.tb_writer.as_default():
                return self._tf.summary.scalar(*args, **kwargs)
        else:
            return self.tb_writer.add_scalar(*args, **kwargs)

    def tb_scalar(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_tb_scalar` method available directly under
        the ``scalar`` alias.

        Args:
            *args: All args to pass to the :meth:`add_tb_scalar` method.
            **kwargs: All kwargs to pass to the :meth:`add_tb_scalar` method.
        """
        self.add_tb_scalar(*args, **kwargs)

    def add_tb_histogram(self, *args, **kwargs):
        """Add histogram to tb summary.

        Wrapper that calls the
        :obj:`torch.utils.tensorboard.SummaryWriter.add_histogram` or
        :obj:`tf.summary.histogram` (depending on the backend) method while first making
        sure a SummaryWriter object exits. These methods can be used to add a histogram
        to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists.
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step", self._global_step)
            with self.tb_writer.as_default():
                return self._tf.summary.histogram(*args, **kwargs)
        else:
            return self.tb_writer.add_histogram(*args, **kwargs)

    def tb_histogram(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_tb_histogram` method available directly
        under the ``tb_histogram`` alias.

        Args:
            *args: All args to pass to the :meth:`add_tb_histogram` method.
            **kwargs: All kwargs to pass to the :meth:`add_tb_histogram` method.
        """
        self.add_tb_histogram(*args, **kwargs)

    def add_tb_image(self, *args, **kwargs):
        """Add image to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_image`
        or :obj:`tf.summary.image` (depending on the backend) method while first making
        sure a SummaryWriter object exits. These methods can be used to add a image
        to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists.
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step", self._global_step)
            with self.tb_writer.as_default():
                return self._tf.summary.image(*args, **kwargs)
        else:
            return self.tb_writer.add_image(*args, **kwargs)

    def tb_image(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_tb_image` method available directly under
        the ``tb_image`` alias.

        Args:
            *args: All args to pass to the :meth:`add_tb_image` method.
            **kwargs: All kwargs to pass to the :meth:`add_tb_image` method.
        """
        self.add_tb_image(*args, **kwargs)

    def add_tb_text(self, *args, **kwargs):
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
        self.use_tensorboard = True  # Make sure SummaryWriter exists.
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step", self._global_step)
            with self.tb_writer.as_default():
                return self._tf.summary.text(*args, **kwargs)
        else:
            return self.tb_writer.add_text(*args, **kwargs)

    def tb_text(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_tb_text` method available directly under
        the ``text`` alias.

        Args:
            *args: All args to pass to the :meth:`add_tb_text` method.
            **kwargs: All kwargs to pass to the :meth:`add_tb_text` method.
        """
        self.add_tb_text(*args, **kwargs)

    def add_tb_graph(self, *args, **kwargs):
        """Add graph to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_graph`
        or :obj:`tf.summary.add_graph` (depending on the backend) method while first
        making sure a SummaryWriter object exits. These methods can be used to add a
        graph to the summary.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.
        """
        self.use_tensorboard = True
        if self._use_tf_backend:
            kwargs["step"] = kwargs.pop("global_step", self._global_step)
            with self.tb_writer.as_default():
                return self._tf.summary.graph(*args, **kwargs)
        else:
            return self.tb_writer.add_graph(*args, **kwargs)

    def tb_graph(self, *args, **kwargs):
        """Wrapper for making the :meth:`add_tb_graph` method available directly under
        the ``tb_graph`` alias.

        Args:
            *args: All args to pass to the :meth:`add_tb_graph` method.
            **kwargs: All kwargs to pass to the :meth:`add_tb_graph` method.
        """
        self.add_tb_graph(*args, **kwargs)

    def add_tb_hparams(self, *args, **kwargs):
        """Adds hyper parameters to tb summary.

        Wrapper that calls the :obj:`torch.utils.tensorboard.SummaryWriter.add_hparams`
        method while first making sure a SummaryWriter object exits. The ``add_hparams``
        method adds a set of hyperparameters to be compared in TensorBoard.

        Args:
            *args: All args to pass to the Summary/SummaryWriter object.
            **kwargs: All kwargs to pass to the Summary/SummaryWriter object.

        Raises:
            NotImplementedError: Raised if you try to call this method when using the
                TensorFlow backend.
        """
        self.use_tensorboard = True  # Make sure SummaryWriter exists.
        if self._use_tf_backend:
            raise NotImplementedError(
                "The 'add_tb_hparams' method is not available when using the "
                "'tensorflow' backend."
            )
        return self.tb_writer.add_hparams(*args, **kwargs)

    def flush_tb_writer(self, *args, **kwargs):
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
        self.use_tensorboard = True  # Make sure SummaryWriter exists.
        if self._use_tf_backend:
            with self.tb_writer.as_default():
                return self._tf.summary.flush(*args, **kwargs)
        else:
            return self.tb_writer.flush(*args, **kwargs)


class EpochLogger(Logger):
    """
    A variant of :class:`Logger` tailored for tracking average values over epochs.

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
           :class:`EpochLogger` current state.
    """

    def __init__(self, *args, **kwargs):
        """Initialise a EpochLogger."""
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
        """Save something into the :class:`EpochLogger`'s current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.

        Args:
            tb_write (Union[bool, dict], optional): Boolean or dict of key boolean pairs
                specifying whether you also want to write the value to the TensorBoard
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
            # Store variable values in epoch_dict and increase global step count.
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
                self._step_count_dict[k] = 0
            if extend:
                self.epoch_dict[k].extend(v)
            else:
                self.epoch_dict[k].append(v)

            # Increase the step count for all the keys.
            # NOTE: This is done in such a way that two values of a given key do not
            # get the same global step value assigned to them.
            self._step_count_dict[k] = (
                self._step_count_dict[k] + 1
                if self._step_count_dict[k] + 1 >= self._global_step
                else self._global_step
            )

            # Check if a alias was given for the current parameter.
            var_name = k if k not in tb_aliases.keys() else tb_aliases[k]

            # Write variable value to TensorBoard.
            tb_write_key = (
                (tb_write[k] if k in tb_write.keys() else False)
                if isinstance(tb_write, dict)
                else tb_write
            )
            if tb_write_key:
                global_step = (
                    global_step if global_step is not None else self._global_step
                )  # Use internal counter if global_step is None.
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
        """Log a diagnostic to TensorBoard. This function takes or a list of keys or a
        key-value pair. If only keys are supplied, averages will be calculated using
        the new data found in the Loggers internal storage. If a key-value pair is
        supplied, this pair will be directly logged to TensorBoard.

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
        if val is not None:  # When key and value are supplied use direct write.
            super().log_to_tb(
                keys,
                val,
                tb_prefix=tb_prefix,
                tb_alias=tb_alias,
                global_step=global_step,
            )
        else:  # When only keys are supplied use internal storage.
            keys = [keys] if not isinstance(keys, list) else keys
            for key in keys:
                if global_step is None:  # Retrieve global step if not supplied.
                    if self._n_table_dumps >= 1:
                        global_step_tmp = self._global_step
                    elif key in self.epoch_dict.keys():
                        global_step_tmp = len(self.epoch_dict[key])
                else:
                    global_step_tmp = global_step

                self._log_tb_diagnostics(
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
                the value to the TensorBoard logfile. Defaults to False.
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
        """Small wrapper around the :meth:`Logger.dump_tabular` method which
        makes sure that the TensorBoard index track dictionary is reset after the table
        is dumped.

        Args:
            *args: All args to pass to parent method.
            **kwargs: All kwargs to pass to parent method.
        """
        super().dump_tabular(*args, **kwargs)
        self._n_table_dumps += 1
        self._tb_index_dict = {
            key: 0 for key in self._tb_index_dict.keys()
        }  # Reset TensorBoard logging index storage dict.

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

    def _log_tb_diagnostics(
        self,
        key,
        with_min_and_max=False,
        average_only=False,
        tb_prefix=None,
        tb_alias=None,
        global_step=None,
    ):
        """Calculates the diagnostics of a given key from all the new data found in the
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
