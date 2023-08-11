"""Responsible for creating the CLI for the stable_learning_control package. It can
be used to start the training of an algorithm, or run any of the other utilities.
"""
import os
import os.path as osp
import subprocess
import sys
from copy import deepcopy
from textwrap import dedent
import inspect

import ruamel.yaml as yaml

from stable_learning_control.common.helpers import (
    flatten,
    get_unique_list,
    friendly_err,
)
from stable_learning_control.user_config import DEFAULT_BACKEND
from stable_learning_control.utils.gym_utils import validate_gym_env
from stable_learning_control.utils.import_utils import tf_installed
from stable_learning_control.utils.log_utils.helpers import log_to_std_out
from stable_learning_control.utils.run_utils import ExperimentGrid
from stable_learning_control.utils.safer_eval_util import safer_eval
from stable_learning_control.version import __version__

# Command line args that will go to ExperimentGrid.run, and must possess unique values
# (therefore must be treated separately).
RUN_KEYS = ["num_cpu", "data_dir", "datestamp"]

# Command line sweetener, allowing short-form flags for common, longer flags.
SUBSTITUTIONS = {
    "env": "env_name",
    "env_k": "env_kwargs",
    "hid": "ac_kwargs:hidden_sizes",
    "hid_a": "ac_kwargs:hidden_sizes:actor",
    "hid_c": "ac_kwargs:hidden_sizes:critic",
    "act": "ac_kwargs:activation",
    "act_out": "ac_kwargs:output_activation",
    "act_a": "ac_kwargs:activation:actor",
    "act_out_a": "ac_kwargs:output_activation:actor",
    "act_c": "ac_kwargs:activation:critic",
    "act_out_c": "ac_kwargs:output_activation:critic",
    "cpu": "num_cpu",
    "dt": "datestamp",
    "q": "quiet",
    "use_tb": "logger_kwargs:use_tensorboard",
    "use_wandb": "logger_kwargs:use_wandb",
    "wandb_job_type": "logger_kwargs:wandb_job_type",
    "wandb_proj": "logger_kwargs:wandb_project",
    "wandb_group": "logger_kwargs:wandb_group",
    "wandb_run_name": "logger_kwargs:wandb_run_name",
    "save_cps": "logger_kwargs:save_checkpoints",
    "tb_log_freq": "logger_kwargs:tb_log_freq",
    "quiet": "logger_kwargs:quiet",
    "verbose_fmt": "logger_kwargs:verbose_fmt",
    "verbose_vars": "logger_kwargs:verbose_vars",
}

# Only some algorithms can be parallelized (have num_cpu > 1).
MPI_COMPATIBLE_ALGOS = []

# Algo names (used in a few places).
BASE_ALGO_NAMES = ["sac", "lac", "latc"]


def _parse_hyperparameter_variants(exp_val):
    """Function parses exp config values to make sure that comma/space separated
    strings (i.e. ``5, 3, 2`` or ``5 3 2``)) are recognized as hyperparameter variants.

    Args:
        exp_val (object): The variable to parse.

    Returns:
        union[:obj:`str`, :obj:`list`, :obj:`None`]: A hyper parameter string or list.
            Returns ``None`` if ``exp_val`` is ``None``.
    """
    if exp_val is None:
        return None
    if not isinstance(exp_val, str):
        return str(exp_val)
    else:
        return get_unique_list(exp_val.replace(" ", ",").split(","))


def _parse_exp_cfg(cmd_line_args):
    """This function parses the cmd line args to see if it contains the ``exp_cfg``
    flag. If this flag is present it uses the ``exp_cfg`` file path (next cmd_line arg)
    to add any hyperparameters found in this experimental configuration file to the cmd
    line arguments.

    Args:
        cmd_line_args (list): The cmd line input arguments.

    Returns:
        list: Modified cmd line argument list that also contains any hyperparameters
            that were specified in a experimental cfg file.

    .. note::
        This function assumes comma/space separated strings (i.e. ``5, 3, 2`` or
        ``5 3 2``)) to be hyperparmeter variants.
    """
    if "--exp_cfg" in cmd_line_args:
        exp_cfg_idx = cmd_line_args.index("--exp_cfg")
        cmd_line_args.pop(exp_cfg_idx)  # Remove exp_cfg argument.

        # Validate config path.
        cfg_error = False
        try:
            exp_cfg_file_path = cmd_line_args.pop(exp_cfg_idx)
            if exp_cfg_file_path.startswith("--"):
                raise IndexError
            exp_cfg_file_path = (
                exp_cfg_file_path
                if exp_cfg_file_path.endswith(".yml")
                else exp_cfg_file_path + ".yml"
            )
            if not osp.exists(exp_cfg_file_path):
                project_path = osp.abspath(osp.join(osp.dirname(__file__), ".."))
                exp_cfg_file_path = osp.abspath(
                    osp.join(project_path, "experiments", exp_cfg_file_path)
                )
                if not osp.exists(exp_cfg_file_path):
                    raise FileNotFoundError(
                        "No experiment configuration file found at "
                        f"'{exp_cfg_file_path}'."
                    )
        except (IndexError, FileNotFoundError) as e:
            cfg_error = True
            if isinstance(e, IndexError):
                log_to_std_out(
                    "You did not supply a experiment configuration path. As a result "
                    "the '--exp_cfg' argument has been ignored.",
                    type="warning",
                )
            else:
                log_to_std_out(
                    (
                        e.args[0]
                        + " As a result the '--exp_cfg' argument has been "
                        + "ignored."
                    ),
                    type="warning",
                )

        # Read configuration values.
        if not cfg_error:
            # Load exp config.
            with open(exp_cfg_file_path) as stream:
                try:
                    exp_cfg_params = yaml.safe_load(stream)
                except yaml.YAMLError:
                    log_to_std_out(
                        "Something went wrong while trying to load the experiment  "
                        f"config in '{exp_cfg_file_path}. As a result the --exp_cfg "
                        "argument has been ignored.",
                        type="warning",
                    )

            # Retrieve values from exp config.
            log_to_std_out(
                f"Experiment hyperparameters loaded from '{exp_cfg_file_path}'",
                type="info",
            )
            if not exp_cfg_params:
                log_to_std_out(
                    "No hyperparameters were found in your experiment config. "
                    "As a result the '--exp_cfg' flag has been ignored.",
                    type="warning",
                )
            else:
                # Remove 'alg_name' from exp_cfg if '--exp_cfg' was not the first arg.
                if exp_cfg_idx == 1:
                    if "alg_name" in exp_cfg_params.keys():
                        cmd_line_args.insert(1, exp_cfg_params.pop("alg_name", None))
                else:
                    exp_cfg_params.pop("alg_name")

                # Append cfg hyperparameters to the input arguments.
                # NOTE: Here we assume comma or space separated strings to be variants.
                exp_cfg_params = {
                    (key if key.startswith("--") else "--" + key): val
                    for key, val in exp_cfg_params.items()
                }
                exp_cfg_params = list(
                    flatten(
                        [
                            [str(key), _parse_hyperparameter_variants(val)]
                            for key, val in exp_cfg_params.items()
                        ]
                    )
                )

                # Remove None values so that they are treated as on/off flags (see
                # https://docs.python.org/3/library/argparse.html#core-functionality).
                exp_cfg_params = [val for val in exp_cfg_params if val is not None]
            cmd_line_args.extend(exp_cfg_params)

    return cmd_line_args


def _parse_eval_cfg(cmd_line_args):
    """This function parses the cmd line args to see if it contains the ``eval_cfg``
    flag. If this flag is present it uses the ``eval_cfg`` file path (next cmd_line arg)
    to add any hyperparameters found in this eval configuration file to the cmd
    line arguments.

    Args:
        cmd_line_args (list): The cmd line input arguments.

    Returns:
        list: Modified cmd line argument list that also contains any hyperparameters
            that were specified in a eval cfg file.

    .. note::
        This function assumes comma/space separated strings (i.e. ``5, 3, 2`` or
        ``5 3 2``)) to be hyperparmeter variants.
    """
    if "--eval_cfg" in cmd_line_args:
        # If 'eval_robustness' is not first argument add it to the front.
        if "eval_robustness" not in cmd_line_args:
            cmd_line_args.insert(1, "eval_robustness")
        else:
            eval_robustness_idx = cmd_line_args.index("eval_robustness")
            if eval_robustness_idx != 1:
                cmd_line_args.pop("eval_robustness", None)
                cmd_line_args.insert(1, "eval_robustness")

        # Retrieve '--eval_cfg' argument.
        eval_cfg_idx = cmd_line_args.index("--eval_cfg")
        cmd_line_args.pop(eval_cfg_idx)  # Remove eval_cfg argument.

        # Validate config path.
        cfg_error = False
        try:
            eval_cfg_file_path = cmd_line_args.pop(eval_cfg_idx)
            if eval_cfg_file_path.startswith("--"):
                raise IndexError
            eval_cfg_file_path = (
                eval_cfg_file_path
                if eval_cfg_file_path.endswith(".yml")
                else eval_cfg_file_path + ".yml"
            )
            if not osp.exists(eval_cfg_file_path):
                project_path = osp.abspath(osp.join(osp.dirname(__file__), ".."))
                eval_cfg_file_path = osp.abspath(
                    osp.join(project_path, "experiments", eval_cfg_file_path)
                )
                if not osp.exists(eval_cfg_file_path):
                    raise FileNotFoundError(
                        "No eval configuration file found at "
                        f"'{eval_cfg_file_path}'."
                    )
        except (IndexError, FileNotFoundError) as e:
            cfg_error = True
            if isinstance(e, IndexError):
                log_to_std_out(
                    "You did not supply a eval configuration path. As a result "
                    "the '--eval_cfg' argument has been ignored.",
                    type="warning",
                )
            else:
                log_to_std_out(
                    (
                        e.args[0]
                        + " As a result the '--eval_cfg' argument has been "
                        + "ignored."
                    ),
                    type="warning",
                )

        # Read configuration values.
        if not cfg_error:
            # Load eval config.
            with open(eval_cfg_file_path) as stream:
                try:
                    eval_cfg_params = yaml.safe_load(stream)
                except yaml.YAMLError:
                    log_to_std_out(
                        "Something went wrong while trying to load the eval config "
                        f"in '{eval_cfg_file_path}. As a result the --eval_cfg "
                        "argument has been ignored.",
                        type="warning",
                    )

            # Retrieve values from eval config.
            log_to_std_out(
                f"Eval parameters loaded from '{eval_cfg_file_path}'",
                type="info",
            )
            if not eval_cfg_params:
                log_to_std_out(
                    "No eval parameters were found in your eval config. As a result "
                    "the '--eval_cfg' flag has been ignored.",
                    type="warning",
                )
            else:
                # Ensure that the eval config contains a disturber key.
                if "disturber" not in eval_cfg_params.keys():
                    raise ValueError(
                        "The eval config does not contain a 'disturber' key."
                    )

                # Remove the disturber key and extend the cmd line arguments with it.
                disturber = eval_cfg_params.pop("disturber")
                cmd_line_args.append(disturber)

                # Append other cfg parameters to the input arguments.
                # NOTE: Here we assume comma or space separated strings to be variants.
                eval_cfg_params = {
                    (key if key.startswith("--") else "--" + key): val
                    for key, val in eval_cfg_params.items()
                }
                eval_cfg_params = list(
                    flatten(
                        [
                            [str(key), _parse_hyperparameter_variants(val)]
                            for key, val in eval_cfg_params.items()
                        ]
                    )
                )

                # Remove None values so that they are treated as on/off flags (see
                # https://docs.python.org/3/library/argparse.html#core-functionality).
                eval_cfg_params = [val for val in eval_cfg_params if val is not None]
            cmd_line_args.extend(eval_cfg_params)

    return cmd_line_args


def _add_backend_to_cmd(cmd):
    """Adds the backend suffix to the input command.

    Args:
        cmd (str): The cmd string.

    Returns:
        (tuple): tuple containing:

            - cmd (:obj:`str`): The new cmd.
            - backend (:obj:`str`): The used backend (options: ``tf2`` or ``pytorch``).

    Raises:
        AssertError:
            Raised when a the TensorFlow backend is requested but TensorFlow is not
            installed.
    """
    if cmd in BASE_ALGO_NAMES:
        backend = DEFAULT_BACKEND[cmd]
        print("\n\nUsing default backend (%s) for %s.\n" % (backend, cmd))
        cmd = cmd + "_" + backend
    else:
        backend = cmd.split("_")[-1]

    # Throw error if tf algorithm is requested but TensorFlow is not installed.
    if backend.lower() == "tf2":
        if not tf_installed():
            raise AssertionError(
                friendly_err(
                    "You requested a TensorFlow algorithm but TensorFlow is not "
                    "installed. Did you run the `pip install .[tf2]` command?"
                )
            )
    return cmd, backend


def _process_arg(arg, backend=None):
    """Process an arg by eval-ing it, so users can specify more than just strings at
    the command line (eg allows for users to give functions as args).

    Args:
        arg (str): Input argument.
        backend (str): The machine learning backend you want to use. Options are ``tf2``
            or ``torch``. By default ``None``, meaning no backend is assumed.

    Returns:
        obj: Processed input argument.
    """
    try:
        return safer_eval(arg, backend=backend)
    except Exception:
        return arg


def _add_with_backends(algo_list):
    """Helper function to build lists with backend-specific function names

    Args:
        algo_list (list): List of algorithms.

    Returns:
       list: The algorithms with their backends.
    """
    algo_list_with_backends = deepcopy(algo_list)
    for algo in algo_list:
        algo_list_with_backends += [algo + "_tf2", algo + "_pytorch"]
    return algo_list_with_backends


def _parse_and_execute_grid_search(cmd, args):
    """Interprets algorithm name and cmd line args into an ExperimentGrid.

    Args:
        cmd (str): The requested CLI command.
        args (list): The command arguments.

    Raises:
        ImportError: A custom import error if TensorFlow is not installed.
    """
    cmd, backend = _add_backend_to_cmd(cmd)

    # warning.
    algo = safer_eval("stable_learning_control." + cmd, backend=backend)

    # Before all else, check to see if any of the flags is 'help'.
    valid_help = ["--help", "-h", "help"]
    if any([arg in valid_help for arg in args]):
        print("\n\nShowing docstring for stable_learning_control." + cmd + ":\n")
        print("\t{}\n\n".format(inspect.cleandoc(algo.__doc__).replace("\n", "\n\t")))
        sys.exit()

    # Make first pass through args to build base arg_dict. Anything
    # with a '--' in front of it is an argument flag and everything after,
    # until the next flag, is a possible value.
    arg_dict = dict()
    for i, arg in enumerate(args):
        assert i > 0 or "--" in arg, friendly_err("You didn't specify a first flag.")
        if "--" in arg:
            arg_key = arg.lstrip("-")
            arg_dict[arg_key] = []
        else:
            arg_dict[arg_key].append(_process_arg(arg, backend=backend))

    # Make second pass through, to catch flags that have no vals.
    # Assume such flags indicate that a boolean parameter should have
    # value True.
    for k, v in arg_dict.items():
        if len(v) == 0:
            v.append(True)

    # Third pass: check for user-supplied shorthands, where a key has
    # the form --keyname[kn]. The thing in brackets, 'kn', is the
    # shorthand. NOTE: modifying a dict while looping through its
    # contents is dangerous, and breaks in 3.6+. We loop over a fixed list
    # of keys to avoid this issue.
    given_shorthands = dict()
    fixed_keys = list(arg_dict.keys())
    for k in fixed_keys:
        p1, p2 = k.find("["), k.find("]")
        if p1 >= 0 and p2 >= 0:
            # Both '[' and ']' found, so shorthand has been given
            k_new = k[:p1]
            shorthand = k[p1 + 1 : p2]
            given_shorthands[k_new] = shorthand
            arg_dict[k_new] = arg_dict[k]
            del arg_dict[k]

    # Penultimate pass: sugar. Allow some special shortcuts in arg naming,
    # eg treat "env" the same as "env_name". This is super specific
    # to Stable Learning Control implementations, and may be hard to maintain.
    # These special shortcuts are described by SUBSTITUTIONS.
    for special_name, true_name in SUBSTITUTIONS.items():
        if special_name in arg_dict:
            # swap it in arg dict.
            arg_dict[true_name] = arg_dict[special_name]
            del arg_dict[special_name]

        if special_name in given_shorthands:
            # point the shortcut to the right name.
            given_shorthands[true_name] = given_shorthands[special_name]
            del given_shorthands[special_name]

    # Determine experiment name. If not given by user, will be determined
    # by the algorithm name.
    if "exp_name" in arg_dict:
        assert len(arg_dict["exp_name"]) == 1, friendly_err(
            "You can only provide one value for exp_name."
        )
        exp_name = arg_dict["exp_name"][0]
        del arg_dict["exp_name"]
    else:
        exp_name = "cmd_" + cmd

    # Special handling for environment: make sure that env_name is a real,
    # registered gymnasium environment.
    validate_gym_env(arg_dict)

    # Final pass: check for the special args that go to the 'run' command
    # for an experiment grid, separate them from the arg dict, and make sure
    # that they have unique values. The special args are given by RUN_KEYS.
    run_kwargs = dict()
    for k in RUN_KEYS:
        if k in arg_dict:
            val = arg_dict[k]
            assert len(val) == 1, friendly_err(
                "You can only provide one value for %s." % k
            )
            run_kwargs[k] = val[0]
            del arg_dict[k]

    # Make sure that if num_cpu > 1, the algorithm being used is compatible
    # with MPI.
    if "num_cpu" in run_kwargs and not (run_kwargs["num_cpu"] == 1):
        assert cmd in _add_with_backends(MPI_COMPATIBLE_ALGOS), friendly_err(
            "This algorithm can't be run with num_cpu > 1."
        )

    # Construct and execute the experiment grid.
    eg = ExperimentGrid(name=exp_name)
    for k, v in arg_dict.items():
        eg.add(k, v, shorthand=given_shorthands.get(k))
    eg.run(algo, **run_kwargs)


def run(input_args):
    """Function that is used to run the experiments. I modified this component
    compared to the SpiningUp such that I can import it in other
    modules.

    Args:
        input_args (list): List with command line argument.
    """
    valid_algos = _add_with_backends(BASE_ALGO_NAMES)
    valid_utils = ["plot", "test_policy", "eval_robustness"]
    valid_help = ["--help", "-h", "help"]
    valid_version = ["--version"]
    valid_specials = ["--exp_cfg", "--eval_cfg"]
    valid_cmds = valid_algos + valid_utils + valid_help + valid_version + valid_specials

    # Load hyperparameters from a experimental/eval configuration file if supplied.
    sys.argv = _parse_exp_cfg(sys.argv)
    sys.argv = _parse_eval_cfg(sys.argv)
    cmd = sys.argv[1] if len(input_args) > 1 else "help"

    if cmd not in valid_cmds:
        raise ValueError(
            friendly_err(
                "Input argument '{}' is invalid. Please select an algorithm or ".format(
                    cmd
                )
                + "utility which is implemented in the stable_learning_control "
                "package."
            )
        )

    if cmd in valid_help:
        # Before all else, check to see if any of the flags is 'help'.

        # List commands that are available.
        str_valid_cmds = "\n\t" + "\n\t".join(valid_algos + valid_utils + valid_version)
        help_msg = (
            dedent(
                """
            Experiment in Stable Learning Control from the command line with

            \tpython -m stable_learning_control.run CMD [ARGS...]

            where CMD is a valid command. Current valid commands are:
            """
            )
            + str_valid_cmds
        )
        print(help_msg)

        # Provide some useful details for algorithm running.
        subs_list = [
            "--" + k.ljust(17) + "for".ljust(8) + "--" + v
            for k, v in SUBSTITUTIONS.items()
        ]
        str_valid_subs = "\n\t" + "\n\t".join(subs_list)
        special_info = (
            dedent(
                """
            FYI: When running an algorithm, any keyword argument to the
            algorithm function can be used as a flag, eg

            \tpython -m stable_learning_control.run sac --env HalfCheetah-v2 --clip_ratio 0.1

            If you need a quick refresher on valid kwargs, get the docstring
            with

            \tpython -m stable_learning_control.run [algo] --help

            See the "Running Experiments" docs page for more details.

            Also: Some common but long flags can be substituted for shorter
            ones. Valid substitutions are:
            """  # noqa: E501
            )
            + str_valid_subs
        )
        print(special_info)
    elif cmd in valid_version:
        print("stable_learning_control, version {}".format(__version__))
    elif cmd in valid_utils:
        # Execute the correct utility file.
        runfile = osp.join(osp.abspath(osp.dirname(__file__)), "utils", cmd + ".py")
        args = [sys.executable if sys.executable else "python", runfile] + input_args[
            2:
        ]
        subprocess.check_call(args, env=os.environ)
    else:
        # Assume that the user plans to execute an algorithm. Run custom
        # parsing on the arguments and build a grid search to execute.
        args = input_args[2:]
        _parse_and_execute_grid_search(cmd, args)


if __name__ == "__main__":
    """
    This is a wrapper allowing command-line interfaces to individual algorithms and the
    plot, test policy and test robustness utilities.

    For utilities, it only checks which thing to run, and calls the appropriate file,
    passing all arguments through.

    For algorithms, it sets up an ExperimentGrid object and uses the ExperimentGrid run
    routine to execute each possible experiment.
    """
    run(sys.argv)
