"""Contains utilities and helper functions/classes that can be used for calling
experiments.

.. note::
    This module was based on
    `spinningup repository <https://github.com/openai/spinningup/blob/master/spinup/utils/run_utils.py>`__.
"""  # noqa

import base64
import json
import os
import os.path as osp
import string
import subprocess
import sys
import time
import zlib
from subprocess import CalledProcessError
from textwrap import dedent

import cloudpickle
import numpy as np
import psutil
from bayesian_learning_control.common.helpers import all_bools, valid_str
from bayesian_learning_control.user_config import DEFAULT_SHORTHAND, WAIT_BEFORE_LAUNCH
from bayesian_learning_control.utils.log_utils import (
    colorize,
    friendly_err,
    setup_logger_kwargs,
)
from bayesian_learning_control.utils.mpi_utils.mpi_tools import mpi_fork
from bayesian_learning_control.utils.serialization_utils import convert_json
from tqdm import trange

DIV_LINE_WIDTH = 80


def call_experiment(
    exp_name, thunk, seed=0, num_cpu=1, data_dir=None, datestamp=False, **kwargs
):
    """Run a function (thunk) with hyperparameters (:obj:`kwargs`), plus configuration.

    This wraps a few pieces of functionality which are useful when you want
    to run many experiments in sequence, including logger configuration and
    splitting into multiple processes for MPI.

    There's also a SpinningUp-specific convenience added into executing the
    thunk: if ``env_name`` is one of the **kwargs** passed to call_experiment, it's
    assumed that the thunk accepts an argument called ``env_fn``, and that
    the ``env_fn`` should make a gym environment with the given ``env_name``.

    The way the experiment is actually executed is slightly complicated: the
    function is serialised to a string, and then ``run_entrypoint.py`` is
    executed in a subprocess call with the serialised string as an argument.
    ``run_entrypoint.py`` unserializes the function call and executes it.
    We choose to do it this way---instead of just calling the function
    directly here---to avoid leaking state between successive experiments.

    Args:
        exp_name (str): Name for experiment.
        thunk (callable): A python function.
        seed (int): Seed for random number generators.
        num_cpu (int): Number of MPI processes to split into. Also accepts
            'auto', which will set up as many procs as there are cpus on
            the machine.
        data_dir (str): Used in configuring the logger, to decide where
            to store experiment results. Note: if left as None, data_dir will
            default to ``DEFAULT_DATA_DIR`` from
            :mod:`bayesian_learning_control.user_config`.
        datestamp (bool): Whether a datestamp should be added to the experiment name.
        kwargs: All kwargs to pass to thunk.
    """
    # Determine number of CPU cores to run on
    num_cpu = psutil.cpu_count(logical=False) if num_cpu == "auto" else num_cpu

    # Send random seed to thunk
    kwargs["seed"] = seed

    # Be friendly and print out your kwargs, so we all know what's up
    print(colorize("Running experiment:\n", color="cyan", bold=True))
    print(exp_name + "\n")
    print(colorize("with kwargs:\n", color="cyan", bold=True))
    kwargs_json = convert_json(kwargs)
    print(json.dumps(kwargs_json, separators=(",", ":\t"), indent=4, sort_keys=True))
    print("\n")

    # Set up logger output directory
    if "logger_kwargs" not in kwargs:
        kwargs["logger_kwargs"] = setup_logger_kwargs(
            exp_name, seed, data_dir, datestamp
        )

    else:
        print("Note: Call experiment is not handling logger_kwargs.\n")
        kwargs["logger_kwargs"] = setup_logger_kwargs(
            exp_name,
            seed=seed,
            data_dir=data_dir,
            datestamp=datestamp,
            **kwargs["logger_kwargs"],
        )

    # Force algorithm default if verbose_fmt is line
    # NOTE: Done since otherwise the stdout gets cluttered.
    if kwargs["logger_kwargs"]["verbose_fmt"] == "line":
        kwargs["logger_kwargs"]["verbose_vars"] = None

    def thunk_plus():
        """Setup environment used in the experiment."""
        # Make 'env_fn' from 'env_name'
        if "env_name" in kwargs:
            # Import gym environments
            import gym

            # import bayesian_learning_control.simzoo.simzoo  # noqa: F401
            try:
                import bayesian_learning_control.env_config  # noqa: F401
            except Exception as e:
                raise Exception(
                    friendly_err(
                        "Something went wrong when trying to import the 'env_config' "
                        " file."
                    )
                ) from e

            env_name = kwargs.pop("env_name")
            env_kwargs = kwargs.pop("env_kwargs", {})
            kwargs["env_fn"] = lambda: gym.make(env_name, **env_kwargs)

        # Fork into multiple processes
        mpi_fork(num_cpu)

        # Run thunk
        thunk(**kwargs)

    # Prepare to launch a script to run the experiment
    pickled_thunk = cloudpickle.dumps(thunk_plus)
    encoded_thunk = base64.b64encode(zlib.compress(pickled_thunk)).decode("utf-8")

    entrypoint = osp.join(osp.abspath(osp.dirname(__file__)), "run_entrypoint.py")
    cmd = [sys.executable if sys.executable else "python", entrypoint, encoded_thunk]
    try:
        subprocess.check_call(cmd, env=os.environ)
    except CalledProcessError:
        err_msg = (
            "\n" * 3
            + "=" * DIV_LINE_WIDTH
            + "\n"
            + dedent(
                """

            There appears to have been an error in your experiment.

            Check the traceback above to see what actually went wrong. The
            traceback below, included for completeness (but probably not useful
            for diagnosing the error), shows the stack leading up to the
            experiment launch.

            """
            )
            + "=" * DIV_LINE_WIDTH
            + "\n" * 3
        )
        print(err_msg)
        raise

    # Tell the user about where results are, and how to check them
    logger_kwargs = kwargs["logger_kwargs"]

    plot_cmd = (
        "python -m bayesian_learning_control.run plot " + logger_kwargs["output_dir"]
    )
    plot_cmd = colorize(plot_cmd, "green")

    test_cmd = (
        "python -m bayesian_learning_control.run test_policy "
        + logger_kwargs["output_dir"]
    )
    test_cmd = colorize(test_cmd, "green")

    eval_cmd = (
        "python -m bayesian_learning_control.run eval_robustness "
        + logger_kwargs["output_dir"]
    )
    eval_cmd = colorize(eval_cmd, "green")

    output_msg = (
        "\n" * 5
        + "=" * DIV_LINE_WIDTH
        + "\n"
        + dedent(
            """\
    End of experiment.


    Plot results from this run with:

    %s


    Watch the trained agent with:

    %s


    Evaluate it's robustness with:

    %s

    """
            % (plot_cmd, test_cmd, eval_cmd)
        )
        + "=" * DIV_LINE_WIDTH
        + "\n" * 5
    )

    print(output_msg)


class ExperimentGrid:
    """Tool for running many experiments given hyperparameter ranges."""

    def __init__(self, name=""):
        """Initiate object.

        Args:
            name (str): Experimental grid id.
        """
        self.keys = []
        self.vals = []
        self.shs = []
        self.in_names = []
        self.name(name)

    def name(self, _name):
        """Validate grid id.

        Args:
            _name (object): Input object.
        """
        assert isinstance(_name, str), "Name has to be a string."
        self._name = _name

    def print(self):
        """Print a helpful report about the experiment grid."""
        print("=" * DIV_LINE_WIDTH)

        # Prepare announcement at top of printing. If the ExperimentGrid has a
        # short name, write this as one line. If the name is long, break the
        # announcement over two lines.
        base_msg = "ExperimentGrid %s runs over parameters:\n"
        name_insert = "[" + self._name + "]"
        if len(base_msg % name_insert) <= 80:
            msg = base_msg % name_insert
        else:
            msg = base_msg % (name_insert + "\n")
        print(colorize(msg, color="green", bold=True))

        # List off parameters, shorthands, and possible values.
        for k, v, sh in zip(self.keys, self.vals, self.shs):
            color_k = colorize(k.ljust(40), color="cyan", bold=True)
            print("", color_k, "[" + sh + "]" if sh is not None else "", "\n")
            for i, val in enumerate(v):
                print("\t" + str(convert_json(val)))
            print()

        # Count up the number of variants. The number counting seeds
        # is the total number of experiments that will run; the number not
        # counting seeds is the total number of otherwise-unique configs
        # being investigated.
        nvars_total = int(np.prod([len(v) for v in self.vals]))
        if "seed" in self.keys:
            num_seeds = len(self.vals[self.keys.index("seed")])
            nvars_seedless = int(nvars_total / num_seeds)
        else:
            nvars_seedless = nvars_total
        print(" Variants, counting seeds: ".ljust(40), nvars_total)
        print(" Variants, not counting seeds: ".ljust(40), nvars_seedless)
        print()
        print("=" * DIV_LINE_WIDTH)

    def _default_shorthand(self, key):
        """Create grid key shorthands.

        Create a default shorthand for the key, built from the first three letters of
        each colon-separated part. But if the first three letters contains something
        which isn't alphanumeric, shear that off.

        Args:
            key (str): Full grid key name.

        Returns:
            str: Generated shorthand.
        """
        valid_chars = "%s%s" % (string.ascii_letters, string.digits)

        def shear(x):
            return "".join(z for z in x[:3] if z in valid_chars)

        sh = "-".join([shear(x) for x in key.split(":")])
        return sh

    def add(self, key, vals, shorthand=None, in_name=False):
        """Add a parameter (key) to the grid config, with potential values (vals).

        By default, if a shorthand isn't given, one is automatically generated
        from the key using the first three letters of each colon-separated
        term. To disable this behavior, change :attr:`DEFAULT_SHORTHAND` in the
        :mod:`bayesian_learning_control.user_config` file to ``False``.

        Args:
            key (str): Name of parameter.
            vals (value or list of values): Allowed values of parameter.
            shorthand (str): Optional, shortened name of parameter. For
                example, maybe the parameter ``steps_per_epoch`` is shortened
                to ``steps``.
            in_name (bool): When constructing variant names, force the
                inclusion of this parameter into the name.
        """
        assert isinstance(key, str), "Key must be a string."
        assert shorthand is None or isinstance(
            shorthand, str
        ), "Shorthand must be a string."
        if not isinstance(vals, list):
            vals = [vals]
        if DEFAULT_SHORTHAND and shorthand is None:
            shorthand = self._default_shorthand(key)
        self.keys.append(key)
        self.vals.append(vals)
        self.shs.append(shorthand)
        self.in_names.append(in_name)

    def variant_name(self, variant):
        """Given a variant (dict of valid param/value pairs), make an exp_name.

        A variant's name is constructed as the grid name (if you've given it
        one), plus param names (or shorthands if available) and values
        separated by underscores.

        Note: if ``seed`` is a parameter, it is not included in the name.

        args:
            variant (str): The variant name.
        """

        def get_val(v, k):
            # Utility method for getting the correct value out of a variant
            # given as a nested dict. Assumes that a parameter name, k,
            # describes a path into the nested dict, such that k='a:b:c'
            # corresponds to value=variant['a']['b']['c']. Uses recursion
            # to get this.
            if k in v:
                return v[k]
            else:
                splits = k.split(":")
                k0, k1 = splits[0], ":".join(splits[1:])
                return get_val(v[k0], k1)

        # Start the name off with the name of the variant generator.
        var_name = self._name

        # Build the rest of the name by looping through all parameters,
        # and deciding which ones need to go in there.
        for k, v, sh, inn in zip(self.keys, self.vals, self.shs, self.in_names):

            # Include a parameter in a name if either 1) it can take multiple
            # values, or 2) the user specified that it must appear in the name.
            # Except, however, when the parameter is 'seed'. Seed is handled
            # differently so that runs of the same experiment, with different
            # seeds, will be grouped by experiment name.
            if (len(v) > 1 or inn) and not (k == "seed"):

                # Use the shorthand if available, otherwise the full name.
                param_name = sh if sh is not None else k
                param_name = valid_str(param_name)

                # Get variant value for parameter k
                variant_val = get_val(variant, k)

                # Append to name
                if all_bools(v):
                    # If this is a param which only takes boolean values,
                    # only include in the name if it's True for this variant.
                    var_name += ("_" + param_name) if variant_val else ""
                else:
                    var_name += "_" + param_name + valid_str(variant_val)

        return var_name.lstrip("_")

    def _variants(self, keys, vals):
        """Recursively builds list of valid variants.

        Args:
            keys (object): Hyperparameter key name.
            vals (object): Grid value.

        Returns:
            list: List of valid variants.
        """
        if len(keys) == 1:
            pre_variants = [dict()]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])

        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                v = {}
                v[keys[0]] = val
                v.update(pre_v)
                variants.append(v)
        return variants

    def variants(self):
        """Makes a list of dicts, where each dict is a valid config in the grid.

        There is special handling for variant parameters whose names take
        the form

            ``'full:param:name'``.

        The colons are taken to indicate that these parameters should
        have a nested dict structure. eg, if there are two params,

            ====================  ===
            Key                   Val
            ====================  ===
            ``'base:param:a'``    1
            ``'base:param:b'``    2
            ====================  ===

        the variant dict will have the structure

        .. parsed-literal::

            variant = {
                base: {
                    param : {
                        a : 1,
                        b : 2
                        }
                    }
                }
        """
        flat_variants = self._variants(self.keys, self.vals)

        def unflatten_var(var):
            """
            Build the full nested dict version of var, based on key names.
            """
            new_var = dict()
            unflatten_set = set()

            for k, v in var.items():
                if ":" in k:
                    splits = k.split(":")
                    k0 = splits[0]
                    assert k0 not in new_var or isinstance(
                        new_var[k0], dict
                    ), "You can't assign multiple values to the same key."

                    if not (k0 in new_var):
                        new_var[k0] = dict()

                    sub_k = ":".join(splits[1:])
                    new_var[k0][sub_k] = v
                    unflatten_set.add(k0)
                else:
                    assert not (
                        k in new_var
                    ), "You can't assign multiple values to the same key."
                    new_var[k] = v

            # Make sure to fill out the nested dicts.
            for k in unflatten_set:
                new_var[k] = unflatten_var(new_var[k])

            return new_var

        new_variants = [unflatten_var(var) for var in flat_variants]
        return new_variants

    def run(self, thunk, num_cpu=1, data_dir=None, datestamp=False):
        """Run each variant in the grid with function 'thunk'.

        Note: 'thunk' must be either a callable function, or a string. If it is
        a string, it must be the name of a parameter whose values are all
        callable functions.

        Uses :meth:`call_experiment` to actually launch each experiment, and gives
        each variant a name using :meth:`variant_name`.

        Maintenance note: the args for ExperimentGrid.run should track closely
        to the args for call_experiment. However, ``seed`` is omitted because
        we presume the user may add it as a parameter in the grid.

        Args:
            thunk (callable): A python function.
            seed (int): Seed for random number generators.
            num_cpu (int): Number of MPI processes to split into. Also accepts
                'auto', which will set up as many procs as there are cpus on
                the machine.
            data_dir (str): Used in configuring the logger, to decide where
                to store experiment results. Note: if left as None, data_dir will
                default to ``DEFAULT_DATA_DIR`` from
                :mod:`bayesian_learning_control.user_config`.
            datestamp (bool): Whether a datestamp should be added to the experiment
                name.
        """
        # Print info about self.
        self.print()

        # Make the list of all variants.
        variants = self.variants()

        # Print variant names for the user.
        var_names = set([self.variant_name(var) for var in variants])
        var_names = sorted(list(var_names))
        line = "=" * DIV_LINE_WIDTH
        preparing = colorize(
            "Preparing to run the following experiments...", color="green", bold=True
        )
        joined_var_names = "\n".join(var_names)
        announcement = f"\n{preparing}\n\n{joined_var_names}\n\n{line}"
        print(announcement)

        if WAIT_BEFORE_LAUNCH > 0:
            delay_msg = (
                colorize(
                    dedent(
                        """
            Launch delayed to give you a few seconds to review your experiments.

            To customise or disable this behaviour, change WAIT_BEFORE_LAUNCH in
            ``bayesian_learning_control/user_config.py``.

            """
                    ),
                    color="cyan",
                    bold=True,
                )
                + line
            )
            print(delay_msg)
            wait, steps = WAIT_BEFORE_LAUNCH, 100
            prog_bar = trange(
                steps,
                desc="Launching in...",
                leave=False,
                ncols=DIV_LINE_WIDTH,
                mininterval=0.25,
                bar_format="{desc}: {bar}| {remaining} {elapsed}",
            )
            for _ in prog_bar:
                time.sleep(wait / steps)

        # Run the variants.
        for var in variants:
            exp_name = self.variant_name(var)

            # Figure out what the thunk is.
            if isinstance(thunk, str):
                # Assume one of the variant parameters has the same
                # name as the string you passed for thunk, and that
                # variant[thunk] is a valid callable function.
                thunk_ = var[thunk]
                del var[thunk]
            else:
                # Assume thunk is given as a function.
                thunk_ = thunk

            call_experiment(
                exp_name,
                thunk_,
                num_cpu=num_cpu,
                data_dir=data_dir,
                datestamp=datestamp,
                **var,
            )
