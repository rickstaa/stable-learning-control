"""Module used for plotting the training results.

.. note::
    This module was based on
    `Spinning Up repository <https://github.com/openai/spinningup/tree/master/spinup/utils/plot.py>`__.
"""  # noqa

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stable_learning_control.common.helpers import friendly_err

DIV_LINE_WIDTH = 50
VALID_DATA_FILES = ["progress.txt", "progress.csv"]

# Global vars for tracking and labelling data at load time.
exp_idx = 0
units = dict()


def plot_data(
    data,
    xaxis="Epoch",
    value="AverageEpRet",
    condition="Condition1",
    errorbar="sd",
    smooth=1,
    font_scale=1.5,
    style="darkgrid",
    **kwargs,
):
    """Function used to plot data.

    Args:
        data (obj): The data you want to plot.
        xaxis (str): Pick what column from data is used for the x-axis.
            Defaults to ``TotalEnvInteracts``.
        value (str): Pick what columns from data to graph on the y-axis.
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the
            off-policy algorithms. The plotter will automatically figure out
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for
            each separate logdir.
        condition (str, optional): The condition to search for. By default
            ``Condition1``.
        errorbar (str): The error bar you want to use for the plot. Defaults
            to ``sd``.
        smooth (int): Smooth data by averaging it over a fixed window. This
            parameter says how wide the averaging window will be.
        font_scale (int): The font scale you want to use for the plot text.
        style (str): The style you want to use for the plot.
    """
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style=style, font_scale=font_scale)
    sns.lineplot(
        data=data, x=xaxis, y=value, hue=condition, errorbar=errorbar, **kwargs
    )
    plt.legend(loc="best").set_draggable(True)

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    :class:`~stable_learning_control.utils.log_utils.logx.EpochLogger`.

    Assumes that any file ``progress.(csv|txt)`` is a valid hit.

    Args:
        logdir (str): The log directory to search in.
        condition (str, optional): The condition to search for. By default ``None``.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        data_file = [file for file in files if file in VALID_DATA_FILES]
        if data_file:
            exp_name = None
            try:
                config_path = open(osp.join(root, "config.json"))
                config = json.load(config_path)
                if "exp_name" in config:
                    exp_name = config["exp_name"]
            except Exception:
                print("No file named config.json")
            condition1 = condition or exp_name or "exp"
            condition2 = condition1 + "-" + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(
                    osp.join(root, data_file[0]), sep=None, engine="python"
                )
            except Exception:
                print("Could not read from %s" % osp.join(root, data_file[0]))
                continue
            performance = (
                "AverageTestEpRet" if "AverageTestEpRet" in exp_data else "AverageEpRet"
            )
            exp_data.insert(len(exp_data.columns), "Unit", unit)
            exp_data.insert(len(exp_data.columns), "Condition1", condition1)
            exp_data.insert(len(exp_data.columns), "Condition2", condition2)
            exp_data.insert(len(exp_data.columns), "Performance", exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.

    Args:
        all_logdirs (list): A list of lig directories you want to use.
        legend (list[str]): Optional way to specify legend for the plot. The
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one
            match for a given logdir prefix, and you will need to provide a
            legend string for each one of those matches---unless you have
            removed some of them as candidates via selection or exclusion
            rules (below).)
        select (list[str]): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.
        exclude (list[str]): Optional exclusion rule: plotter will only show
            curves from logdirs that do not contain these substrings.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)  # noqa: E731
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    # Verify logdirs.
    print("Plotting from...\n" + "=" * DIV_LINE_WIDTH + "\n")
    for logdir in logdirs:
        print(logdir)
    print("\n" + "=" * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs.
    assert not (legend) or (len(legend) == len(logdirs)), friendly_err(
        "Must give a legend title for each set of experiments."
    )

    # Load data from logdirs.
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(
    all_logdirs,
    legend=None,
    xaxis=None,
    values=None,
    count=False,
    font_scale=1.5,
    style="darkgrid",
    smooth=1,
    select=None,
    exclude=None,
    estimator="mean",
):
    """Function used for generating the plots.

    Args:
        logdir (str): As many log directories (or prefixes to log
            directories, which the plotter will autocomplete internally) as
            you'd like to plot from.
        legend (list[str]): Optional way to specify legend for the plot. The
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one
            match for a given logdir prefix, and you will need to provide a
            legend string for each one of those matches---unless you have
            removed some of them as candidates via selection or exclusion
            rules (below).)
        xaxis (str): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.
        values (list): Pick what columns from data to graph on the y-axis.
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the
            off-policy algorithms. The plotter will automatically figure out
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for
            each separate logdir.
        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``,
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves
            separately, use the ``--count`` flag.
        font_scale (int): The font scale you want to use for the plot text.
        style (str): The style you want to use for the plot.
        smooth (int): Smooth data by averaging it over a fixed window. This
            parameter says how wide the averaging window will be.
        select (list[str]): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.
        exclude (list[str]): Optional exclusion rule: plotter will only show
            curves from logdirs that do not contain these substrings.
        estimator (str): The estimator you want to use in your plot (ie. mean, min max).
    """
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = "Condition2" if count else "Condition1"
    estimator = getattr(
        np, estimator
    )  # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(
            data,
            xaxis=xaxis,
            value=value,
            condition=condition,
            smooth=smooth,
            estimator=estimator,
            font_scale=font_scale,
            style=style,
        )
    plt.show()


def plot():
    """Run the plot utility.

    Args:
        logdir (list[str]): As many log directories (or prefixes to log
            directories, which the plotter will autocomplete internally) as
            you'd like to plot from.
        legend (list[str]): Optional way to specify legend for the plot. The
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one
            match for a given logdir prefix, and you will need to provide a
            legend string for each one of those matches---unless you have
            removed some of them as candidates via selection or exclusion
            rules (below).)
        xaxis (str): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.
        value (str): Pick what columns from data to graph on the y-axis.
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the
            off-policy algorithms. The plotter will automatically figure out
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for
            each separate logdir.
        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``,
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves
            separately, use the ``--count`` flag.
        smooth (int): Smooth data by averaging it over a fixed window. This
            parameter says how wide the averaging window will be.
        select (list[str]): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.
        exclude (list[str]): Optional exclusion rule: plotter will only show
            curves from logdirs that do not contain these substrings.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logdir",
        nargs="*",
        help="The directory where the results are that you want to plot",
    )
    parser.add_argument(
        "--legend",
        "-l",
        nargs="*",
        help="Optional way to specify a legend for the plot",
    )
    parser.add_argument(
        "--xaxis",
        "-x",
        default="TotalEnvInteracts",
        help="Data used for the x-axis (default: TotalEnvInteracts)",
    )
    parser.add_argument(
        "--value",
        "-y",
        default="Performance",
        nargs="*",
        help="Data used for the y-axis (default: performance)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Whether you want to average over all the results (default: False)",
    )
    parser.add_argument(
        "--smooth", "-s", type=int, default=1, help="The size of the averaging window"
    )
    parser.add_argument(
        "--select", nargs="*", help="Log directories to include in your plot"
    )
    parser.add_argument(
        "--exclude", nargs="*", help="Log directories to exclude in your plot"
    )
    parser.add_argument(
        "--est",
        default="mean",
        help="The estimator you want to use in your plot (ie. mean, min max)",
    )
    args = parser.parse_args()

    make_plots(
        args.logdir,
        args.legend,
        args.xaxis,
        args.value,
        args.count,
        smooth=args.smooth,
        select=args.select,
        exclude=args.exclude,
        estimator=args.est,
    )


if __name__ == "__main__":
    plot()
