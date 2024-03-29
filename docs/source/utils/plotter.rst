=======
Plotter
=======

.. contents:: Table of Contents

.. _plot:

Plot utility
============

SLC ships with a simple plotting utility that can be used to plot diagnostics from experiments. Run it with:

.. parsed-literal::

    python -m stable_learning_control.run plot [path/to/output_directory ...] [-h] [--legend [LEGEND [LEGEND ...]]] 
        [--xaxis XAXIS] [--value [VALUE [VALUE ...]]] [--count] [--smooth SMOOTH]
        [--select [SELECT [SELECT ...]]] [--exclude [EXCLUDE [EXCLUDE ...]]] [--est EST]

**Positional Arguments:**

.. option:: logdir

    :obj:`list of strings`. As many log directories (or prefixes to log directories, which the plotter will autocomplete internally) as you'd like to plot from. Logdirs will be searched recursively for experiment outputs.

    .. admonition:: You Should Know

        The internal autocompleting is handy! Suppose you have run several experiments to compare performance between different algorithms, resulting in a log directory structure of:

        .. parsed-literal::

            data/
                bench_algo1/
                    bench_algo1-seed0/
                    bench_algo1-seed10/
                bench_algo2/
                    bench_algo2-seed0/
                    bench_algo2-seed10/

        You can quickly produce a graph comparing algo1 and algo2 with:

        .. parsed-literal::

            python stable_learning_control/utils/plot.py data/bench_algo

        relying on the autocomplete to find both ``data/bench_algo1`` and ``data/bench_algo2``.

**Optional Arguments:**

.. option:: -l, --legend=[LEGEND ...]

    :obj:`list of strings`. Optional way to specify legend for the plot. The plotter legend will automatically use the ``exp_name`` from the ``config.json`` file, unless you tell it otherwise through this flag. This only works if you provide a name for each directory that will get plotted. (Note: this may not be the same as the number of logdir args you provide! Recall that the plotter looks for autocompletes of the logdir args: there may be more than one match for a given logdir prefix, and you will need to provide a legend string for each one of those matches---unless you have removed some of them as candidates via selection or exclusion rules (below).)

.. option:: -x, --xaxis=XAXIS, default='step'

    :obj:`str`. Pick what column from the data is used for the x-axis.

.. option:: -y, --value=[VALUE ...], default='Performance'

    :obj:`list of strings`. Pick what columns from the data to graph on the y-axis. Submitting multiple values will produce multiple graphs. Defaults to
    ``Performance``, which is not an actual output of any algorithm. Instead, ``Performance`` refers to either ``AverageEpRet``, the
    correct performance measure for the on-policy algorithms, or ``AverageTestEpRet``, the correct performance measure for the off-policy
    algorithms. The plotter will automatically figure out which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for each logdir.

.. option:: --count, default=False

    Optional flag. By default, the plotter shows y-values averaged across all results that share an ``exp_name``, typically a set of identical experiments that only vary in the random seed. But if you'd like to see all of those curves separately, use the ``--count`` flag.

.. option:: -s, --smooth=S, default=1

    :obj:`int`. Smooth data by averaging it over a fixed window. This parameter says how wide the averaging window will be.

.. option:: --select=[SEL ...]

    :obj:`list of strings`. Optional selection rule: the plotter will only show curves from logdirs containing all these substrings.

.. option:: --exclude=[EXC ...]

    :obj:`list of strings`. Optional exclusion rule: plotter will only show curves from logdirs that do not contain these substrings.

.. option:: --est=[EST], default='mean'

    :obj:`str`. The estimator you want to use for the plot. Options are ``mean`` (default), ``min``, ``max``.
