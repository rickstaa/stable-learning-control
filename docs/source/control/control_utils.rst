=========
Utilities
=========

.. contents:: Table of Contents

.. _plot:

Plot utility
============

BLC ships with a simple plotting utility that can be used to plot diagnostics from experiments. Run it with:

.. parsed-literal::

    python -m bayesian_learning_control.run plot [path/to/output_directory ...] [--legend [LEGEND ...]]
        [--xaxis XAXIS] [--value [VALUE ...]] [--count] [--smooth S]
        [--select [SEL ...]] [--exclude [EXC ...]]


**Positional Arguments:**

.. option:: logdir

    *strings*. As many log directories (or prefixes to log directories, which the plotter will autocomplete internally) as you'd like to plot from. Logdirs will be searched recursively for experiment outputs.

    .. admonition:: You Should Know

        The internal autocompleting is really handy! Suppose you have run several experiments to compare performance between different algorithms, resulting in a log directory structure of:

        .. parsed-literal::

            data/
                bench_algo1/
                    bench_algo1-seed0/
                    bench_algo1-seed10/
                bench_algo2/
                    bench_algo2-seed0/
                    bench_algo2-seed10/

        You can easily produce a graph comparing algo1 and algo2 with:

        .. parsed-literal::

            python bayesian_learning_control/control/utils/plot.py data/bench_algo

        relying on the autocomplete to find both ``data/bench_algo1`` and ``data/bench_algo2``.

**Optional Arguments:**

.. option:: -l, --legend=[LEGEND ...]

    *strings*. Optional way to specify legend for the plot. The plotter legend will automatically use the ``exp_name`` from the ``config.json`` file, unless you tell it otherwise through this flag. This only works if you provide a name for each directory that will get plotted. (Note: this may not be the same as the number of logdir args you provide! Recall that the plotter looks for autocompletes of the logdir args: there may be more than one match for a given logdir prefix, and you will need to provide a legend string for each one of those matches---unless you have removed some of them as candidates via selection or exclusion rules (below).)

.. option:: -x, --xaxis=XAXIS, default='step'

    *string*. Pick what column from data is used for the x-axis.

.. option:: -y, --value=[VALUE ...], default='Performance'

    *strings*. Pick what columns from data to graph on the y-axis. Submitting multiple values will produce multiple graphs. Defaults to
    ``Performance``, which is not an actual output of any algorithm. Instead, ``Performance`` refers to either ``AverageEpRet``, the
    correct performance measure for the on-policy algorithms, or ``AverageTestEpRet``, the correct performance measure for the off-policy
    algorithms. The plotter will automatically figure out which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for each separate logdir.

.. option:: --count

    Optional flag. By default, the plotter shows y-values that are averaged across all results that share an ``exp_name``, typically a set of identical experiments that only vary in random seed. But if you'd like to see all of those curves separately, use the ``--count`` flag.

.. option:: -s, --smooth=S, default=1

    *int*. Smooth data by averaging it over a fixed window. This parameter says how wide the averaging window will be.

.. option:: --select=[SEL ...]

    *strings*. Optional selection rule: the plotter will only show curves from logdirs that contain all of these substrings.

.. option:: --exclude=[EXC ...]

    *strings*. Optional exclusion rule: plotter will only show curves from logdirs that do not contain these substrings.

.. _test_policy:

Test policy utility
===================

Environment Found
-----------------

BLC ships with an evaluation utility that can be used to check a trained policy's performance. For cases where the environment
is successfully saved alongside the agent, it's a cinch to watch the trained agent act in the environment using:

.. parsed-literal::

    python -m bayesian_learning_control.run test_policy path/to/output_directory


There are a few flags for options:


.. option:: -l L, --len=L, default=0

    *int*. Maximum length of test episode / trajectory / rollout. The default of 0 means no maximum episode length---episodes only end when the agent has
    reached a terminal state in the environment. (Note: setting L=0 will not prevent Gym envs wrapped by TimeLimit wrappers from ending when they reach
    their pre-set maximum episode length.)

.. option:: -n N, --episodes=N, default=100

    *int*. Number of test episodes to run the agent for.

.. option:: -nr, --norender

    Do not render the test episodes to the screen. In this case, ``test_policy`` will only print the episode returns and lengths. (Use case: the renderer
    slows down the testing process, and you just want to get a fast sense of how the agent is performing, so you don't particularly care to watch it.)

.. option:: -i I, --itr=I, default=-1

    Specify the snapshot (checkpoint) for which you want to see the policy performance. Use case: Sometimes, it's nice to watch trained agents from many
    different points in training (eg watch at iteration 50, 100, 150, etc.). The default value of this flag means "use the latest snapshot."

    .. important::
        This option only works if snapshots were saved while training the agent (i.e. the ``--save_checkpoints`` flag was set). For more information on
        storing these snapshots see :ref:`alg_flags`.

.. option:: -d, --deterministic

    Another special case, which is only used for the :ref:`SAC <sac>` and :ref:`LAC <lac>` algorithms. The BLC implementation trains a stochastic
    policy, but is evaluated using the deterministic *mean* of the action distribution. ``test_policy`` will default to using the stochastic policy trained
    by SAC, but you should set the deterministic flag to watch the deterministic mean policy (the correct evaluation policy for SAC). This flag is not used
    for any other algorithms.


Environment Not Found Error
---------------------------

If the environment wasn't saved successfully, you can expect ``test_policy.py`` to crash with something that looks like

.. parsed-literal::

    Traceback (most recent call last):
      File "bayesian_learning_control/control/utils/test_policy.py", line 153, in <module>
        run_policy(env, get_action, args.len, args.episodes, not(args.norender))
      File "bayesian_learning_control/control/utils/test_policy.py", line 114, in run_policy
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " +
    AssertionError: Environment not found!

    It looks like the environment wasn't saved, and we can't run the agent in it. :(

    Check out the readthedocs page on Experiment Outputs for how to handle this situation.


In this case, watching your agent perform is slightly more of a pain but not impossible, as long as you can recreate your environment easily. Try the following in IPython:

>>> from bayesian_learning_control.control.utils.test_policy import load_policy_and_env, run_policy
>>> import your_env
>>> _, get_action = load_policy_and_env('/path/to/output_directory')
>>> env = your_env.make()
>>> run_policy(env, get_action)
Logging data to /tmp/experiments/1536150702/progress.txt
Episode 0    EpRet -163.830      EpLen 93
Episode 1    EpRet -346.164      EpLen 99
...


Using Trained Value Functions
-----------------------------

The ``test_policy.py`` tool doesn't help you look at trained value functions, and if you want to use those, you will have
to load the policy manually. Please see the :ref:`manual_policy_testing` documentation for an example on how to do this.

.. _robustness_eval:

Robustness eval utility
=======================


ExperimentGrid utility
======================

BLC ships with a tool called ExperimentGrid for making hyperparameter ablations easier. This is based on (but simpler than) `the rllab tool`_ called VariantGenerator.

.. _`the rllab tool`: https://github.com/rll/rllab/blob/master/rllab/misc/instrument.py#L173

.. autoclass:: bayesian_learning_control.control.utils.run_utils.ExperimentGrid
    :members:


Calling Experiments utility
===========================

.. autofunction:: bayesian_learning_control.control.utils.run_utils.call_experiment
