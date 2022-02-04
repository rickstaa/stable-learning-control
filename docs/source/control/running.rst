===================
Running Experiments
===================

.. contents:: Table of Contents

Launching from the Command Line
===============================


BLC ships with, a convenient :ref:`command line interface (CLI) <runner>` that lets you easily
launch any algorithm (with any choices of hyperparameters) from the command line. It also serves as a thin wrapper over
the utilities for watching/evaluating the trained policies and plotting, although we will not discuss that functionality on this page
(for those details, see the pages on `experiment outputs`_, `robustness eval`_ and `plotting`_).

The standard way to run a BLC algorithm from the command line is

.. parsed-literal::

    python -m bayesian_learning_control.run [algo name] [experiment flags]

eg:

.. parsed-literal::

    python -m bayesian_learning_control.run sac --env Walker2d-v2 --exp_name walker

.. _`experiment outputs`: ../control/saving_and_loading.html
.. _`robustness eval`: ../control/eval_robustness.html
.. _`plotting`: ../control/plotting.html

.. admonition:: You Should Know

    If you are using ZShell: ZShell interprets square brackets as special characters. BLC uses square brackets
    in a few ways for command-line arguments; make sure to escape them or try the solution recommended
    `here <http://kinopyo.com/en/blog/escape-square-bracket-by-default-in-zsh>`_ if you want to escape them by default.

.. admonition:: Detailed Quickstart Guide

    .. parsed-literal::

        python -m bayesian_learning_control.run sac --exp_name sac_ant --env Ant-v2 --clip_ratio 0.1 0.2
            --hid[h] [32,32] [64,32] --act torch.nn.Tanh --seed 0 10 20 --dt
            --data_dir path/to/data

    runs SAC in the ``Ant-v2`` Gym environment, with various settings controlled by the flags.

    By default, the PyTorch version will run. Substitute ``sac`` with ``sac_tf2`` for the Tensorflow version.

    ``clip_ratio``, ``hid``, and ``act`` are flags to set some algorithm hyperparameters. You can provide multiple values for hyperparameters to run
    multiple experiments. Check the docs to see what hyperparameters you can set (click here for the `SAC documentation`_).

    ``hid`` and ``act`` are :ref:`special shortcut flags <shortcut_flags>` for setting the hidden sizes and activation function for the neural networks trained by the algorithm.

    The ``seed`` flag sets the seed for the random number generator. RL algorithms have high variance, so try multiple seeds to get a feel for how performance varies.

    The ``dt`` flag ensures that the save directory names will have timestamps in them (otherwise they don't, unless you set ``FORCE_DATESTAMP=True`` in :mod:`bayesian_learning_control.user_config`).

    The ``data_dir`` flag allows you to set the save folder for results. The default value is set by ``DEFAULT_DATA_DIR`` in :mod:`bayesian_learning_control.user_config`, which will be a subfolder
    ``data`` in the ``bayesian_learning_control`` folder (unless you change it).

    `Save directory names`_ are based on ``exp_name`` and any flags which have multiple values. Instead of the full flag, a shorthand will appear in the directory name. Shorthands can be provided
    by the user in square brackets after the flag, like ``--hid[h]``; otherwise, shorthands are substrings of the flag (``clip_ratio`` becomes ``cli``). To illustrate, the save directory for the
    run with ``clip_ratio=0.1``, ``hid=[32,32]``, and ``seed=10`` will be:

    .. parsed-literal::

        path/to/data/YY-MM-DD_sac_ant_cli0-1_h32-32/YY-MM-DD_HH-MM-SS-sac_ant_cli0-1_h32-32_seed10

.. _`SAC documentation`: ../control/algorithms/sac.html#documentation
.. _`special shortcut flags`: ../control/running.html#shortcut-flags
.. _`Save directory names`: ../control/running.html#where-results-are-saved


Choosing PyTorch or Tensorflow from the Command Line
----------------------------------------------------

To use a PyTorch version of an algorithm, run with

.. parsed-literal::

    python -m bayesian_learning_control.run [algo]_pytorch

To use a Tensorflow version of an algorithm, run with

.. parsed-literal::

    python -m bayesian_learning_control.run [algo]_tf2

If you run ``python -m bayesian_learning_control.run [algo]`` without ``_pytorch`` or ``_tf2``, the runner will look in ``bayesian_learning_control/user_config.py`` for which version it should
default to for that algorithm.

Setting Hyperparameters from the Command Line
---------------------------------------------

Every hyperparameter in every algorithm can be controlled directly from the command line. If ``kwarg`` is a valid keyword arg for the function call of an algorithm, you can set values for
it with the flag ``--kwarg``.

To find out what keyword args are available, see either the docs page for :ref:`an algorithm <algorithms>`, :ref:`api` or try

.. parsed-literal::

    python -m bayesian_learning_control.run [algo name] --help

to see a readout of the docstring.

.. admonition:: You Should Know

    Values pass through :meth:`~bayesian_learning_control.control.utils.safer_eval.safer_eval()` before being used, so you can describe some functions and objects directly from
    the command line. For example:

    .. parsed-literal::

        python -m bayesian_learning_control.run SAC --env Walker2d-v2 --exp_name walker --act torch.nn.ELU

    sets ``torch.nn.ELU`` as the activation function. (Tensorflow equivalent: run ``sac_tf`` with ``--act tf.nn.relu``.)

.. admonition:: You Should Know

    There's some nice handling for kwargs that take dict values. Instead of having to provide

    .. parsed-literal::

        --key dict(v1=value_1, v2=value_2)

    you can give

    .. parsed-literal::

        --key:v1 value_1 --key:v2 value_2

    to get the same result.

Launching Multiple Experiments at Once
--------------------------------------

You can launch multiple experiments, to be executed **in series**, by simply providing more than one value for a given argument. (An experiment for each possible combination of values will be launched.)

For example, to launch otherwise-equivalent runs with different random seeds (0, 10, and 20), do:

.. parsed-literal::

    python -m bayesian_learning_control.run sac --env Walker2d-v2 --exp_name walker --seed 0 10 20

Experiments don't launch in parallel because they soak up enough resources that executing several at the same time wouldn't get a speedup.

Special Flags
-------------

A few flags receive special treatment.

Environment Flags
^^^^^^^^^^^^^^^^^

.. option:: --env, --env_name

    :obj:`str`. The name of an environment in the OpenAI Gym. All BLC algorithms are implemented as functions that accept ``env_fn`` as an argument, where ``env_fn``
    must be a callable function that builds a copy of the RL environment. Since the most common use case is Gym environments, though, all of which are built through ``gym.make(env_name)``,
    we allow you to just specify ``env_name`` (or ``env`` for short) at the command line, which gets converted to a lambda-function that builds the correct gym environment.

.. option:: --env_kwargs

    :obj:`object`. Additional keyword arguments you want to pass to the gym environment.

.. _alg_flags:

Algorithm Flags
^^^^^^^^^^^^^^^

General Flags
~~~~~~~~~~~~~

.. option:: --save_checkpoints

    By default, only the most recent state of the agent and environment is saved. When the ``--save_checkpoints`` flag is supplied, a snapshot (checkpoint) of the agent
    and environment will be saved at each epoch. These snapshots are saved in a ``checkpoints`` folder inside the Logger output directory (for more information, see
    :ref:`Saving and Loading Experiment Outputs <checkpoints>`).

.. _`shortcut_flags`:

Shortcut Flags
~~~~~~~~~~~~~~

Some algorithm arguments are relatively long, and we enabled shortcuts for them:

.. option:: --hid, --ac_kwargs:hidden_sizes

    :obj:`list of ints`. Sets the sizes of the hidden layers in the neural networks of both the actor and critic.

.. option:: --hid_a, --ac_kwargs:hidden_sizes:actor

    :obj:`list of ints`. Sets the sizes of the hidden layers in the neural networks of the actor.

.. option:: --hid_c, --ac_kwargs:hidden_sizes:critic

    :obj:`list of ints`. Sets the sizes of the hidden layers in the neural networks of the critic.

.. option:: --act, --ac_kwargs:activation

    :obj:`tf op`. The activation function for the neural networks in the actor and critic.

.. option:: --act_out, --ac_kwargs:output_activation

   :obj:`tf op`. The activation function for the neural networks in the actor and critic.

.. option:: --act_a, --ac_kwargs:activation:actor

   :obj:`tf op`. The activation function for the neural networks in the actor.

.. option:: --act_c, --ac_kwargs:activation:critic

   :obj:`tf op`. The activation function for the neural networks in the critic.

.. option:: --act_out_a, --ac_kwargs:output_activation:actor

   :obj:`tf op`. The activation function for the output activation function of the actor.

.. option:: --act_out_c, --ac_kwargs:output_activation:critic

   :obj:`tf op`. The activation function for the output activation function of the critic.

These flags are valid for all current BLC algorithms.


Config Flags
^^^^^^^^^^^^

These flags are not hyperparameters of any algorithm but change the experimental configuration in some way.

.. option:: --cpu, --num_cpu

    :obj:`int`. If this flag is set, the experiment is launched with this many processes, one per cpu, connected by MPI. Some algorithms are amenable to this sort of parallelization but not all.
    An error will be raised if you try setting ``num_cpu`` > 1 for an incompatible algorithm. You can also set ``--num_cpu auto``, which will automatically use as many CPUs as are available on the machine.

.. option:: --exp_name

    :obj:`str`. The experiment name. This is used in naming the save directory for each experiment. The default is "cmd" + [algo name].

.. option:: --data_dir

    :obj:`path str`. Set the base save directory for this experiment or set of experiments. If none is given, the ``DEFAULT_DATA_DIR`` in ``bayesian_learning_control/user_config.py`` will be used.

.. option:: --datestamp

    :obj:`bool`. Include date and time in the name for the save directory of the experiment.

Logger Flags
^^^^^^^^^^^^

The CLI also contains several (shortcut) flags that can be used to change the behavior of the :class:`bayesian_learning_control.utils.log_utils.logx.EpochLogger`.

.. option:: --use_tensorboard, --logger_kwargs:use_tensorboard

    :obj:`bool`. Enables tensorboard logging.

.. option:: --tb_log_freq, --logger_kwargs:tb_log_freq

    :obj:`str`. The tensorboard log frequency. Options are ``low`` (Recommended: logs at every epoch) and ``high`` (logs at every SGD update
    batch). Defaults to ``low`` since this is less resource intensive.

.. option:: --verbose, --logger_kwargs:verbose

    :obj:`bool`. Whether you want to log to the std_out. Defaults to ``True``.

.. option:: --verbose_fmt, --logger_kwargs:verbose_fmt

    :obj:`bool`. The format in which the statistics are displayed to the terminal. Options are ``table`` which supplies them as a table and ``line`` which prints
    them in one line. Defaults to ``line``.

.. option:: --verbose_vars, --logger_kwargs:verbose_vars

    :obj:`list`. A list of variables you want to log to the std_out. By default all variables are logged.

.. important::

    The verbose_vars list should be supplied as a list that can be evaluated in python (e.g. ``--verbose_vars ["Lr_a", "Lr_c"]``).


Using experimental configuration files (yaml)
---------------------------------------------

The BLC CLI comes with a handy configuration file loader that can be used to load `YAML`_ configuration files. These configuration files provide a convenient way to store your experiments'
hyperparameter such that results can be reproduced. You can supply the CLI with an experiment configuration file using the ``--exp_cfg`` flag.

.. option:: --exp_cfg

    :obj:`path str`. Sets the path to the ``yml`` config file used for loading experiment hyperparameter.

For example, we can use the following command to train a SAC algorithm using the original hyperparameters used by `Haarnoja et al., 2019`_.

.. code-block:: bash

    python -m bayesian_learning_control.run --exp_cfg ./experiments/haarnoja_et_al_2019.yml


.. important::

    Please note that if you want to run multiple hyperparameter variants, for example, multiple seeds or learning rates, you have to use
    comma/space-separated strings in your configuration file:

    .. code-block:: yaml
        :emphasize-lines: 3, 8

        alg_name: lac
        exp_name: my_experiment
        seed: 0 12345 342699
        ac_kwargs:
        hidden_sizes:
            actor: [64, 64]
            critic: [256, 256, 16]
        lr_a: "1e-4, 1e-3, 1e-2"

.. _`YAML`: https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html
.. _`Haarnoja et al., 2019`: https://arxiv.org/abs/1801.01290

Where Results are Saved
-----------------------

Results for a particular experiment (a single run of a configuration of hyperparameters) are stored in

::

    data_dir/[outer_prefix]exp_name[suffix]/[inner_prefix]exp_name[suffix]_s[seed]

where

* ``data_dir`` is the value of the ``--data_dir`` flag (defaults to ``DEFAULT_DATA_DIR`` from ``bayesian_learning_control/user_config.py`` if ``--data_dir`` is not given),
* the ``outer_prefix`` is a ``YY-MM-DD_`` timestamp if the ``--datestamp`` flag is raised, otherwise nothing,
* the ``inner_prefix`` is a ``YY-MM-DD_HH-MM-SS-`` timestamp if the ``--datestamp`` flag is raised, otherwise nothing,
* and ``suffix`` is a special string based on the experiment hyperparameters.

How is Suffix Determined?
^^^^^^^^^^^^^^^^^^^^^^^^^

Suffixes are only included if you run multiple experiments at once, and they only include references to hyperparameters that differ across experiments, except for random seed. The goal is to
ensure that results for similar experiments (ones that share all params except seed) are grouped in the same folder.

Suffixes are constructed by combining *shorthands* for hyperparameters with their values, where a shorthand is either 1) constructed automatically from the hyperparameter name or 2) supplied by
the user. The user can supply a shorthand by writing in square brackets after the kwarg flag.

For example, consider:

.. parsed-literal::

    python -m bayesian_learning_control.run sac_tf --env Hopper-v2 --hid[h] [300] [128,128] --act tf.nn.tanh tf.nn.relu

Here, the ``--hid`` flag is given a **user-supplied shorthand**, ``h``. The ``--act`` flag is not given a shorthand by the user, so one will be constructed for it automatically.

The suffixes produced in this case are:

.. parsed-literal::
    _h128-128_ac-actrelu
    _h128-128_ac-acttanh
    _h300_ac-actrelu
    _h300_ac-acttanh

Note that the ``h`` was given by the user. the ``ac-act`` shorthand was constructed from ``ac_kwargs:activation`` (the true name for the ``act`` flag).

Extra
-----

.. admonition:: You Don't Actually Need to Know This One

    Each individual algorithm is located in a file ``bayesian_learning_control/algos/BACKEND/ALGO_NAME/ALGO_NAME.py``, and these files can be run directly from the command line
    with a limited set of arguments (some of which differ from what's available to ``bayesian_learning_control/run.py``). The command line support in thet in the individual algorithm files
    is essentially vestigial, however, and this is **not** a recommended way to perform experiments.

    This documentation page will not describe those command line calls and *only* describe calls through ``bayesian_learning_control/run.py``.

Launching from Scripts
======================

Each algorithm is implemented as a python function, which can be imported directly from the ``bayesian_learning_control`` package, eg

.. code-block::

    >>> from bayesian_learning_control.control import sac_pytorch as sac

See the documentation page for each algorithm for a complete account of possible arguments. These methods can be used to set up specialized custom experiments, for example:

.. code-block:: python

    from bayesian_learning_control.control import sac_tf2 as sac
    import tensorflow as tf
    import gym

    env_fn = lambda : gym.make('LunarLander-v2')

    ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

    logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

    sac(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)

Use transfer learning
=====================

The ``start_policy`` command-line flag allows you to use an already trained algorithm as the starting point for your new algorithm:

.. option:: --start_policy

    *str*. This flag can be used to train your policy while taking an already started policy as the starting point. It should contain the path to the folder
    where the already trained policy is found.

Using custom environments
=========================

There are two methods for adding custom environments to the BLC package. The first and easiest way is to make use of `OpenAi gym`_ it's internal module import
mechanism:

.. parsed-literal::

    python -m bayesian_learning_control.run sac --env custom_env_module:CustomEnv-v0

This imports the ``custom_env_module`` and then looks for the ``CustomEnv-v0`` in this environment.

.. warning::

    This method only works if you created your environment according to the `OpenAi gym custom gym environment guide`_.

.. _`OpenAi gym`: https://gym.openai.com/
.. _`OpenAi gym custom gym environment guide`: https://github.com/openai/gym/blob/master/docs/creating-environments.md

Additionally you can also add the setup code for registering your environment in the :mod:`bayesian_learning_control.env_config` module.
