.. _running_experiments:

===================
Running Experiments
===================

.. contents:: Table of Contents

One of the best ways to get a feel for deep RL is to run the algorithms and see how they
perform on different tasks. The SLC library makes small-scale (local) experiments easy to
do, and in this section, we'll discuss two ways to run them: either from the command line
or through function calls in scripts.

Launching from the Command Line
===============================

SLC ships with a convenient :ref:`command line interface (CLI) <runner>` that lets you
quickly launch any algorithm (with any choices of hyperparameters) from the command line.
It also serves as a thin wrapper over the utilities for watching/evaluating the trained
policies and plotting. However, that functionality is not discussed on this page (for
those details, see the pages on :ref:`experiment outputs <saving_and_loading>`, 
:ref:`robustness evaluation <robustness_eval>` and :ref:`plotting`).

The standard way to run an SLC algorithm from the command line is

.. parsed-literal::

    python -m stable_learning_control.run [algo name] [experiment flags]

eg:

.. parsed-literal::

    python -m stable_learning_control.run sac --env Walker2d-v2 --exp_name walker

.. admonition:: You Should Know

    If you are using ZShell: ZShell interprets square brackets as special characters. SLC
    uses square brackets in a few ways for command-line arguments; make sure to escape
    them or try the solution recommended `here`_
    if you want to escape them by default.

.. _`here`: https://kinopyo.com/en/blog/escape-square-bracket-by-default-in-zsh

.. admonition:: Detailed Quickstart Guide

    .. parsed-literal::

        python -m stable_learning_control.run sac --exp_name sac_ant --env Ant-v2 --clip_ratio 0.1 0.2
            --hid[h] [32,32] [64,32] --act torch.nn.Tanh --seed 0 10 20 --dt
            --data_dir path/to/data

    runs SAC in the ``Ant-v2`` gymnasium environment, with various settings controlled by the flags.

    By default, the PyTorch version will run. You can, however, substitute ``sac`` with
    ``sac_tf2`` for the TensorFlow version.

    ``clip_ratio``, ``hid``, and ``act`` are flags to set some algorithm hyperparameters. You
    can provide multiple values for hyperparameters to run multiple experiments. Check the docs
    to see what hyperparameters you can set (click here for the :ref:`SAC documentation <sac>`).

    ``hid`` and ``act`` are :ref:`special shortcut flags <shortcut_flags>` for setting the
    hidden sizes and activation function for the neural networks trained by the algorithm.

    The ``seed`` flag sets the seed for the random number generator. RL algorithms have
    high variance, so try multiple seeds to get a feel for how performance varies.

    The ``dt`` flag ensures that the save directory names will have timestamps in them
    (otherwise, they don't, unless you set ``FORCE_DATESTAMP=True`` in :mod:`stable_learning_control.user_config`).

    The ``data_dir`` flag allows you to set the save folder for results. The default
    value is set by ``DEFAULT_DATA_DIR`` in :mod:`stable_learning_control.user_config`,
    which will be a subfolder ``data`` in the ``stable_learning_control`` folder (unless you change it).

    The `Save directory names`_ are based on ``exp_name`` and any flags which have multiple
    values. Instead of the full flag, a shorthand will appear in the directory name.
    Shorthands can be provided by the user in square brackets after the flag, like
    ``--hid[h]``; otherwise, shorthands are substrings of the flag (``clip_ratio``
    becomes ``cli``). To illustrate, the save directory for the run with 
    ``clip_ratio=0.1``, ``hid=[32,32]``, and ``seed=10`` will be:

    .. parsed-literal::

        path/to/data/YY-MM-DD_sac_ant_cli0-1_h32-32/YY-MM-DD_HH-MM-SS-sac_ant_cli0-1_h32-32_seed10

.. _`special shortcut flags`: #shortcut-flags
.. _`Save directory names`: #where-results-are-saved

Choosing PyTorch or TensorFlow from the Command Line
----------------------------------------------------

To use a PyTorch version of an algorithm, run with

.. parsed-literal::

    python -m stable_learning_control.run [algo]_pytorch

To use a TensorFlow version of an algorithm, run with

.. parsed-literal::

    python -m stable_learning_control.run [algo]_tf2

If you run ``python -m stable_learning_control.run [algo]`` without ``_pytorch`` or ``_tf2``,
the runner will look in ``stable_learning_control/user_config.py`` for which version it should 
default to that algorithm.

.. attention::
    The TensorFlow version is still experimental. It is not guaranteed to work, and it is not
    guaranteed to be up-to-date with the PyTorch version.

Setting Hyperparameters from the Command Line
---------------------------------------------

Every hyperparameter in every algorithm can be controlled directly from the command line. If **kwarg**
is a valid keyword arg for the function call of an algorithm, you can set values for it with the flag
``--kwarg``.

To find out what keyword args are available, see either the docs page for :ref:`an algorithm <algos>`, 
:ref:`the API reference <autoapi>` or try

.. parsed-literal::

    python -m stable_learning_control.run [algo name] --help

to see a readout of the docstring.

.. admonition:: You Should Know

    Values pass through :meth:`~stable_learning_control.utils.safer_eval_util.safer_eval()` before
    being used so that you can describe some functions and objects directly from the command line.
    For example:

    .. parsed-literal::

        python -m stable_learning_control.run SAC --env Walker2d-v2 --exp_name walker --act torch.nn.ReLU

    sets ``torch.nn.ReLU`` as the activation function. (TensorFlow equivalent: run ``sac_tf`` with ``--act tf.nn.relu``.)

.. admonition:: You Should Know

    There's some excellent handling for kwargs that take :obj:`dict` values. Instead of having to provide

    .. parsed-literal::

        --key dict(v1=value_1, v2=value_2)

    you can give

    .. parsed-literal::

        --key:v1 value_1 --key:v2 value_2

    to get the same result.

.. _running_multiple_experiments:

Launching Multiple Experiments at Once
--------------------------------------

You can launch multiple experiments, to be executed **in series**, by simply providing more than
one value for a given argument. (An experiment for each possible combination of values will
be launched.)

For example, to launch otherwise-equivalent runs with different random seeds (0, 10, and 20), do:

.. parsed-literal::

    python -m stable_learning_control.run sac --env Walker2d-v2 --exp_name walker --seed 0 10 20

Experiments don't launch in parallel because they soak up enough resources that executing several
simultaneously wouldn't get a speedup.

Special Flags
-------------

A few flags receive special treatment.

.. _env_flags:

Environment Flags
^^^^^^^^^^^^^^^^^

.. option:: --env, --env_name

    :obj:`str`. The name of an environment in gymnasium. All SLC algorithms are implemented as
    functions that accept ``env_fn`` as an argument, where ``env_fn`` must be a callable function
    that builds a copy of the RL environment. Since the most common use case is gymnasium
    environments, though, all of which are built through ``gym.make(env_name)``, we allow
    you to specify ``env_name`` (or ``env`` for short) at the command line, which gets
    converted to a lambda-function that builds the correct gymnasium environment. You can
    prefix the environment name with a module name, separated by a colon, to specify a
    custom gymnasium environment (i.e. ``--env stable_gym:Oscillator-v1``).

.. option:: --env_k, --env_kwargs

    :obj:`object`. Additional keyword arguments you want to pass to the gym environment. If 
    you, for example, want to change the forward reward weight and healthy reward of the
    `Walker2d-v2`_ environment, you can do so by passing ``--env_kwargs "{'forward_reward_weight': 0.5, 'healthy_reward': 0.5}"``
    to the run command.

.. _`Walker2d-v2`: https://mgoulao.github.io/gym-docs/environments/mujoco/walker2d/

.. _alg_flags:

Algorithm Flags
^^^^^^^^^^^^^^^

General Flags
~~~~~~~~~~~~~

.. option:: --save_cps, --save_checkpoints, default: False

    :obj:`bool`. Only the most recent state of the agent and environment is saved by default. When the
    ``--save_checkpoints`` flag is supplied, a snapshot (checkpoint) of the agent and
    environment will be saved at each epoch. These snapshots are saved in a ``checkpoints``
    folder inside the Logger output directory (for more information, see
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

    :mod:`torch.nn` or :mod:`tf.nn`. The activation function for the neural networks in the actor and critic.

.. option:: --act_out, --ac_kwargs:output_activation

   :mod:`torch.nn` or :mod:`tf.nn`. The activation function for the neural networks in the actor and critic.

.. option:: --act_a, --ac_kwargs:activation:actor

   :mod:`torch.nn` or :mod:`tf.nn`. The activation function for the neural networks in the actor.

.. option:: --act_c, --ac_kwargs:activation:critic

   :mod:`torch.nn` or :mod:`tf.nn`. The activation function for the neural networks in the critic.

.. option:: --act_out_a, --ac_kwargs:output_activation:actor

   :mod:`torch.nn` or :mod:`tf.nn`. The activation function for the output activation function of the actor.

.. option:: --act_out_c, --ac_kwargs:output_activation:critic

   :mod:`torch.nn` or :mod:`tf.nn`. The activation function for the output activation function of the critic.

These flags are valid for all current SLC algorithms.

Config Flags
^^^^^^^^^^^^

These flags are not hyperparameters of any algorithm but change the experimental configuration in some way.

.. option:: --cpu, --num_cpu

    :obj:`int`. If this flag is set, the experiment is launched with this many processes, one per CPU,
    connected by MPI. Some algorithms are amenable to this sort of parallelization, but not all. If you
    try setting ``num_cpu`` > 1 for an incompatible algorithm, an error will be raised. You can also
    set ``--num_cpu auto``, which will automatically use as many CPUs as are available on the machine.

.. option:: --exp_name

    :obj:`str`. The experiment name. This is used in naming the save directory for each experiment. The default
    is "cmd" + [algo name].

.. option:: --data_dir

    :obj:`path str`. Set the base save directory for this experiment or set of experiments. If none is given, 
    the ``DEFAULT_DATA_DIR`` in ``stable_learning_control/user_config.py`` will be used.

.. option:: --dt, --datestamp

    :obj:`bool`. Include the date and time in the name for the save directory of the experiment.

Logger Flags
^^^^^^^^^^^^

The CLI also contains several (shortcut) flags that can be used to change the behaviour of the
:class:`stable_learning_control.utils.log_utils.logx.EpochLogger`.

.. option:: --use_tb, --logger_kwargs:use_tensorboard, default=False

    :obj:`bool`. Enables TensorBoard logging.

.. option:: --tb_log_freq, --logger_kwargs:tb_log_freq, default='low'

    :obj:`str`. The TensorBoard log frequency. Options are ``low`` (Recommended: logs at every epoch) and
    ``high`` (logs at every SGD update batch). Defaults to ``low`` since this is less resource intensive.

.. option:: --use_wandb, --logger_kwargs:use_wandb, default=False

    :obj:`bool`. Enables Weights & Biases logging.

.. option:: --wandb_job_type, --logger_kwargs:wandb_job_type, default='train'

    :obj:`str`. The Weights & Biases job type.

.. option:: --wandb_project, --logger_kwargs:wandb_project, default='stable_learning_control'
    
        :obj:`str`. The Weights & Biases project name.

.. option:: --wandb_group, --logger_kwargs:wandb_group, default=None

    :obj:`str`. The Weights & Biases group name.

.. option:: --quiet, --logger_kwargs:quiet, default=False

    :obj:`bool`. Suppress logging of diagnostics to the stdout.

.. option:: --verbose_fmt, --logger_kwargs:verbose_fmt, default='line'

    :obj:`bool`. The format in which the diagnostics are displayed to the terminal when ``quiet`` is ``False``. 
    Options are ``table``, which supplies them as a table and ``line``, which prints them in one line. 

.. option:: --verbose_vars, --logger_kwargs:verbose_vars, default=None

    :obj:`list`. A list of variables you want to log to the stdout when ``quiet`` is ``False``. The default :obj:`None` means that all variables are logged.

.. important::
    The verbose_vars list should be supplied as a list that can be evaluated in Python (e.g. 
    ``--verbose_vars ["Lr_a", "Lr_c"]``).

.. _exp_cfg:

Using experimental configuration files (yaml)
---------------------------------------------

The SLC CLI comes with a handy configuration file loader that can be used to load `YAML`_ configuration files.
These configuration files provide a convenient way to store your experiments' hyperparameter such that results
can be reproduced. You can supply the CLI with an experiment configuration file using the ``--exp_cfg`` flag.

.. option:: --exp_cfg

    :obj:`path str`. Sets the path to the ``yml`` config file used for loading experiment hyperparameter.

For example, we can use the following command to train a SAC algorithm using the original hyperparameters used
by `Haarnoja et al., 2019`_.

.. code-block:: bash

    python -m stable_learning_control.run --exp_cfg ./experiments/haarnoja_et_al_2019.yml


.. important::
    Please note that if you want to run multiple hyperparameter variants, for example, multiple seeds or
    learning rates, you have to use comma/space-separated strings in your configuration file:

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

    Additionally, if you want to specify a `on/off`_ flag, you can supply an empty key.

.. _`YAML`: https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html
.. _`Haarnoja et al., 2019`: https://arxiv.org/abs/1801.01290
.. _`on/off`: https://docs.python.org/dev/library/argparse.html#core-functionality

Where Results are Saved
-----------------------

Results for a particular experiment (a single run of a configuration of hyperparameters) are stored in

::

    data_dir/[outer_prefix]exp_name[suffix]/[inner_prefix]exp_name[suffix]_s[seed]

where

* ``data_dir`` is the value of the ``--data_dir`` flag (defaults to ``DEFAULT_DATA_DIR`` from 
  ``stable_learning_control/user_config.py`` if ``--data_dir`` is not given),
* the ``outer_prefix`` is a ``YY-MM-DD_`` timestamp if the ``--datestamp`` flag is raised, otherwise nothing,
* the ``inner_prefix`` is a ``YY-MM-DD_HH-MM-SS-`` timestamp if the ``--datestamp`` flag is raised, otherwise nothing,
* and ``suffix`` is a special string based on the experiment hyperparameters.

How is Suffix Determined?
^^^^^^^^^^^^^^^^^^^^^^^^^

Suffixes are only included if you run multiple experiments at once, and they only have references to hyperparameters
that differ across experiments, except for the random seed. The goal is to ensure that results for similar experiments
(ones that share all parameters except the seed) are grouped in the same folder.

Suffixes are constructed by combining *shorthands* for hyperparameters with their values, where a shorthand is either 
1) constructed automatically from the hyperparameter name or 2) supplied by the user. The user can write a shorthand
2) in square brackets after the ``kwarg`` flag.

For example, consider:

.. parsed-literal::

    python -m stable_learning_control.run sac_tf --env Hopper-v2 --hid[h] [300] [128,128] --act tf.nn.tanh tf.nn.relu

Here, the ``--hid`` flag is given a **user-supplied shorthand**, ``h``. The user does not provide the ``--act``
flag with a shorthand, so one will be constructed for it automatically.

The suffixes produced in this case are:

.. parsed-literal::
    _h128-128_ac-actrelu
    _h128-128_ac-acttanh
    _h300_ac-actrelu
    _h300_ac-acttanh

Note that the ``h`` was given by the user. the ``ac-act`` shorthand was constructed from ``ac_kwargs:activation``
(the true name for the ``act`` flag).

Extra
-----

.. admonition:: You don't actually Need to Know This One

    Each individual algorithm is located in a file ``stable_learning_control/algos/BACKEND/ALGO_NAME/ALGO_NAME.py``,
    and these files can be run directly from the command line with a limited set of arguments (some of which differ
    from what's available to ``stable_learning_control/run.py``). However, the command line support in the individual
    algorithm files is vestigial, which is **not** a recommended way to perform experiments.

    This documentation page will not describe those command line calls and *only* describe calls through
    ``stable_learning_control/run.py``.

Use transfer learning
---------------------

The ``start_policy`` command-line flag allows you to use an already trained algorithm as the starting point for
your new algorithm:

.. option:: --start_policy

    :obj:`str`. This flag can be used to train your policy while taking an already-started policy as the starting point.
    It should contain the path to the folder
    where the already trained policy is found.

Using custom environments
-------------------------

The SLC package can be used with any :gymnasium:`Gymnasium-based <>` environment. To use a custom environment, you need
to ensure it inherits from the :class:`gym.Env` class and implements the following methods:

* ``reset(self)``: Reset the environment's state. Returns ``observation, info``.
* ``step(self, action)``: Step the environment by one timestep. Returns ``observation, reward, terminated, truncated, info``.

Additionally, you must ensure that your environment is registered in the :gymnasium:`Gymnasium registry <>`. This
can be done by adding the following lines to your environment file:

.. code-block:: python

    import gymnasium as gym
    from gymnasium.envs.registration import register

    register(
        id='CustomEnv-v0',
        entry_point='path.to.your.env:CustomEnv',
    )

After these requirements are met, you can use it with the SLC package by passing the environment
name to the ``--env`` command-line flag. For example, if your environment is called ``CustomEnv`` and is located in
the file ``custom_env_module.py``, you can run the SLC package with your environment by running:

.. parsed-literal::

    python -m stable_learning_control.run sac --env custom_env_module:CustomEnv-v0

Launching from Scripts
======================

Each algorithm is implemented as a Python function, which can be imported directly from the ``stable_learning_control``
package, eg.

.. code-block::

    >>> from stable_learning_control.control import sac_pytorch as sac

See the documentation page for each algorithm for a complete account of possible arguments. These methods can be used
to set up specialized custom experiments, for example:

.. code-block:: python

    from stable_learning_control.control import sac_tf2 as sac
    import tensorflow as tf
    import gymnasium as gym

    env_fn = lambda : gym.make('LunarLander-v2')

    ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

    logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

    sac(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)

Using ExperimentGrid
--------------------

An easy way to find good hyperparameters is to run the same algorithm with many possible hyperparameters. LC ships with
a simple tool for facilitating this, called :ref:`ExperimentGrid <exp_grid_utility>`.

Consider the example in ``stable_learning_control/examples/pytorch/sac_exp_grid_search.py``:

.. literalinclude:: /../../examples/pytorch/sac_exp_grid_search.py
   :language: python
   :linenos:
   :lines: 16-
   :emphasize-lines: 22-28, 31

After making the ExperimentGrid object, parameters are added to it with

.. parsed-literal::

    eg.add(param_name, values, shorthand, in_name)

where ``in_name`` forces a parameter to appear in the experiment name, even if it has the same value across all experiments.

After all parameters have been added,

.. parsed-literal::

    eg.run(thunk, \*\*run_kwargs)

runs all experiments in the grid (one experiment per valid configuration), by providing the configurations as kwargs to the
function ``thunk``. ``ExperimentGrid.run`` uses a function named :ref:`call_experiment <exp_call_utility>` to launch ``thunk``, and ``**run_kwargs``
specify behaviors for ``call_experiment``. See :ref:`the documentation page <exp_grid_utility>` for details.

Except for the absence of shortcut kwargs (you can't use ``hid`` for ``ac_kwargs:hidden_sizes`` in ``ExperimentGrid``), the
basic behaviour of ``ExperimentGrid`` is the same as running things from the command line.
(In fact, ``stable_learning_control.run`` uses an ``ExperimentGrid`` under the hood.)

..  note::
    
    An equivalent TensorFlow example is available in ``stable_learning_control/examples/tf2/sac_exp_grid_search.py``.
