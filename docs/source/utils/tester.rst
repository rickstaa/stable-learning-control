.. _tester:

=============
Policy tester
=============

.. _test_policy:

Policy eval utility
===================

SLC ships with an evaluation utility that can be used to check a trained policy's performance. In cases where the environment
is successfully saved alongside the agent, it's a cinch to watch the trained agent act in the environment using:

.. parsed-literal::

    python -m stable_learning_control.run test_policy path/to/output_directory

There are a few flags for options:

.. option:: -l L, --len=L, default=0

    :obj:`int`. Maximum length of test episode/trajectory/rollout. The default of ``0`` 
    means no maximum episode length (i.e. episodes only end when the agent has reached a terminal state in the environment).

.. option:: -n N, --episodes=N, default=100

    :obj:`int`. Number of test episodes to run the agent for.

.. option:: -nr, --norender, default=False

    :obj:`bool`. Do not render the test episodes to the screen. In this case, ``test_policy`` will only print the episode returns and lengths. (Use case: the renderer
    slows down the testing process, and you want to get a fast sense of how the agent is performing, so you don't particularly care to watch it.)

.. option:: -i I, --itr=I, default=-1

    :obj:`int`. Specify the snapshot (checkpoint) for which you want to see the policy performance. Use case: Sometimes, it's nice to watch trained agents from many
    different training points (eg watch at iteration 50, 100, 150, etc.). The default value of this flag means "use the latest snapshot."

    .. important::

        This option only works if snapshots were saved while training the agent (i.e. the ``--save_checkpoints`` flag was set). For more information on
        storing these snapshots see :ref:`alg_flags`.

.. option:: -d, --deterministic, default=True

    :obj:`bool`. Another special case, which is only used for the :ref:`SAC <sac>` and :ref:`LAC <lac>` algorithms. The SLC implementation trains a stochastic
    policy, but is evaluated using the deterministic *mean* of the action distribution. ``test_policy`` will default to using the stochastic policy trained
    by SAC, but you should set the deterministic flag to watch the deterministic mean policy (the correct evaluation policy for SAC). This flag is not used
    for any other algorithms.

.. seealso::

    If you receive an "Environment not found" error, see :ref:`manual_policy_testing`.

.. _eval_robustness:

Robustness eval utility
=======================

SLC ships with an evaluation utility that can be used to check the robustness of the trained policy. In cases where the environment
is successfully saved alongside the agent, the robustness can be evaluated using the following command:

.. parsed-literal::

    python -m stable_learning_control.run eval_robustness path/to/output_directory

There are a few flags for options:

.. option:: --data_dir

    :obj:`str`. The folder to which you want to store the robustness eval results.

.. option:: -l L, --len=L, default=0

    :obj:`int`. Maximum length of evaluation episode / trajectory / rollout. The default of ``0`` means no maximum episode length---episodes only end when the agent has
    reached a terminal state in the environment. (Note: setting L=0 will not prevent gymnasium envs wrapped by TimeLimit wrappers from ending when they reach
    their pre-set maximum episode length.)

.. option:: -n N, --episodes=N, default=10

    :obj:`int`. Number of evaluation episodes to run for each disturbance.

.. option:: -r, --render, default=False

    :obj:`bool`. Do also render the evaluation episodes to the screen.

.. option:: -i I, --itr=I, default=-1

    :obj:`int`. Specify the snapshot (checkpoint) for which you want to see the policy performance. Use case: Sometimes, it's nice to evaluate the robustness of the agent from many
    different points in training (e.g. at iteration 50, 100, 150, etc.). The default value of ``-1`` means "use the latest snapshot."

    .. important::

        This option only works if snapshots were saved while training the agent (i.e. the ``--save_checkpoints`` flag was set). For more information on
        storing these snapshots, see :ref:`alg_flags`.

.. option:: -d, --deterministic, default=True

    :obj:`bool`. Another special case, which is only used for the :ref:`SAC <sac>` and :ref:`LAC <lac>` algorithms. The SLC implementation trains a stochastic
    policy, but is evaluated using the deterministic *mean* of the action distribution. ``test_policy`` will default to using the stochastic policy trained
    by SAC, but you should set the deterministic flag to watch the deterministic mean policy (the correct evaluation policy for SAC). This flag is not used
    for any other algorithms.

.. option:: --save_result, default=False

    :obj:`bool`. Whether you want to save the robustness evaluation data frame to disk. It can be useful for creating custom plots see :ref:`robust_custom_plots`.

.. option:: --list_disturbance_types, default=False

    :obj:`bool`. Lists the available disturbance types for the trained agent and stored environment.

.. option:: --list_disturbance_variants, default=False

    :obj:`bool`. Lists the available disturbance variants that are available for a given disturbance type.

.. option:: -d_type, --disturbance_type

    :obj:`str`. The disturbance type you want to apply. This type should be implemented in the :class:`~stable_gym.common.disturber.Disturber`
    your gym environment inherits from. See :ref:`env_add`.

.. option:: -d_variant, --disturbance_variant

    :obj:`str`. The disturbance variant you want to apply. This argument is only required for some disturbance types. The variant should be implemented in the
    :class:`~stable_gym.common.disturber.Disturber`
    your gym environment inherits from. See :ref:`env_add`.

.. option:: --disable_baseline, default=False

    :obj:`bool`. Specifies whether you want to automatically disable the baseline (i.e., zero disturbance) from being added to the disturbance array.

.. option:: --obs, default=None

    *:obj:`list of ints`*. The observations you want to show in the observations/reference plots. The default value of :obj:`None` means all observations will be shown.

.. option:: --merged, default=False

    :obj:`bool`. Specifies whether you want to merge all observations into one plot. By default, observations under each disturbance are shown in a separate subplot.

.. option:: --save_figs, default=True

    :obj:`bool`. Specifies whether you want to save the generated plots to disk.

.. option:: --figs_fmt, default=pdf

    :obj:`bool`. The file format you want to use for saving the plot.

.. option:: --font_scale, default=1.5

    :obj:`float`. The font scale you want to use for the plot text.

.. seealso::

    If you receive an "Environment not found" error, see :ref:`manual_policy_testing`.
