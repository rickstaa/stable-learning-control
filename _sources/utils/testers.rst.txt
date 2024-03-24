.. _testers:

==============
Policy testers
==============

.. _test_policy:

Policy eval utility
===================

SLC ships with an evaluation utility that can be used to check a trained policy's performance. In cases where the environment
is successfully saved alongside the agent, it's a cinch to watch the trained agent act in the environment using:

.. parsed-literal::

    python -m stable_learning_control.run test_policy [path/to/output_directory] [-h] 
        [--len LEN] [--episodes EPISODES] [--norender] [--itr ITR] [--deterministic] 

**Positional Arguments:**

.. option:: output_dir

    :obj:`str`. The path to the output directory where the agent and environment were saved. 

**Optional Arguments:**

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

    python -m stable_learning_control.run eval_robustness [path/to/output_directory] [disturber] [-h] [--list_disturbers]
        [--disturber_config DISTURBER_CONFIG] [--data_dir DATA_DIR] [--itr ITR]
        [--len LEN] [--episodes EPISODES] [--render] [--deterministic] [--disable_baseline]
        [--observations [OBSERVATIONS [OBSERVATIONS ...]]] [--references [REFERENCES [REFERENCES ...]]]
        [--reference_errors [REFERENCE_ERRORS [REFERENCE_ERRORS ...]]] [--absolute_reference_errors]
        [--merge_reference_errors] [--use_subplots] [--use_time] [--save_result] [--save_plots]
        [--figs_fmt FIGS_FMT] [--font_scale FONT_SCALE]

**Positional Arguments:**

.. option:: output_dir

    :obj:`str`. The path to the output directory where the agent and environment were saved. 

.. option:: disturber

    :obj:`str`. The name of the disturber you want to evaluate. Can include an unloaded module in 'module:disturber_name' style.

**Optional Arguments:**

.. option:: --list, --list_disturbers, default=False

    :obj:`bool`. Lists the available disturbers found in the SLC package.

.. option:: --cfg, --disturber_config DISTURBER_CONFIG, default=None

    :obj:`str`. The configuration you want to pass to the disturber. It sets up the range of disturbances you wish to evaluate. Expects a dictionary that depends on the specified disturber (e.g. ``"{'mean': [0.25, 0.25], 'std': [0.05, 0.05]}"`` for :class:`~stable_learning_control.disturbers.ObservationRandomNoiseDisturber` disturber).

.. option:: --data_dir

    :obj:`str`. The folder to which you want to store the robustness eval results, meaning the data frame and the plots.

.. option:: -i I, --itr=I, default=-1

    :obj:`int`. Specify the snapshot (checkpoint) for which you want to see the policy performance. Use case: Sometimes, it's nice to evaluate the robustness of the agent from many
    different points in training (e.g. at iteration 50, 100, 150, etc.). The default value of ``-1`` means "use the latest snapshot."

    .. important::
        This option only works if snapshots were saved while training the agent (i.e. the ``--save_checkpoints`` flag was set). For more information on
        storing these snapshots, see :ref:`alg_flags`.

.. option:: -l L, --len=L, default=None

    :obj:`int`. Maximum length of evaluation episode / trajectory / rollout. The default of ``None`` means no maximum episode length---episodes only end when the agent has
    reached a terminal state in the environment.

.. option:: -n N, --episodes=N, default=100

    :obj:`int`. Number of evaluation episodes to run for each disturbance variant.

.. option:: -r, --render, default=False

    :obj:`bool`. Do also render the evaluation episodes to the screen.

.. option:: -d, --deterministic, default=False

    :obj:`bool`. Another special case, which is only used for the :ref:`SAC <sac>` and :ref:`LAC <lac>` algorithms. The SLC implementation trains a stochastic
    policy, but is evaluated using the deterministic *mean* of the action distribution. ``test_policy`` will default to using the stochastic policy trained
    by SAC, but you should set the deterministic flag to watch the deterministic mean policy (the correct evaluation policy for SAC). This flag is not used
    for any other algorithms.

.. option:: --disable_baseline, default=False

    :obj:`bool`. Disable the baseline evaluation. The baseline evaluation is a special case where the agent is evaluated without any disturbance applied. This
    is useful for comparing the performance of the agent with and without the disturbance.

.. option:: --obs, --observations, default=None

    *:obj:`list of ints`*. The observations you want to show in the observations/reference plots. The default value of :obj:`None` means all observations will be shown.

.. option:: --refs, --references, default=None

    *:obj:`list of ints`*. The references you want to show in the observations/reference plots. The default value of :obj:`None` means all references will be shown.

.. option:: --ref_errs, --reference_errors, default=None

    *:obj:`list of ints`*. The reference errors you want to show in the reference error plots. The default value of :obj:`None` means all reference errors will be shown.

.. option:: --abs_ref_errs, --absolute_reference_errors, default=False

    :obj:`bool`. Whether you want to show the absolute reference errors in the reference error plots. The default value of :obj:`False` means the relative reference errors will be shown.

.. option:: --merge_ref_errs, --merge_reference_errors, default=False

    :obj:`bool`. Whether you want to merge the reference errors into one reference error. The default value of :obj:`False` means the reference errors will be shown separately.

.. option:: --use_subplots, default=False

    :obj:`bool`. Whether you want to use subplots for the plots. The default value of :obj:`False` means the plots will be shown separately.

.. option:: --use_time, default=False

    :obj:`bool`. Whether you want to use time as the x-axis for the plots. The default value of :obj:`False` means the x-axis will show the steps.

.. option:: --save_result, default=False

    :obj:`bool`. Whether you want to save the robustness evaluation data frame to disk. It can be useful for creating custom plots see :ref:`robust_custom_plots`.

.. option:: --save_plots, default=False

    :obj:`bool`. Specifies whether you want to save the generated plots to disk.

.. option:: --figs_fmt, default=pdf

    :obj:`bool`. The file format you want to use for saving the plot.

.. option:: --font_scale, default=1.5

    :obj:`float`. The font scale you want to use for the plot text.

.. option:: --use_wandb, default=False

    :obj:`bool`. Whether you want log the results to Weights & Biases.

.. option:: --wandb_job_type, default=eval

    :obj:`str`. The job type you want to use for the Weights & Biases logging.

.. option:: --wandb_project, default=stable-learning-control
    
        :obj:`str`. The name of the Weights & Biases project you want to log to.

.. option:: --wandb_group, default=None

    :obj:`str`. The name of the Weights & Biases group you want to log to.

.. option:: --wandb_run_name, default=None

    :obj:`str`. The name of the Weights & Biases run you want to log to. If not
    specified, the run name will be automatically generated based on the policy
    directory and disturber.

.. seealso::
    If you receive an "Environment not found" error, see :ref:`manual_policy_testing`.
