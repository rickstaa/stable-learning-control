"""A set of functions that can be used to evaluate the stability and robustness of an
algorithm. This is done by evaluating an algorithm's performance under two types of
disturbances: A disturbance that is applied during the environment step and a
perturbation added to the environmental parameters. For the functions in this
module to work work, these disturbances should be implemented as methods on the
environment. The Simzoo package contains a
:class:`~bayesian_learning_control.simzoo.simzoo.common.disturber.Disturber` class from which a Gym environment
can inherit to add these methods. See the
`Robustness Evaluation Documentation <https://rickstaa.github.io/bayesian-learning-control/control/eval_robustness.html>`_
for more information.
"""  # noqa: E501
# TODO: Add ability to set output folder

import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bayesian_learning_control.control.utils.test_policy import load_policy_and_env
from bayesian_learning_control.utils.log_utils import (
    EpochLogger,
    friendly_err,
    log_to_std_out,
)

REQUIRED_DISTURBER_OBJECTS = {
    "methods": [
        "init_disturber",
        "disturbed_step",
        "next_disturbance",
    ],
    "attributes": [
        "disturber_done",
    ],
}


def _validate_observations(observations, obs_dataframe):
    """Checks if the request observations exist in the ``obs_dataframe`` displays a
    warning if they do not.

    Args:
        observations (list): The requested observations.
        obs_dataframe (pandas.DataFrame): The dataframe with the observations that are
            present.

    Returns:
        list: List with the observations that are present in the dataframe.
    """
    valid_vals = obs_dataframe.observation.unique()
    if observations is None:
        return list(valid_vals)
    else:
        invalid_vals = [obs for obs in map(int, observations) if obs not in valid_vals]
        valid_observations = [
            obs for obs in map(int, observations) if obs in valid_vals
        ]
        if len(observations) == len(invalid_vals):
            log_to_std_out(
                "{} not valid. All observations plotted instead.".format(
                    f"Observations {invalid_vals} are"
                    if len(invalid_vals) > 1
                    else f"Observation {invalid_vals[0]} is"
                ),
                type="warning",
            )
            valid_observations = list(valid_vals)
        elif invalid_vals:
            log_to_std_out(
                "{} not valid.".format(
                    f"Observations {invalid_vals} could not plotted as they are"
                    if len(invalid_vals) > 1
                    else f"Observation {invalid_vals[0]} could not be plotted as it is"
                ),
                type="warning",
            )
        return valid_observations


def _disturber_implemented(env):
    """Checks if the environment inherits from the
    :class:`~bayesian_learning_control.simzoo.common.disturber.Disturber`
    class or that the methods and attributes that are required for the robustness
    evaluation are present.

    Returns:
        (tuple): tuple containing:

            - compatible (:obj:`bool`): Wether the environment is compatible with the
                robustness evaluation tool.
            - missing_attributes (:obj:`dict`): Dictionary that contains the 'methods'
                and 'attributes' that are missing.
    """
    missing_methods_mask = [
        not hasattr(env, obj) for obj in REQUIRED_DISTURBER_OBJECTS["methods"]
    ]
    missing_attributes_mask = [
        not hasattr(env, obj) for obj in REQUIRED_DISTURBER_OBJECTS["attributes"]
    ]
    missing_methods = [
        i
        for (i, v) in zip(REQUIRED_DISTURBER_OBJECTS["methods"], missing_methods_mask)
        if v
    ]
    missing_attributes = [
        i
        for (i, v) in zip(
            REQUIRED_DISTURBER_OBJECTS["attributes"], missing_attributes_mask
        )
        if v
    ]
    return not all(missing_methods_mask + missing_attributes_mask), {
        "methods": missing_methods,
        "attributes": missing_attributes,
    }


def run_disturbed_policy(  # noqa: C901
    env,
    policy,
    disturbance_type,
    disturbance_variant=None,
    max_ep_len=None,
    num_episodes=10,
    render=True,
    deterministic=True,
    save_result=False,
    output_dir=None,
):
    """Evaluates the disturbed policy inside a given gym environment. This function
    loops to all the disturbances that are specified in the environment and outputs the
    results of all these episodes in a pandas dataframe.

    Args:
        env (:obj:`gym.env`): The gym environment.
        policy (Union[tf.keras.Model, torch.nn.Module]): The policy.
        disturbance_type (str): The disturbance type you want to apply. Valid options
            are the onces that are implemented in the gym environment (e.g.
            ``env``, ``input``, ``output``, ``combined``, ...).
        disturbance_variant (str, optional): The variant of the disturbance (e.g.
            ``impulse``, ``periodic``, ``noise``,...)
        max_ep_len (int, optional): The maximum episode length. Defaults to None.
        num_episodes (int, optional): Number of episodes you want to perform in the
            environment. Defaults to 100.
        render (bool, optional): Whether you want to render the episode to the screen.
            Defaults to ``True``.
        deterministic (bool, optional): Whether you want the action from the policy to
            be deterministic. Defaults to ``True``.
        save_result (bool, optional): Whether you want to save the dataframe with the
            results to disk.
        output_dir (str, optional): A directory for saving the diagnostics to. If
            ``None``, defaults to a temp directory of the form
            ``/tmp/experiments/somerandomnumber``.

    Returns:
        :obj:`pandas.DataFrame`:
            Dataframe that contains information about all the episodes and disturbances.

    Raises:
        RuntimeError: Thrown when the environment you specified is not compatible with
            the robustness evaluation tool.
        ValueError: Thrown when the disturbance type or variant is not supported by the
            disturber.
        TypeError: Thrown when no disturbance variant supplied and it is needed for the
            requested disturbance type.
    """  # noqa: E501
    # Validate environment, environment disturber
    assert env is not None, friendly_err(
        "Environment not found!\n\n It looks like the environment wasn't saved, "
        + "and we can't run the agent in it. :( \n\n Check out the documentation "
        + "page on the robustness evaluation utility for how to handle this situation."
    )
    disturber_implemented, missing_objects = _disturber_implemented(env)
    if not disturber_implemented:
        missing_keys = [key for key, item in missing_objects.items() if len(item) >= 1]
        missign_methods_str = [
            key if len(item) > 1 else key[:-1]
            for key, item in missing_objects.items()
            if len(item) >= 1
        ]
        missing_warn_string = (
            (
                f"{missing_objects[missing_keys[0]]} "
                f"{missign_methods_str[0]} and "
                f"{missing_objects[missing_keys[1]]} "
                f"{missign_methods_str[1]}"
            )
            if len(missing_keys) > 1
            else (f"{missing_objects[missing_keys[0]]} " f"{missign_methods_str[0]}")
        )
        raise RuntimeError(
            "The environment does not seem to be compatible with the robustness "
            f"evaluation tool. The tool expects to find the {missing_warn_string} but "
            "they are not implemented. Please check the Robustness Evaluation "
            "documentation and try again."
        )

    output_dir = (
        Path(output_dir).joinpath("eval") if output_dir is not None else output_dir
    )
    logger = EpochLogger(
        verbose_fmt="table", output_dir=output_dir, output_fname="eval_statistics.csv"
    )

    if max_ep_len is None:
        max_ep_len = env._max_episode_steps
    else:
        if max_ep_len > env._max_episode_steps:
            logger.log(
                (
                    f"You defined your 'max_ep_len' to be {max_ep_len} "
                    "while the environment 'max_episide_steps' is "
                    f"{env._max_episode_steps}. As a result the environment "
                    f"'max_episode_steps' has been increased to {max_ep_len}"
                ),
                type="warning",
            )
            env._max_episode_steps = max_ep_len

    # Try to retrieve default type if type not given
    if disturbance_type is None:
        if hasattr(env.unwrapped, "_disturber_cfg"):
            if "default_type" in env.unwrapped._disturber_cfg.keys():
                disturbance_type = env.unwrapped._disturber_cfg["default_type"]
                log_to_std_out(
                    (
                        "INFO: No disturbance type given default type "
                        f"'{disturbance_type}' used instead."
                    ),
                    type="info",
                )

    # Initialize the disturber!
    try:
        env.init_disturber(disturbance_type, disturbance_variant),
    except (ValueError, TypeError) as e:
        if len(e.args) > 1:
            raise Exception(
                friendly_err(
                    "You did not give a valid value for --{}! Please try again.".format(
                        e.args[1]
                    )
                )
            ) from e
        else:
            raise e
    disturbance_type = (
        env.disturbance_info["type"]
        if hasattr(env, "disturbance_info") and "type" in env.disturbance_info.keys()
        else disturbance_type
    )  # Retrieve used disturbance type
    disturbance_variant = (
        env.disturbance_info["variant"]
        if hasattr(env, "disturbance_info") and "variant" in env.disturbance_info.keys()
        else disturbance_variant
    )  # Retrieve used disturbance variant

    # Loop though all disturbances till disturber is done
    logger.log("Starting robustness evaluation...", type="info")
    render_error = False
    path = {
        "o": [],
        "r": [],
        "reference": [],
        "state_of_interest": [],
    }
    o_episodes_dfs, r_episodes_dfs, soi_episodes_dfs, ref_episodes_dfs = [], [], [], []
    (
        o_disturbances_dfs,
        r_disturbances_dfs,
        soi_disturbances_dfs,
        ref_disturbances_dfs,
    ) = ([], [], [], [])
    n_disturbance = 0
    soi_found, ref_found = True, True
    while not env.disturber_done:
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
        supports_deterministic = True  # Only supported with gaussian algorithms
        while n < num_episodes:
            # Render env if requested
            if render and not render_error:
                try:
                    env.render()
                    time.sleep(1e-3)
                except NotImplementedError:
                    render_error = True
                    logger.log(
                        (
                            "WARNING: Nothing was rendered since no render method was "
                            f"implemented for the '{env.unwrapped.spec.id}' "
                            "environment."
                        ),
                        type="warning",
                    )

            # Retrieve action
            if deterministic and supports_deterministic:
                try:
                    a = policy.get_action(o, deterministic=deterministic)
                except TypeError:
                    supports_deterministic = False
                    logger.log(
                        "Input argument 'deterministic' ignored as the algorithm does "
                        "not support deterministic actions. This is only supported for "
                        "gaussian  algorithms.",
                        type="warning",
                    )
                    a = policy.get_action(o)
            else:
                a = policy.get_action(o)

            # Perform (disturbed) action in the environment and store result
            if disturbance_type == "env":
                o, r, d, info = env.step(a)
            else:
                o, r, d, info = env.disturbed_step(a)
            ep_ret += r
            ep_len += 1

            # Store path, cost, reference and state of interest
            path["o"].append(o)
            path["r"].append(r)
            if "reference" in info.keys():
                path["reference"].append(info["reference"])
            else:
                ref_found = False
            if "state_of_interest" in info.keys():
                path["state_of_interest"].append(info["state_of_interest"])
            else:
                soi_found = False
            if any([not ref_found, not soi_found]):
                warning_str = (
                    " 'state_of_interest' and 'reference' "
                    if all([not ref_found, not soi_found])
                    else (" 'state_of_interest' " if not soi_found else " 'reference' ")
                )
                logger.log(
                    (
                        f"No{warning_str}variable found in the info dictionary that "
                        "was returned from the environment step method. In order to "
                        "use all feature from the robustness evaluation tool, please "
                        "make sure your environment returns the reference."
                    ),
                    type="warning",
                )

            # Store performance measurements
            if d or (ep_len == max_ep_len):
                died = ep_len < max_ep_len
                logger.store(EpRet=ep_ret, EpLen=ep_len, DeathRate=(float(died)))
                logger.log(
                    "Episode %d \t EpRet %.3f \t EpLen %d \t Died %s"
                    % (n, ep_ret, ep_len, died)
                )

                # Store observations
                o_episode_df = pd.DataFrame(path["o"])
                o_episode_df.insert(0, "step", range(0, ep_len))
                o_episode_df = pd.melt(
                    o_episode_df,
                    id_vars="step",
                    var_name="observation",
                )  # Flatten dataframe
                o_episodes_dfs.append(o_episode_df)

                # Store episode rewards
                r_episode_df = pd.DataFrame(
                    {
                        "step": range(0, ep_len),
                        "reward": path["r"],
                    }
                )
                r_episode_df.insert(len(r_episode_df.columns), "episode", n)
                r_episodes_dfs.append(r_episode_df)

                # Store states of interest
                if soi_found:
                    soi_episode_df = pd.DataFrame(path["state_of_interest"])
                    soi_episode_df.insert(0, "step", range(0, ep_len))
                    soi_episode_df = pd.melt(
                        soi_episode_df,
                        id_vars="step",
                        var_name="state_of_interest",
                        value_name="error",
                    )  # Flatten dataframe
                    soi_episodes_dfs.append(soi_episode_df)

                # Store reference
                if ref_found:
                    ref_episode_df = pd.DataFrame(path["reference"])
                    ref_episode_df.insert(0, "step", range(0, ep_len))
                    ref_episode_df = pd.melt(
                        ref_episode_df,
                        id_vars="step",
                        var_name="reference",
                    )  # Flatten dataframe
                    ref_episodes_dfs.append(ref_episode_df)

                # Increment counters and reset storage variables
                n += 1
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                path = {
                    "o": [],
                    "r": [],
                    "reference": [],
                    "state_of_interest": [],
                }

        # Print robustness evaluation diagnostics
        if hasattr(env, "disturbance_info") and "type" in env.disturbance_info.keys():
            logger.log_tabular(
                "DisturbanceType",
                env.disturbance_info["type"].replace("_disturbance", ""),
            )
        if (
            hasattr(env, "disturbance_info")
            and "variant" in env.disturbance_info.keys()
        ):
            logger.log_tabular("DisturbanceVariant", env.disturbance_info["variant"])
        if hasattr(env, "disturbance_info") and (
            "variable" in env.disturbance_info.keys()
            and "value" in env.disturbance_info.keys()
        ):
            if isinstance(env.disturbance_info["value"], dict):
                for key, val in env.disturbance_info["value"].items():
                    logger.log_tabular(
                        "{}_{}".format(env.disturbance_info["variable"], key), val
                    )
            else:
                logger.log_tabular(
                    env.disturbance_info["variable"], env.disturbance_info["value"]
                )
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("DeathRate")
        log_to_std_out("")
        logger.dump_tabular()

        # Add extra disturbance information to the robustness eval dataframe
        disturbance_label = (
            env.disturbance_info["label"]
            if (
                hasattr(env, "disturbance_info")
                and "label" in env.disturbance_info.keys()
            )
            else "Disturbance: {}".format(str(n_disturbance + 1))
        )
        o_disturbance_df = pd.concat(o_episodes_dfs, ignore_index=True)
        o_disturbance_df.insert(
            len(o_disturbance_df.columns), "disturbance", disturbance_label
        )
        o_disturbance_df.insert(
            len(o_disturbance_df.columns), "disturbance_index", n_disturbance
        )
        o_disturbances_dfs.append(o_disturbance_df)
        r_disturbance_df = pd.concat(r_episodes_dfs, ignore_index=True)
        r_disturbance_df.insert(
            len(r_disturbance_df.columns), "disturbance", disturbance_label
        )
        r_disturbance_df.insert(
            len(r_disturbance_df.columns), "disturbance_index", n_disturbance
        )
        r_disturbances_dfs.append(r_disturbance_df)
        soi_disturbance_df = pd.concat(soi_episodes_dfs, ignore_index=True)
        soi_disturbance_df.insert(
            len(soi_disturbance_df.columns), "disturbance", disturbance_label
        )
        soi_disturbance_df.insert(
            len(soi_disturbance_df.columns), "disturbance_index", n_disturbance
        )
        soi_disturbances_dfs.append(soi_disturbance_df)
        ref_disturbance_df = pd.concat(ref_episodes_dfs, ignore_index=True)
        ref_disturbance_df.insert(
            len(ref_disturbance_df.columns), "disturbance", disturbance_label
        )
        ref_disturbance_df.insert(
            len(ref_disturbance_df.columns), "disturbance_index", n_disturbance
        )
        ref_disturbances_dfs.append(ref_disturbance_df)

        # Reset storage buckeets and go to next disturbance
        o_episodes_dfs = []
        r_episodes_dfs = []
        soi_episodes_dfs = []
        ref_episodes_dfs = []
        env.next_disturbance()
        n_disturbance += 1

    # Merge robustness evaluation information for all disturbances
    o_disturbances_df = pd.concat(o_disturbances_dfs, ignore_index=True)
    r_disturbances_df = pd.concat(r_disturbances_dfs, ignore_index=True)
    soi_disturbances_df = pd.concat(soi_disturbances_dfs, ignore_index=True)
    ref_disturbances_df = pd.concat(ref_disturbances_dfs, ignore_index=True)

    # Return/save robustness evaluation dataframe
    o_disturbances_df.insert(len(o_disturbances_df.columns), "variable", "observation")
    r_disturbances_df.insert(len(r_disturbances_df.columns), "variable", "reward")
    soi_disturbances_df.insert(
        len(soi_disturbances_df.columns), "variable", "state_of_interest"
    )
    ref_disturbances_df.insert(
        len(ref_disturbances_df.columns), "variable", "reference"
    )
    robustness_eval_df = pd.concat(
        [
            o_disturbances_df,
            r_disturbances_df,
            soi_disturbances_df,
            ref_disturbances_df,
        ],
        ignore_index=True,
    )
    robustness_eval_df.insert(
        len(robustness_eval_df.columns),
        "disturbance_type",
        disturbance_type,
    )
    robustness_eval_df.insert(
        len(robustness_eval_df.columns),
        "disturbance_variant",
        disturbance_variant,
    )

    # Save robustness evaluation dataframe and return it to the user
    if save_result:
        results_path = logger.output_dir.joinpath("results.csv")
        logger.log(
            f"Saving robustness evaluation results to path: {results_path}", type="info"
        )
        robustness_eval_df.to_csv(results_path, index=False)
    return robustness_eval_df


def plot_robustness_results(  # noqa: C901
    dataframe,
    save_figs,
    output_dir,
    font_scale=1.5,
    observations=None,
    figs_fmt="pdf",
):
    """Creates several usefull plots out of the dataframe that was collected in the
    :meth:`run_disturbed_policy` method.

    Args:
        dataframe (pandas.DataFrame): The data frame that contains the robustness
            information.
        save_figs (bool): Whether you want to save the created plots to disk.
        output_dir (str): The directory where you want to save the output figures to.
        font_scale (int): The font scale you want to use for the plot text. Defaults to
            ``1.5``.
        observations (list): The observations you want to show in the observations plot.
            By default all observations are shown.
        figs_fmt (str, optional): In which format you want to save the plots. Defaults
            to ``pdf``.
    """
    output_dir = Path(output_dir).joinpath("eval")
    figs = {
        "observations": [],
        "costs": [],
        "states_of_interest": [],
    }  # Store all plots (Needed for save)
    log_to_std_out("Showing robustness evaluation plots...", type="info")
    sns.set(style="darkgrid", font_scale=font_scale)
    time_instant = None
    if hasattr(env, "disturbance_info") and "cfg" in env.disturbance_info.keys():
        time_instant_keys = [
            key for key in env.disturbance_info["cfg"].keys() if "_instant" in key
        ]
        time_instant = (
            env.disturbance_info["cfg"][time_instant_keys[0]]
            if time_instant_keys
            else None
        )

    # Unpack required data from dataframe
    obs_found, rew_found, soi_found, ref_found = True, True, True, True
    o_disturbances_df, ref_disturbances_df = pd.DataFrame(), pd.DataFrame()
    if "observation" in dataframe["variable"].unique():
        o_disturbances_df = dataframe.query("variable == 'observation'").dropna(
            axis=1, how="all"
        )
    else:
        obs_found = False
    if "reward" in dataframe["variable"].unique():
        r_disturbances_df = dataframe.query("variable == 'reward'").dropna(
            axis=1, how="all"
        )
    else:
        rew_found = False
    if "state_of_interest" in dataframe["variable"].unique():
        soi_disturbances_df = dataframe.query("variable == 'state_of_interest'").dropna(
            axis=1, how="all"
        )
    else:
        soi_found = False
    if "state_of_interest" in dataframe["variable"].unique():
        ref_disturbances_df = dataframe.query("variable == 'reference'").dropna(
            axis=1, how="all"
        )
    else:
        ref_found = False

    # Merge observations and references
    if obs_found:
        obs_df_tmp = o_disturbances_df.copy(deep=True)
        obs_df_tmp["signal"] = "obs_" + (obs_df_tmp["observation"] + 1).astype(str)
        obs_df_tmp.insert(len(obs_df_tmp.columns), "type", "observation")

        # Retrieve the requested observations
        observations = _validate_observations(observations, o_disturbances_df)
        observations = [obs - 1 for obs in observations]  # Humans count from 1
        obs_df_tmp = obs_df_tmp.query(f"observation in {observations}")
    if ref_found:
        ref_df_tmp = ref_disturbances_df.copy(deep=True)
        ref_df_tmp["signal"] = "ref_" + (ref_df_tmp["reference"] + 1).astype(str)
        ref_df_tmp.insert(len(ref_df_tmp.columns), "type", "reference")
    obs_ref_df = pd.concat([obs_df_tmp, ref_df_tmp], ignore_index=True)

    # Loop though all disturbances and plot the observations and references in one plot
    num_plots = len(obs_ref_df.disturbance.unique())
    total_cols = 3
    total_rows = math.ceil(num_plots / total_cols)
    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=total_cols,
        figsize=(7 * total_cols, 7 * total_rows),
        tight_layout=True,
        sharex=True,
    )
    fig.suptitle(
        "{} under several {}{}.".format(
            "Observation and reference"
            if all([obs_found, ref_found])
            else ("Observation" if obs_found else "reference"),
            "{} disturbances".format(obs_ref_df.disturbance_variant[0])
            if "disturbance_variant" in obs_ref_df.keys()
            else "disturbances",
            f" at step {time_instant}" if time_instant else "",
        )
    )
    obs_ref_df.loc[obs_ref_df["disturbance_index"] == 0, "disturbance"] = (
        obs_ref_df.loc[obs_ref_df["disturbance_index"] == 0, "disturbance"]
        + " (original)"
    )  # Append original to original value
    for ii, var in enumerate(obs_ref_df.disturbance.unique()):
        row = ii // total_cols
        pos = ii % total_cols
        sns.lineplot(
            data=obs_ref_df.query(f"disturbance == '{var}'"),
            x="step",
            y="value",
            ci="sd",
            hue="signal",
            style="type",
            ax=axes[row][pos],
            legend="auto" if ii == 0 else False,
        ).set_title(var)
    plt.figlegend(loc="center right")
    axes[0][0].get_legend().remove()
    figs["observations"].append(fig)

    # Plot mean cost
    if rew_found:
        fig = plt.figure(tight_layout=True)
        figs["costs"].append(fig)
        r_disturbances_df.loc[
            r_disturbances_df["disturbance_index"] == 0, "disturbance"
        ] = (
            r_disturbances_df.loc[
                r_disturbances_df["disturbance_index"] == 0, "disturbance"
            ]
            + " (original)"
        )  # Append original to original value
        sns.lineplot(
            data=r_disturbances_df, x="step", y="reward", ci="sd", hue="disturbance"
        ).set_title(
            "Mean cost under several {}{}.".format(
                "{} disturbances".format(obs_ref_df.disturbance_variant[0])
                if "disturbance_variant" in obs_ref_df.keys()
                else "disturbances",
                f" at step {time_instant}" if time_instant else "",
            )
        )
    else:
        log_to_std_out(
            (
                "Mean costs plot could not we shown as no 'rewards' field was found ",
                "in the supplied dataframe.",
            ),
            type="warning",
        )

    # Plot states of interest
    if soi_found:
        n_soi = soi_disturbances_df["state_of_interest"].max() + 1
        soi_disturbances_df.loc[
            soi_disturbances_df["disturbance_index"] == 0, "disturbance"
        ] = (
            soi_disturbances_df.loc[
                soi_disturbances_df["disturbance_index"] == 0, "disturbance"
            ]
            + " (original)"
        )  # Append original to original value
        for index in range(0, n_soi):
            fig = plt.figure(tight_layout=True)
            figs["states_of_interest"].append(fig)
            sns.lineplot(
                data=soi_disturbances_df.query(f"state_of_interest == {index}"),
                x="step",
                y="error",
                ci="sd",
                hue="disturbance",
            ).set_title(
                "{} under several {}{}.".format(
                    "State of interest" if n_soi == 1 else f"State of interest {index}",
                    "{} disturbances".format(obs_ref_df.disturbance_variant[0])
                    if "disturbance_variant" in obs_ref_df.keys()
                    else "disturbances",
                    f" at step {time_instant}" if time_instant else "",
                )
            )
        plt.show()
    else:
        log_to_std_out(
            (
                "State of interest plot could not we shown as no 'state_of_interest' "
                "field was found in the supplied dataframe.",
            ),
            type="warning",
        )

    # Save plots
    if save_figs:
        figs_path = output_dir.joinpath("figures")
        figs_extension = figs_fmt[1:] if figs_fmt.startswith(".") else figs_fmt
        os.makedirs(figs_path, exist_ok=True)
        log_to_std_out("Saving plots...", type="info")
        log_to_std_out(f"Saving figures to path: {figs_path}", type="info")
        if obs_found or ref_found:
            figs["observations"][0].savefig(
                output_dir.joinpath("figures", f"obserations.{figs_extension}"),
                bbox_inches="tight",
            )
        if rew_found:
            figs["costs"][0].savefig(
                output_dir.joinpath("figures", f"costs.{figs_extension}"),
                bbox_inches="tight",
            )
        if soi_found:
            for index, fig in enumerate(figs["states_of_interest"]):
                fig.savefig(
                    output_dir.joinpath(
                        "figures",
                        f"soi.{figs_extension}" if n_soi == 1 else f"soi_{index}.pdf",
                    ),
                    bbox_inches="tight",
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str, help="The path where the policy is stored")
    parser.add_argument(
        "--len",
        "-l",
        type=int,
        default=None,
        help="The episode length (defaults to environment max_episode_steps)",
    )
    parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=10,
        help="The number of episodes you want to run per disturbance (default: 10)",
    )
    parser.add_argument(
        "--render",
        "-r",
        action="store_true",
        help="Whether you want to render the environment step (default: True)",
    )
    parser.add_argument(
        "--itr",
        "-i",
        type=int,
        default=-1,
        help="The policy iteration (epoch) you want to use (default: 'last')",
    )
    parser.add_argument(
        "--deterministic",
        "-d",
        action="store_true",
        help="Wether you want to use a deterministic policy (default: True)",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help=(
            "Whether you want to save the robustness evaluation dataframe to disk "
            "(default: False)"
        ),
    )
    parser.add_argument(
        "--disturbance_type",
        "-d_type",
        type=str,
        help="The disturbance type you want to investigate",
    )
    parser.add_argument(
        "--disturbance_variant",
        "-d_variant",
        type=str,
        default=None,
        help=(
            "The disturbance variant you want to investigate (only required for some "
            "disturbance types)"
        ),
    )
    parser.add_argument(
        "--obs",
        default=None,
        nargs="+",
        help="The observations you want to show in the observations/references plot",
    )
    parser.add_argument(
        "--save_figs",
        action="store_true",
        help="Whether you want to save the plots (default: True)",
    )
    parser.add_argument(
        "--figs_fmt",
        default="pdf",
        help="The filetype you want to use for the plots (default: pdf)",
    )
    parser.add_argument(
        "--font_scale",
        default=1.5,
        help="The font scale you want to use for the plot text.",
    )
    parser.add_argument(
        "--list_disturbance_types",
        action="store_true",
        help=(
            "Lists the available disturbance types for the trained agent and stored "
            "environment."
        ),
    )
    parser.add_argument(
        "--list_disturbance_variants",
        action="store_true",
        help=(
            "Lists the available disturbance variants that are available for a given "
            "disturbance type."
        ),
    )
    args = parser.parse_args()

    # Load policy and environment
    env, policy = load_policy_and_env(args.fpath, args.itr if args.itr >= 0 else "last")

    # List d_type or d_variant if requested
    if args.list_disturbance_types or args.list_disturbance_variants:
        if hasattr(env.unwrapped, "_disturber_cfg"):
            if args.list_disturbance_types:
                d_type_info_msg = (
                    "The following disturbance types are implemented for the "
                    f"'{policy.__class__.__name__}' in the '{env.unwrapped.spec.id}' "
                    "environment:"
                )
                for item in {
                    k
                    for k in env.unwrapped._disturber_cfg.keys()
                    if k not in ["default_type"]
                }:
                    d_type_info_msg += f"\t\n - {item}"
                log_to_std_out(friendly_err(d_type_info_msg))
            if args.list_disturbance_variants and args.disturbance_type:
                try:
                    d_variant_msg = (
                        "The following disturbance types are implemented for "
                        f"disturbance '{policy.__class__.__name__}' in the "
                        f"'{env.unwrapped.spec.id}' environment when using disturbance "
                        f"type '{args.disturbance_type}':"
                    )
                    for item in {
                        k
                        for k in env.unwrapped._disturber_cfg[
                            args.disturbance_type
                        ].keys()
                        if k not in ["default_type"]
                    }:
                        d_variant_msg += f"\t\n - {item}"
                    log_to_std_out(
                        friendly_err(
                            d_variant_msg, prepend=(not args.list_disturbance_types)
                        ),
                    )
                except KeyError:
                    error_msg = (
                        f"Disturbance type {args.disturbance_type} does not exist for "
                        f"'{policy.__class__.__name__}' in the "
                        f"'{env.unwrapped.spec.id}' environment. Please specify a  "
                        "valid disturbance type and try again. You can check all the "
                        "valid disturbance types using the --list_disturbance_types "
                        "flag."
                    )
                    log_to_std_out(
                        friendly_err(
                            error_msg, prepend=(not args.list_disturbance_types)
                        ),
                    )
            elif args.list_disturbance_variants and not args.disturbance_type:
                error_msg = (
                    "Disturbance variants could not be retrieved as no disturbance "
                    "type was given. Please specify a disturbance type and try again."
                )
                log_to_std_out(
                    friendly_err(error_msg, prepend=(not args.list_disturbance_types))
                )
        else:
            error_msg = (
                "{} could not be listed as no disturber config ".format(
                    "Disturbance types/variants"
                    if (args.list_disturbance_types and args.list_disturbance_variants)
                    else (
                        "Disturbance types"
                        if args.list_disturbance_types
                        else "Disturbance variants"
                    )
                )
                + "'disturber_cfg' attribute was found on the environment. Please "
                "make sure your environment is compatible with the robustness eval "
                "utility and try again."
            )
            log_to_std_out(friendly_err(error_msg))
        sys.exit()

    # Perform robustness evaluation
    run_results_df = run_disturbed_policy(
        env,
        policy,
        args.disturbance_type,
        disturbance_variant=args.disturbance_variant,
        max_ep_len=args.len,
        num_episodes=args.episodes,
        render=args.render,
        deterministic=args.deterministic,
        save_result=args.save_result,
        output_dir=args.fpath,
    )
    plot_robustness_results(
        run_results_df,
        observations=args.obs,
        output_dir=args.fpath,
        font_scale=args.font_scale,
        save_figs=args.save_figs,
        figs_fmt=args.figs_fmt,
    )
