"""Script version of the eval robustness tool. This can be used if you don't want to
implement a disturber class.
"""

import math
import os
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bayesian_learning_control.control.common.helpers import validate_observations
from bayesian_learning_control.control.utils.test_policy import load_policy_and_env
from bayesian_learning_control.utils.log_utils import EpochLogger, log_to_std_out

# Disturbance settings
# NOTE: In this example we add a noise disturbance to the action
disturbance_type = "test1"
disturbance_variant = "test"  # MAYBE REMOVE
disturbance_range = {
    "mean": np.linspace(0.0, 0.0, num=4, dtype=np.float32),
    "std": np.linspace(0.0, 20.0, num=4, dtype=np.float32),
}


def noise_disturbance(mean, std):
    """Returns a random noise specified mean and a standard deviation.

    Args:
        mean (union[float, :obj:`numpy.ndarray`]): The mean value of the noise.
        std (union[float, :obj:`numpy.ndarray`]): The standard deviation of the noise.

    Returns:
        numpy.ndarray: The disturbance array.
    """
    return np.random.normal(mean, std)


if __name__ == "__main__":  # noqa: C901
    import argparse

    # Retrieve the policy you want to load
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str, help="The path where the policy is stored")
    parser.add_argument(
        "--itr",
        "-i",
        type=int,
        default=-1,
        help="The policy iteration (epoch) you want to use (default: 'last')",
    )
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
        "--deterministic",
        "-d",
        action="store_true",
        help="Whether you want to use a deterministic policy (default: True)",
    )
    parser.add_argument(
        "--render",
        "-r",
        action="store_true",
        help="Whether you want to render the environment step (default: True)",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help=(
            "Whether you want to save the robustness evaluation robustness_eval_df to "
            "disk (default: False)"
        ),
    )
    parser.add_argument(
        "--obs",
        default=None,
        nargs="+",
        help="The observations you want to show in the observations/references plot",
    )
    parser.add_argument(
        "--merged",
        default=None,
        action="store_true",
        help=(
            "Merge all observations into one plot. By default observations under each "
            "disturbance are show in a separate subplot."
        ),
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
    args = parser.parse_args()

    # Load policy and environment
    try:
        env, policy = load_policy_and_env(
            args.fpath, args.itr if args.itr >= 0 else "last"
        )
    except Exception:
        log_to_std_out(
            (
                "Environment and policy could not be loaded. Please check the 'fpath' "
                "and try again."
            ),
            type="error",
        )
        sys.exit(0)

    # Remove action clipping if present
    if hasattr(env.unwrapped, "_clipped_action"):
        env.unwrapped._clipped_action = False

    # Setup logger
    output_dir = Path(args.fpath).joinpath("eval")
    logger = EpochLogger(
        verbose_fmt="table", output_dir=output_dir, output_fname="eval_statistics.csv"
    )

    # Set max episode length
    if args.len is None:
        max_ep_len = env._max_episode_steps
    else:
        if args.len > env._max_episode_steps:
            logger.log(
                (
                    f"You defined your 'max_ep_len' to be {args.len} "
                    "while the environment 'max_episide_steps' is "
                    f"{env._max_episode_steps}. As a result the environment "
                    f"'max_episode_steps' has been increased to {args.len}"
                ),
                type="warning",
            )
            env._max_episode_steps = args.len

    ############################################################
    # Collect disturbed episodes ###############################
    ############################################################
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
    disturbances_length = len(disturbance_range["mean"])
    soi_found, ref_found = True, True
    supports_deterministic = True  # Only supported with gaussian algorithms
    log_to_std_out("Adding random observation noise.", type="info")
    for _ in range(0, disturbances_length):
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

        ################################################
        # Get disturbance variables ####################
        ################################################
        mean = disturbance_range["mean"][n_disturbance]
        std = disturbance_range["std"][n_disturbance]
        if not isinstance(mean, np.ndarray):
            mean = np.repeat(mean, env.action_space.shape)
        if not isinstance(std, np.ndarray):
            std = np.repeat(std, env.action_space.shape)
        log_to_std_out(
            f"Disturbance {n_disturbance}: mean: {mean}, std: {std}", type="info"
        )

        # Perform disturbed episodes
        while n < args.episodes:
            # Render env if requested
            if args.render and not render_error:
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
            if args.deterministic and supports_deterministic:
                try:
                    a = policy.get_action(o, deterministic=args.deterministic)
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

            ################################################
            # Perform disturbed step #######################
            ################################################
            # Perform (disturbed) action in the environment and store result
            # NOTE: Add your disturbance here or in the environment!
            a += noise_disturbance(
                mean, std
            )  # NOTE: In this example we add a small random noise to the action
            o, r, d, info = env.step(a)

            # Increment counters
            ep_ret += r
            ep_len += 1
            ################################################

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
                )  # Flatten robustness_eval_df
                o_episodes_dfs.append(o_episode_df)

                # Store episode rewards
                r_episode_df = pd.DataFrame(
                    {"step": range(0, ep_len), "reward": path["r"]}
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
                    )  # Flatten robustness_eval_df
                    soi_episodes_dfs.append(soi_episode_df)

                # Store reference
                if ref_found:
                    ref_episode_df = pd.DataFrame(path["reference"])
                    ref_episode_df.insert(0, "step", range(0, ep_len))
                    ref_episode_df = pd.melt(
                        ref_episode_df,
                        id_vars="step",
                        var_name="reference",
                    )  # Flatten robustness_eval_df
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
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("DeathRate")
        log_to_std_out("")
        logger.dump_tabular()

        # Add extra disturbance information to the robustness eval robustness_eval_df
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

        # Reset storage buckets and go to next disturbance
        o_episodes_dfs = []
        r_episodes_dfs = []
        soi_episodes_dfs = []
        ref_episodes_dfs = []

        ################################################
        # Perform disturbed step #######################
        ################################################
        # NOTE: Add here your disturbance change logic
        n_disturbance += 1
        ################################################

    # Merge robustness evaluation information for all disturbances
    o_disturbances_df = pd.concat(o_disturbances_dfs, ignore_index=True)
    r_disturbances_df = pd.concat(r_disturbances_dfs, ignore_index=True)
    soi_disturbances_df = pd.concat(soi_disturbances_dfs, ignore_index=True)
    ref_disturbances_df = pd.concat(ref_disturbances_dfs, ignore_index=True)

    # Return/save robustness evaluation robustness_eval_df
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

    # Save robustness evaluation robustness_eval_df and return it to the user
    if args.save_result:
        results_path = logger.output_dir.joinpath("results.csv")
        logger.log(
            f"Saving robustness evaluation results to path: {results_path}", type="info"
        )
        robustness_eval_df.to_csv(results_path, index=False)

    ############################################################
    # Create plots #############################################
    ############################################################
    figs = {
        "observations": [],
        "costs": [],
        "states_of_interest": [],
    }  # Store all plots (Needed for save)
    log_to_std_out("Showing robustness evaluation plots...", type="info")
    sns.set(style="darkgrid", font_scale=args.font_scale)

    # Unpack required data from robustness_eval_df
    obs_found, rew_found, soi_found, ref_found = True, True, True, True
    o_disturbances_df, ref_disturbances_df = (
        pd.DataFrame(),
        pd.DataFrame(),
    )
    if "observation" in robustness_eval_df["variable"].unique():
        o_disturbances_df = robustness_eval_df.query(
            "variable == 'observation'"
        ).dropna(axis=1, how="all")
    else:
        obs_found = False
    if "reward" in robustness_eval_df["variable"].unique():
        r_disturbances_df = robustness_eval_df.query("variable == 'reward'").dropna(
            axis=1, how="all"
        )
    else:
        rew_found = False
    if "state_of_interest" in robustness_eval_df["variable"].unique():
        soi_disturbances_df = robustness_eval_df.query(
            "variable == 'state_of_interest'"
        ).dropna(axis=1, how="all")
    else:
        soi_found = False
    if "state_of_interest" in robustness_eval_df["variable"].unique():
        ref_disturbances_df = robustness_eval_df.query(
            "variable == 'reference'"
        ).dropna(axis=1, how="all")
    else:
        ref_found = False

    # Merge observations and references
    if obs_found:
        obs_df_tmp = o_disturbances_df.copy(deep=True)
        obs_df_tmp["signal"] = "obs_" + (obs_df_tmp["observation"] + 1).astype(str)
        obs_df_tmp.insert(len(obs_df_tmp.columns), "type", "observation")

        # Retrieve the requested observations
        observations = args.observations if hasattr(args, "observations") else None
        observations = validate_observations(observations, o_disturbances_df)
        observations = [obs - 1 for obs in observations]  # Humans count from 1
        obs_df_tmp = obs_df_tmp.query(f"observation in {observations}")
    if ref_found:
        ref_df_tmp = ref_disturbances_df.copy(deep=True)
        ref_df_tmp["signal"] = "ref_" + (ref_df_tmp["reference"] + 1).astype(str)
        ref_df_tmp.insert(len(ref_df_tmp.columns), "type", "reference")
    obs_ref_df = pd.concat([obs_df_tmp, ref_df_tmp], ignore_index=True)

    # Loop though all disturbances and plot the observations and references in one plot
    fig_title = "{} under several {}.".format(
        "Observation and reference"
        if all([obs_found, ref_found])
        else ("Observation" if obs_found else "reference"),
        "{} disturbances".format(obs_ref_df.disturbance_variant[0])
        if "disturbance_variant" in obs_ref_df.keys()
        else "disturbances",
    )
    obs_ref_df.loc[obs_ref_df["disturbance_index"] == 0, "disturbance"] = (
        obs_ref_df.loc[obs_ref_df["disturbance_index"] == 0, "disturbance"]
        + " (original)"
    )  # Append original to original value
    if not args.merged:
        num_plots = len(obs_ref_df.disturbance.unique())
        total_cols = 3
        total_rows = math.ceil(num_plots / total_cols)
        fig, axes = plt.subplots(
            nrows=total_rows,
            ncols=total_cols,
            figsize=(7 * total_cols, 7 * total_rows),
            tight_layout=True,
            sharex=True,
            squeeze=False,
        )
        fig.suptitle(fig_title)
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
    else:
        fig = plt.figure(tight_layout=True)
        sns.lineplot(
            data=obs_ref_df, x="step", y="value", ci="sd", hue="disturbance"
        ).set_title(fig_title)
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
            "Mean cost under several {}.".format(
                "{} disturbances".format(obs_ref_df.disturbance_variant[0])
                if "disturbance_variant" in obs_ref_df.keys()
                else "disturbances",
            )
        )
    else:
        log_to_std_out(
            (
                "Mean costs plot could not we shown as no 'rewards' field was found ",
                "in the supplied robustness_eval_df.",
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
                "{} under several {}.".format(
                    "State of interest" if n_soi == 1 else f"State of interest {index}",
                    "{} disturbances".format(obs_ref_df.disturbance_variant[0])
                    if "disturbance_variant" in obs_ref_df.keys()
                    else "disturbances",
                )
            )
        plt.show()
    else:
        log_to_std_out(
            (
                "State of interest plot could not we shown as no 'state_of_interest' "
                "field was found in the supplied robustness_eval_df.",
            ),
            type="warning",
        )

    # Save plots
    if args.save_figs:
        figs_path = output_dir.joinpath("figures")
        figs_extension = (
            args.figs_fmt[1:] if args.figs_fmt.startswith(".") else args.figs_fmt
        )
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
