"""Script version of the eval robustness tool. This can be used to manually evaluate the
disturbance if you don't want to implement a disturber.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stable_learning_control.common.helpers import get_env_id
from stable_learning_control.utils.log_utils import EpochLogger
from stable_learning_control.utils.test_policy import load_policy_and_env

# Example disturbance config.
# NOTE: In this example we add a random noise disturbance to each action dimension.
disturbance_config = {
    "mean": np.round(np.linspace(0.0, 0.0, num=4, dtype=np.float64), 2),
    "std": np.round(np.linspace(0.0, 20.0, num=4, dtype=np.float32), 2),
}


def noise_disturbance(mean, std):
    """Returns a random noise disturbance.

    Args:
        mean (union[float, :obj:`numpy.ndarray`]): The mean value of the noise.
        std (union[float, :obj:`numpy.ndarray`]): The standard deviation of the noise.

    Returns:
        numpy.ndarray: The disturbance array.
    """
    return np.random.normal(mean, std)


if __name__ == "__main__":
    import argparse

    # Retrieve the policy you want to load.
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str, help="The path where the policy is stored")
    parser.add_argument(
        "--data_dir",
        type=str,
        help=(
            "The path where you want to store to store the robustness evaluation "
            "results, meaning the dataframe and the plots (default: 'DEFAULT_DATA_DIR' "
            "parameter from the 'user_config' file)"
        ),
    )
    parser.add_argument(
        "--itr",
        "-i",
        type=int,
        default=-1,
        help="The policy iteration (epoch) you want to use (default: last)",
    )
    parser.add_argument(
        "--len",
        "-l",
        type=int,
        default=None,
        help="The episode length (defaults to environment 'max_episode_steps')",
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
        help="Whether you want to render the environment step (default: False)",
    )
    parser.add_argument(
        "--deterministic",
        "-d",
        action="store_true",
        help=(
            "Whether you want to use a deterministic policy. Only available for "
            "Gaussian policies (default: False)"
        ),
    )
    parser.add_argument(
        "--save_result",
        "--save",
        action="store_true",
        help=(
            "Whether you want to save the robustness evaluation dataframe to disk "
            "(default: False)"
        ),
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Whether you want to save the plots (default: False)",
    )
    parser.add_argument(
        "--figs_fmt",
        default="pdf",
        help="The filetype you want to use for the plots (default: pdf)",
    )
    parser.add_argument(
        "--font_scale",
        default=1.5,
        help="The font scale you want to use for the plot text",
    )
    args = parser.parse_args()

    # Setup logger.
    if not args.data_dir:
        args.data_dir = args.fpath
    output_dir = str(Path(args.data_dir).joinpath("eval"))
    logger = EpochLogger(
        verbose_fmt="table",
        output_dir=output_dir,
        output_fname="eval_statistics.csv",
    )

    # Load policy and environment.
    env, policy = load_policy_and_env(args.fpath, args.itr if args.itr >= 0 else "last")

    # Increase action space.
    # NOTE: Needed to prevent the disturbance from being clipped by the action space.
    env.unwrapped.action_space.high = np.array(
        [
            np.finfo(env.unwrapped.action_space.dtype).max
            for _ in env.unwrapped.action_space.high
        ]
    )
    env.unwrapped.action_space.low = np.array(
        [
            np.finfo(env.unwrapped.action_space.dtype).min
            for _ in env.unwrapped.action_space.low
        ]
    )

    # Apply episode length and set render mode.
    max_ep_len = args.len
    if max_ep_len is not None and max_ep_len != 0:
        if max_ep_len > env.env._max_episode_steps:
            logger.log(
                (
                    f"You defined your 'max_ep_len' to be {max_ep_len} "
                    "while the environment 'max_episode_steps' is "
                    f"{env.env._max_episode_steps}. As a result the environment "
                    f"'max_episode_steps' has been increased to {max_ep_len}"
                ),
                type="warning",
            )
        env.env._max_episode_steps = max_ep_len
    else:
        max_ep_len = env.env._max_episode_steps
    if args.render:
        render_modes = env.unwrapped.metadata.get("render_modes", [])
        if render_modes:
            env.unwrapped.render_mode = "human" if "human" in render_modes else None
        else:
            logger.log(
                (
                    f"Nothing was rendered since the '{get_env_id(env)}' "
                    f"environment does not contain a 'human' render mode."
                ),
                type="warning",
            )

    ############################################################
    # Collect disturbed episodes ###############################
    ############################################################
    logger.log("Starting robustness evaluation...", type="info")

    # Setup storage variables.
    path = {
        "o": [],
        "r": [],
        "reference": [],
        "reference_error": [],
    }
    variant_df = pd.DataFrame()
    variants_df = pd.DataFrame()
    ref_found, ref_error_found, supports_deterministic = True, True, True
    time_attribute = None
    time_step_attribute = None

    # Evaluate each disturbance variant.
    disturbances_length = len(disturbance_config["mean"])
    logger.log("Adding random observation noise.", type="info")
    for i in range(0, disturbances_length):
        o, _ = env.reset()
        r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0

        ################################################
        # Get disturbance variables ####################
        ################################################
        mean = disturbance_config["mean"][i]
        std = disturbance_config["std"][i]
        mean = np.repeat(mean, env.action_space.shape)
        std = np.repeat(std, env.action_space.shape)
        logger.log(
            f"Action disturbance {i+1}: mean: {mean[0]}, std: {std[0]}", type="info"
        )
        ################################################

        # Perform episodes.
        while n < args.episodes:
            # Retrieve action.
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
            # Perform (disturbed) action in the environment.
            # NOTE: Add your disturbance here.
            a += noise_disturbance(
                mean, std
            )  # NOTE: In this example we add a small random noise to the action.
            o, r, d, truncated, info = env.step(a)
            ################################################

            # Store path, cost, reference and state of interest
            ep_ret += r
            ep_len += 1
            path["o"].append(o)
            path["r"].append(r)
            if ref_found and "reference" in info.keys():
                path["reference"].append(info["reference"])
            else:
                ref_found = False
            if ref_error_found and "reference_error" in info.keys():
                path["reference_error"].append(info["reference_error"])
            else:
                ref_error_found = False

            # Store episode information.
            if d or truncated:
                died = not truncated
                logger.store(EpRet=ep_ret, EpLen=ep_len, DeathRate=(float(died)))
                logger.log(
                    "Episode %d \t EpRet %.3f \t EpLen %d \t Died %s"
                    % (n, ep_ret, ep_len, died)
                )

                # Store episode data.
                episode_df = pd.DataFrame(np.arange(0, ep_len), columns=["step"])
                episode_df.insert(len(episode_df.columns), "episode", n)
                if isinstance(path["o"][0], np.ndarray):
                    episode_df = pd.concat(
                        [
                            episode_df,
                            pd.DataFrame(
                                np.array(path["o"]),
                                columns=[
                                    f"observation_{i}"
                                    for i in range(1, len(path["o"][0]) + 1)
                                ],
                            ),
                        ],
                        axis=1,
                    )
                else:
                    episode_df.insert(len(episode_df.columns), "observation", path["o"])
                episode_df.insert(len(episode_df.columns), "cost", path["r"])
                if ref_found:
                    if isinstance(path["reference"][0], np.ndarray):
                        episode_df = pd.concat(
                            [
                                episode_df,
                                pd.DataFrame(
                                    np.array(path["reference"]),
                                    columns=[
                                        f"reference_{i}"
                                        for i in range(1, len(path["reference"][0]) + 1)
                                    ],
                                ),
                            ],
                            axis=1,
                        )
                    else:
                        episode_df.insert(
                            len(episode_df.columns), "reference", path["reference"]
                        )
                if ref_error_found:
                    if isinstance(path["reference_error"][0], np.ndarray):
                        episode_df = pd.concat(
                            [
                                episode_df,
                                pd.DataFrame(
                                    np.array(path["reference_error"]),
                                    columns=[
                                        f"reference_error_{i}"
                                        for i in range(
                                            1, len(path["reference_error"][0]) + 1
                                        )
                                    ],
                                ),
                            ],
                            axis=1,
                        )
                    else:
                        episode_df.insert(
                            len(episode_df.columns),
                            "reference_error",
                            path["reference_error"],
                        )
                variant_df = pd.concat(
                    [variant_df, episode_df], axis=0, ignore_index=True
                )

                # Reset env, episode storage buckets and increment episode counter.
                o, info = env.reset()
                r, d, ep_ret, ep_len, n, t = 0, False, 0, 0, n + 1, 0
                path = {
                    "o": [],
                    "r": [],
                    "reference": [],
                    "reference_error": [],
                }

        # Print variant diagnostics.
        logger.log_tabular("Disturbance", "RandomActionNoise")
        logger.log_tabular("Disturbance_mean", mean[0])
        logger.log_tabular("Disturbance_std", std[0])
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("DeathRate")
        logger.log("")
        logger.dump_tabular()
        logger.log("")

        # Add disturbance information to the robustness evaluation dataframe.
        variant_df.insert(len(variant_df.columns), "disturber", "RandomActionNoise")
        variant_df.insert(
            len(variant_df.columns),
            "disturber_mean",
            mean[0],
        )
        variant_df.insert(
            len(variant_df.columns),
            "disturber_std",
            std[0],
        )
        variants_df = pd.concat([variants_df, variant_df], axis=0, ignore_index=True)

        # Reset variant storage buckets.
        variant_df = pd.DataFrame()

    # Save robustness evaluation dataframe and return it to the user.
    if args.save_result:
        logger.log("Saving robustness evaluation dataframe...", type="info")
        results_path = Path(logger.output_dir).joinpath("eval_results.csv")
        logger.log(
            f"Saving robustness evaluation results to path: {results_path}", type="info"
        )
        variants_df.to_csv(results_path, index=False)
        logger.log("Robustness evaluation dataframe saved.", type="info")

    ############################################################
    # Create plots #############################################
    ############################################################
    # Create a dictionary to store all plots.
    figs = {
        "observation": [],
        "cost": [],
        "reference_error": [],
    }

    # Initialize Seaborn style and font scale.
    logger.log("Showing robustness evaluation plots...", type="info")
    sns.set(style="darkgrid", font_scale=args.font_scale)

    # Retrieve available observations, references and reference errors.
    available_obs = (
        [1]
        if "observation" in variants_df.columns
        else [
            int(col.replace("observation_", ""))
            for col in variants_df.columns
            if col.startswith("observation_")
        ]
    )
    available_refs = (
        [1]
        if "reference" in variants_df.columns
        else [
            int(col.replace("reference_", ""))
            for col in variants_df.columns
            if col.startswith("reference_")
            and not col.replace("reference_", "").startswith("error")
        ]
    )
    available_ref_errors = (
        [1]
        if "reference_error" in variants_df.columns
        else [
            int(col.replace("reference_error_", ""))
            for col in variants_df.columns
            if col.startswith("reference_error_")
        ]
    )

    # Create disturbance variants and add disturbance label to dataframe.
    logger.log("Creating disturbance variants...", type="info")
    disturbance_variants = [
        {"mean": val[0], "std": val[1]}
        for val in list(
            variants_df.groupby(["disturber_mean", "disturber_std"]).groups.keys()
        )
    ]
    variants_df["disturbance_label"] = variants_df.apply(
        lambda x: "mean_{}_std_{}".format(
            round(x["disturber_mean"], 2), round(x["disturber_std"], 2)
        ),
        axis=1,
    )

    # Plot mean observations and references per disturbance variant.
    if len(available_obs) > 0 or len(available_refs) > 0:
        logger.log(
            "Plotting mean observations and references per disturbance variant...",
            type="info",
        )
        n_plots = len(disturbance_variants)
        figs_tmp = []
        for variant in disturbance_variants:
            # Get observations in long format.
            disturbance_df = variants_df[
                variants_df["disturber_mean"] == variant["mean"]
            ]
            obs_value_vars = (
                ["observation"]
                if "observation" in variants_df.columns
                else [f"observation_{obs}" for obs in available_obs]
            )
            obs_disturbance_df = disturbance_df.melt(
                id_vars=["step"],
                value_vars=obs_value_vars,
                var_name="observation",
                value_name="value",
            )

            # Get references in long format.
            refs_value_vars = (
                ["reference"]
                if "reference" in variants_df.columns
                else [f"reference_{ref}" for ref in available_refs]
            )
            refs_disturbance_df = disturbance_df.melt(
                id_vars=["step"],
                value_vars=refs_value_vars,
                var_name="reference",
                value_name="value",
            )

            # Replace observations and references with short names.
            obs_disturbance_df["observation"] = obs_disturbance_df["observation"].apply(
                lambda x: (
                    "observation"
                    if x == "observation"
                    else x.replace("observation_", "obs_")
                )
            )
            refs_disturbance_df["reference"] = refs_disturbance_df["reference"].apply(
                lambda x: (
                    "reference" if x == "reference" else x.replace("reference_", "ref_")
                )
            )

            # Initialize plot.
            fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

            # Create plot title.
            if len(available_obs) > 0 and len(available_refs) > 0:
                plot_title = "Mean {} and {} ".format(
                    "observations" if len(available_obs) > 1 else "observation",
                    "references" if len(available_refs) > 1 else "reference",
                )
            elif len(available_obs) > 0:
                plot_title = "Mean {} ".format(
                    "observations" if len(available_obs) > 1 else "observation"
                )
            else:
                plot_title = "Mean {} ".format(
                    "references" if len(available_refs) > 1 else "reference"
                )
            plot_title += (
                "under 'RandomActionNoise' disturber with mean {} and std {}.".format(
                    round(variant["mean"], 2), round(variant["std"], 2)
                )
            )
            fig.canvas.manager.resize(*fig.canvas.manager.window.maxsize())

            # Add observations to plot.
            if len(available_obs) > 0:
                sns.lineplot(
                    x="step",
                    y="value",
                    hue="observation",
                    data=obs_disturbance_df,
                    ax=ax,
                    legend="full",
                    palette="tab10",
                )

            # Add references to plot.
            if len(available_refs) > 0:
                sns.lineplot(
                    x="step",
                    y="value",
                    hue="reference",
                    data=refs_disturbance_df,
                    ax=ax,
                    legend="full",
                    palette="hls",
                    linestyle="--",
                )

            # Apply plot settings.
            ax.set_xlabel("step")
            ax.set_ylabel("Value")
            ax.set_title(
                plot_title,
            )
            ax.get_legend().set_title(None)

            # Store figure.
            figs_tmp.append(fig)

        # Store plot.
        figs["observation"] = figs_tmp
    else:
        logger.log(
            (
                "No observations or references available in dataframe. Skipping "
                "observation plot."
            ),
            type="warning",
        )

    # Plot mean cost per disturbance variant in one plot if available.
    if "cost" in variants_df.columns:
        logger.log("Plotting mean cost per disturbance variant...", type="info")
        fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
        fig.canvas.manager.resize(*fig.canvas.manager.window.maxsize())
        sns.lineplot(
            x="step",
            y="cost",
            hue="disturbance_label",
            data=variants_df,
            ax=ax,
            legend="full",
            palette="tab10",
        )
        ax.set_xlabel("step")
        ax.set_ylabel("Cost")
        ax.set_title("Mean cost under 'RandomActionNoise' disturber")
        ax.get_legend().set_title(None)
        figs["cost"].append(fig)
    else:
        logger.log(
            (
                "Mean cost not plotted since no cost information was found in the "
                "supplied dataframe. Please ensure the dataframe contains the 'cost' "
                "key."
            ),
            type="warning",
        )

    # Plot mean reference error per disturbance variant.
    if len(available_ref_errors) > 0:
        logger.log(
            "Plotting mean reference error per disturbance variant...", type="info"
        )
        n_plots = len(disturbance_variants)
        figs_tmp = []
        for variant in disturbance_variants:
            # Get reference error in long format.
            disturbance_df = variants_df[
                variants_df["disturber_mean"] == variant["mean"]
            ]
            ref_errors_value_vars = (
                ["reference_error"]
                if "reference_error" in variants_df.columns
                else [
                    f"reference_error_{ref_error}" for ref_error in available_ref_errors
                ]
            )
            ref_errors_disturbance_df = disturbance_df.melt(
                id_vars=["step"],
                value_vars=ref_errors_value_vars,
                var_name="reference_error",
                value_name="value",
            )

            # Replace reference error with short names.
            ref_errors_disturbance_df["reference_error"] = ref_errors_disturbance_df[
                "reference_error"
            ].apply(
                lambda x: (
                    "reference_error"
                    if x == "reference_error"
                    else x.replace("reference_error_", "ref_error_")
                )
            )

            # Initialize plot.
            fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

            # Create plot title.
            plot_title = "Mean {} ".format(
                (
                    "reference errors"
                    if len(available_ref_errors) > 1
                    else "reference error"
                ),
            )
            plot_title += (
                "under 'RandomActionNoise' disturber with mean {} and std {}.".format(
                    round(variant["mean"], 2), round(variant["std"], 2)
                )
            )
            fig.canvas.manager.resize(*fig.canvas.manager.window.maxsize())

            # Add reference error to plot.
            sns.lineplot(
                x="step",
                y="value",
                hue="reference_error",
                data=ref_errors_disturbance_df,
                ax=ax,
                legend="full",
                palette="tab10",
            )

            # Configure plot.
            ax.set_xlabel("step")
            ax.set_ylabel("Value")
            ax.set_title(
                plot_title,
            )
            ax.get_legend().set_title(None)

            # Add figure title if using subplots.
            figs_tmp.append(fig)

        # Store plot.
        figs["reference_error"] = figs_tmp
    else:
        logger.log(
            (
                "Mean reference error not plotted since no reference error "
                "information was found in the supplied dataframe. Please ensure "
                "the dataframe contains the 'reference_error' key."
            ),
            type="warning",
        )

    # Save plots.
    if args.save_plots:
        figs_path = Path(logger.output_dir).joinpath("figures")
        figs_extension = args.figs_fmt[1:] if args.startswith(".") else args
        os.makedirs(figs_path, exist_ok=True)
        logger.log("Saving plots...", type="info")
        logger.log(f"Saving figures to path: {figs_path}", type="info")
        if figs["observation"]:
            for idx, fig in enumerate(figs["observation"]):
                fig_id = "random_action_noise-subplots-fig_{}".format(idx + 1)
                fig.savefig(
                    figs_path.joinpath(f"observations-{fig_id}.{figs_extension}"),
                    bbox_inches="tight",
                )
        if figs["cost"]:
            figs["cost"][0].savefig(
                figs_path.joinpath(
                    f"cost-random_action_noise.{figs_extension}",
                ),
                bbox_inches="tight",
            )
        if figs["reference_error"]:
            for idx, fig in enumerate(figs["reference_error"]):
                fig_id = "random_action_noise-subplots-fig_{}".format(idx + 1)
                fig.savefig(
                    figs_path.joinpath(f"reference_error-{fig_id}.{figs_extension}"),
                    bbox_inches="tight",
                )
        logger.log("Plots saved.", type="info")

    # Wait for user to close plots before continuing.
    plt.show()
