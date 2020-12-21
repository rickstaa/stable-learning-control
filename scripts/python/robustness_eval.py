"""Script that can be used to display the performance and robustness of a trained
agent."""

import os
import os.path as osp

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools

from lac import LAC

from utils import get_env_from_name, colorize, get_log_path, validate_indices
from variant import EVAL_PARAMS, ENVS_PARAMS, ENV_NAME, ENV_SEED, REL_PATH

def validate_req_policies(req_policies, policies):
    """Validates whether the requested policies are valid.

    Args:
        req_policies (list): Requested policies.

        policies (list): Available policies.

    Returns:
        list: The requested policies which are valid.
    """

    # Validate whether the policy is a number and does exist
    valid_policies = []
    invalid_policies = []
    for policy in req_policies:
        if isinstance(policy, str):
            if not policy.isnumeric():
                invalid_policies.append(policy)
            else:
                if int(policy) in policies:
                    valid_policies.append(int(policy))
                else:
                    invalid_policies.append(int(policy))
        elif isinstance(policy, (float, int)):
            if int(policy) in policies:
                valid_policies.append(int(policy))
            else:
                invalid_policies.append(int(policy))
        else:
            invalid_policies.append(policy)

    # Display warning if policies were not found
    if len(invalid_policies) != 0:
        policy_strs = (
            ["policy", "it"] if len(invalid_policies) <= 1 else ["policies", "they"]
        )
        print(
            colorize(
                (
                    f"WARN: Skipping {policy_strs[0]} {invalid_policies} as "
                    f"{policy_strs[1]} not exist. Please re-check the list of "
                    "policies you supplied."
                ),
                "yellow",
                bold=True,
            )
        )

    # Return valid policies
    return valid_policies


def validate_req_sio(req_sio, sio_array, refs_array):
    """Validates whether the requested states of interest exists. Throws a warning
    message if a requested state is not found.

    Args:
        req_sio (list): The requested states of interest.

        sio_array (numpy.ndarray): The array with the states of interest.

        refs_array (numpy.ndarray): The array with the references.

    Returns:
        tuple: Lists with the requested states of interest and reference which are
            valid.
    """

    # Check if the requested states of interest and references are present
    req_sio = [sio - 1 for sio in req_sio]  # Translate indices to python format
    valid_sio, invalid_sio = validate_indices(req_sio, sio_array)
    valid_refs, invalid_refs = validate_indices(req_sio, refs_array)
    valid_sio = [sio + 1 for sio in valid_sio]  # Translate back to hunan format
    invalid_sio = [sio + 1 for sio in invalid_sio]
    valid_refs = [ref + 1 for ref in valid_refs]
    invalid_refs = [ref + 1 for ref in invalid_refs]

    # Display warning if not found
    if invalid_sio and invalid_refs:
        warning_str = (
            "{} {} and {} {}".format(
                "states of interest" if len(invalid_sio) > 1 else "state of interest",
                invalid_sio,
                "references" if len(invalid_refs) > 1 else "reference",
                invalid_refs,
            )
            + " could not be plotted as they did not exist."
        )
        print(colorize("WARN: " + warning_str.capitalize(), "yellow"))
    elif invalid_sio:
        warning_str = "WARN: {} {}".format(
            "States of interest" if len(invalid_sio) > 1 else "State of interest",
            invalid_sio,
        ) + " could not be plotted as {} does not exist.".format(
            "they" if len(invalid_sio) > 1 else "it",
        )
        print(colorize(warning_str, "yellow"))
    elif invalid_refs:
        warning_str = "WARN: {} {}".format(
            "References" if len(invalid_refs) > 1 else "Reference", invalid_refs,
        ) + " {} not be plotted as {} does not exist.".format(
            "could" if len(invalid_refs) > 1 else "can",
            "they" if len(invalid_refs) > 1 else "it",
        )
        print(colorize(warning_str, "yellow"))

    # Return valid states of interest and references
    return valid_sio, valid_refs


def validate_req_obs(req_obs, obs_array):
    """Validates whether the requested observations exists. Throws a warning
    message if a requested state is not found.

    Args:
        req_obs (list): The requested observations.

        obs_array (numpy.ndarray): The array with the observations.

    Returns:
        list: The requested observations which are valid.
    """

    # Check if the requested observations are present
    req_obs = [obs - 1 for obs in req_obs]  # Translate indices to python format
    valid_obs, invalid_obs = validate_indices(req_obs, obs_array)
    valid_obs = [obs + 1 for obs in valid_obs]  # Translate back to hunan format
    invalid_obs = [obs + 1 for obs in invalid_obs]

    # Display warning if req obs were not found
    if invalid_obs:
        warning_str = (
            "WARN: {} {}".format(
                "Observations" if len(invalid_obs) > 1 else "Observations", invalid_obs,
            )
            + " could not be plotted as they does not exist."
        )
        print(colorize(warning_str, "yellow"))

    # Return valid observations
    return valid_obs


def validate_req_costs(req_costs, costs_array):
    """Validates whether the requested observations exists. Throws a warning
    message if a requested state is not found.

    Args:
        req_costs (list): The requested observations.

        costs_array (numpy.ndarray): The array with the costs.

    Returns:
        list: The requested observations which are valid.
    """

    # Check if the requested observations are present
    req_costs = [cost - 1 for cost in req_costs]  # Translate indices to python format
    valid_costs, invalid_costs = validate_indices(req_costs, costs_array)
    valid_costs = [cost + 1 for cost in valid_costs]  # Translate back to hunan format
    invalid_costs = [cost + 1 for cost in invalid_costs]

    # Display warning if req obs were not found
    if invalid_costs:
        warning_str = (
            "WARN: {} {}".format(
                "Costs" if len(invalid_costs) > 1 else "Cost", invalid_costs,
            )
            + " could not be plotted as they does not exist."
        )
        print(colorize(warning_str, "yellow"))

    # Return valid costs
    return valid_costs


def get_distrubance_function(env_name):
    """Returns a compatible disturber for a given environment.

    Args:
        env_name (str): The environment name.

    Raises:
        NameError: If environment name does not exist.

    Returns:
        function: A function that can be used as to perform a disturbed step into the
        environment.
    """

    if "cartpole_cost" in env_name:
        from disturbers import cartpole_disturber
        disturbance_step = cartpole_disturber
    elif "oscillator" in env_name:
        from disturbers import oscillator_disturber
        disturbance_step = oscillator_disturber
    else:
        print("no disturber designed for " + env_name)
        raise NameError
    return disturbance_step


if __name__ == "__main__":

    # Parse cmdline arguments
    parser = argparse.ArgumentParser(
        description="Evaluate trained the LAC agents in a given environment."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=EVAL_PARAMS["eval_list"],
        help="The name of the model you want to evaluate.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=ENV_NAME,
        help="The name of the env you want to evaluate.",
    )
    parser.add_argument(
        "-p",
        "--policies",
        nargs="+",
        default=EVAL_PARAMS["which_policy_for_inference"],
        help="The policies you want to use in the inference.",
    )
    parser.add_argument(
        "--plot-s",
        type=bool,
        default=EVAL_PARAMS["plot_soi"],
        help="Whether want to plot the states of reference.",
    )
    parser.add_argument(
        "--plot-o",
        type=bool,
        default=EVAL_PARAMS["plot_obs"],
        help="Whether you want to plot the observations.",
    )
    parser.add_argument(
        "--plot-c",
        type=bool,
        default=EVAL_PARAMS["plot_cost"],
        help="Whether want to plot the cost.",
    )
    parser.add_argument(
        "--save-figs",
        type=bool,
        default=EVAL_PARAMS["save_figs"],
        help="Whether you want to save the figures to pdf.",
    )
    args = parser.parse_args()

    # Validate specified figure output file type
    sup_file_types = ["pdf", "svg", "png", "jpg"]
    if EVAL_PARAMS["fig_file_type"] not in sup_file_types:
        file_Type = EVAL_PARAMS["fig_file_type"]
        print(
            colorize(
                (
                    f"ERROR: The requested figure save file type {file_Type} "
                    "is not supported file types are {sup_file_types}."
                ),
                "red",
                bold=True,
            )
        )
        sys.exit(0)

    # Retrieve available policies
    eval_agents = (
        [args.model_name] if not isinstance(args.model_name, list) else args.model_name
    )

    ####################################################
    # Perform Inference for all agents #################
    ####################################################
    print("\n=========Performing inference evaluation=========")
    print(f"Evaluationing agents: {eval_agents}")

    # Loop though USER defined agents list
    for name in eval_agents:

        # Create agent policy and log paths
        if REL_PATH:
            model_path = get_log_path(env_name=args.env_name, agent_name=name)
            log_path = "/".join([model_path, "figure"])
            os.makedirs(log_path, exist_ok=True)
        else:
            model_path = get_log_path(env_name=args.env_name, agent_name=name)
            log_path = osp.abspath(osp.join(model_path, "figure"))
            os.makedirs(log_path, exist_ok=True)
        print("\n====Evaluation agent " + name + "====")
        print(colorize(f"INFO: Using model folder {model_path}.", "cyan", bold=True))

        # Create environment
        print(colorize(f"INFO: Using environment {args.env_name}.", "cyan", bold=True))
        env = get_env_from_name(args.env_name, ENV_SEED)

        # Check if specified agent exists
        if not osp.exists(model_path):
            warning_str = (
                f"WARN: Inference could not be run for agent `{name}` as it was not "
                f"found for the `{args.env_name}` environment."
            )
            print(colorize(warning_str, "yellow"))
            continue

        # Get environment action and observation space dimensions
        a_lowerbound = env.action_space.low
        a_upperbound = env.action_space.high
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]

        # Initiate the LAC policy
        policy = LAC(
            a_dim, s_dim, act_limits={"low": a_lowerbound, "high": a_upperbound}
        )

        # Retrieve all trained policies for a given agent
        print("Looking for policies...")
        policy_list = os.listdir(model_path)
        policy_list = [
            policy_name
            for policy_name in policy_list
            if osp.exists(
                osp.abspath(osp.join(model_path, policy_name, "policy/model.pth"))
            )
        ]
        policy_list = [int(item) for item in policy_list if item.isnumeric()]
        policy_list.sort()

        # Check if a given policy exists for the current agent
        if not policy_list:
            warning_str = (
                f"WARN: Inference could not be run for agent `{name}` in the "
                f"`{args.env_name}` environment as no policies were found."
            )
            print(colorize(warning_str, "yellow"))
            continue

        # Retrieve *user* defined policy list
        input_policies = args.policies

        # Validate *user* defined policy list
        valid_policies = validate_req_policies(input_policies, policy_list)

        # Check if valid policies were found
        if not valid_policies:
            print(
                colorize(
                    (
                        f"WARN: Skipping agent `{name}` since no valid policies were "
                        "found. Please re-check the list policies you supplied."
                    ),
                    "yellow",
                    bold=True,
                )
            )
            continue  # Skip agent

        ############################################
        # Run policy inference #####################
        ############################################
        print(f"Using policies: {valid_policies}")
        # Perform a number of paths in each policy and store them.
        policies_paths = {}
        for policy_name in valid_policies:

            # Policy paths storage bucket
            policy_paths = {
                "s": [],
                "r": [],
                "s_": [],
                "state_of_interest": [],
                "reference": [],
                "episode_length": [],
                "return": [],
                "death_rate": 0.0,
            }

            # Load current policy
            retval = policy.restore(
                osp.abspath(osp.join(model_path, str(policy_name), "policy"))
            )
            policy_type = "LAC" if policy.use_lyapunov else "SAC"
            if not retval:
                print(
                    f"Policy {policy_name} could not be loaded. Continuing to the next "
                    "policy."
                )
                continue

            # Perform a number of paths in the environment
            for i in range(
                math.ceil(EVAL_PARAMS["num_of_paths"] / len(input_policies))
            ):

                # Path storage bucket
                episode_path = {
                    "s": [],
                    "r": [],
                    "s_": [],
                    "state_of_interest": [],
                    "reference": [],
                }

                # Reset environment
                # NOTE (rickstaa): This check was added since some of the supported
                # environments have a different reset when running the inference.
                if ENVS_PARAMS[ENV_NAME]["eval_reset"]:
                    s = env.reset(eval=True)
                else:
                    s = env.reset()

                # Retrieve path
                for j in range(ENVS_PARAMS[args.env_name]["max_ep_steps"]):

                    # Retrieve (scaled) action based on the current policy
                    # NOTE (rickstaa): The scaling operation is already performed inside
                    # the policy based on the `act_limits` you supplied.
                    a = policy.choose_action(s, True)

                    # Perform action in the environment
                    s_, r, done, info = env.step(a)

                    # Store observations in path
                    episode_path["s"].append(s)
                    episode_path["r"].append(r)
                    episode_path["s_"].append(s_)
                    if "state_of_interest" in info.keys():
                        episode_path["state_of_interest"].append(
                            np.array([info["state_of_interest"]])
                        )
                    if "reference" in info.keys():
                        episode_path["reference"].append(np.array(info["reference"]))

                    # Terminate if max step has been reached
                    done = False  # Ignore done signal from env because inference
                    if j == (ENVS_PARAMS[args.env_name]["max_ep_steps"] - 1):
                        done = True

                    # Update current state
                    s = s_

                    # Check if episode is done and break loop
                    if done:
                        break

                # Append path to policy paths list
                policy_paths["s"].append(episode_path["s"])
                policy_paths["r"].append(episode_path["r"])
                policy_paths["s_"].append(episode_path["s_"])
                policy_paths["state_of_interest"].append(
                    episode_path["state_of_interest"]
                )
                policy_paths["reference"].append(episode_path["reference"])
                policy_paths["episode_length"].append(len(episode_path["s"]))
                policy_paths["return"].append(np.sum(episode_path["r"]))

            # Calculate policy death rate
            policy_paths["death_rate"] = sum(
                [
                    episode <= (ENVS_PARAMS[args.env_name]["max_ep_steps"] - 1)
                    for episode in policy_paths["episode_length"]
                ]
            ) / len(policy_paths["episode_length"])

            # Store policy results in policy dictionary
            policies_paths["policy " + str(policy_name)] = policy_paths

        ############################################
        # Calculate policy statistics ##############
        ############################################
        eval_paths = {}
        policies_diag = {}
        for pol, val in policies_paths.items():

            # Calculate policy statistics
            policies_diag[pol] = {}
            policies_diag[pol]["mean_return"] = np.mean(val["return"])
            policies_diag[pol]["mean_episode_length"] = np.mean(val["episode_length"])
            policies_diag[pol]["death_rate"] = val.pop("death_rate")

            # concatenate current policy to eval dictionary
            for key, val in val.items():
                if key not in eval_paths.keys():
                    eval_paths[key] = val
                else:
                    eval_paths[key].extend(val)

        ############################################
        # Display policy diagnostics ###############
        ############################################

        # Display policy diagnostics
        print("Printing policies diagnostics...")
        print("\n==Policies diagnostics==")
        eval_diagnostics = {}
        for pol, diag_val in policies_diag.items():
            print(f"{pol}:")
            for key, val in diag_val.items():
                print(f"- {key}: {val}")
                if key not in eval_diagnostics:
                    eval_diagnostics[key] = [val]
                else:
                    eval_diagnostics[key].append(val)
            print("")
        print("all policies:")
        for key, val in eval_diagnostics.items():
            print(f" - {key}: {np.mean(val)}")
            print(f" - {key}_std: {np.std(val)}")

        ############################################
        # Display performance figures ##############
        ############################################
        print("\n==Policies inference plots==")

        # Create figure storage dictionary and time axis
        figs = {
            "states_of_interest": [],
            "states": [],
            "costs": [],
        }  # Store all figers (Needed for save)
        t = range(max(eval_paths["episode_length"]))

        ####################################
        # Plot mean path and std for #######
        # states of interest and ###########
        # references. ######################
        ####################################
        if args.plot_s:
            print("Plotting states of interest mean path and standard deviation...")

            # Retrieve USER defined sates of reference list
            req_sio = EVAL_PARAMS["soi"]

            # Calculate mean path of the state of interest
            soi_trimmed = [
                path
                for path in eval_paths["state_of_interest"]
                if len(path) == max(eval_paths["episode_length"])
            ]  # Trim unfinished paths
            soi_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(soi_trimmed), axis=0))
            )
            soi_std_path = np.transpose(
                np.squeeze(np.std(np.array(soi_trimmed), axis=0))
            )

            # Calculate mean path of the of the reference
            ref_trimmed = [
                path
                for path in eval_paths["reference"]
                if len(path) == max(eval_paths["episode_length"])
            ]  # Trim unfinished paths
            ref_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(ref_trimmed), axis=0))
            )
            ref_std_path = np.transpose(
                np.squeeze(np.std(np.array(ref_trimmed), axis=0))
            )

            # Make sure mean path and std arrays are the right shape
            soi_mean_path = (
                np.expand_dims(soi_mean_path, axis=0)
                if len(soi_mean_path.shape) == 1
                else soi_mean_path
            )
            soi_std_path = (
                np.expand_dims(soi_std_path, axis=0)
                if len(soi_std_path.shape) == 1
                else soi_std_path
            )
            ref_mean_path = (
                np.expand_dims(ref_mean_path, axis=0)
                if len(ref_mean_path.shape) == 1
                else ref_mean_path
            )
            ref_std_path = (
                np.expand_dims(ref_std_path, axis=0)
                if len(ref_std_path.shape) == 1
                else ref_std_path
            )

            # Check if requested state_of_interest exists
            valid_sio, valid_refs = validate_req_sio(
                req_sio, soi_mean_path, ref_mean_path
            )

            # Plot mean path and std for states of interest and references
            if valid_sio or valid_refs:  # Check if any sio or refs were found
                print(
                    "Using: {}...".format(
                        (
                            "states of interest "
                            if len(valid_sio) > 1
                            else "state of interest "
                        )
                        + str(valid_sio)
                        + (" and " if valid_sio and valid_refs else "")
                        + ("references " if len(valid_refs) > 1 else "reference ")
                        + str(valid_refs)
                    )
                )

                # Plot sio/ref mean path and std
                if EVAL_PARAMS["sio_merged"]:  # Add all soi in one figure
                    fig = plt.figure(
                        figsize=(9, 6),
                        num=(
                            f"{policy_type}_TORCH_"
                            + str(len(list(itertools.chain(*figs.values()))) + 1)
                        ),
                    )
                    ax = fig.add_subplot(111)
                    colors = "bgrcmk"
                    cycol = itertools.cycle(colors)
                    figs["states_of_interest"].append(fig)  # Store figure reference
                for i in range(0, max(soi_mean_path.shape[0], ref_mean_path.shape[0])):
                    if (i + 1) in req_sio or not req_sio:
                        if not EVAL_PARAMS[
                            "sio_merged"
                        ]:  # Create separate figs for each sio
                            fig = plt.figure(
                                figsize=(9, 6),
                                num=(
                                    f"{policy_type}_TORCH_"
                                    + str(
                                        len(list(itertools.chain(*figs.values()))) + 1
                                    )
                                ),
                            )
                            ax = fig.add_subplot(111)
                            color1 = "red"
                            color2 = "blue"
                            figs["states_of_interest"].append(
                                fig
                            )  # Store figure reference
                        else:
                            color1 = color2 = next(cycol)

                        # Plot states of interest
                        if i <= (len(soi_mean_path) - 1):
                            ax.plot(
                                t,
                                soi_mean_path[i],
                                color=color1,
                                linestyle="dashed",
                                label=f"state_of_interest_{i+1}_mean",
                            )
                            ax.fill_between(
                                t,
                                soi_mean_path[i] - soi_std_path[i],
                                soi_mean_path[i] + soi_std_path[i],
                                color=color1,
                                alpha=0.3,
                                label=f"state_of_interest_{i+1}_std",
                            )

                        # Plot references
                        if i <= (len(ref_mean_path) - 1):
                            ax.plot(
                                t,
                                ref_mean_path[i],
                                color=color2,
                                label=f"reference_{i+1}",
                            )
                            # ax.fill_between(
                            #     t,
                            #     ref_mean_path[i] - ref_std_path[i],
                            #     ref_mean_path[i] + ref_std_path[i],
                            #     color=color2,
                            #     alpha=0.3,
                            #     label=f"reference_{i+1}_std",
                            # )  # Should be zero

                        # Add figure legend and title (Separate figures)
                        if not EVAL_PARAMS["sio_merged"]:
                            ax_title = (
                                EVAL_PARAMS["soi_title"]
                                if (
                                    EVAL_PARAMS["soi_title"]
                                    and isinstance(EVAL_PARAMS["soi_title"], str)
                                )
                                else "True and Estimated Quatonian"
                            )
                            ax.set_title(f"{ax_title} {i+1}")
                            handles, labels = ax.get_legend_handles_labels()
                            ax.legend(
                                handles, labels, loc=2, fancybox=False, shadow=False
                            )

                # Add figure legend and title (Merged figure)
                if EVAL_PARAMS["sio_merged"]:
                    ax_title = (
                        EVAL_PARAMS["soi_title"]
                        if (
                            EVAL_PARAMS["soi_title"]
                            and isinstance(EVAL_PARAMS["soi_title"], str)
                        )
                        else "True and Estimated Quatonian"
                    )
                    ax.set_title(f"{ax_title}")
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
            else:
                print(
                    colorize(
                        "WARN: No states of interest or references were found.",
                        "yellow",
                    )
                )

        ####################################
        # Plot mean path and std for #######
        # the observations #################
        ####################################
        if args.plot_o:
            print("Plotting observations mean path and standard deviation...")

            # Retrieve USER defined observations list
            req_obs = EVAL_PARAMS["obs"]

            # Calculate mean observation path and std
            obs_trimmed = [
                path
                for path in eval_paths["s"]
                if len(path) == max(eval_paths["episode_length"])
            ]
            obs_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(obs_trimmed), axis=0))
            )
            obs_std_path = np.transpose(
                np.squeeze(np.std(np.array(obs_trimmed), axis=0))
            )

            # Make sure mean path and std arrays are the right shape
            obs_mean_path = (
                np.expand_dims(obs_mean_path, axis=0)
                if len(obs_mean_path.shape) == 1
                else obs_mean_path
            )
            obs_std_path = (
                np.expand_dims(obs_std_path, axis=0)
                if len(obs_std_path.shape) == 1
                else obs_std_path
            )

            # Check if USER requested observation exists
            valid_obs = validate_req_obs(req_obs, obs_mean_path)

            # Plot mean observations path and std
            print(
                "Using: {}...".format(
                    ("observations " if len(valid_obs) > 1 else "observation ")
                    + str(valid_obs)
                )
            )
            if valid_obs:  # Check if any sio or refs were found
                if EVAL_PARAMS["obs_merged"]:
                    fig = plt.figure(
                        figsize=(9, 6),
                        num=(
                            f"{policy_type}_TORCH_"
                            + str(len(list(itertools.chain(*figs.values()))) + 1)
                        ),
                    )
                    colors = "bgrcmk"
                    cycol = itertools.cycle(colors)
                    ax2 = fig.add_subplot(111)
                    figs["states"].append(fig)  # Store figure reference
                for i in range(0, obs_mean_path.shape[0]):
                    if (i + 1) in req_obs or not req_obs:
                        if not EVAL_PARAMS[
                            "obs_merged"
                        ]:  # Create separate figs for each obs
                            fig = plt.figure(
                                figsize=(9, 6),
                                num=(
                                    f"{policy_type}_TORCH_"
                                    + str(
                                        len(list(itertools.chain(*figs.values()))) + 1
                                    )
                                ),
                            )
                            ax2 = fig.add_subplot(111)
                            color = "blue"
                            figs["states"].append(fig)  # Store figure reference
                        else:
                            color = next(cycol)

                        # Plot observations
                        ax2.plot(
                            t,
                            obs_mean_path[i],
                            color=color,
                            linestyle="dashed",
                            label=(f"s_{i+1}"),
                        )
                        ax2.fill_between(
                            t,
                            obs_mean_path[i] - obs_std_path[i],
                            obs_mean_path[i] + obs_std_path[i],
                            color=color,
                            alpha=0.3,
                            label=(f"s_{i+1}_std"),
                        )

                        # Add figure legend and title (Separate figures)
                        if not EVAL_PARAMS["obs_merged"]:
                            ax2_title = (
                                EVAL_PARAMS["obs_title"]
                                if (
                                    EVAL_PARAMS["obs_title"]
                                    and isinstance(EVAL_PARAMS["obs_title"], str)
                                )
                                else "Observation"
                            )
                            ax2.set_title(f"{ax2_title} {i+1}")
                            handles2, labels2 = ax2.get_legend_handles_labels()
                            ax2.legend(
                                handles2, labels2, loc=2, fancybox=False, shadow=False
                            )

                # Add figure legend and title (merged figure)
                if EVAL_PARAMS["obs_merged"]:
                    ax2_title = (
                        EVAL_PARAMS["obs_title"]
                        if (
                            EVAL_PARAMS["obs_title"]
                            and isinstance(EVAL_PARAMS["obs_title"], str)
                        )
                        else "Observations"
                    )
                    ax2.set_title(f"{ax2_title}")
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(handles2, labels2, loc=2, fancybox=False, shadow=False)
            else:
                print(colorize("WARN: No observations were found.", "yellow"))

        ####################################
        # Plot mean cost and std for #######
        # the observations #################
        ####################################
        if args.plot_c:
            print("Plotting mean cost and standard deviation...")

            # Retrieve USER defined observations list
            req_costs = EVAL_PARAMS["costs"]

            # Calculate mean cost and std
            costs_trimmed = [
                path
                for path in eval_paths["r"]
                if len(path) == max(eval_paths["episode_length"])
            ]
            costs_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(costs_trimmed), axis=0))
            )
            costs_std_path = np.transpose(
                np.squeeze(np.std(np.array(costs_trimmed), axis=0))
            )

            # Make sure mean path and std arrays are the right shape
            costs_mean_path = (
                np.expand_dims(costs_mean_path, axis=0)
                if len(costs_mean_path.shape) == 1
                else costs_mean_path
            )
            costs_std_path = (
                np.expand_dims(costs_std_path, axis=0)
                if len(costs_std_path.shape) == 1
                else costs_std_path
            )

            # Check if USER requested observation exists
            valid_costs = validate_req_costs(req_costs, costs_mean_path)

            # Plot mean observations path and std
            print(
                "Using: {}...".format(
                    ("costs " if len(valid_costs) > 1 else "cost ") + str(valid_costs)
                )
            )
            if valid_costs:  # Check if any sio or refs were found
                if EVAL_PARAMS["costs_merged"]:
                    fig = plt.figure(
                        figsize=(9, 6),
                        num=(
                            f"{policy_type}_TORCH_"
                            + str(len(list(itertools.chain(*figs.values()))) + 1)
                        ),
                    )
                    colors = "bgrcmk"
                    cycol = itertools.cycle(colors)
                    ax3 = fig.add_subplot(111)
                    figs["costs"].append(fig)  # Store figure reference
                for i in range(0, costs_mean_path.shape[0]):
                    if (i + 1) in req_costs or not req_costs:
                        if not EVAL_PARAMS[
                            "costs_merged"
                        ]:  # Create separate figs for each cost
                            fig = plt.figure(
                                figsize=(9, 6),
                                num=(
                                    f"{policy_type}_TORCH_"
                                    + str(
                                        len(list(itertools.chain(*figs.values()))) + 1
                                    )
                                ),
                            )
                            ax3 = fig.add_subplot(111)
                            color = "blue"
                            figs["costs"].append(fig)  # Store figure reference
                        else:
                            color = next(cycol)

                        # Plot mean costs
                        ax3.plot(
                            t,
                            costs_mean_path[i],
                            color=color,
                            linestyle="dashed",
                            label=(f"cost_{i+1}"),
                        )
                        ax3.fill_between(
                            t,
                            costs_mean_path[i] - costs_std_path[i],
                            costs_mean_path[i] + costs_std_path[i],
                            color=color,
                            alpha=0.3,
                            label=(f"cost_{i+1}_std"),
                        )

                        # Add figure legend and title (Separate figures)
                        if not EVAL_PARAMS["costs_merged"]:
                            ax3_title = (
                                EVAL_PARAMS["costs_title"]
                                if (
                                    EVAL_PARAMS["costs_title"]
                                    and isinstance(EVAL_PARAMS["costs_title"], str)
                                )
                                else "Mean cost"
                            )
                            ax3.set_title(f"{ax3_title} {i+1}")
                            handles3, labels3 = ax3.get_legend_handles_labels()
                            ax3.legend(
                                handles3, labels3, loc=2, fancybox=False, shadow=False
                            )

                # Add figure legend and title (merged figure)
                if EVAL_PARAMS["obs_merged"]:
                    ax3_title = (
                        EVAL_PARAMS["costs_title"]
                        if (
                            EVAL_PARAMS["costs_title"]
                            and isinstance(EVAL_PARAMS["costs_title"], str)
                        )
                        else "Mean Cost"
                    )
                    ax3.set_title(f"{ax3_title}")
                    handles3, labels3 = ax3.get_legend_handles_labels()
                    ax3.legend(handles3, labels3, loc=2, fancybox=False, shadow=False)
            else:
                print(colorize("WARN: No costs were found.", "yellow"))

        # Show figures
        plt.show()

        # Save figures if requested
        print("Saving plots...")
        print(colorize(f"INFO: Save path: {log_path}", "cyan", bold=True))
        if args.save_figs:
            for index, fig in enumerate(figs["states_of_interest"]):
                save_path = (
                    osp.join(
                        log_path,
                        "Quatonian_"
                        + str(index + 1)
                        + "."
                        + EVAL_PARAMS["fig_file_type"],
                    )
                    if not EVAL_PARAMS["sio_merged"]
                    else osp.join(
                        log_path, "Quatonians" + "." + EVAL_PARAMS["fig_file_type"],
                    )
                )
                fig.savefig(
                    save_path, bbox_inches="tight",
                )
            for index, fig in enumerate(figs["states"]):
                save_path = (
                    osp.join(
                        log_path,
                        "State_" + str(index + 1) + "." + EVAL_PARAMS["fig_file_type"],
                    )
                    if not EVAL_PARAMS["obs_merged"]
                    else osp.join(
                        log_path, "States" + "." + EVAL_PARAMS["fig_file_type"],
                    )
                )
                fig.savefig(
                    save_path, bbox_inches="tight",
                )
            for index, fig in enumerate(figs["costs"]):
                fig.savefig(
                    osp.join(log_path, "Cost" + "." + EVAL_PARAMS["fig_file_type"]),
                    bbox_inches="tight",
                )
