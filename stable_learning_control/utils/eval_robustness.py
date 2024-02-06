"""Contains a utility that can be used to evaluate the stability and robustness of an
algorithm. See the :ref:`Robustness Evaluation Documentation <robustness_eval>` for
more information.
"""

import ast
import copy
import importlib
import inspect
import os
import sys
from pathlib import Path, PurePath
from textwrap import dedent

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stable_learning_control.common.helpers import (
    convert_to_snake_case,
    convert_to_wandb_config,
    friendly_err,
    get_env_id,
)
from stable_learning_control.utils.log_utils.helpers import log_to_std_out
from stable_learning_control.utils.log_utils.logx import EpochLogger
from stable_learning_control.utils.test_policy import load_policy_and_env


def get_human_readable_disturber_label(disturber_label):
    """Get a human readable label for a given disturber label.

    Args:
        disturber_label (str): The disturber label.

    Returns:
        str: The human readable disturber label.
    """
    human_str = ""
    for i, s in enumerate(disturber_label.split("_")):
        if i == 0:
            human_str += f"{s}"
        elif i % 2 == 0:
            human_str += f", {s}"
        else:
            human_str += f"={s}"
    return human_str


def add_disturbance_label_column(dataframe):
    """Add a column that contains a disturbance label. This label will be created
    by concatenating all disturber parameters.

    .. note::
        This function will not add a disturbance label if the dataframe already contains
        a disturbance label.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the disturber parameters.

    Returns:
        pd.DataFrame: The dataframe with the disturbance label column.

    Raises:
        ValueError: If the dataframe does not contain a 'disturber' column.
    """
    # Return dataframe if it already contains a disturbance label.
    if "disturbance_label" in dataframe.columns:
        return dataframe["disturbance_label"].fillna("baseline")

    # Create disturbance label.
    if "disturber" in dataframe.columns:
        disturbance_label_df = dataframe.apply(
            lambda row: (
                "_".join(
                    [
                        "{key}_{value}".format(
                            key=key.split("_")[1],
                            value=(
                                round(value, 2) if isinstance(value, float) else value
                            ),
                        )
                        for key, value in row.items()
                        if key.startswith("disturber_")
                    ]
                )
                if row["disturber"] != "baseline"
                else "baseline"
            ),
            axis=1,
        )

        return disturbance_label_df
    elif "disturber" not in dataframe.columns:
        raise ValueError("The dataframe does not contain a 'disturber' column. ")


def get_available_disturbers():
    """Get all disturbers that are available in the ``stable_learning_control`` package.

    Returns:
        list: List with all available disturbers.
    """
    disturber_module = importlib.import_module("stable_learning_control.disturbers")
    disturber_names = [
        name
        for name, obj in inspect.getmembers(disturber_module)
        if inspect.isclass(obj) and issubclass(obj, gym.Wrapper)
    ]
    return disturber_names


def get_available_disturbers_string():
    """Get a string with all available disturbers that are available in the
    ``stable_learning_control`` package.

    Returns:
        str: String with all available disturbers.
    """
    available_disturbers = get_available_disturbers()
    str_valid_disturbers = ""
    for disturber_name in available_disturbers:
        str_valid_disturbers += f" - {disturber_name}\n"
    list_disturbers_msg = (
        dedent(
            """
            Currently available disturbers in the SLC package are:
            """
        )
        + str_valid_disturbers
        + dedent(
            """
            You can use the '--disturbance_type' argument to specify the disturber and
            the '--help' argument to get more information about a given disturber.

            For example:

                python -m stable_learning_control.run eval_robustness [path/to/output_directory] [disturber] --help
            """  # noqa: E501
        )
    )
    return list_disturbers_msg


def print_available_disturbers():
    """Print all available disturbers that are available in the
    ``stable_learning_control`` package.
    """
    available_disturbers_string = get_available_disturbers_string()
    print(available_disturbers_string)


def load_disturber(disturber_id):
    """Load a given disturber. The disturber name can include an unloaded module in
    "module:disturber_name" style. If no module is given, the disturber is loaded from
    the ``stable_learning_control.disturbers`` module.

    Args:
        disturber_id (str): The name of the disturber you want to load.

    Returns:
        The loaded disturber object.

    Raises:
        ModuleNotFoundError: If the given module could not be imported.
        TypeError: If the given disturber is not a subclass of the :class:`gym.Wrapper`
            class.
        AttributeError: If the given disturber does not exist in the given module.
    """
    module, disturber_name = (
        (None, disturber_id) if ":" not in disturber_id else disturber_id.split(":")
    )

    # Load disturber from the given module.
    if module is not None:
        # Try to import the module.
        try:
            disturber_module = importlib.import_module(module)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                friendly_err(
                    f"{e}. Disturber retrieval via importing a module failed. "
                    f"Check whether '{module}' contains '{disturber_name}'."
                )
            ) from e

        # Load disturber from the module.
        try:
            disturber = getattr(disturber_module, disturber_name)
        except AttributeError as e:
            raise AttributeError(
                friendly_err(
                    f"{e}. Disturber '{disturber_name}' could not be found in "
                    f"module '{module}'."
                )
            ) from e

    else:
        # Load disturber from the SLC disturbers.
        disturber_module = importlib.import_module("stable_learning_control.disturbers")
        try:
            disturber = getattr(disturber_module, disturber_name)
        except AttributeError as e:
            raise AttributeError(
                friendly_err(
                    f"Disturber '{disturber_name}' could not be found in "
                    f"module '{disturber_module}'.\n"
                    f"{get_available_disturbers_string()}"
                )
            ) from e

    # Validate whether the disturber is a subclass of the gym.Wrapper class.
    if not issubclass(disturber, gym.Wrapper):
        raise TypeError(
            friendly_err(
                f"Disturber '{disturber_name}' is not a valid disturber since it "
                "does not inherit from the 'gym.Wrapper' class."
            )
        )

    return disturber


def retrieve_disturber_variants(disturber_range_dict):
    """Retrieves all disturber variants from the given disturber configuration
    dictionary. Variants are created by combining the key values over indexes.

    Args:
        disturber_range_dict (dict): The disturber configuration dictionary.

    Returns:
        list: List with all disturber variants.

    Raises:
        TypeError: Thrown when the values in the disturber configuration variables are
            not of type ``float``, ``int`` or ``list``.
        ValueError: Thrown when the values in the disturber configuration variables do
            not have the same length.
    """
    # Throw warning if values are not float int or list.
    if not all(
        [
            isinstance(value, (float, int, list))
            for value in disturber_range_dict.values()
        ]
    ):
        raise TypeError(
            friendly_err(
                "All values in the 'disturber_range_dict' dictionary should be of type "
                "'float', 'int' or 'list'."
            )
        )

    # Convert all values to lists.
    disturber_range_dict = {
        key: [value] if not isinstance(value, list) else value
        for key, value in disturber_range_dict.items()
    }

    # Throw a warning if the values don't have the same length.
    if not all(
        [
            len(value) == len(list(disturber_range_dict.values())[0])
            for value in disturber_range_dict.values()
        ]
    ):
        raise ValueError(
            friendly_err(
                "All values in the '--disturber_config' dictionary should have the "
                "same length."
            )
        )

    # Get the disturber variants.
    # NOTE: Combines key values over indexes.
    disturber_variants = [
        {key: value[ii] for key, value in disturber_range_dict.items()}
        for ii in range(len(list(disturber_range_dict.values())[0]))
    ]

    return disturber_variants


def run_disturbed_policy(
    env,
    policy,
    disturber,
    disturber_config,
    include_baseline=True,
    max_ep_len=None,
    num_episodes=100,
    render=True,
    deterministic=True,
    save_result=False,
    output_dir=None,
    use_wandb=False,
    wandb_job_type=None,
    wandb_project=None,
    wandb_group=None,
    wandb_run_name=None,
):
    """Evaluates the disturbed policy inside a given gymnasium environment. This
    function loops to all the disturbances that are specified in the environment and
    outputs the results of all these episodes as a :obj:pandas.Dataframe`.

    Args:
        env (:obj:`gym.env`): The gymnasium environment.
        policy (Union[tf.keras.Model, torch.nn.Module]): The policy.
        disturber (obj:`gym.Wrapper`): The disturber you want to use.
        disturber_config (dict): The disturber configuration dictionary. Contains the
            variables that you want to pass to the disturber. It sets up the range of
            disturbances you wish to evaluate.
        include_baseline (bool): Whether you want to automatically add the baseline
            (i.e. zero disturbance) when it not present. Defaults to ``True``.
        max_ep_len (int, optional): The maximum episode length. Defaults to ``None``.
            Meaning the maximum episode length of the environment is used.
        num_episodes (int, optional): Number of episodes you want to perform in the
            environment. Defaults to ``100``.
        render (bool, optional): Whether you want to render the episode to the screen.
            Defaults to ``True``.
        deterministic (bool, optional): Whether you want the action from the policy to
            be deterministic. Defaults to ``True``.
        save_result (bool, optional): Whether you want to save the dataframe with the
            results to disk. Defaults to ``False``.
        output_dir (str, optional): A directory for saving the diagnostics to. If
            ``None``, defaults to a temp directory of the form
            ``/tmp/experiments/somerandomnumber``.
        use_wandb (bool, optional): Whether to use Weights & Biases for logging.
            Defaults to ``False``.
        wandb_job_type (str, optional): The type of job you are running. Defaults
            to ``None``.
        wandb_project (str, optional): The name of the Weights & Biases
            project. Defaults to ``None`` which means that the project name is
            automatically generated.
        wandb_group (str, optional): The name of the Weights & Biases group you want
            to assign the run to. Defaults to ``None``.
        wandb_run_name (str, optional): The name of the Weights & Biases run. Defaults
            to ``None`` which means that the run name is automatically generated.

    Returns:
        :obj:`pandas.DataFrame`: Dataframe that contains information about all the
            episodes and disturbances.

    Raises:
        AssertionError: Thrown when the environment or policy is not found.
    """
    eval_args = copy.deepcopy(locals())

    # Retrieve disturber variants.
    disturber_variants = (
        retrieve_disturber_variants(disturber_config)
        if disturber_config is not None
        else []
    )

    # Throw error if baseline is disabled and no disturber variants are present.
    if not include_baseline and not disturber_variants:
        raise ValueError(
            friendly_err(
                "You disabled the baseline evaluation and no disturber variants are "
                "present. Please check the '--disturber_config' dictionary and try "
                "again."
            )
        )

    # Let the user know that only the baseline is evaluated and change output dir.
    if not disturber_variants:
        log_to_std_out(
            (
                "You did not supply a '--disturber_config' dictionary. As a result only "
                "the baseline will be evaluated."
            ),
            type="warning",
        )

    # Setup logger.
    if output_dir is not None:
        output_dir = str(
            Path(output_dir).joinpath(
                "eval/{}".format(
                    convert_to_snake_case(disturber.__name__)
                    if disturber_variants
                    else "baseline"
                )
            )
        )
    logger = EpochLogger(
        verbose_fmt="table",
        output_dir=output_dir,
        output_fname="eval_statistics.csv",
    )

    assert env is not None, friendly_err(
        "Environment not found!\n\n It looks like the environment wasn't saved, and we "
        "can't run the agent in it. :( \n\n Check out the documentation page on the "
        "page on the robustness evaluation utility for how to handle this situation."
    )
    assert policy is not None, friendly_err(
        "Policy not found!\n\n It looks like the policy could not be loaded. :( \n\n "
        "Check out the documentation page on the robustness evaluation utility for how "
        "to handle this situation."
    )

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
    if render:
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

    # Create environment list for each disturbance variant.
    variants_envs = []
    if include_baseline:
        variants_envs.append(env)
    for variant in disturber_variants:
        try:
            disturbance_variant = disturber(env, **variant)
        except Exception as e:
            if isinstance(e, TypeError):
                raise TypeError(
                    friendly_err(
                        "Something went wrong when trying to initialize the "
                        f"'{disturber.__name__}' disturber with arguments `{variant}`. "
                        "Please check the disturber arguments supplied in the "
                        "'disturber_config' dictionary and try again. You can find more "
                        f"information about the '{disturber.__name__}' disturber "
                        "through the '--help' flag."
                    )
                )
            else:
                raise friendly_err(e)
        variants_envs.append(disturbance_variant)

    # Setup storage variables.
    path = {
        "o": [],
        "r": [],
        "reference": [],
        "reference_error": [],
    }
    time = []
    variant_df = pd.DataFrame()
    variants_df = pd.DataFrame()
    ref_found, ref_error_found, supports_deterministic = True, True, True
    time_attribute = None
    time_step_attribute = None
    disturbance_diagnostics = pd.DataFrame()

    # Check if environment has time attribute.
    if hasattr(env.unwrapped, "time") or hasattr(env.unwrapped, "t"):
        time_attribute = "time" if hasattr(env.unwrapped, "time") else "t"
    if (
        hasattr(env.unwrapped, "dt")
        or hasattr(env.unwrapped, "tau")
        or hasattr(env.unwrapped, "timestep")
        or hasattr(env.unwrapped, "time_step")
    ):
        time_step_attribute = (
            "dt"
            if hasattr(env.unwrapped, "dt")
            else (
                "tau"
                if hasattr(env.unwrapped, "tau")
                else "timestep" if hasattr(env.unwrapped, "timestep") else "time_step"
            )
        )
    if time_attribute is None and time_step_attribute is None:
        logger.log(
            (
                "No time attributes (i.e. 'time', 't', 'dt', 'tau', 'timestep', "
                "'time_step') were found on the environment. As a result a time "
                "step of '1' will be used."
            ),
            type="warning",
        )

    # Evaluate each disturbance variant.
    logger.log("\n")
    logger.log("Performing robustness evaluation...", type="info")
    for variant_idx, variant_env in enumerate(variants_envs):
        if include_baseline and variant_idx == 0:
            logger.log("Evaluating baseline...", type="info")
        else:
            logger.log(
                (
                    f"Evaluating '{disturber.__name__}' with "
                    f"`{disturber_variants[variant_idx-1]}` parameters..."
                ),
                type="info",
            )

        # Perform episodes.
        o, _ = variant_env.reset()
        r, d, ep_ret, ep_len, n, t = 0, False, 0, 0, 0, 0
        while n < num_episodes:
            # Retrieve action.
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

            # Perform action in the environment and store result.
            o, r, d, truncated, info = variant_env.step(a)

            # Track time.
            time.append(t)
            if time_attribute is not None:
                t = variant_env.unwrapped.__getattribute__(time_attribute)
            elif time_step_attribute is not None:
                t += variant_env.unwrapped.__getattribute__(time_step_attribute)
            else:
                t += 1

            # Store step information.
            ep_ret += r
            ep_len += 1
            path["o"].append(o)
            path["r"].append(r)
            if ref_found and "reference" in info.keys():
                path["reference"].append(info["reference"])
            else:
                if ref_found:
                    logger.log(
                        (
                            "No 'reference' found in info dictionary. Please add the "
                            "reference to the info dictionary of the environment if "
                            "you want to have it show up in the evaluation results."
                        ),
                        type="warning",
                    )
                ref_found = False
            if ref_error_found and "reference_error" in info.keys():
                path["reference_error"].append(info["reference_error"])
            else:
                if ref_error_found:
                    logger.log(
                        (
                            "No 'reference_error' found in info dictionary. Please add "
                            "the reference to the info dictionary of the environment "
                            "if you want to have it show up in the evaluation results."
                        ),
                        type="warning",
                    )
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
                episode_df.insert(len(episode_df.columns), "time", time)
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
                o, info = variant_env.reset()
                r, d, ep_ret, ep_len, n, t = 0, False, 0, 0, n + 1, 0
                path = {
                    "o": [],
                    "r": [],
                    "reference": [],
                    "reference_error": [],
                }
                time = []

        # Print variant diagnostics.
        logger.log_tabular(
            "Disturber",
            "baseline" if include_baseline and variant_idx == 0 else disturber.__name__,
        )
        if len(disturber_variants) > 1:
            for var_name, var_value in disturber_variants[
                max(0, variant_idx - 1) if include_baseline else variant_idx
            ].items():
                if isinstance(var_value, list):
                    for i, var_value_i in enumerate(var_value):
                        logger.log_tabular(
                            f"Disturber_{var_name}_{i+1}",
                            (
                                np.nan
                                if include_baseline and variant_idx == 0
                                else var_value_i
                            ),
                        )
                else:
                    logger.log_tabular(
                        f"Disturber_{var_name}",
                        np.nan if include_baseline and variant_idx == 0 else var_value,
                    )
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("DeathRate")
        logger.log("")
        disturbance_diagnostics = pd.concat(
            [
                disturbance_diagnostics,
                pd.DataFrame(
                    logger.log_current_row,
                    index=[variant_idx],
                ),
            ],
            axis=0,
            ignore_index=True,
        )  # Store variant diagnostics in dataframe.
        logger.dump_tabular()
        logger.log("")

        # Add disturbance information to the robustness evaluation dataframe.
        variant_df.insert(
            len(variant_df.columns),
            "disturber",
            "baseline" if include_baseline and variant_idx == 0 else disturber.__name__,
        )
        if len(disturber_variants) > 1:
            for var_name, var_value in disturber_variants[
                max(0, variant_idx - 1) if include_baseline else variant_idx
            ].items():
                if isinstance(var_value, list):
                    for i, var_value_i in enumerate(var_value):
                        variant_df.insert(
                            len(variant_df.columns),
                            f"disturber_{var_name}_{i+1}",
                            (
                                np.nan
                                if include_baseline and variant_idx == 0
                                else var_value_i
                            ),
                        )
                else:
                    variant_df.insert(
                        len(variant_df.columns),
                        f"disturber_{var_name}",
                        np.nan if include_baseline and variant_idx == 0 else var_value,
                    )
        if hasattr(
            variant_env, "disturbance_label"
        ):  # NOTE: Add disturbance label if available in the environment.
            variant_df.insert(
                len(variant_df.columns),
                "disturbance_label",
                variant_env.disturbance_label,
            )
        variants_df = pd.concat([variants_df, variant_df], axis=0, ignore_index=True)

        # Reset variant storage buckets.
        variant_df = pd.DataFrame()

    # Save robustness evaluation dataframe and return it to the user.
    logger.log("Saving robustness evaluation dataframe...", type="info")
    results_path = Path(logger.output_dir).joinpath("eval_results.csv")
    if save_result:
        logger.log(
            f"Saving robustness evaluation results to path: {results_path}", type="info"
        )
        variants_df.to_csv(results_path, index=False)
        logger.log("Robustness evaluation dataframe saved.", type="info")

    # Log robustness evaluation results to Weights & Biases.
    if use_wandb:
        logger.log("Storing robustness evaluation results in Weights & Biases...")
        import wandb

        # Initialize Weights & Biases if not already initialized.
        if wandb.run is None:
            wandb.init(
                project=wandb_project,
                group=wandb_group,
                job_type=wandb_job_type,
                name=wandb_run_name,
                config=convert_to_wandb_config(eval_args),
            )

        # Create disturbance diagnostics table and bar plots.
        wandb.log({"disturbance_diagnostics": disturbance_diagnostics})
        bar_plots = {}
        for column_name in disturbance_diagnostics.columns:
            if column_name == "Disturber":
                continue
            bar_plots[f"{column_name}_bar"] = wandb.plot.bar(
                wandb.Table(
                    data=disturbance_diagnostics[["Disturber", column_name]],
                    columns=["Disturber", column_name],
                ),
                label="Disturber",
                value=column_name,
                title=f"{column_name} per disturbance",
            )
        wandb.log(bar_plots)

        # Store robustness evaluation dataframe as artifact.
        eval_results_artifact = wandb.Artifact(
            name="robustness_evaluation_results",
            type="dataframe",
            description="Contains the results of the robustness evaluation runs.",
        )
        eval_results_artifact.add_file(results_path)
        wandb.log_artifact(eval_results_artifact)
        logger.log("Robustness evaluation results stored in Weights & Biases.")

    return variants_df


def plot_robustness_results(
    dataframe,
    observations=None,
    references=None,
    reference_errors=None,
    absolute_reference_errors=False,
    merge_reference_errors=False,
    use_subplots=False,
    use_time=False,
    save_plots=False,
    font_scale=1.5,
    figs_fmt="pdf",
    output_dir=None,
    use_wandb=False,
    wandb_job_type=None,
    wandb_project=None,
    wandb_group=None,
    wandb_run_name=None,
):
    """Creates several useful plots out of a robustness evaluation dataframe that was
    collected in the :meth:`run_disturbed_policy` method.

    Args:
        dataframe (pandas.DataFrame): The data frame that contains the robustness
            evaluation information information.
        observations (list): The observations you want to show in the observations plot.
            By default for clarity only the first 6 observations are shown.
        references (list): The references you want to show in the reference plot. By
            default for clarity only the first references is shown.
        reference_errors (list): The reference errors you want to show in the reference
            error plot. By default for clarity only the first reference error is shown.
        absolute_reference_errors (bool): Whether you want to plot the absolute
            reference errors instead of relative reference errors. Defaults to
            ``False``.
        merge_reference_errors (bool): Whether you want to merge the reference errors
            into one reference error. Defaults to ``False``.
        use_subplots (bool): Whether you want to use subplots instead of separate
            figures. Defaults to ``False``.
        use_time (bool): Whether you want to use the time as the x-axis. Defaults to
            ``False``.
        save_plots (bool): Whether you want to save the created plots to disk. Defaults
            to ``False``.
        font_scale (int): The font scale you want to use for the plot text. Defaults to
            ``1.5``.
        figs_fmt (str, optional): In which format you want to save the plots. Defaults
            to ``pdf``.
        output_dir (str, optional):The directory where you want to save the output
            figures to. If ``None``, defaults to a temp directory of the form
            ``/tmp/experiments/somerandomnumber``.
        use_wandb (bool, optional): Whether to use Weights & Biases for logging.
            Defaults to ``False``.
        wandb_job_type (str, optional): The type of job you are running. Defaults
            to ``None``.
        wandb_project (str, optional): The name of the Weights & Biases
            project. Defaults to ``None`` which means that the project name is
            automatically generated.
        wandb_group (str, optional): The name of the Weights & Biases group you want
            to assign the run to. Defaults to ``None``.
        wandb_run_name (str, optional): The name of the Weights & Biases run. Defaults
            to ``None`` which means that the run name is automatically generated.
    """
    # Retrieve disturbances and disturber name from dataframe.
    dataframe["disturbance_label"] = add_disturbance_label_column(dataframe)
    disturber_name = dataframe["disturber"].unique()[-1]

    # Setup logger, x-axis variable and output directory.
    if output_dir is not None:
        output_dir = str(
            Path(output_dir).joinpath(
                "eval/{}".format(
                    convert_to_snake_case(disturber.__name__)
                    if disturber_name != "baseline"
                    else "baseline"
                )
            )
        )
    logger = EpochLogger(
        output_dir=output_dir,
        output_dir_exists_warning=False,
    )
    x_axis_var = "time" if use_time else "step"

    # Create a dictionary to store all plots.
    figs = {
        "observation": [],
        "cost": [],
        "reference_error": [],
    }

    # Absolute reference errors if requested.
    if absolute_reference_errors:
        for col in dataframe.columns:
            if col.startswith("reference_error_") or col == "reference_error":
                dataframe[col] = dataframe[col].abs()

    # Merge reference errors if requested.
    if merge_reference_errors:
        if "reference_error" not in dataframe.columns:
            dataframe["reference_error"] = dataframe[
                [col for col in dataframe.columns if col.startswith("reference_error_")]
            ].sum(axis=1)
            dataframe = dataframe.drop(
                [
                    col
                    for col in dataframe.columns
                    if col.startswith("reference_error_")
                ],
                axis=1,
            )
        else:
            logger.log(
                "Only one reference error present in dataframe, skipping merging...",
                type="warn",
            )

    # Initialize Seaborn style and font scale.
    logger.log("\n")
    logger.log("Showing robustness evaluation plots...", type="info")
    sns.set(style="darkgrid", font_scale=font_scale)

    # Retrieve available observations, references and reference errors.
    available_obs = (
        [1]
        if "observation" in dataframe.columns
        else [
            int(col.replace("observation_", ""))
            for col in dataframe.columns
            if col.startswith("observation_")
        ]
    )
    available_refs = (
        [1]
        if "reference" in dataframe.columns
        else [
            int(col.replace("reference_", ""))
            for col in dataframe.columns
            if col.startswith("reference_")
            and not col.replace("reference_", "").startswith("error")
        ]
    )
    available_ref_errors = (
        [1]
        if "reference_error" in dataframe.columns
        else [
            int(col.replace("reference_error_", ""))
            for col in dataframe.columns
            if col.startswith("reference_error_")
        ]
    )

    # Filter observations if requested.
    if len(available_obs) > 0:
        if observations is not None:
            # Validate requested observations.
            if not all([obs.isdigit() for obs in observations]):
                raise ValueError(
                    "Observations must be a list of integers. Please check your input."
                )
            observations = [int(obs) for obs in observations]
            if any([obs < 1 for obs in observations]):
                raise ValueError(
                    "Observations must be a list of integers greater than '0'. Please "
                    "check your input."
                )
            if any([obs > max(available_obs) for obs in observations]):
                raise ValueError(
                    f"Observations must be a list of integers smaller than or equal to "
                    f"'{max(available_obs)}'. Please check your input."
                )

            # Remove observation columns that are not requested.
            if "observation" not in dataframe.columns:
                unwanted_obs = [
                    col
                    for col in dataframe.columns
                    if col.startswith("observation_")
                    and int(col.replace("observation_", "")) not in observations
                ]
            elif "observation" in dataframe.columns:
                unwanted_obs = ["observation"] if len(observations) == 0 else []
            dataframe = dataframe.drop(unwanted_obs, axis=1)
            available_obs = observations
        else:  # Show only the first 6 observations by default.
            available_obs = available_obs[:6]

    # Filter references if requested.
    if len(available_refs) > 0:
        if references is not None:
            # Validate requested references.
            if not all([ref.isdigit() for ref in references]):
                raise ValueError(
                    "References must be a list of integers. Please check your input."
                )
            references = [int(ref) for ref in references]
            if any([ref < 1 for ref in references]):
                raise ValueError(
                    "References must be a list of integers greater than '0'. Please "
                    "check your input."
                )
            if any([ref > max(available_refs) for ref in references]):
                raise ValueError(
                    f"References must be a list of integers smaller than or equal to "
                    f"'{max(available_refs)}'. Please check your input."
                )

            # Remove reference columns that are not requested.
            if "reference" not in dataframe.columns:
                unwanted_refs = [
                    col
                    for col in dataframe.columns
                    if col.startswith("reference_")
                    and not col.replace("reference_", "").startswith("error")
                    and int(col.replace("reference_", "")) not in references
                ]
            elif "reference" in dataframe.columns:
                unwanted_refs = ["reference"] if len(references) == 0 else []
            dataframe = dataframe.drop(unwanted_refs, axis=1)
            available_refs = references
        else:  # Show only the first 6 references by default.
            available_refs = available_refs[:6]

    # Filter reference errors if requested.
    if len(available_ref_errors) > 0:
        if reference_errors is not None:
            # Validate requested reference errors.
            if not all([ref_err.isdigit() for ref_err in reference_errors]):
                raise ValueError(
                    "Reference errors must be a list of integers. Please check your "
                    "input."
                )
            reference_errors = [int(ref_err) for ref_err in reference_errors]
            if any([ref_err < 1 for ref_err in reference_errors]):
                raise ValueError(
                    "Reference errors must be a list of integers greater than '0'. "
                    "Please check your input."
                )
            if any(
                [ref_err > max(available_ref_errors) for ref_err in reference_errors]
            ):
                raise ValueError(
                    f"Reference errors must be a list of integers smaller than or "
                    f"equal to '{max(available_ref_errors)}'. Please check your "
                    "input."
                )

            # Remove reference error columns that are not requested.
            if "reference_error" not in dataframe.columns:
                unwanted_ref_errors = [
                    col
                    for col in dataframe.columns
                    if col.startswith("reference_error_")
                    and int(col.replace("reference_error_", "")) not in reference_errors
                ]
            elif "reference_error" in dataframe.columns:
                unwanted_ref_errors = (
                    ["reference_error"] if len(reference_errors) == 0 else []
                )
            dataframe = dataframe.drop(unwanted_ref_errors, axis=1)
            available_ref_errors = reference_errors
        else:
            available_ref_errors = available_ref_errors[:6]

    # Plot mean observations and references per disturbance variant.
    if len(available_obs) > 0 or len(available_refs) > 0:
        logger.log(
            "Plotting mean observations and references per disturbance variant...",
            type="info",
        )
        n_plots = len(dataframe["disturbance_label"].unique())
        figs_tmp = []
        for disturbance_idx, disturbance in enumerate(
            dataframe["disturbance_label"].unique()
        ):
            # Get observations in long format.
            disturbance_df = dataframe[dataframe["disturbance_label"] == disturbance]
            obs_value_vars = (
                ["observation"]
                if "observation" in dataframe.columns
                else [f"observation_{obs}" for obs in available_obs]
            )
            obs_disturbance_df = disturbance_df.melt(
                id_vars=[x_axis_var, "disturbance_label"],
                value_vars=obs_value_vars,
                var_name="observation",
                value_name="value",
            )

            # Get references in long format.
            refs_value_vars = (
                ["reference"]
                if "reference" in dataframe.columns
                else [f"reference_{ref}" for ref in available_refs]
            )
            refs_disturbance_df = disturbance_df.melt(
                id_vars=[x_axis_var, "disturbance_label"],
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

            # Initialize plot/subplots and title.
            if use_subplots:
                if disturbance_idx % 6 == 0:
                    fig, axs = plt.subplots(
                        ncols=min(n_plots - disturbance_idx, 3),
                        nrows=min(int(np.ceil((n_plots - disturbance_idx) / 3)), 2),
                        figsize=(12, 6),
                        tight_layout=True,
                    )
                    plt.suptitle(
                        f"Mean observations and references under '{disturber_name}' "
                        "disturber"
                    )
                ax = (
                    axs.flatten()[disturbance_idx % 6]
                    if isinstance(axs, np.ndarray)
                    else axs
                )

                # Create plot title.
                plot_title = get_human_readable_disturber_label(disturbance)
            else:
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
                if disturbance == "baseline":
                    plot_title += "baseline conditions"
                else:
                    plot_title += (
                        f"under '{disturber_name}' disturber with "
                        f"`{get_human_readable_disturber_label(disturbance)}`"
                    )  # Fix when disturbance label.
            fig.canvas.manager.resize(*fig.canvas.manager.window.maxsize())

            # Add observations to plot.
            if len(available_obs) > 0:
                sns.lineplot(
                    data=obs_disturbance_df,
                    x=x_axis_var,
                    y="value",
                    hue="observation",
                    palette="tab10",
                    legend="full",
                    ax=ax,
                )

            # Add references to plot.
            if len(available_refs) > 0:
                sns.lineplot(
                    data=refs_disturbance_df,
                    x=x_axis_var,
                    y="value",
                    hue="reference",
                    palette="hls",
                    legend="full",
                    ax=ax,
                    linestyle="--",
                )

            # Apply plot settings.
            ax.set_xlabel(x_axis_var)
            ax.set_ylabel("Value")
            ax.set_title(
                plot_title,
            )
            ax.get_legend().set_title(None)

            # Store figure.
            if (
                use_subplots
                and disturbance_idx % 6 == 5
                or disturbance_idx == n_plots - 1
            ):
                figs_tmp.append(fig)
            else:
                figs_tmp.append(fig)

        # Store plot.
        figs["observation"] = figs_tmp
    else:
        if not (observations and references):
            logger.log(
                (
                    "No observations or references available in dataframe. Skipping "
                    "observation plot."
                ),
                type="warning",
            )

    # Plot mean cost per disturbance variant in one plot if available.
    if "cost" in dataframe.columns:
        logger.log("Plotting mean cost per disturbance variant...", type="info")
        fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
        fig.canvas.manager.resize(*fig.canvas.manager.window.maxsize())
        sns.lineplot(
            data=dataframe,
            x=x_axis_var,
            y="cost",
            hue="disturbance_label",
            palette="tab10",
            legend="full",
            ax=ax,
        )
        ax.set_xlabel(x_axis_var)
        ax.set_ylabel("Cost")
        ax.set_title(f"Mean cost under '{disturber_name}' disturber")
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
        n_plots = len(dataframe["disturbance_label"].unique())
        figs_tmp = []
        for disturbance_idx, disturbance in enumerate(
            dataframe["disturbance_label"].unique()
        ):
            # Get reference error in long format.
            disturbance_df = dataframe[dataframe["disturbance_label"] == disturbance]
            ref_errors_value_vars = (
                ["reference_error"]
                if "reference_error" in dataframe.columns
                else [
                    f"reference_error_{ref_error}" for ref_error in available_ref_errors
                ]
            )
            ref_errors_disturbance_df = disturbance_df.melt(
                id_vars=[x_axis_var, "disturbance_label"],
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

            # Initialize plot/subplots and title.
            if use_subplots:
                if disturbance_idx % 6 == 0:
                    fig, axs = plt.subplots(
                        ncols=min(n_plots - disturbance_idx, 3),
                        nrows=min(int(np.ceil((n_plots - disturbance_idx) / 3)), 2),
                        figsize=(12, 6),
                        tight_layout=True,
                    )
                ax = (
                    axs.flatten()[disturbance_idx % 6]
                    if isinstance(axs, np.ndarray)
                    else axs
                )

                # Create plot title.
                plot_title = get_human_readable_disturber_label(disturbance)
            else:
                fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

                # Create plot title.
                plot_title = "Mean {} ".format(
                    (
                        "reference errors"
                        if len(available_ref_errors) > 1
                        else "reference error"
                    ),
                )
                if disturbance == "baseline":
                    plot_title += "baseline conditions"
                else:
                    plot_title += (
                        f"under '{disturber_name}' disturber with "
                        f"`{get_human_readable_disturber_label(disturbance)}`"
                    )
            fig.canvas.manager.resize(*fig.canvas.manager.window.maxsize())

            # Add reference error to plot.
            sns.lineplot(
                data=ref_errors_disturbance_df,
                x=x_axis_var,
                y="value",
                hue="reference_error",
                palette="tab10",
                legend="full",
                ax=ax,
            )

            # Configure plot.
            ax.set_xlabel(x_axis_var)
            ax.set_ylabel("Value")
            ax.set_title(
                plot_title,
            )
            ax.get_legend().set_title(None)

            # Add figure title if using subplots.
            if use_subplots:
                plt.suptitle(f"Mean reference error under '{disturber_name}' disturber")

            # Store figure.
            if (
                use_subplots
                and disturbance_idx % 6 == 5
                or disturbance_idx == n_plots - 1
            ):
                figs_tmp.append(fig)
            else:
                figs_tmp.append(fig)

        # Store plot.
        figs["reference_error"] = figs_tmp
    else:
        if not reference_errors:
            logger.log(
                (
                    "Mean reference error not plotted since no reference error "
                    "information was found in the supplied dataframe. Please ensure "
                    "the dataframe contains the 'reference_error' key."
                ),
                type="warning",
            )

    # Initialize Weights & Biases if requested and not already initialized.
    if use_wandb:
        import wandb

        if wandb.run is None:
            wandb.init(
                project=wandb_project,
                group=wandb_group,
                job_type=wandb_job_type,
                name=wandb_run_name,
            )

    # Store plots.
    if save_plots or use_wandb:
        figs_path = Path(logger.output_dir).joinpath("figures")
        figs_extension = figs_fmt[1:] if figs_fmt.startswith(".") else figs_fmt
        os.makedirs(figs_path, exist_ok=True)
        logger.log("Saving plots...", type="info")
        if save_plots:
            logger.log(f"Saving figures to path: {figs_path}", type="info")
        if use_wandb:
            logger.log("Saving figures to Weights & Biases...", type="info")

        # Store observation plots.
        if figs["observation"]:
            for idx, fig in enumerate(figs["observation"]):
                if not use_subplots:
                    fig_id = (
                        "baseline"
                        if convert_to_snake_case(disturber_name) == "baseline"
                        else "{}-{}".format(
                            convert_to_snake_case(disturber_name),
                            dataframe["disturbance_label"].unique()[idx],
                        )
                    )
                else:
                    fig_id = "{}-subplots-fig_{}".format(
                        convert_to_snake_case(disturber_name), idx + 1
                    )
                fig_name = f"observations-{fig_id}"

                # Save figures to disk.
                if save_plots:
                    fig.savefig(
                        figs_path.joinpath(f"{fig_name}.{figs_extension}"),
                        bbox_inches="tight",
                    )

                # Save figures to Weights & Biases.
                if use_wandb:
                    # IMPROVE: Can be replaced with wandb.log({fig_name: fig}) once
                    # https://github.com/wandb/wandb/issues/3987 has been solved.
                    wandb.log(
                        {fig_name: wandb.Image(fig)},
                    )

        # Store cost plots.
        if figs["cost"]:
            fig_name = f"cost-{convert_to_snake_case(disturber_name)}"

            # Save figures to disk.
            if save_plots:
                figs["cost"][0].savefig(
                    figs_path.joinpath(
                        f"{fig_name}.{figs_extension}",
                    ),
                    bbox_inches="tight",
                )

            # Save figures to Weights & Biases.
            if use_wandb:
                # IMPROVE: Can be replaced with wandb.log({fig_name: fig}) once
                # https://github.com/wandb/wandb/issues/3987 has been solved.
                wandb.log(
                    {fig_name: wandb.Image(figs["cost"][0])},
                )

        # Store reference error plots.
        if figs["reference_error"]:
            for idx, fig in enumerate(figs["reference_error"]):
                if not use_subplots:
                    fig_id = "{}-{}".format(
                        convert_to_snake_case(disturber_name),
                        dataframe["disturbance_label"].unique()[idx],
                    )
                else:
                    fig_id = "{}-subplots-fig_{}".format(
                        convert_to_snake_case(disturber_name), idx + 1
                    )
                fig_name = f"reference_error-{fig_id}"

                # Save figures to disk.
                if save_plots:
                    fig.savefig(
                        figs_path.joinpath(f"{fig_name}.{figs_extension}"),
                        bbox_inches="tight",
                    )

                # Save figures to Weights & Biases.
                if use_wandb:
                    # IMPROVE: Can be replaced with wandb.log({fig_name: fig}) once
                    # https://github.com/wandb/wandb/issues/3987 has been solved.
                    wandb.log(
                        {fig_name: wandb.Image(fig)},
                    )

        logger.log("Plots saved.", type="info")

    # Wait for user to close plots before continuing.
    plt.show()


if __name__ == "__main__":
    import argparse

    # Print available disturbers if '--list_disturbers'/'--list' argument is supplied.
    args = sys.argv[1:]
    if "--list_disturbers" in args or "--list" in args:
        print_available_disturbers()
        sys.exit()

    # Check if '--help', '-h' or 'help' is supplied after the disturber name.
    disturber_help = False
    if len(args) == 3:
        disturber_name = args[1]
        disturber_help = args[2] in ["--help", "-h", "help"]
        if disturber_help:
            # Remove '--help', '-h' or 'help' from args to prevent argparse error.
            sys.argv.pop(3)

    # Parse other arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str, help="The path where the policy is stored")
    parser.add_argument(
        "disturber",
        type=str,
        help=(
            "The disturber you want to use for the robustness evaluation (e.g. "
            "'ObservationRandomNoiseDisturber'). Can include an unloaded module "
            "in 'module:disturber_name' style."
        ),
    )
    parser.add_argument(
        "--list_disturbers",
        "--list",
        action="store_true",
        help="List available disturbers in the SLC package",
    )
    parser.add_argument(
        "--disturber_config",
        "--cfg",
        type=str,
        help=(
            "The configuration you want to pass to the disturber. It sets up the range "
            "of disturbances you wish to evaluate. Expects a dictionary that depends "
            "on the specified disturber (e.g. \"{'mean': [0.25, 0.25], 'std': [0.05, "
            "0.05]}\" for 'ObservationRandomNoiseDisturber' disturber)"
        ),
    )
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
        help="The episode length (default: environment 'max_episode_steps')",
    )
    parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=100,
        help=(
            "The number of episodes you want to run per disturbance variant "
            "(default: 100)"
        ),
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
        "--disable_baseline",
        default=None,
        action="store_true",
        help=(
            "Whether you want to disable the baseline (i.e. zero disturbance) from "
            "being added to the disturbance array automatically (default: False)"
        ),
    )
    parser.add_argument(
        "--observations",
        "--obs",
        default=None,
        nargs="*",
        help=(
            "The observations you want to show in the observations/references plot "
            "(default: only the first 6 observations). Keep empty to hide observations"
        ),
    )
    parser.add_argument(
        "--references",
        "--refs",
        default=None,
        nargs="*",
        help=(
            "The references you want to show in the observations/references plot "
            "(default: only the first reference). Keep empty to hide references"
        ),
    )
    parser.add_argument(
        "--reference_errors",
        "--ref_errs",
        default=None,
        nargs="*",
        help=(
            "The reference errors you want to show in the reference errors plot "
            "(default: only the first reference error). Keep empty to hide reference "
            "errors"
        ),
    )
    parser.add_argument(
        "--absolute_reference_errors",
        "--abs_ref_errs",
        action="store_true",
        help=(
            "Whether you want to use absolute reference errors instead of relative "
            "reference errors (default: False)"
        ),
    )
    parser.add_argument(
        "--merge_reference_errors",
        "--merge_ref_errs",
        action="store_true",
        help=(
            "Whether you want to merge the reference errors into one reference error "
            "(default: False)"
        ),
    )
    parser.add_argument(
        "--use_subplots",
        action="store_true",
        help=(
            "Whether you want to use subplots instead of separate figures "
            "(default: False)"
        ),
    )
    parser.add_argument(
        "--use_time",
        action="store_true",
        help=(
            "Whether you want to use time as the x-axis instead of steps "
            "(default: False)"
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
        help="The font scale you want to use for the plot text (default: 1.5)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="use Weights & Biases logging (default: False)",
    )
    parser.add_argument(
        "--wandb_job_type",
        type=str,
        default="eval",
        help="the Weights & Biases job type (default: eval)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stable-learning-control",
        help="the name of the wandb project (default: stable-learning-control)",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help=(
            "the name of the Weights & Biases group you want to assign the run to "
            "(defaults: None)"
        ),
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help=(
            "the name of the Weights & Biases run (defaults: None, will be "
            "automatically generated based on the policy directory and disturber"
        ),
    )
    args = parser.parse_args()

    # Load policy and environment.
    env, policy = load_policy_and_env(args.fpath, args.itr if args.itr >= 0 else "last")

    # Load the disturber or throw an error if it does not exist.
    disturber = load_disturber(args.disturber)

    # Return disturber help if requested.
    if disturber_help:
        print(f"\n\nShowing docstring for '{args.disturber}:\n'")
        print(
            "\t{}\n".format(inspect.cleandoc(disturber.__doc__).replace("\n", "\n\t"))
        )
        print(
            "\t{}\n\n".format(
                inspect.cleandoc(disturber.__init__.__doc__).replace("\n", "\n\t")
            )
        )
        sys.exit()

    # Retrieve disturber configuration.
    try:
        if args.disturber_config is not None:
            args.disturber_config = ast.literal_eval(args.disturber_config)
    except ValueError:
        raise ValueError(
            friendly_err(
                "The supplied '--disturber_config' is not a valid dictionary. Please "
                "make sure you supply a valid dictionary."
            )
        )

    # Set data directory to the fpath directory if not specified.
    if args.data_dir is None:
        args.data_dir = args.fpath

    # Create wandb run name if not specified.
    wandb_run_name = (
        (
            "{}_{}".format(
                PurePath(args.fpath).parts[-1],
                (
                    convert_to_snake_case(args.disturber)
                    if args.disturber is not None
                    else "baseline"
                ),
            )
            if args.wandb_run_name is None
            else args.wandb_run_name
        )
        if args.use_wandb
        else None
    )

    # Perform robustness evaluation and plot results.
    run_results_df = run_disturbed_policy(
        env,
        policy,
        disturber,
        disturber_config=args.disturber_config,
        include_baseline=(not args.disable_baseline),
        max_ep_len=args.len,
        num_episodes=args.episodes,
        render=args.render,
        deterministic=args.deterministic,
        save_result=args.save_result,
        output_dir=args.data_dir,
        use_wandb=args.use_wandb,
        wandb_job_type=args.wandb_job_type,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_run_name=wandb_run_name,
    )
    plot_robustness_results(
        run_results_df,
        observations=args.observations,
        references=args.references,
        reference_errors=args.reference_errors,
        absolute_reference_errors=args.absolute_reference_errors,
        merge_reference_errors=args.merge_reference_errors,
        use_subplots=args.use_subplots,
        use_time=args.use_time,
        save_plots=args.save_plots,
        font_scale=args.font_scale,
        figs_fmt=args.figs_fmt,
        output_dir=args.data_dir,
        use_wandb=args.use_wandb,
        wandb_job_type=args.wandb_job_type,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_run_name=wandb_run_name,
    )
