"""This script is used to perform the data analysis of the alpha3 tuning experiments of
my master thesis.
"""

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from stable_learning_control.utils.plot import get_all_datasets, plot_data

# Script parameters.
DATA_DIRS = [
    # "data/cartpole",
    # "data/han_reproduction/oscillator",
    # "data/comp_oscillator",
    # "data/fetch_reach_finite_horizon",
    # "data/fetch_reach_infinite_horizon",
    # "data/han_reproduction_extra_seeds/oscillator"
    # "data/han_reproduction_extra_seeds/comp_oscillator"
    # "data/han_reproduction_small_critic/comp_oscillator",
    # "data/han_reproduction_extra_long/comp_oscillator"
    "data/han_reproduction_extra_long/oscillator"
]  # List of data directories to analyze.
PLOT = True
SAVE_DATA = False
SAVE_STATISTICS = True
SAVE_PLOT = True
# ALPHA3_VALUES = None  # Specify alpha3 values to filter by, or use all if None.
# ALPHA3_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
ALPHA3_VALUES = [0.1, 0.3, 1.0]
LAST_N_EPOCHS = [
    10,
    10,
    10,
    10,
    10,
]  # Epoch count for average final metric value calculation per condition.
CONVERGENCE_THRESHOLD = 0.95  # Define convergence threshold for metric.
MOVING_AVERAGE_WINDOW = 5  # Set moving average window for metric.
# LAST_N_PLOT_EPOCHS = 5  # Epochs to plot from training end. None plots all.
LAST_N_PLOT_EPOCHS = None  # Epochs to plot from training end. None plots all.
EXTRA_SUFFIX = "_extra_long"  # Add extra file suffix.

# Select the metric to analyse.
METRIC = "Performance"
# METRIC = "AverageTestEpLen"
# METRIC = "AverageLambda"
X_AXIS = "TotalEnvInteracts"
# X_AXIS = "Epoch"


def create_legend_strings(data_folders):
    """Create legend strings for the alpha3 tuning plots.

    Args:
        data_folders (List[str]): List of data folders.

    Returns:
        List[str]: List of legend strings.
    """
    return [
        f'$\\alpha_{3}$={folder.split("_")[-1].replace("alp", "").replace("-", ".")}'
        for folder in data_folders
    ]


def filter_folders_by_alpha3(data_folders, alpha3_values):
    """Filter the data folders by alpha3 value.

    Args:
        data_folders (List[str]): List of data folders.
        alpha3_values (list[float]): Alpha value to filter by.

    Returns:
        List[str]: List of filtered data folders.
    """
    return [
        folder
        for folder in data_folders
        if any(
            f"alp{str(alpha).replace('.', '-')}" in folder for alpha in alpha3_values
        )
    ]


def calculate_condition_performance_statistics(
    df, metric="AverageTestEpRet", num_epochs=10
):
    """Calculate the mean, std and max of the metric for each condition.

    Args:
        df (pd.DataFrame): DataFrame to calculate the statistics from.
        metric (str, optional): The metric to calculate the statistics for. Defaults to
            "AverageTestEpRet".
        num_epochs (int, optional): Number of epochs to consider. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with the mean, std and max of the metric for each
            condition.
    """
    # Filter the data to include only the last N epochs for each seed.
    last_n_df = df.groupby(["Condition1", "Condition2"]).tail(num_epochs)

    # Calculate mean and std for each seed.
    seed_means_stds = (
        last_n_df.groupby(["Condition1", "Condition2"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Calculate mean, max and std across seeds for each condition.
    condition_means_stds = (
        seed_means_stds.groupby("Condition1")["mean"]
        .agg(["mean", "std", "max"])
        .reset_index()
    )
    condition_means_stds.columns = [
        "Condition",
        "Mean of Seed Means",
        "Std of Seed Means",
        "Max of Seed Means",
    ]
    return condition_means_stds


def calculate_condition_convergence_statistics(
    df,
    metric="Performance",
    moving_avg_window=1,
    convergence_threshold=0.95,
    num_epochs_for_average=10,
):
    """Calculate the mean and std of the convergence epoch for each condition.

    Args:
        df (pd.DataFrame): DataFrame to calculate the statistics from.
        metric (str, optional): The metric to calculate the statistics for. Defaults to
            "Performance".
        moving_avg_window (int, optional): Moving average window for the metric.
            Defaults to 1.
        convergence_threshold (float, optional): Convergence threshold for the metric.
            Defaults to 0.95.
        num_epochs_for_average (int, optional): Number of epochs to consider for the
            average metric value. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with the mean and std of the convergence epoch for each
            condition.
    """
    # Calculate the convergence epoch per seed.
    convergence_epochs = pd.DataFrame(
        index=pd.MultiIndex.from_arrays([[], []], names=["Condition1", "Condition2"])
    )
    for (condition, seed), group in df.groupby(["Condition1", "Condition2"]):
        mean = group[metric].rolling(window=moving_avg_window).mean()
        metric_baseline = mean.tail(num_epochs_for_average).mean()
        metric_convergence_threshold = (mean.max() - metric_baseline) * (
            1 - convergence_threshold
        ) + metric_baseline
        converged_epoch = None
        for epoch, metric_value in zip(group["Epoch"], mean):
            if metric_value <= metric_convergence_threshold:
                if converged_epoch is None:
                    converged_epoch = epoch
            else:
                converged_epoch = None
        # Add the convergence epoch to the DataFrame
        convergence_epochs.loc[(condition, seed), "Convergence Epoch"] = converged_epoch

    # Calculate the mean and std of convergence epochs across seeds for each condition.
    convergence_epoch = (
        convergence_epochs.groupby("Condition1")["Convergence Epoch"]
        .agg(["mean", "std"])
        .reset_index()
    )
    convergence_epoch.columns = [
        "Condition",
        "Convergence Epoch Mean",
        "Convergence Epoch Std",
    ]
    return convergence_epoch


if __name__ == "__main__":
    print(f"\nCreating '{METRIC}' plots...")
    for idx, DATA_DIR in enumerate(DATA_DIRS):
        parent_dir = Path(DATA_DIR).parent
        env_name = DATA_DIR.split("/")[-1]
        data_dir = f"{parent_dir}/plots/{env_name}"
        metric_str = METRIC.replace(" ", "_").lower()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        print("Creating {} plot...".format(env_name.replace("_", " ").title()))

        # Retrieve data directories and add legend column.
        if not Path(DATA_DIR).exists():
            raise FileNotFoundError(f"The data directory {DATA_DIR} does not exist.")
        all_data_folders = sorted(
            [str(f) for f in Path(DATA_DIR).iterdir() if f.is_dir()]
        )
        if ALPHA3_VALUES:
            data_folders = filter_folders_by_alpha3(all_data_folders, ALPHA3_VALUES)
        else:
            data_folders = all_data_folders
        file_suffix = "_filtered" if len(data_folders) < len(all_data_folders) else ""
        legend = create_legend_strings(data_folders)
        data = get_all_datasets(data_folders, legend=legend)

        # Store the data in csv file.
        # NOTE: Can be useful to inspect the data in a spreadsheet software.
        if SAVE_DATA:
            data_file_path = (
                f"{data_dir}/alpha3_tune_{metric_str}_{env_name}_plot_data"
                f"{file_suffix}{EXTRA_SUFFIX}.csv"
            )
            data_tmp = pd.concat(data, ignore_index=True)
            print(f"Saving plot input data to {data_file_path}.")
            data_tmp.to_csv(data_file_path, index=False)

        # Compute the statistics.
        data_concat = pd.concat(data, ignore_index=True)
        condition_performance_statistics = calculate_condition_performance_statistics(
            data_concat, metric=METRIC, num_epochs=LAST_N_EPOCHS[idx]
        )
        condition_convergence_statistics = calculate_condition_convergence_statistics(
            data_concat,
            metric=METRIC,
            moving_avg_window=MOVING_AVERAGE_WINDOW,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            num_epochs_for_average=LAST_N_EPOCHS[idx],
        )

        # Store the statistics in csv file.
        statistics = pd.concat(
            [condition_performance_statistics, condition_convergence_statistics], axis=1
        )
        statistics = statistics.loc[:, ~statistics.columns.duplicated()]
        statistics = statistics.rename(columns={"Condition": "Alpha3"})
        statistics["Alpha3"] = statistics["Alpha3"].str.replace(
            r"\$\\alpha_3\$=", "", regex=True
        )
        statistics_file_path = (
            f"{data_dir}/alpha3_tune_{metric_str}_{env_name}_statistics"
            f"{file_suffix}{EXTRA_SUFFIX}.csv"
        )
        if SAVE_STATISTICS:
            print(f"Saving statistics to {statistics_file_path}.")
            statistics.to_csv(statistics_file_path, index=False)

        # Only keep the last N epochs for the plot.
        if LAST_N_PLOT_EPOCHS:
            data = [
                df[df["Epoch"] >= df["Epoch"].max() - LAST_N_PLOT_EPOCHS] for df in data
            ]

        # Visualize the data using the SCL plot utilities.
        if PLOT:
            fig_name = env_name.replace("_", " ").title()
            fig = plt.figure(figsize=(10, 8), num=f"Alpha3 Tuning ({fig_name})")
            palette = sns.color_palette("tab20", n_colors=len(legend))
            plt.tight_layout()
            plot_data(
                data,
                xaxis=X_AXIS,
                value=METRIC,
                errorbar="ci",
                smooth=MOVING_AVERAGE_WINDOW,
                style="ticks",
                palette=palette,
            )
            plt.grid()

            # Save the plot as png file.
            if SAVE_PLOT:
                prefix = "partial_" if LAST_N_PLOT_EPOCHS is not None else ""
                plot_file_path = (
                    f"{data_dir}/alpha3_tune_{metric_str}_{env_name}_{prefix}plot"
                    f"{file_suffix}{EXTRA_SUFFIX}.png"
                )
                print(f"Saving plot to {plot_file_path}.")
                plt.savefig(plot_file_path, bbox_inches="tight", dpi=300)

        plt.show()
    print("Analysis completed.")
