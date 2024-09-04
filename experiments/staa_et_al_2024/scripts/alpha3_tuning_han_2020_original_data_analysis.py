"""This script is used to perform the data analysis of the alpha3 tuning experiments of
my master thesis that were conducted using the original code of Han et al. (2020).
"""
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from stable_learning_control.utils.plot import get_all_datasets, plot_data

# Script parameters.
DATA_DIRS = [
    "data/han_2020_original/cartpole_cost",
    "data/han_2020_original/fetch_reach",
    "data/han_2020_original/oscillator",
    "data/han_2020_original/oscillator_complicated",
]  # List of data directories to analyze.
SAVE_PLOT = True
ALPHA3_VALUES = [0.1, 0.3, 1.0]
CONVERGENCE_THRESHOLD = 0.95  # Define convergence threshold for metric.
MOVING_AVERAGE_WINDOW = 5  # Set moving average window for metric.
EXTRA_SUFFIX = ""  # Add extra file suffix.

# Select the metric to analyse.
# METRIC = "Performance"
METRIC = "lambda"
# METRIC = "AverageLambda"
X_AXIS = "total_timesteps"


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
        if not Path(DATA_DIR).resolve().exists():
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

        # Visualize the data using the SCL plot utilities.
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
            plot_file_path = (
                f"{data_dir}/alpha3_tune_{metric_str}_{env_name}_plot"
                f"{file_suffix}{EXTRA_SUFFIX}.png"
            )
            print(f"Saving plot to {plot_file_path}.")
            plt.savefig(plot_file_path, bbox_inches="tight", dpi=300)

        plt.show()
    print("Analysis completed.")
