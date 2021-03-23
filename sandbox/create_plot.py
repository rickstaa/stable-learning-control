import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str, help="The path where the policy is stored")
    args = parser.parse_args()

    # Retrieve dataframe
    robustness_eval_df = pd.read_csv(Path(args.fpath).absolute())

    # Retrieve observation and reference data from the dataframe
    o_disturbances_df = robustness_eval_df.query("variable == 'observation'").dropna(
        axis=1, how="all"
    )
    ref_disturbance_df = robustness_eval_df.query("variable == 'reference'").dropna(
        axis=1, how="all"
    )

    # Merge observations and references into one dataframe
    obs_df_tmp = o_disturbances_df.query("observation == 3")
    obs_df_tmp["signal"] = "obs_" + (obs_df_tmp["observation"] + 1).astype(str)
    obs_df_tmp.insert(len(obs_df_tmp.columns), "type", "observation")
    ref_df_tmp = ref_disturbance_df.query("reference == 0")
    ref_df_tmp["signal"] = "ref_" + (ref_df_tmp["reference"] + 1).astype(str)
    ref_df_tmp.insert(len(ref_df_tmp.columns), "type", "reference")
    obs_ref_df = pd.concat([obs_df_tmp, ref_df_tmp], ignore_index=True)

    # Plot observation 2 and reference 1 for different disturbance values
    fig = plt.figure(tight_layout=True)
    sns.lineplot(
        data=obs_ref_df,
        x="step",
        y="value",
        ci="sd",
        hue="disturbance",
        style="type",
    ).set_title("Observation 2 and reference 1 for several disturbance values")
    plt.show()
