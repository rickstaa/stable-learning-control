"""Run training script. This script servers as a wrapper for the original run module.
"""
import argparse
import os.path as osp
import sys

import ruamel.yaml as yaml
from machine_learning_control.run import run
from machine_learning_control.control.utils.eval_utils import flatten
from machine_learning_control.control.utils.logx import colorize

# Retrieve exp config path folder
EXP_CFG_PATH = osp.join(osp.abspath(osp.dirname(__file__)), "../../cfg/exp_cfgs")

# FIXME: Add configuration folder

if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(description="Train a agent.")
    parser.add_argument("--exp_cfg", type=str, default="")
    # IMPROVE: Pass other arguments
    args = parser.parse_args()

    # Load experiment config if supplied
    cmd = sys.argv
    if args.exp_cfg:

        # Remove cfg arg
        try:
            cmd.remove("--exp_cfg")
            cmd.remove(args.exp_cfg)
        except ValueError:
            cmd.remove("--exp_cfg=" + str(args.exp_cfg))

        # Append yaml to file_name if not yet present
        exp_cfg_name = (
            args.exp_cfg if args.exp_cfg.endswith(".yml") else args.exp_cfg + ".yml"
        )

        # Create exp config path
        if not osp.exists(exp_cfg_name):
            project_path = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
            exp_cfg_path = osp.join(project_path, "cfg/exp_cfgs", exp_cfg_name)
        else:
            exp_cfg_path = exp_cfg_name
        with open(exp_cfg_path) as stream:
            try:
                # Retrieve run arguments from the config file
                print(
                    colorize(
                        (
                            "\nInfo: Loading experimental parameters from "
                            f"`{exp_cfg_path}`."
                        ),
                        "cyan",
                        bold=True,
                    )
                )
                exp_cfg = yaml.safe_load(stream)
                try:
                    exp_cfg = exp_cfg["train"]
                except KeyError:
                    print(
                        colorize(
                            (
                                "WARN: No `train` field was found in your "
                                "experiment config. Please add your "
                                "hyperparameters to a field named `train` and try "
                                "again."
                            ),
                            "yellow",
                            bold=True,
                        )
                    )
                    sys.exit(0)
                if not exp_cfg:
                    print(
                        colorize(
                            (
                                "WARN: No hyperparameters were found in the "
                                "`train` field of your experiment config. Using cmd"
                                "line arguments instead."
                            ),
                            "yellow",
                            bold=True,
                        )
                    )
                else:
                    if "alg_name" in exp_cfg.keys():
                        cmd.insert(1, exp_cfg["alg_name"])
                        exp_cfg.pop("alg_name", None)
                    exp_cfg = {
                        (key if key.startswith("--") else "--" + key): val
                        for key, val in exp_cfg.items()
                    }
                    exp_cfg = list(
                        flatten([[key, val] for key, val in exp_cfg.items()])
                    )  # NOTE (rickstaa): Doesn't work with nested dictionaries
                    exp_cfg = [str(item) for item in exp_cfg]
                    cmd.extend(exp_cfg)
            except yaml.YAMLError:
                print(
                    colorize(
                        (
                            "\nWarning: No experiment config was found in "
                            f"`{exp_cfg_path}`. As a result the --exp-config "
                            "argument has been ignored."
                        ),
                        "yellow",
                        bold=True,
                    )
                )

    # Run the experiments using the Elpg run module.
    run(cmd)
