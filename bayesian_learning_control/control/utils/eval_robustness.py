"""A set of functions that can be used to evaluate the stability and robustness of an
algorithm. This is done by evaluating an algorithm's performance under two types of
disturbances: A disturbance that is applied during the environment step and a
perturbation added to the environmental parameters. For the functions in this
module to work work, these disturbances should be implemented as methods on the
environment. See the
`Robustness Evaluation Documentation <https://rickstaa.github.io/bayesian-learning-control/control/robustness_eval.html>`
on how this is done.
"""  # noqa: E501

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bayesian_learning_control.control.utils.test_policy import load_policy_and_env
from bayesian_learning_control.utils.log_utils import EpochLogger, log_to_std_out


def run_robustness_eval(
    env,
    policy,
    max_ep_len=None,
    num_episodes=100,
    render=True,
    deterministic=False,
    output_dir=None,
):
    """Evaluates a policy inside a given gym environment.

    Args:
        env (:obj:`gym.env`): The gym environment.
        policy (Union[tf.keras.Model, torch.nn.Module]): The policy.
        max_ep_len (int, optional): The maximum episode length. Defaults to None.
        num_episodes (int, optional): Number of episodes you want to perform in the
            environment. Defaults to 100.
        deterministic (bool, optional): Whether you want the action from the policy to
            be deterministic. Defaults to ``False``.
        render (bool, optional): Whether you want to render the episode to the screen.
            Defaults to ``True``.
        output_dir (str, optional): A directory for saving the diagnostics to. If
            ``None``, defaults to a temp directory of the form
            ``/tmp/experiments/somerandomnumber``.
    """
    assert env is not None, (
        "Environment not found!\n\n It looks like the environment wasn't saved, "
        + "and we can't run the agent in it. :( \n\n Check out the readthedocs "
        + "page on Experiment Outputs for how to handle this situation."
    )

    output_dir = (
        Path(output_dir).joinpath("eval") if output_dir is not None else output_dir
    )
    logger = EpochLogger(
        verbose_fmt="table", output_dir=output_dir, output_fname="eval_statistics.csv"
    )

    max_ep_len = env._max_episode_steps if max_ep_len is None else max_ep_len

    # Initialize the disturber
    # TODO: Add check if distruber is implemented for the environment.
    disturbance_type = "step_disturbance"
    disturbance_variant = "impulse"
    env.init_disturber(disturbance_type, disturbance_variant)

    # Loop though all disturbances till disturber is done
    render_error = False
    path = {
        "o": [],
        "r": [],
        "reference": [],
        "state_of_interest": [],
    }
    r_episodes_dfs = []
    soi_episodes_dfs = []
    r_disturbances_dfs = []
    soi_disturbances_dfs = []
    n_disturbance = 0
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
                    log_to_std_out(
                        (
                            "WARNING: Nothing was rendered since no render method was "
                            f"implemented for the '{env.unwrapped.spec.id}' environment."
                        ),
                        type="warning",
                    )

            # Retrieve action
            if deterministic and supports_deterministic:
                try:
                    a = policy.get_action(o, deterministic=deterministic)
                except TypeError:
                    supports_deterministic = False
                    log_to_std_out(
                        "Input argument 'deterministic' ignored as the algorithm does "
                        "not support deterministic actions. This is only supported for "
                        "gaussian  algorithms.",
                        type="warning",
                    )
                    a = policy.get_action(o)
            else:
                a = policy.get_action(o)

            # Perform action in the environment and store result
            if disturbance_type == "env_disturbance":
                o, r, d, info = env.step(a)
            else:
                o, r, d, info = env.disturbed_step(a)
            ep_ret += r
            ep_len += 1

            # Store path, cost, reference and state of interest
            # TODO: Fix for multiple states of interest
            path["o"].append(o)
            path["r"].append(r)
            path["reference"].append(info["reference"])
            path["state_of_interest"].append(info["state_of_interest"])

            # Store performance measurements
            if d or (ep_len == max_ep_len):
                died = ep_len < max_ep_len
                logger.store(EpRet=ep_ret, EpLen=ep_len, DeathRate=(float(died)))
                logger.log(
                    "Episode %d \t EpRet %.3f \t EpLen %d \t Died %s"
                    % (n, ep_ret, ep_len, died)
                )

                # Store episode information
                r_episode_df = pd.DataFrame(
                    {"reward": path["r"], "step": range(0, ep_len)}
                )
                r_episode_df.insert(len(r_episode_df.columns), "episode", n)
                r_episodes_dfs.append(r_episode_df)
                soi_episode_df = pd.DataFrame(
                    {
                        "error": path["state_of_interest"],
                        "step": range(0, ep_len),
                    }
                )
                soi_episode_df.insert(len(soi_episode_df.columns), "episode", n)
                soi_episodes_dfs.append(soi_episode_df)

                # Increment counters and reset storage variables
                n += 1
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                path = {
                    "o": [],
                    "r": [],
                    "reference": [],
                    "state_of_interest": [],
                }

        # Print diagnostics
        if "type" in env.disturbance_info.keys():
            logger.log_tabular("DisturbanceType", env.disturbance_info["type"])
        if "variant" in env.disturbance_info.keys():
            logger.log_tabular("DisturbanceVariant", env.disturbance_info["variant"])
        if (
            "variable" in env.disturbance_info.keys()
            and "value" in env.disturbance_info.keys()
        ):
            logger.log_tabular(
                env.disturbance_info["variable"], env.disturbance_info["value"]
            )
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("DeathRate")
        print("")
        logger.dump_tabular()

        # Store disturbance information
        disturbance_label = (
            env.disturbance_info["label"]
            if env.disturbance_info["label"]
            else "Disturbance: {}".format(str(n_disturbance + 1))
        )
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
        r_episodes_dfs = []
        soi_episodes_dfs = []

        # Go to next disturbance
        env.next_disturbance()
        n_disturbance += 1

    # Store robustness evaluation information
    # TODO: Add support for multiple soi
    r_disturbances_df = pd.concat(r_disturbances_dfs, ignore_index=True)
    soi_disturbances_df = pd.concat(soi_disturbances_dfs, ignore_index=True)

    # TODO: Store full database

    # Plot results
    sns.set(style="darkgrid", font_scale=1.5)  # TODO: Add as input arguments
    reward_plot = sns.lineplot(
        data=r_disturbances_df, x="step", y="reward", ci="sd", hue="disturbance"
    ).set_title("Mean costs")
    plt.show()
    soi_plot = sns.lineplot(
        data=soi_disturbances_df, x="step", y="error", ci="sd", hue="disturbance"
    ).set_title("State of interest")
    plt.show(aspect="auto")

    # Store figures
    # TODO: Enable option to set format and disable figures!
    figs_path = logger.output_dir.joinpath("figures")
    os.makedirs(figs_path, exist_ok=True)
    reward_plot.get_figure().savefig(
        logger.output_dir.joinpath("figures", "rewards.pdf"), bbox_inches="tight"
    )
    soi_plot.get_figure().savefig(
        logger.output_dir.joinpath("figures", "soi.pdf"), bbox_inches="tight"
    )

    # Store robustness evaluation dataframe
    # TODO: Make saving optional with input argument!
    r_disturbances_df.insert(len(r_disturbances_df.columns), "variable", "reward")
    soi_disturbances_df.insert(
        len(soi_disturbances_df.columns), "variable", "state_of_interest"
    )
    robustness_eval_df = pd.concat(
        [
            r_disturbances_df,
            soi_disturbances_df,
        ],
        ignore_index=True,
    )
    robustness_eval_df.to_csv(logger.output_dir.joinpath("results.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str)
    parser.add_argument("--len", "-l", type=int, default=800)
    # parser.add_argument("--len", "-l", type=int, default=0)
    # parser.add_argument("--episodes", "-n", type=int, default=100)
    parser.add_argument("--episodes", "-n", type=int, default=10)
    parser.add_argument("--norender", "-nr", action="store_true")
    parser.add_argument("--itr", "-i", type=int, default=-1)
    parser.add_argument("--deterministic", "-d", action="store_true")
    parser.add_argument("--disturbance_type", default=None)  # TODO: Add special flag
    args = parser.parse_args()
    env, policy = load_policy_and_env(args.fpath, args.itr if args.itr >= 0 else "last")
    run_robustness_eval(
        env, policy, args.len, args.episodes, not (args.norender), output_dir=args.fpath
    )  # TODO: Return results
    # plot_results() # Implement

# TODO: Change loging output!
