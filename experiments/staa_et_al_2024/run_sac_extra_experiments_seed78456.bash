# This script executes the lambda_lr_check_experiment for seed 78456 for all environments
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/fetch_reach/sac_extra/bigger_initial_alpha/han2020_reproduction_sac_fetch_reach_alpha3_tune_experiment_seed78456_bigger_initial_alpha.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/fetch_reach/sac_extra/different_steps_per_update/han2020_reproduction_sac_fetch_reach_alpha3_tune_experiment_seed78456_different_steps_per_update.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/fetch_reach/sac_extra/lac_critic/han2020_reproduction_sac_fetch_reach_alpha3_tune_experiment_seed78456_lac_critic.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/fetch_reach/sac_extra/sac_extra_all/han2020_reproduction_sac_fetch_reach_alpha3_tune_experiment_seed78456_sac_extra_all.yml
