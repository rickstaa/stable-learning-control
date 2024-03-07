# This script executes the lambda_lr_check_experiment for seed 234 for all environments
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/cartpole/lambda_lr_check/han2020_reproduction_lac_cartpole_cost_alpha3_tune_experiment_seed234_lambda_lr_check.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/comp_oscillator/lambda_lr_check/han2020_reproduction_lac_oscillator_complicated_alpha3_tune_experiment_seed234_lr_lambda_check.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/fetch_reach/lambda_lr_check/han2020_reproduction_lac_fetch_reach_alpha3_tune_experiment_seed234_lambda_lr_check.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/fetch_reach/infinite_horizon/lambda_lr_check/han2020_reproduction_lac_fetch_reach_alpha3_tune_infinite_horizon_experiment_seed234_lambda_lr_check.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/oscillator/lambda_lr_check/han2020_reproduction_lac_oscillator_alpha3_tune_experiment_seed234_lambda_lr_check.yml
