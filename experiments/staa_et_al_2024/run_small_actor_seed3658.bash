# This script executes the lambda_lr_check_experiment for seed 3658 for all environments
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/cartpole/small_actor/han2020_reproduction_lac_cartpole_cost_alpha3_tune_experiment_seed3658_small_actor.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/comp_oscillator/small_actor/han2020_reproduction_lac_oscillator_complicated_alpha3_tune_experiment_seed3658.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/fetch_reach/small_actor/han2020_reproduction_lac_fetch_reach_alpha3_tune_experiment_seed3658.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/fetch_reach/infinite_horizon/small_actor/han2020_reproduction_lac_fetch_reach_alpha3_tune_infinite_horizon_experiment_seed3658.yml
python -m stable_learning_control.run --exp_cfg experiments/staa_et_al_2024/oscillator/small_actor/han2020_reproduction_lac_oscillator_alpha3_tune_experiment_seed3658.yml
