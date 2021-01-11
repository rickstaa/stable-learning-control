conda activate machine_learning_control
python "./machine_learning_control/control/algos/lac/lac.py" --env="Oscillator-v1" --lr_a="1e-4" --lr_c="3e-4" --gamma="0.995" --batch-size="256" --replay-size="1000000" --l_a="2" --l_c="2" --hid_c="256" --hid_a="256"
