conda activate machine_learning_control
python "/home/ricks/Development/machine_learning_control_ws/src/machine_learning_control/control/algos/sac/sac.py" --env="Walker2d-v2" --lra="3e-4" --lrc="3e-4" --gamma="0.99" --batch-size="256" --replay-size="1000000" --l="2" --hid="256"
