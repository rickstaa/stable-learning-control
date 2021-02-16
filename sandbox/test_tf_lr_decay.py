from machine_learning_control.control.algos.tf2.common import get_lr_scheduler

lr_start = 1.0
lr_final = 0.5
N = 100

lr_scheduler = get_lr_scheduler("exponential", lr_start, lr_final, N)
lr = []

for ii in range(0, N):
    lr.append(lr_scheduler(ii))
print("jan")


lr_scheduler_linear = get_lr_scheduler("linear", lr_start, lr_final, N)
lr_linear = []

for ii in range(0, N):
    lr_linear.append(lr_scheduler_linear(ii))
print("jan")
