from machine_learning_control.control.algos.pytorch.common import get_lr_scheduler
import torch
from torch.optim import Adam
import torch.nn as nn

lr_start = 1e-4
lr_final = 1e-10
N = 2

test_var = nn.Parameter(torch.tensor(1.0), requires_grad=True)

optimizer = Adam([test_var], lr=lr_start)
lr_scheduler = get_lr_scheduler(optimizer, "exponential", lr_start, lr_final, N)
lr = []

for ii in range(0, N - 1):
    lr_scheduler.step()
    lr.append(optimizer.param_groups[0]["lr"])
print("jan")


lr_scheduler_linear = get_lr_scheduler(optimizer, "linear", lr_start, lr_final, N)
lr_linear = []

for ii in range(0, N - 1):
    lr_scheduler_linear.step()
    lr_linear.append(optimizer.param_groups[0]["lr"])
print("jan")
