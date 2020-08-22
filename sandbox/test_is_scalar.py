import numpy as np
import torch

from machine_learning_control.control.utils.helpers import is_scalar

print(is_scalar(1), "True")
print(is_scalar([1]), "False")
print(is_scalar(np.array(2)), "False")  # Validate
print(is_scalar(np.array([2])), "True")
print(is_scalar(np.array([2, 2])), "False")
print(is_scalar(np.array(["2.2"])), "True")
print(is_scalar(np.array(["2,2"])), "False")
print(is_scalar(torch.Tensor((2))), "False")
print(is_scalar(torch.Tensor([2])), "True")
print(is_scalar(torch.Tensor([2, 2])), "False")
pass
