import torch
import gym

env = gym.make("Asteroids-ram-v0")

# Clamp does not work with arrays
a = torch.randn(4) ** 5
print(a)
min_tensor = torch.Tensor([-0.5, -0.2, 0.0, 0.5])
max_tensor = torch.Tensor([-0.5, -0.2, 0.0, 0.5])
clamped_version = torch.clamp(a, min=-0.5, max=0.5)
print(clamped_version)

# Torch max
clipped = torch.max(torch.min(a, max_tensor), min_tensor)

# Torch where
A = torch.rand(1, 10, 8, 8)
Means = torch.mean(torch.mean(A, dim=3), dim=2).unsqueeze(-1).unsqueeze(-1)
Ones = torch.ones(A.size())
Zeros = torch.zeros(A.size())
Thresholded = torch.where(A > Means, Ones, Zeros)
