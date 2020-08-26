import torch
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

dataset = datasets.FakeData(size=200, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

max_t = torch.tensor(100)
model = nn.Linear(3 * 224 * 224, 10)
optimizer = optim.SGD(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.MultiplicativeLR(
    optimizer, lr_lambda=lambda epoch: 1.0 - (epoch - 1.0) / 100
)
criterion = nn.NLLLoss()

# print(optimizer.param_groups[0]["lr"])
for batch_idx, (data, target) in enumerate(loader):
    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {batch_idx}, lr {lr}")
    optimizer.zero_grad()
    output = model(data.view(10, -1))
    loss = criterion(output, target.long())
    loss.backward()
    optimizer.step()
    scheduler.step()
