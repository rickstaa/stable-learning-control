import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from machine_learning_control.control.utils.log_utils.logx import EpochLogger


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Simple script for training an CNN on CIFAR10.
def train_cifar10(
    epochs=2,
    batch_size=4,
    lr=1e-3,
    logger_kwargs=dict(),
    save_freq=1,
):
    # Setup logger and save hyperparameters
    logger = EpochLogger(**logger_kwargs, verbose_fmt="tab")
    logger.save_config(locals())

    # Load and preprocess CIFAR10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # print information about the dataset
    total_samples = len(trainset)
    n_iterations = math.ceil(total_samples / batch_size)
    logger.log(
        "We perform {} epochs on our dataset that contains {} samples ".format(
            epochs,
            total_samples,
        )
        + "and each epoch has {} iterations.".format(n_iterations),
        type="info",
    )

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    logger.log(
        "labels:" + " ".join("%5s" % classes[labels[j]] for j in range(4)), type="info"
    )  # print labels

    # Define a Convolutional Neural Network
    net = Net()

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # Setup model saving
    logger.setup_pytorch_saver(net)

    # Run main training loop
    start_time = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            logger.store(Loss=loss)
            loss.backward()
            optimizer.step()

            # calculate accuracy and increment running loss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).float().sum()
            accuracy = 100 * correct / len(trainset)
            running_loss += loss.item()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                logger.log(
                    "[%d, %5d] loss: %.3f, acc: %.3f"
                    % (epoch + 1, i + 1, running_loss / 2000, accuracy)
                )
                logger.store(Loss=loss, Acc=accuracy)
                running_loss = 0.0
                correct = 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state(state_dict=dict(), itr=None)

        # Log info about epoch
        logger.log_tabular("Epoch", epoch, tb_write=True)
        logger.log_tabular("Acc", with_min_and_max=True, tb_write=True)
        logger.log_tabular("Loss", average_only=True, tb_write=True)
        logger.log_tabular(
            "TotalGradientSteps", (epoch + 1) * total_samples, tb_write=True
        )
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()

    logger.log("Finished Training")


if __name__ == "__main__":
    train_cifar10()
