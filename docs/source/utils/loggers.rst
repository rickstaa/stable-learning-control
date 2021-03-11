.. _loggers:

=======
Loggers
=======

.. contents:: Table of Contents

Using a Logger
==============

BLC ships with basic logging tools, implemented in the classes
:class:`~bayesian_learning_control.utils.log_utils.logx.Logger`
and :class:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger`. The Logger
class contains most of the basic functionality for saving diagnostics,
hyperparameter configurations, the state of a training run, and the trained model. The EpochLogger
class adds a thin layer on top of that to make it easy to track the average, standard deviation, min,
and max value of a diagnostic over each epoch and across MPI workers.

.. admonition:: You Should Know

    All BLC algorithm implementations use an EpochLogger.


These loggers allow you to write these diagnostic to the ``stdout``, a ``csv/txt`` file and/or :tb:`Tensorboard <>`.

Examples
--------

First, let's look at a simple example of how an :class:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger`
keeps track of a diagnostic value:

>>> from bayesian_learning_control.control.utils.logx import EpochLogger
>>> epoch_logger = EpochLogger()
>>> for i in range(10):
        epoch_logger.store(Test=i)
>>> epoch_logger.log_tabular('Test', with_min_and_max=True)
>>> epoch_logger.dump_tabular()
-------------------------------------
|     AverageTest |             4.5 |
|         StdTest |            2.87 |
|         MaxTest |               9 |
|         MinTest |               0 |
-------------------------------------

The :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.store` method is used to save all
values of ``Test`` to the :class:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger`'s internal
state. Then, when :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.log_tabular` is called,
it computes the average, standard deviation, min, and max of ``Test`` over all of the values in the internal state.
The internal state is wiped clean after the call to :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.log_tabular`
(to prevent leakage into the statistics at the next epoch). Finally, :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.dump_tabular`
is called to write the diagnostics to file, Tensorboard and/or stdout.

Next, let's use the `Pytorch Classifier`_ tutorial to look at a full training procedure with the logger embedded, to highlight configuration and model
saving as well as diagnostic logging:

.. _`Pytorch Classifier`: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

.. code-block:: python
   :linenos:
   :emphasize-lines: 13, 52-53, 81-88, 96-98, 108, 138-141, 142, 148, 151-158, 160

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

    from bayesian_learning_control.utils.log_utils.logx import EpochLogger


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

In this example, observe that

* On line 52, the :class:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger` object is initiated. We set the ``std_out`` format to the "tabular" format.
* On line 53, :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.save_config` is used to save the hyperparameter configuration to a JSON file.
* On lines 81-88, 96-98, 138-141 and 160 we log information to the ``std_out``.
* On lines 108 :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.setup_pytorch_saver` is used to prepare the logger to save the key elements of the CNN model.
* On line 142, diagnostics are saved to the logger's internal state via :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.store`.
* On line 148, the CNN model saved once per epoch via :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.save_state`.
* On lines 61-66, :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.log_tabular` and :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.dump_tabular` are used to write the epoch diagnostics to file. Note that the keys passed into :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.log_tabular` are the same as the keys passed into :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.store`. We use the ``tb_write=True`` option to enable Tensorboard logging.

After this script has finished, the following files should be added to a ``/tmp/experiments/<TIMESTAMP>`` folder:

* ``events.out.tfevents.*``: The tensorboard event file containing the logged diagnostics.
* ``progress.csv``: Contains the logged diagnostics.
* ``config.json``: Contains the hyperparameter configuration.
* ``vars.pkl``: Contains some additional saved variables used for easy loading.

You can then use the ``tensorboard --logdir=/tmp/experiments`` command to see the diagnostics in tensorboard.

Logging and Tensorflow
----------------------

The preceding example was given in Pytorch. For Tensorflow, everything is the same except for L42-43:
instead of :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.setup_pytorch_saver`, you would
use :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.setup_tf_saver` and you would pass it
`a Tensorflow Module`_ (the algorithm you are training) as an argument.

The behavior of :meth:`~bayesian_learning_control.utils.log_utils.logx.EpochLogger.save_state` is the same as in the
PyTorch case: each time it is called,
it'll save the latest version of the Tensorflow module.

.. _`a Tensorflow module`: https://www.tensorflow.org/api_docs/python/tf/nn

Logging and MPI
---------------

.. admonition:: You Should Know

    Several RL algorithms are easily parallelized by using MPI to average gradients and/or other
    key quantities. The BLC loggers are designed to be well-behaved when using
    MPI: things will only get written to stdout, file or Tensorboard from the process with rank 0. But
    information from other processes isn't lost if you use the EpochLogger: everything which
    is passed into EpochLogger via ``store``, regardless of which process it's stored in, gets
    used to compute average/std/min/max values for a diagnostic.


Logger Classes
==============


.. autoclass:: bayesian_learning_control.utils.log_utils.logx.Logger
    :members:

    .. automethod:: bayesian_learning_control.utils.log_utils.logx.Logger.__init__

.. autoclass:: bayesian_learning_control.utils.log_utils.logx.EpochLogger
    :show-inheritance:
    :members:

Logger helper functions
=======================

.. autofunction:: bayesian_learning_control.utils.log_utils.setup_logger_kwargs