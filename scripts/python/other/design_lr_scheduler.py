"""Script that can be used to design your learning rate decay scheduler.

It allows you to see the learning rate decay of multiple learning rate decay strategies
so that you can decide which strategy you want to use with the `Pytorch`_ and
`Tensorflow`Learning rate schedulers.

.. _`Pytorch`_: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
.. _`Tensorflow`_: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
"""  # noqa: E501

import decimal

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim

from machine_learning_control.control.utils.helpers import (
    calc_gamma_lr_decay,
    calc_linear_lr_decay,
)

# Script settings
epochs = 100
dummy_var = torch.tensor(np.log(1.0), requires_grad=True)
lr_init = 3e-4

# Chose scheduler type you want to see
# NOTE: 1: LambdaLR scheduler, 2: MultiplicativeLR scheduler ,3: Exponential scheduler
scheduler_type = 1

# Setup optimizer optimizer
optimizer = optim.Adam([dummy_var], lr=lr_init)

#################################################
# Choose and tune your scheduler  ###############
#################################################

if __name__ == "__main__":

    #########################################
    # LambdaLR scheduler ####################
    #########################################
    # NOTE: Sets the learning rate of each parameter group to the initial lr times a
    # given function.
    if scheduler_type == 1:
        print("Show results for LambdaLR scheduler.")

        # Set l_final
        lr_final = 1e-10

        # Create lambda learning rate decay function
        def lambda_function(epoch):
            """Returns the current value of the factor by which the learning rate is
            multiplied when scheduler.step is called.

            Args:
                epoch (int): The current epoch.
            """
            # lr_ff = 1.0 - (epoch - 1.0) / epochs  # Linear decaying learning rate
            lr_f = decimal.Decimal(1.0) - (
                calc_linear_lr_decay(lr_init, lr_final, epochs) * decimal.Decimal(epoch)
            )  # Bounded linearly decaying learning rate
            # lr_f = epoch // 30
            # lr_f = 0.95 ** epoch
            return np.longdouble(lr_f)

        # Plot lambda learning rate decay function
        lr_factor_array = []
        for epoch in range(epochs):
            lr_factor_array.append(lambda_function(epoch))
        print("Plotting learning rate decay function.")
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(
            range(0, epochs), lr_factor_array, color="green", marker=".", linestyle="--"
        )
        plt.title("Learning rate decay function")
        plt.xlabel("Epoch")
        plt.ylabel("Decay value")

        # Plot learning rate decay function against initial lr
        lr_factor_array = []
        for epoch in range(epochs):
            lr_factor_array.append(lr_init * np.longdouble(lambda_function(epoch)))
        print("Plotting learning rate decay function * init learning rate.")
        plt.subplot(1, 3, 2)
        plt.plot(
            range(0, epochs), lr_factor_array, color="green", marker=".", linestyle="--"
        )
        plt.axhline(y=lr_final, color="green")
        plt.text(
            0, lr_final + 0.01 * (lr_init - lr_final), f"{lr_final:.2E}", rotation=0
        )
        plt.title("Learning rate decay function * initial lr")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")

        # Create MultiplicativeLR scheduler
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_function)

        # Simulate a training loop
        lr_array = []
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]["lr"]
            # lr = scheduler.get_lr()[0]
            print(f"Epoch {epoch}, lr {lr:.2E}")
            lr_array.append(lr)
            loss = dummy_var
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Plot Learning rate
        print("Simulate learning rate.")
        plt.subplot(1, 3, 3)
        plt.plot(range(0, epochs), lr_array, color="green", marker=".", linestyle="--")
        plt.axhline(y=lr_final, color="green")
        plt.text(
            0, lr_final + 0.01 * (lr_init - lr_final), f"{lr_final:.2E}", rotation=0
        )
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("Learning rate (LabmdaLR decay)")
        plt.show()

    #########################################
    # MultiplicativeLR ######################
    #########################################
    # NOTE: Multiply the learning rate decay function with the current learning rate.
    if scheduler_type == 2:
        print("Show results for MultiplicativeLR scheduler")

        # Set scheduler settings
        lr_final = 1e-10  # Final learning rate

        # Create learning rate factor multiplication function
        def lr_factor(epoch):
            """Returns the current value of the factor by which the learning rate is
            multiplied when scheduler.step is called.

            Args:
                epoch (int): The current epoch.
            """
            lr_ff = 1.0 - (epoch - 1.0) / epochs
            # lr_ff = epoch // 30
            # lr_ff = 0.95 ** epoch
            return lr_ff

        # Plot learning rate factor
        lr_factor_array = []
        lr = []
        for epoch in range(epochs):
            lr_factor_array.append(lr_factor(epoch))
        print("Plotting learning rate decay factor function.")
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(
            range(0, epochs),
            lr_factor_array,
            color="blue",
            marker=".",
            linestyle="solid",
        )
        plt.title("Learning rate decay multiplication factor.")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate decay factor")

        # Plot learning rate decay function against initial lr
        lr_factor_array = []
        lr = lr_init
        for epoch in range(epochs):
            lr *= lr_factor(epoch)
            lr_factor_array.append(lr)
        print("Plotting learning rate decay function * current learning rate.")
        plt.subplot(1, 3, 2)
        plt.plot(
            range(0, epochs), lr_factor_array, color="blue", marker=".", linestyle="--"
        )
        plt.title("Learning rate decay function * current lr")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")

        # Create MultiplicativeLR scheduler
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_factor)

        # Simulate a training loop
        lr_array = []
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]["lr"]
            # lr = scheduler.get_lr()[0]
            print(f"Epoch {epoch}, lr {lr:.2E}")
            lr_array.append(lr)
            loss = dummy_var
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Plot Learning rate
        print("Simulate learning rate.")
        plt.subplot(1, 3, 3)
        plt.plot(
            range(0, epochs), lr_array, color="blue", marker=".", linestyle="solid"
        )
        plt.title("Learning rate (Multiplicative decay)")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.show()

    #########################################
    # ExponentialLR #########################
    #########################################
    # NOTE: Decays the learning rate of each parameter group by gamma every epoch. For a
    # visualisation see https://www.desmos.com/calculator/3fisjexbvp.
    #
    # Function:
    #   lr(epoch) = lr_init * gamma**(epoch)
    #       gamma: Exponential decay factor.
    if scheduler_type == 3:
        print("Show results for ExponentialLR scheduler")

        # Set scheduler settings
        lr_final = 1e-10  # Final learning rate
        gamma = np.longdouble(
            calc_gamma_lr_decay(lr_init, lr_final, epochs)
        )  # The decay exponent
        print("gamma: {gamma}")

        # Create ExponentialLR scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

        # Plot learning rate decay function against initial lr
        lr_factor_array = []
        lr = lr_init
        for epoch in range(epochs):
            lr = lr_init * gamma ** epoch
            lr_factor_array.append(lr)
        print("Plotting learning rate decay function * current learning rate.")
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(
            range(0, epochs), lr_factor_array, color="blue", marker=".", linestyle="--"
        )
        plt.axhline(y=lr_final)
        plt.text(
            0, lr_final + 0.01 * (lr_init - lr_final), f"{lr_final:.2E}", rotation=0
        )
        plt.title(f"Learning rate decay function (gamma={gamma}) * init_lr")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")

        # Simulate a training loop
        lr_array = []
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]["lr"]
            # lr = scheduler.get_lr()[0]
            print(f"Epoch {epoch}, lr {lr:.2E}")
            lr_array.append(lr)
            loss = dummy_var
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Plot Learning rate
        print("Simulate learning rate.")
        plt.subplot(1, 2, 2)
        plt.plot(range(0, epochs), lr_array, color="red", marker=".", linestyle="solid")
        plt.axhline(y=lr_final, color="red")
        plt.text(
            0, lr_final + 0.01 * (lr_init - lr_final), f"{lr_final:.2E}", rotation=0
        )
        plt.title(f"Learning rate (Exponential decay) (gamma={gamma})")
        plt.xlabel("Epoch")
        plt.ylabel("Lr")
        plt.show()
