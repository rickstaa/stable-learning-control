import multiprocessing
import os

import numpy as np
import torch
from machine_learning_control.control.utils.mpi_tools import (
    broadcast,
    mpi_avg,
    num_procs,
    proc_id,
)
from mpi4py import MPI


def setup_pytorch_for_mpi():
    """Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    # print(
    #     "Proc %d: Reporting original number of Torch threads as %d."
    #     % (proc_id(), torch.get_num_threads()),
    #     flush=True,
    # )
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    # print(
    #     "Proc %d: Reporting new number of Torch threads as %d."
    #     % (proc_id(), torch.get_num_threads()),
    #     flush=True,
    # )


def mpi_avg_grads(module):
    """Average contents of gradient buffers across MPI processes.

    Args:
        module (object): Python object for which you want to average the gradients.
    """
    if num_procs() == 1:
        return

    # Sync torch module parameters
    if hasattr(module, "parameters"):
        for p in module.parameters():

            # Sync network grads
            p_grad_numpy = p.grad.numpy()
            avg_p_grad = mpi_avg(p.grad)
            p_grad_numpy[:] = avg_p_grad[:]
    elif isinstance(module, torch.Tensor):

        # Sync network grads
        p_grad_numpy = module.grad.numpy()
        avg_p_grad = mpi_avg(module.grad)
        if isinstance(avg_p_grad, list):
            p_grad_numpy[:] = avg_p_grad[:]
        else:
            p_grad_numpy = avg_p_grad
    else:
        raise TypeError(
            (
                "The gradients of parameter with type {} could not be synced accord "
                "the MPI processes as objects of this type are not yet supported. "
            ).format(type(module))
        )


def sync_params(module):
    """Sync all parameters of module across all MPI processes.

    Args:
        module (object): Python object for which you want to average the gradients.
    """
    if num_procs() == 1:
        return

    # Sync torch module parameters
    if hasattr(module, "parameters"):

        # Sync network parameters
        for p in module.parameters():
            p_numpy = p.data.numpy()
            broadcast(p_numpy)
    elif isinstance(module, torch.Tensor):

        # Sync pytorch parameter
        p_numpy = module.data.numpy()
        broadcast(p_numpy)
        return
    elif isinstance(module, np.ndarray):

        # Sync numpy parameters
        broadcast(module)
    else:
        raise TypeError(
            (
                "Parameter of type {} could not be synced accord the MPI processes "
                "as objects of this type are not yet supported. "
            ).format(type(module))
        )
