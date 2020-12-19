"""Module used for managing MPI processes.

This module was cloned from the
`spinningup repository <https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_tools.py>`_.
"""

import os
import subprocess
import sys

import numpy as np
from mpi4py import MPI


def mpi_fork(n, bind_to_core=False):
    """Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.
        bind_to_core (bool, optional): Bind each MPI process to a core. Defaults to False.
    """
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", IN_MPI="1")
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=""):
    """Send message from one MPI process to the other.

    Args:
        m (string): Message you want to send.
        string (str, optional): Additional process description. Defaults to "".
    """
    print(("Message from %d: %s \t " % (MPI.COMM_WORLD.Get_rank(), string)) + str(m))


def pprint(input_str="", end="\n", comm=MPI.COMM_WORLD):
    """Print for MPI parallel programs: Only rank 0 prints *str*."""
    if comm.rank == 0:
        print(str(input_str) + end)


def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def allreduce(*args, **kwargs):
    """Reduced results of a operation across all processes.

    Args:
        *args: All args to pass to thunk.
        **kwargs: All kwargs to pass to thunk.

    Returns:
        object: Result object.
    """
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def num_procs():
    """Count active MPI processes.

    Returns:
        int: The number of mpi processes.
    """
    return MPI.COMM_WORLD.Get_size()


def broadcast(x, root=0):
    """Broadcast variable to other MPI processes.

    Args:
        x (object): Variable you want to broadcast.
        root (int, optional): Rank of the root process. Defaults to 0.
    """
    MPI.COMM_WORLD.Bcast(x, root=root)


def mpi_op(x, op):
    """Perform a MPI operation.

    Args:
        x (object): Python variable.
        op (mpi4py.MPI.Op): Operation type

    Returns:
        object: Reduced mpi operation result.
    """
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    """Take the sum of a scalar or vector over MPI processes.

    Args:
        x (object): Python variable.

    Returns:
        object: Reduced sum.
    """
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    """Average a scalar or vector over MPI processes.

    Args:
        x (object): Python variable.

    Returns:
        object: Reduced average.
    """
    return mpi_sum(x) / num_procs()


def mpi_statistics_scalar(x, with_min_and_max=False):
    """Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool, optional): If true, return min and max of x in
            addition to mean and std. Defaults to False.

    Returns:
        tuple: Reduced mean and standard deviation.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean) ** 2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
