"""Helper methods for managing TF2 MPI processes.

.. note::
    This module is not yet translated to TF2. It is not used by any of the current
    algorithms, but is kept here for future reference.
"""

# import numpy as np
# import tensorflow as tf

# from mpi4py import MPI
# from spinup.utils.mpi_tools import broadcast


def flat_concat(xs):  # noqa: DC102, D103
    raise NotImplementedError("The mpi_tf2 tools have not yet been implemented.")
    # NOTE: Old tf1 code
    # return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def assign_params_from_flat(x, params):  # noqa: DC102, D103
    raise NotImplementedError("The mpi_tf2 tools have not yet been implemented.")
    # NOTE: Old tf1 code
    # flat_size = lambda p: int(
    #     np.prod(p.shape.as_list())
    # )  # the 'int' is important for scalars
    # splits = tf.split(x, [flat_size(p) for p in params])
    # new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    # return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])


def sync_params(params):  # noqa: DC102, D103
    raise NotImplementedError("The mpi_tf2 tools have not yet been implemented.")
    # NOTE: Old tf1 code
    # get_params = flat_concat(params)
    # def _broadcast(x):
    #     broadcast(x)
    #     return x
    # synced_params = tf.py_func(_broadcast, [get_params], tf.float32)
    # return assign_params_from_flat(synced_params, params)


def sync_all_params():
    """Sync all tf variables across MPI processes."""
    raise NotImplementedError("The mpi_tf2 tools have not yet been implemented.")
    # NOTE: Old tf1 code
    # return sync_params(tf.global_variables())


# class MpiAdamOptimizer(tf.train.AdamOptimizer):
class MpiAdamOptimizer(object):
    """
    Adam optimizer that averages gradients across MPI processes.

    The compute_gradients method is taken from Baselines `MpiAdamOptimizer`_.
    For documentation on method arguments, see the TensorFlow docs page for
    the base :class:`~tf.keras.optimizers.AdamOptimizer`.

    .. _`MpiAdamOptimizer`: https://github.com/openai/baselines/tree/master/baselines/common/mpi_adam_optimizer.py
    """  # noqa: E501

    def __init__(self, **kwargs):  # noqa: D107, DC104
        raise NotImplementedError("The mpi_tf2 tools have not yet been implemented.")
        # NOTE: Old tf1 code
        # self.comm = MPI.COMM_WORLD
        # tf.train.AdamOptimizer.__init__(self, **kwargs)

    # NOTE: Old tf1 code
    # def compute_gradients(self, loss, var_list, **kwargs):
    #     """
    #     Same as normal compute_gradients, except average grads over processes.
    #     """
    #     grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
    #     grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
    #     flat_grad = flat_concat([g for g, v in grads_and_vars])
    #     shapes = [v.shape.as_list() for g, v in grads_and_vars]
    #     sizes = [int(np.prod(s)) for s in shapes]

    #     num_tasks = self.comm.Get_size()
    #     buf = np.zeros(flat_grad.shape, np.float32)

    #     def _collect_grads(flat_grad):
    #         self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
    #         np.divide(buf, float(num_tasks), out=buf)
    #         return buf

    #     avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
    #     avg_flat_grad.set_shape(flat_grad.shape)
    #     avg_grads = tf.split(avg_flat_grad, sizes, axis=0)

    #     avg_grads_and_vars = [
    #         (tf.reshape(g, v.shape), v)
    #         for g, (_, v) in zip(avg_grads, grads_and_vars)
    #     ]


#     return avg_grads_and_vars

# def apply_gradients(self, grads_and_vars, global_step=None, name=None):
#     """
#     Same as normal apply_gradients, except sync params after update.
#     """
#     opt = super().apply_gradients(grads_and_vars, global_step, name)
#     with tf.control_dependencies([opt]):
#         sync = sync_params([v for g, v in grads_and_vars])
#     return tf.group([opt, sync])
