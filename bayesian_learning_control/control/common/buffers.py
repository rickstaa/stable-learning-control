"""This module contains several replay buffers that are used in multiple Pytorch and
Tensorflow algorithms.
"""

import numpy as np
from bayesian_learning_control.common.helpers import atleast_2d, combine_shapes
from bayesian_learning_control.utils.log_utils import friendly_err, log_to_std_out


class ReplayBuffer:
    """A simple FIFO experience replay buffer.

    Attributes:
        obs_buf (numpy.ndarray): Buffer containing the current state.
        obs_next_buf (numpy.ndarray): Buffer containing the next state.
        act_buf (numpy.ndarray): Buffer containing the current action.
        rew_buf (numpy.ndarray): Buffer containing the current reward.
        done_buf (numpy.ndarray): Buffer containing information whether the episode was
            terminated after the action was taken.
    """

    def __init__(self, obs_dim, act_dim, rew_dim, size):
        """Constructs all the necessary attributes for the FIFO ReplayBuffer object.

        Args:
            obs_dim (tuple): The size of the observation space.
            act_dim (tuple): The size of the action space.
            rew_dim (tuple): The size of the reward space.
            size (int): The replay buffer size.
        """
        # Preallocate memory for experience buffer (s, s', a, r, d)
        # NOTE: Squeeze is needed to remove length 1 axis.
        self.obs_buf = atleast_2d(
            np.zeros(combine_shapes(int(size), obs_dim), dtype=np.float32).squeeze()
        )
        self.obs_next_buf = atleast_2d(
            np.zeros(combine_shapes(int(size), obs_dim), dtype=np.float32).squeeze()
        )
        self.act_buf = atleast_2d(
            np.zeros(combine_shapes(int(size), act_dim), dtype=np.float32).squeeze()
        )
        self.rew_buf = np.zeros(
            combine_shapes(int(size), rew_dim), dtype=np.float32
        ).squeeze()
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)

    def store(self, obs, act, rew, next_obs, done):
        """Add experience tuple to buffer.

        Args:
            obs (numpy.ndarray): Start state (observation).
            act (numpy.ndarray): Action.
            rew (:obj:`numpy.float64`): Reward.
            next_obs (numpy.ndarray): Next state (observation)
            done (bool): Boolean specifying whether the terminal state was reached.
        """
        try:
            self.obs_buf[self.ptr] = obs
            self.obs_next_buf[self.ptr] = next_obs
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure you set the "
                "ReplayBuffer 'obs_dim' equal to your environment 'observation_space'."
            )
            raise ValueError(error_msg)
        try:
            self.act_buf[self.ptr] = act
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure you set the "
                "ReplayBuffer 'act_dim' equal to your environment 'action_space'."
            )
            raise ValueError(error_msg)
        try:
            self.rew_buf[self.ptr] = rew
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure you set the "
                "ReplayBuffer 'rew_dim' equal to your environment 'reward_space'."
            )
            raise ValueError(error_msg)
        try:
            self.done_buf[self.ptr] = done
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure your 'done' ReplayBuffer"
                "element is of dimension 1."
            )
            raise ValueError(error_msg)

        # Increase pointers based on buffer size (first in first out)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        """Retrieve a batch of experiences from buffer.

        Args:
            batch_size (int, optional): The batch size. Defaults to ``32``.
        Returns:
            dict: A batch of experiences.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs_next=self.obs_next_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return batch


class TrajectoryBuffer:
    """A simple FIFO trajectory buffer.

    Attributes:
        obs_buf (numpy.ndarray): Buffer containing the current state.
        obs_next_buf (numpy.ndarray): Buffer containing the next state.
        act_buf (numpy.ndarray): Buffer containing the current action.
        rew_buf (numpy.ndarray): Buffer containing the current reward.
        done_buf (numpy.ndarray): Buffer containing information whether the episode was
            terminated after the action was taken.
        traj_lengths (list): List with the lengths of each trajectory in the
            buffer.
        """

    def __init__(
        self,
        obs_dim,
        act_dim,
        rew_dim,
        size,
        preempt=False,
        min_trajectory_size=3,
        incomplete=False,
    ):
        """Constructs all the necessary attributes for the trajectory buffer object.

        Args:
            obs_dim (tuple): The size of the observation space.
            act_dim (tuple): The size of the action space.
            rew_dim (tuple): The size of the reward space.
            size (int): The replay buffer size.
            preempt (bool): Whether the buffer can be retrieved before it is full.
            min_trajectory_size (int): The minimum trajectory length that can be stored
                in the buffer.
            incomplete (int): Whether the buffer can store incomplete trajectories
                (i.e. trajectories which do not contain the final state).
        """
        if min_trajectory_size < 3:
            log_to_std_out(
                "A minimum trajectory length smaller than 3 steps is not recommended "
                "since it makes the buffer incompatible with Lyapunov based RL "
                "algorithms.",
                type="warning",
            )

        self.obs_buf = atleast_2d(
            np.zeros(combine_shapes(size, obs_dim), dtype=np.float32).squeeze()
        )
        self.obs_next_buf = atleast_2d(
            np.zeros(combine_shapes(size, obs_dim), dtype=np.float32).squeeze()
        )
        self.act_buf = atleast_2d(
            np.zeros(combine_shapes(size, act_dim), dtype=np.float32).squeeze()
        )
        self.rew_buf = np.zeros(
            combine_shapes(int(size), rew_dim), dtype=np.float32
        ).squeeze()
        self.done_buf = np.zeros(int(size), dtype=np.float32)

        # Store buffer attributes
        self.ptr, self.traj_ptr, self.n_traj, self.max_size = 0, 0, 0, size
        self.traj_ptrs = []
        self.traj_lengths = []
        self._preempt = preempt
        self._min_traj_size, self._min_traj_size_warn = min_trajectory_size, False
        self._incomplete, self._incomplete_warn = incomplete, False

    def store(self, obs, act, rew, next_obs, done):
        """Append one timestep of agent-environment interaction to the buffer.

        Args:
            obs (numpy.ndarray): Start state (observation).
            act (numpy.ndarray): Action.
            rew (:obj:`numpy.float64`): Reward.
            next_obs (numpy.ndarray): Next state (observation)
            done (bool): Boolean specifying whether the terminal state was reached.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store

        try:
            self.obs_buf[self.ptr] = obs
            self.obs_next_buf[self.ptr] = next_obs
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure you set the "
                "TrajectoryBuffer 'obs_dim' equal to your environment "
                "'observation_space'."
            )
            raise ValueError(error_msg)
        try:
            self.act_buf[self.ptr] = act
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure you set the "
                "TrajectoryBuffer 'act_dim' equal to your environment 'action_space'."
            )
            raise ValueError(error_msg)
        try:
            self.rew_buf[self.ptr] = rew
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure you set the "
                "TrajectoryBuffer 'rew_dim' equal to your environment 'reward_space'."
            )
            raise ValueError(error_msg)
        try:
            self.done_buf[self.ptr] = done
        except ValueError as e:
            error_msg = (
                f"{e.args[0].capitalize()} please make sure your 'done' ReplayBuffer"
                "element is of dimension 1."
            )
            raise ValueError(error_msg)

        # Increase buffer pointers
        self.ptr += 1

    def finish_path(self):
        """Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending.
        """

        # Store trajectory length and update trajectory pointers
        self.traj_lengths.append(self.ptr - self.traj_ptr)
        self.traj_ptrs.append(self.traj_ptr)
        self.traj_ptr = self.ptr
        self.n_traj += 1

    def get(self, flat=False):
        """Retrieve the trajectory buffer.

        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.

        Args:
            flat (bool, optional): Retrieve a flat buffer (i.e. the trajectories are
                concatenated). Defaults to ``False``.

        .. note:
            If you set flat to `True` all the trajectories will be concatenated into one
            array. You can use the :attr:`~TrajectoryBuffer.traj_lengths` or
            :attr:`traj_ptrs` attributes to split this array into distinct
            trajectories.

        Returns:
            dict: The trajectory buffer.
        """
        if not self._preempt:  # Check if buffer was full
            assert self.ptr == self.max_size

        # Remove incomplete trajectories
        if not self._incomplete and self.traj_ptr != self.ptr:
            if not self._incomplete_warn:
                log_to_std_out(
                    "Incomplete trajectories have been removed from the buffer (i.e. "
                    "Trajectories that did not reach the final state).",
                    type="warning",
                )
                self._incomplete_warn = True
            buffer_end_ptr = self.traj_ptr
        else:
            buffer_end_ptr = self.ptr

        # Remove trajectories that are to short
        if self.traj_lengths[-1] < self._min_traj_size:
            if not self._min_traj_size_warn:
                log_to_std_out(
                    (
                        "Trajectories shorter than {self._min_traj_size} have been "
                        "removed from the buffer."
                    ),
                    type="warning",
                )
                self._min_traj_size_warn = True
            buffer_end_ptr = self.traj_ptr - self.traj_lengths[-1]
            self.traj_lengths = self.traj_lengths[:-1]

        # Create trajectory buffer dictionary
        buff_slice = slice(0, buffer_end_ptr)
        if flat:
            data = dict(
                obs=self.obs_buf[buff_slice],
                next_obs=self.obs_next_buf[buff_slice],
                act=self.act_buf[buff_slice],
                rew=self.rew_buf[buff_slice],
                traj_sizes=self.traj_lengths,
            )
        else:
            data = dict(
                obs=np.split(self.obs_buf[buff_slice], self.traj_ptrs[1:]),
                next_obs=np.split(self.obs_next_buf[buff_slice], self.traj_ptrs[1:]),
                act=np.split(self.act_buf[buff_slice], self.traj_ptrs[1:]),
                rew=np.split(self.rew_buf[buff_slice], self.traj_ptrs[1:]),
                traj_sizes=self.traj_lengths,
            )

        # Reset buffer and traj indexes
        self.ptr, self.traj_ptr, self.traj_ptrs, self.n_traj = 0, 0, [], 0
        self.traj_lengths = []

        # Return experience tuple
        return data


def _split_trajectories():
    print("test")
