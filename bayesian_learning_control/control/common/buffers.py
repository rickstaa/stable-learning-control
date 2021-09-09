"""This module contains several replay buffers that are used in multiple Pytorch and
Tensorflow algorithms.
"""

import numpy as np
from bayesian_learning_control.common.helpers import (
    atleast_2d,
    combine_shapes,
)
from bayesian_learning_control.control.common.helpers import discount_cumsum
from bayesian_learning_control.utils.log_utils import log_to_std_out


class ReplayBuffer:
    """A simple FIFO experience replay buffer.

    Attributes:
        obs_buf (numpy.ndarray): Buffer containing the current state.
        obs_next_buf (numpy.ndarray): Buffer containing the next state.
        act_buf (numpy.ndarray): Buffer containing the current action.
        rew_buf (numpy.ndarray): Buffer containing the current reward.
        done_buf (numpy.ndarray): Buffer containing information whether the episode was
            terminated after the action was taken.
        ptr (int): The current buffer index.
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
        self.ptr, self.size, self._max_size = 0, 0, int(size)

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
        self.ptr = (self.ptr + 1) % self._max_size
        self.size = min(self.size + 1, self._max_size)

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
        ptr (int): The current buffer index.
        traj_ptr (int): The start index of the current trajectory.
        traj_ptrs (list): The start indexes of each trajectory.
        n_traj (int): The number of trajectories currently stored in the buffer.
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
        gamma=0.99,
        lam=0.95,
    ):
        """Constructs all the necessary attributes for the trajectory buffer object.

        Args:
            obs_dim (tuple): The size of the observation space.
            act_dim (tuple): The size of the action space.
            rew_dim (tuple): The size of the reward space.
            size (int): The replay buffer size.
            preempt (bool, optional): Whether the buffer can be retrieved before it is
                full. Defaults to ``False``.
            min_trajectory_size (int, optional): The minimum trajectory length that can
                be stored in the buffer. Defaults to ``3``.
            incomplete (int, optional): Whether the buffer can store incomplete
                trajectories (i.e. trajectories which do not contain the final state).
                Defaults to ``False``.
            gamma (float, optional): The General Advantage Estimate (GAE) discount
                factor (Always between 0 and 1). Defaults to ``0.99``.
            lam (lam, optional): The GAE bias-variance trade-off factor (always between
                0 and 1). Defaults to ``0.95``.
        """
        if min_trajectory_size < 3:
            log_to_std_out(
                "A minimum trajectory length smaller than 3 steps is not recommended "
                "since it makes the buffer incompatible with Lyapunov based RL "
                "algorithms.",
                type="warning",
            )

        # Main buffers
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

        # Optional buffers
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        # Store buffer attributes
        self.ptr, self.traj_ptr, self.n_traj, self._max_size = 0, 0, 0, size
        self.traj_ptrs = []
        self.traj_lengths = []
        self._gamma, self._lam = gamma, lam
        self._contains_vals, self._contains_logp = False, False
        self._preempt = preempt
        self._min_traj_size, self._min_traj_size_warn = min_trajectory_size, False
        self._incomplete, self._incomplete_warn = incomplete, False

    def store(self, obs, act, rew, next_obs, done, val=None, logp=None):  # noqa: C901
        """Append one timestep of agent-environment interaction to the buffer.

        Args:
            obs (numpy.ndarray): Start state (observation).
            act (numpy.ndarray): Action.
            rew (:obj:`numpy.float64`): Reward.
            next_obs (numpy.ndarray): Next state (observation)
            done (bool): Boolean specifying whether the terminal state was reached.
            val (numpy.ndarray, optional): The (action) values. Defaults to ``None``.
            logp (numpy.ndarray, optional): The log probabilities of the actions.
                Defaults to ``None``.
        """
        assert self.ptr < self._max_size  # buffer has to have room so you can store

        # Fill primary buffer
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
                f"{e.args[0].capitalize()} please make sure your 'done' "
                "TrajectoryBuffer element is of dimension 1."
            )
            raise ValueError(error_msg)

        # Fill optional buffer
        if val:
            try:
                self.val_buf[self.ptr] = val
            except ValueError as e:
                error_msg = (
                    f"{e.args[0].capitalize()} please make sure your 'val' "
                    "TrajectoryBuffer element is of dimension 1."
                )
                raise ValueError(error_msg)
            self._contains_vals = True
        if logp:
            try:
                self.logp_buf[self.ptr] = logp
            except ValueError as e:
                error_msg = (
                    f"{e.args[0].capitalize()} please make sure your 'logp' "
                    "TrajectoryBuffer element is of dimension 1."
                )
                raise ValueError(error_msg)
            self._contains_logp = True

        # Increase buffer pointers
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Call this at the end of a trajectory or when one gets cut off by an epoch
        ends. This function increments the buffer pointers and calculates the advantage
        and rewards-to-go if it contains (action) values.

        .. note::
            When (action) values are stored in the buffer, this function looks back in
            the buffer to where the trajectory started and uses rewards and value
            estimates from the whole trajectory to compute advantage estimates with
            GAE-Lambda and compute the rewards-to-go for each state to use as the
            targets for the value function.

            The "last_val" argument should be 0 if the trajectory ended because the
            agent reached a terminal state (died), and otherwise should be V(s_T), the
            value function estimated for the last state. This allows us to bootstrap
            the reward-to-go calculation to account for timesteps beyond the arbitrary
            episode horizon (or epoch cutoff).
        """

        # Calculate the advantage and rewards-to-go if buffer contains vals
        if self._contains_vals:

            # Get the current trajectory
            path_slice = slice(self.traj_ptr, self.ptr)
            rews = np.append(self.rew_buf[path_slice], last_val)
            vals = np.append(self.val_buf[path_slice], last_val)

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self._gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = discount_cumsum(deltas, self._gamma * self._lam)

            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = discount_cumsum(rews, self._gamma)[:-1]

        # Store trajectory length and update trajectory pointers
        self.traj_lengths.append(self.ptr - self.traj_ptr)
        self.traj_ptrs.append(self.traj_ptr)
        self.traj_ptr = self.ptr
        self.n_traj += 1

    def get(self, flat=False):  # noqa: C901
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
            assert self.ptr == self._max_size

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
            if self._contains_vals:
                data["adv"] = self.adv_buf[buff_slice]
                data["ret"] = self.ret_buf[buff_slice]
            if self._contains_logp:
                data["lopg"] = self.logp_buf[buff_slice]
        else:
            data = dict(
                obs=np.split(self.obs_buf[buff_slice], self.traj_ptrs[1:]),
                next_obs=np.split(self.obs_next_buf[buff_slice], self.traj_ptrs[1:]),
                act=np.split(self.act_buf[buff_slice], self.traj_ptrs[1:]),
                rew=np.split(self.rew_buf[buff_slice], self.traj_ptrs[1:]),
                traj_sizes=self.traj_lengths,
            )
            if self._contains_vals:
                data["adv"] = np.split(self.adv_buf[buff_slice], self.traj_ptrs[1:])
                data["ret"] = np.split(self.ret_buf[buff_slice], self.traj_ptrs[1:])
            if self._contains_logp:
                data["lopg"] = np.split(self.logp_buf[buff_slice], self.traj_ptrs[1:])

        # Reset buffer and traj indexes
        self.ptr, self.traj_ptr, self.traj_ptrs, self.n_traj = 0, 0, [], 0
        self.traj_lengths = []

        # Return experience tuple
        return data
