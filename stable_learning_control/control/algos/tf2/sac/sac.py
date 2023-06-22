"""Soft Actor-Critic algorithm

This module contains the Tensorflow 2.x implementation of the SAC algorithm of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.

.. note::
    Code Conventions:
        - We use a `_` suffix to distinguish the next state from the current state.
        - We use a `targ` suffix to distinguish actions/values coming from the target network.

.. rubric:: Class

.. autoclass:: SAC
   :members:

.. rubric:: Function

.. autofunction:: sac
"""  # NOTE: Manual autofunction/class request was added because of bug https://github.com/sphinx-doc/sphinx/issues/7912#issuecomment-786011464  # noqa:E501

import argparse
import os
import os.path as osp
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_learning_control.common.helpers import combine_shapes
from stable_learning_control.control.algos.tf2.common.helpers import (
    count_vars,
    set_device,
)
from stable_learning_control.control.algos.tf2.policies.soft_actor_critic import (
    SoftActorCritic,
)
from stable_learning_control.control.common.buffers import ReplayBuffer
from stable_learning_control.control.common.helpers import heuristic_target_entropy
from stable_learning_control.control.utils.eval_utils import test_agent
from stable_learning_control.control.utils.gym_utils import (
    is_discrete_space,
    is_gym_env,
)
from stable_learning_control.control.utils.safer_eval import safer_eval
from stable_learning_control.utils.import_utils import import_tf, lazy_importer
from stable_learning_control.utils.serialization_utils import save_to_json

from stable_learning_control.control.algos.tf2.common import get_lr_scheduler
from stable_learning_control.utils.log_utils import (
    EpochLogger,
    log_to_std_out,
    setup_logger_kwargs,
)

tf = import_tf()
nn = import_tf(module_name="tensorflow.nn")
Adam = import_tf(module_name="tensorflow.keras.optimizers", class_name="Adam")

# Import ray tuner if installed
tune = lazy_importer(module_name="ray.tune")

# Script settings
SCALE_LAMBDA_MIN_MAX = (
    0.0,
    1.0,
)  # Range of lambda lagrance multiplier
SCALE_ALPHA_MIN_MAX = (0.0, np.inf)  # Range of alpha lagrance multiplier
STD_OUT_LOG_VARS_DEFAULT = [
    "Epoch",
    "TotalEnvInteracts",
    "AverageEpRet",
    "AverageTestEpRet",
    "AverageTestEpLen",
    "AverageAlpha",
    "AverageLossAlpha",
    "AverageQ1Vals",
    "AverageQ2Vals",
    "AverageLossPi",
    "AverageEntropy",
]

# tf.config.run_functions_eagerly(True)  # NOTE: Uncomment for debugging.


class SAC(tf.keras.Model):
    """The Soft Actor Critic algorithm.

    Attributes:
        ac (tf.Module): The (soft) actor critic module.
        ac_ (tf.Module): The (soft) target actor critic module.
        log_alpha (tf.Variable): The temperature lagrance multiplier.
        target_entropy (int): The target entropy.
        device (str): The device the networks are placed on (CPU or GPU).
    """

    def __init__(  # noqa: C901
        self,
        env,
        actor_critic=None,
        ac_kwargs=dict(
            hidden_sizes={"actor": [256] * 2, "critic": [256] * 2},
            activation={"actor": nn.relu, "critic": nn.relu},
            output_activation={"actor": nn.relu, "critic": None},
        ),
        opt_type="maximize",
        alpha=0.99,
        gamma=0.99,
        polyak=0.995,
        target_entropy=None,
        adaptive_temperature=True,
        lr_a=1e-4,
        lr_c=3e-4,
        device="gpu",
        name="SAC",
    ):
        """Soft Actor-Critic (SAC)

        Args:
            env (:obj:`gym.env`): The gymnasium environment the SAC is training in. This is
                used to retrieve the activation and observation space dimensions. This
                is used while creating the network sizes. The environment must satisfy
                the gymnasium API.
            actor_critic (tf.Module, optional): The constructor method for a
                Tensorflow Module with an ``act`` method, a ``pi`` module and several
                ``Q`` or ``L`` modules. The ``act`` method and ``pi`` module should
                accept batches of observations as inputs, and the ``Q*`` and ``L``
                modules should accept a batch of observations and a batch of actions as
                inputs. When called, these modules should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``Q*/L``     (batch,)          | Tensor containing one current estimate
                                               | of ``Q*/L`` for the provided
                                               | observations and actions. (Critical:
                                               | make sure to flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                               | actions in ``a``. Importantly:
                                               | gradients should be able to flow back
                                               | into ``a``.
                ===========  ================  ======================================

                Defaults to
                :class:`~stable_learning_control.control.algos.tf2.policies.soft_actor_critic.SoftActorCritic`
            ac_kwargs (dict, optional): Any kwargs appropriate for the ActorCritic
                object you provided to SAC. Defaults to:

                =======================  ============================================
                Kwarg                    Value
                =======================  ============================================
                ``hidden_sizes_actor``    ``64 x 2``
                ``hidden_sizes_critic``   ``128 x 2``
                ``activation``            :class:`tf.nn.relu`
                ``output_activation``     :class:`tf.nn.relu`
                =======================  ============================================
            opt_type (str, optional): The optimization type you want to use. Options
                ``maximize`` and ``minimize``. Defaults to ``maximize``.
            alpha (float, optional): Entropy regularization coefficient (Equivalent to
                inverse of reward scale in the original SAC paper). Defaults to
                ``0.99``.
            gamma (float, optional): Discount factor. (Always between 0 and 1.).
                Defaults to ``0.99``.
            polyak (float, optional): Interpolation factor in polyak averaging for
                target networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.). Defaults to ``0.995``.
            target_entropy (float, optional): Initial target entropy used while learning
                the entropy temperature (alpha). Defaults to the
                maximum information (bits) contained in action space. This can be
                calculated according to :

                .. math::
                    -{\\prod }_{i=0}^{n}action\\_di{m}_{i}\\phantom{\\rule{0ex}{0ex}}
            adaptive_temperature (bool, optional): Enabled Automating Entropy Adjustment
                for maximum Entropy RL_learning.
            lr_a (float, optional): Learning rate used for the actor. Defaults to
                ``1e-4``.
            lr_c (float, optional): Learning rate used for the (soft) critic.
                Defaults to ``1e-4``.
            device (str, optional): The device the networks are placed on (``cpu``
                or ``gpu``). Defaults to ``cpu``.
        """  # noqa: E501
        super().__init__(name=name)
        self._setup_kwargs = {
            k: v for k, v in locals().items() if k not in ["self", "__class__", "env"]
        }
        self._was_build = False

        # Validate gymnasium env
        # NOTE: The current implementation only works with continuous spaces.
        if not is_gym_env(env):
            raise ValueError("Env must be a valid gymnasium environment.")
        if is_discrete_space(env.action_space) or is_discrete_space(
            env.observation_space
        ):
            raise NotImplementedError(
                "The SAC algorithm does not yet support discrete observation/action "
                "spaces. Please open a feature/pull request on "
                "https://github.com/rickstaa/stable-learning-control/issues if you "
                "need this."
            )

        if hasattr(env.unwrapped.spec, "id"):
            log_to_std_out(
                "You are using the '{}' environment.".format(env.unwrapped.spec.id),
                type="info",
            )
        else:
            log_to_std_out(
                "You are using the '{}' environment.".format(
                    type(env.unwrapped).__name__
                ),
                type="info",
            )
        log_to_std_out("You are using the SAC algorithm.", type="info")
        log_to_std_out(
            "This agent is {}.".format(
                "minimizing the cost"
                if opt_type.lower() == "minimize"
                else "maximizing the return"
            ),
            type="info",
        )

        # Store algorithm parameters
        self._act_dim = env.action_space.shape
        self._obs_dim = env.observation_space.shape
        self._device = set_device(device)
        self._adaptive_temperature = adaptive_temperature
        self._opt_type = opt_type
        self._polyak = polyak
        self._gamma = gamma
        self._lr_a = tf.Variable(lr_a, name="Lr_a")
        if self._adaptive_temperature:
            self._lr_alpha = tf.Variable(lr_a, name="Lr_alpha")
        self._lr_c = tf.Variable(lr_c, name="Lr_c")
        if not isinstance(target_entropy, (float, int)):
            self._target_entropy = heuristic_target_entropy(env.action_space)
        else:
            self._target_entropy = target_entropy

        # Create variables for the Lagrance multipliers
        # NOTE: Clip at 1e-37 to prevent log_alpha/log_lambda from becoming -np.inf
        self.log_alpha = tf.Variable(
            tf.math.log(1e-37 if alpha < 1e-37 else alpha), name="log_alpha"
        )

        # Get default actor critic if no 'actor_critic' was supplied
        actor_critic = SoftActorCritic if actor_critic is None else actor_critic

        # Create actor-critic module and target networks
        self.ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        self.ac_targ = actor_critic(
            env.observation_space,
            env.action_space,
            name="soft_actor_critic_target",
            **ac_kwargs,
        )

        self._init_targets()

        # Create optimizers
        # NOTE: We here optimize for log_alpha instead of alpha because it is more
        # numerically stable (see:
        # https://github.com/rail-berkeley/softlearning/issues/136)
        self._pi_optimizer = Adam(learning_rate=self._lr_a)
        self._pi_params = self.ac.pi.trainable_variables
        if self._adaptive_temperature:
            self._log_alpha_optimizer = Adam(learning_rate=self._lr_alpha)
        # List of parameters for both Q-networks (save this for convenience)
        self._c_params = (
            self.ac.Q1.trainable_variables + self.ac.Q2.trainable_variables
        )  # Chain parameters of the two Q-critics
        self._c_optimizer = Adam(learning_rate=self._lr_c)

    @tf.function
    def call(self, s, deterministic=False):
        """Wrapper around the :meth:`.get_action` method that enables users to also
        receive actions directly by invoking ``SAC(observations)``.

        Args:
            s (numpy.ndarray): The current state.
            deterministic (bool, optional): Whether to return a deterministic action.
                Defaults to ``False``.

        Returns:
            numpy.ndarray: The current action.
        """
        return self.get_action(s, deterministic=deterministic)

    @tf.function
    def get_action(self, s, deterministic=False):
        """Returns the current action of the policy.

        Args:
            s (numpy.ndarray): The current state.
            deterministic (bool, optional): Whether to return a deterministic action.
                Defaults to ``False``.

        Returns:
            numpy.ndarray: The current action.
        """
        # Make sure s is float32 tensorflow tensor
        if not isinstance(s, tf.Tensor):
            s = tf.convert_to_tensor(s, dtype=tf.float32)
        elif s.dtype != tf.float32:
            s = tf.cast(s, dtype=tf.float32)

        return tf.squeeze(
            self.ac.act(s, deterministic)
        )  # NOTE: Squeeze is critical to ensure a has right shape.

    @tf.function
    def update(self, data):
        """Update the actor critic network using stochastic gradient descent.

        Args:
            data (dict): Dictionary containing a batch of experiences.
        """
        diagnostics = dict()
        o, a, r, o_, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs_next"],
            data["done"],
        )
        ################################################
        # Optimize (soft) Q-critic #####################
        ################################################

        # Get target Q values (Bellman-backup)
        # NOTE: Here we use max-clipping instead of min-clipping used in the SAC
        # algorithm when we want to maximize/minimize the return.
        pi_, logp_pi_ = self.ac.pi(
            o_
        )  # NOTE: Target actions coming from *current* policy

        # Get target Q values based on optimization type
        q1_pi_targ = self.ac_targ.Q1([o_, pi_])
        q2_pi_targ = self.ac_targ.Q2([o_, pi_])
        if self._opt_type.lower() == "minimize":
            q_pi_targ = tf.math.maximum(
                q1_pi_targ,
                q2_pi_targ,
            )  # Use max clipping  to prevent overestimation bias.
        else:
            q_pi_targ = tf.math.minimum(
                q1_pi_targ, q2_pi_targ
            )  # Use min clipping to prevent overestimation bias
        q_backup = r + self._gamma * (1 - d) * (q_pi_targ - self.alpha * logp_pi_)

        # Compute the Q-Critic loss gradients
        with tf.GradientTape() as q_tape:
            # Retrieve the current Q values
            q1 = self.ac.Q1([o, a])
            q2 = self.ac.Q2([o, a])

            # Calculate Q-critic MSE loss against Bellman backup
            loss_q1 = 0.5 * tf.reduce_mean((q1 - q_backup) ** 2)  # See Haarnoja eq. 5
            loss_q2 = 0.5 * tf.reduce_mean((q2 - q_backup) ** 2)
            q_loss = loss_q1 + loss_q2

        c_grads = q_tape.gradient(q_loss, self._c_params)
        self._c_optimizer.apply_gradients(zip(c_grads, self._c_params))

        q_info = dict(Q1Vals=q1, Q2Vals=q2)
        diagnostics.update({**q_info, "LossQ": q_loss})
        ################################################
        # Optimize Gaussian actor ######################
        ################################################
        # Compute actor loss gradients
        with tf.GradientTape() as a_tape:
            # Retrieve log probabilities of batch observations based on *current* policy
            pi, logp_pi = self.ac.pi(o)

            # Retrieve current Q values
            # NOTE: Actions come from *current* policy
            q1_pi = self.ac.Q1([o, pi])
            q2_pi = self.ac.Q2([o, pi])
            if self._opt_type.lower() == "minimize":
                q_pi = tf.math.maximum(q1_pi, q2_pi)
            else:
                q_pi = tf.math.minimum(q1_pi, q2_pi)

            # Calculate entropy-regularized policy loss
            if self._opt_type.lower() == "minimize":
                a_loss = tf.reduce_mean(
                    tf.stop_gradient(self.alpha) * logp_pi + q_pi
                )  # Minimization version of Haarnoja eq. 7
            else:
                a_loss = tf.reduce_mean(
                    tf.stop_gradient(self.alpha) * logp_pi - q_pi
                )  # See Haarnoja eq. 7

        a_grads = a_tape.gradient(a_loss, self._pi_params)
        self._pi_optimizer.apply_gradients(zip(a_grads, self._pi_params))

        pi_info = dict(LogPi=logp_pi, Entropy=-tf.reduce_mean(logp_pi))
        diagnostics.update({**pi_info, "LossPi": a_loss})
        ################################################
        # Optimize alpha (Entropy temperature) #########
        ################################################
        if self._adaptive_temperature:
            # Compute alpha loss gradients
            with tf.GradientTape() as alpha_tape:
                # Calculate alpha loss
                alpha_loss = -tf.reduce_mean(
                    self.alpha * tf.stop_gradient(logp_pi + self.target_entropy)
                )  # See Haarnoja eq. 17

            alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self._log_alpha_optimizer.apply_gradients(
                zip(alpha_grads, [self.log_alpha])
            )

            alpha_info = dict(Alpha=self.alpha)
            diagnostics.update({**alpha_info, "LossAlpha": alpha_loss})

        ################################################
        # Update target networks and return ############
        # diagnostics. #################################
        ################################################
        self._update_targets()
        return diagnostics

    def save(self, path, checkpoint_name="checkpoint"):
        """Can be used to save the current model state.

        Args:
            path (str): The path where you want to save the policy.
            checkpoint_name (str): The name you want to use for the checkpoint.

        Raises:
            Exception: Raises an exception if something goes wrong during saving.

        .. note::
            This function saved the model weights using the
            :meth:`tf.keras.Model.save_weights` method (see
            :tf:`keras/Model#save_weights`). The model should therefore be restored
            using the :meth:`tf.keras.Model.load_weights` method (see
            :tf:`keras/Model#load_weights`). If
            you want to deploy the full model use the :meth:`.export` method instead.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_path = path.joinpath(f"policy/{checkpoint_name}")
        try:
            self.save_weights(save_path)
        except Exception as e:
            raise Exception("SAC model could not be saved.") from e

        # Save additional information
        save_info = {
            "alg_name": self.__class__.__name__,
            "setup_kwargs": self._setup_kwargs,
        }
        save_to_json(
            input_object=save_info,
            output_filename="save_info.json",
            output_path=save_path.parent,
        )

    def restore(self, path, restore_lagrance_multipliers=False):
        """Restores a already trained policy. Used for transfer learning.

        Args:
            path (str): The path where the model :attr:`state_dict` of the policy is
                found.
            restore_lagrance_multipliers (bool, optional): Whether you want to restore
                the lagrance multipliers. By fault ``False``.

        Raises:
            Exception: Raises an exception if something goes wrong during loading.
        """
        latest = tf.train.latest_checkpoint(path)
        if latest is None:
            latest = tf.train.latest_checkpoint(osp.join(path, "tf2_save"))
            if latest is None:
                raise Exception(
                    f"No models found in '{path}'. Please check your policy restore"
                    "path and try again."
                )

        # Store initial values in order to ignore them when loading the weights
        lr_a = self._lr_a.value()
        lr_alpha = self._lr_alpha.value()
        lr_c = self._lr_c.value()
        if not restore_lagrance_multipliers:
            log_alpha_init = self.log_alpha.value()

        try:
            self.load_weights(latest)
        except Exception as e:
            raise Exception(
                f"Something went wrong when trying to load model '{latest}'."
            ) from e

        # Make sure learning rates (and lagrance multipliers) are not restored
        self._lr_a.assign(lr_a)
        self._lr_alpha.assign(lr_alpha)
        self._lr_c.assign(lr_c)
        if not restore_lagrance_multipliers:
            self.log_alpha.assign(log_alpha_init)
            log_to_std_out("Restoring lagrance multipliers.", type="info")
        else:
            log_to_std_out(
                "Keeping lagrance multipliers at their initial value.", type="info"
            )

    def export(self, path):
        """Can be used to export the model in the ``SavedModel`` format such that it can
        be deployed to hardware.

        Args:
            path (str): The path where you want to export the policy too.
        """
        if tf.config.functions_run_eagerly():
            log_to_std_out(
                "Exporting the tensorflow model is not supported in eager mode.",
                type="error",
            )
        else:
            # NOTE: Currently we only export the actor as this is what is Useful when
            # deploying the algorithm.
            obs_dummy = tf.random.uniform(
                combine_shapes(1, self._obs_dim), dtype=tf.float32
            )
            self.ac.pi.get_action(obs_dummy)  # Make sure the full graph was traced
            self.ac.pi.save(osp.join(path, "tf2_save"))

    def build(self):
        """Function that can be used to build the full model structure such that it can
        be visualized using the `tf.keras.Model.summary()`.

        .. note::
            This is done by calling the build methods of the submodules.
        """
        obs_dummy = tf.random.uniform(
            combine_shapes(1, self._obs_dim), dtype=tf.float32
        )
        act_dummy = tf.random.uniform(
            combine_shapes(1, self._act_dim), dtype=tf.float32
        )
        self.ac([obs_dummy, act_dummy])
        self.ac_targ([obs_dummy, act_dummy])
        super().build(input_shape=combine_shapes(1, self._obs_dim))
        self(obs_dummy)
        self._was_build = True

    def summary(self):
        """Small wrapper around the :meth:`tf.keras.Model.summary()` method used to
        apply a custom format to the model summary.
        """
        if not self._was_build:
            self.build()
        super().summary()
        print("")
        self.ac.summary()
        print("")
        self.ac.pi.summary()
        print("")
        self.ac.pi.net.summary()
        print("")
        self.ac.Q1.summary()
        print("")
        self.ac.Q1.Q.summary()
        print("")
        self.ac.Q2.summary()
        self.ac.Q2.Q.summary()
        print("")

    def set_learning_rates(self, lr_a=None, lr_c=None, lr_alpha=None):
        """Adjusts the learning rates of the optimizers.

        Args:
            lr_a (float, optional): The learning rate of the actor optimizer. Defaults
                to None.
            lr_c (float, optional): The learning rate of the (soft) Critic. Defaults
                to None.
            lr_alpha (float, optional): The learning rate of the temperature optimizer.
                Defaults to None.
        """
        if lr_a:
            self._pi_optimizer.lr.assign(lr_a)
        if lr_c:
            self._c_optimizer.lr.assign(lr_c)
        if self._adaptive_temperature:
            if lr_alpha:
                self._log_alpha_optimizer.lr.assign(lr_alpha)

    @tf.function
    def _init_targets(self):
        """Updates the target network weights to the main network weights."""
        for pi_main, pi_targ in zip(self.ac.pi.variables, self.ac_targ.pi.variables):
            pi_targ.assign(pi_main)
        for c1_main, c1_targ in zip(self.ac.Q1.variables, self.ac_targ.Q1.variables):
            c1_targ.assign(c1_main)
        for c2_main, c2_targ in zip(self.ac.Q2.variables, self.ac_targ.Q2.variables):
            c2_targ.assign(c2_main)

    @tf.function
    def _update_targets(self):
        """Updates the target networks based on a Exponential moving average
        (Polyak averaging).
        """
        for pi_main, pi_targ in zip(self.ac.pi.variables, self.ac_targ.pi.variables):
            pi_targ.assign(self._polyak * pi_targ + (1 - self._polyak) * pi_main)
        for c1_main, c1_targ in zip(self.ac.Q1.variables, self.ac_targ.Q1.variables):
            c1_targ.assign(self._polyak * c1_targ + (1 - self._polyak) * c1_main)
        for c2_main, c2_targ in zip(self.ac.Q2.variables, self.ac_targ.Q2.variables):
            c2_targ.assign(self._polyak * c2_targ + (1 - self._polyak) * c2_main)

    @property
    def alpha(self):
        """Property used to clip :attr:`alpha` to be equal or bigger than ``0.0`` to
        prevent it from becoming nan when :attr:`log_alpha` becomes ``-inf``. For
        :attr:`alpha` no upper bound is used.
        """
        return tf.clip_by_value(tf.exp(self.log_alpha), *SCALE_ALPHA_MIN_MAX)

    @alpha.setter
    def alpha(self, set_val):
        """Property used to ensure :attr:`alpha` and :attr:`log_alpha` are related."""
        self.log_alpha.assign(
            tf.convert_to_tensor(
                np.log(1e-37 if set_val < 1e-37 else set_val),
                dtype=self.log_alpha.dtype,
            )
        )

    @property
    def target_entropy(self):
        """The target entropy used while learning the entropy temperature
        :attr:`alpha`.
        """
        return self._target_entropy

    @target_entropy.setter
    def target_entropy(self, set_val):
        error_msg = (
            "Changing the 'target_entropy' during training is not allowed."
            "Please open a feature/pull request on "
            "https://github.com/rickstaa/stable-learning-control/issues if you need "
            "this."
        )
        raise AttributeError(error_msg)

    @property
    def device(self):
        """The device the networks are placed on (``cpu`` or ``gpu``). Defaults to
        ``cpu``.
        """
        return self._device

    @device.setter
    def device(self, set_val):
        error_msg = (
            "Changing the computational 'device' during training is not allowed."
        )
        raise AttributeError(error_msg)


def validate_args(**kwargs):
    """Checks if the input argument have valid values.

    Raises:
        ValueError: If a value is invalid.
    """
    if kwargs["update_after"] > kwargs["steps_per_epoch"]:
        raise ValueError(
            "You can not set 'update_after' bigger than the 'steps_per_epoch'. Please "
            "change this and try again."
        )


def sac(  # noqa: C901
    env_fn,
    actor_critic=None,
    ac_kwargs=dict(
        hidden_sizes={"actor": [256] * 2, "critic": [256] * 2},
        activation={"actor": nn.relu, "critic": nn.relu},
        output_activation={"actor": nn.relu, "critic": None},
    ),
    opt_type="maximize",
    max_ep_len=None,
    epochs=100,
    steps_per_epoch=2048,
    start_steps=0,
    update_every=100,
    update_after=1000,
    steps_per_update=100,
    num_test_episodes=10,
    alpha=0.99,
    gamma=0.99,
    polyak=0.995,
    target_entropy=None,
    adaptive_temperature=True,
    lr_a=1e-4,
    lr_c=3e-4,
    lr_a_final=1e-10,
    lr_c_final=1e-10,
    lr_decay_type="linear",
    lr_decay_ref="epoch",
    batch_size=256,
    replay_size=int(1e6),
    seed=None,
    device="cpu",
    logger_kwargs=dict(),
    save_freq=1,
    start_policy=None,
    export=False,
):
    """Trains the sac algorithm in a given environment.

    Args:
        env_fn: A function which creates a copy of the environment.
            The environment must satisfy the gymnasium API.
        actor_critic (tf.Module, optional): The constructor method for a
            Tensorflow Module with an ``act`` method, a ``pi`` module and several ``Q``
            or ``L`` modules. The ``act`` method and ``pi`` module should accept batches
            of observations as inputs, and the ``Q*`` and
            ``L`` modules should accept a batch of observations and a batch of actions
            as inputs. When called, these modules should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)   | Numpy array of actions for each
                                            | observation.
            ``Q*/L``     (batch,)           | Tensor containing one current estimate
                                            | of ``Q*/L`` for the provided
                                            | observations and actions. (Critical:
                                            | make sure to flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)   | Tensor containing actions from policy
                                            | given observations.
            ``logp_pi``  (batch,)           | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly:
                                            | gradients should be able to flow back
                                            | into ``a``.
            ===========  ================  ======================================

            Defaults to
            :class:`~stable_learning_control.control.algos.tf2.policies.soft_actor_critic.SoftActorCritic`
        ac_kwargs (dict, optional): Any kwargs appropriate for the ActorCritic
            object you provided to SAC. Defaults to:

            =======================  ============================================
            Kwarg                    Value
            =======================  ============================================
            ``hidden_sizes_actor``    ``64 x 2``
            ``hidden_sizes_critic``   ``128 x 2``
            ``activation``            :class:`tf.nn.relu`
            ``output_activation``     :class:`tf.nn.relu`
            =======================  ============================================
        opt_type (str, optional): The optimization type you want to use. Options
            ``maximize`` and ``minimize``. Defaults to ``maximize``.
        max_ep_len (int, optional): Maximum length of trajectory / episode /
            rollout. Defaults to the environment maximum.
        epochs (int, optional): Number of epochs to run and train agent. Defaults
            to ``100``.
        steps_per_epoch (int, optional): Number of steps of interaction
            (state-action pairs) for the agent and the environment in each epoch.
            Defaults to ``2048``.
        start_steps (int, optional): Number of steps for uniform-random action
            selection, before running real policy. Helps exploration. Defaults to
            ``0``.
        update_every (int, optional): Number of env interactions that should elapse
            between gradient descent updates. Defaults to ``100``.
        update_after (int, optional): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates. Defaults to ``1000``.
        steps_per_update (int, optional): Number of gradient descent steps that are
            performed for each gradient descent update. This determines the ratio of
            env steps to gradient steps (i.e. :obj:`update_every`/
            :obj:`steps_per_update`). Defaults to ``100``.
        num_test_episodes (int, optional): Number of episodes used to test the
            deterministic policy at the end of each epoch. This is used for logging
            the performance. Defaults to ``10``.
        alpha (float, optional): Entropy regularization coefficient (Equivalent to
            inverse of reward scale in the original SAC paper). Defaults to
            ``0.99``.
        gamma (float, optional): Discount factor. (Always between 0 and 1.).
            Defaults to ``0.99``.
        polyak (float, optional): Interpolation factor in polyak averaging for
            target networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.). Defaults to ``0.995``.
        target_entropy (float, optional): Initial target entropy used while learning
            the entropy temperature (alpha). Defaults to the
            maximum information (bits) contained in action space. This can be
            calculated according to :

            .. math::
                -{\\prod }_{i=0}^{n}action\\_di{m}_{i}\\phantom{\\rule{0ex}{0ex}}
        adaptive_temperature (bool, optional): Enabled Automating Entropy Adjustment
            for maximum Entropy RL_learning.
        lr_a (float, optional): Learning rate used for the actor. Defaults to
            ``1e-4``.
        lr_c (float, optional): Learning rate used for the (soft) critic. Defaults to
            ``1e-4``.
        lr_a_final(float, optional): The final actor learning rate that is achieved
            at the end of the training. Defaults to ``1e-10``.
        lr_c_final(float, optional): The final critic learning rate that is achieved
            at the end of the training. Defaults to ``1e-10``.
        lr_decay_type (str, optional): The learning rate decay type that is used (
            options are: ``linear`` and ``exponential`` and ``constant``). Defaults to
            ``linear``.
        lr_decay_ref (str, optional): The reference variable that is used for decaying
            the learning rate (options: ``epoch`` and ``step``). Defaults to ``epoch``.
        batch_size (int, optional): Minibatch size for SGD. Defaults to ``256``.
        replay_size (int, optional): Maximum length of replay buffer. Defaults to
            ``1e6``.
        seed (int): Seed for random number generators. Defaults to ``None``.
        device (str, optional): The device the networks are placed on (``cpu``
            or ``gpu``). Defaults to ``cpu``.
        logger_kwargs (dict, optional): Keyword args for EpochLogger.
        save_freq (int, optional): How often (in terms of gap between epochs) to save
            the current policy and value function.
        start_policy (str): Path of a already trained policy to use as the starting
            point for the training. By default a new policy is created.
        export (bool): Whether you want to export the model in the ``SavedModel`` format
            such that it can be deployed to hardware. By default ``False``.
    """  # noqa: E501
    total_steps = steps_per_epoch * epochs

    validate_args(**locals())

    env = env_fn()

    # Validate gymnasium env
    # NOTE: The current implementation only works with continuous spaces.
    if not is_gym_env(env):
        raise ValueError("Env must be a valid gymnasium environment.")
    if is_discrete_space(env.action_space) or is_discrete_space(env.observation_space):
        raise NotImplementedError(
            "The SAC algorithm does not yet support discrete observation/action "
            "spaces. Please open a feature/pull request on "
            "https://github.com/rickstaa/stable-learning-control/issues if you "
            "need this."
        )

    env = gym.wrappers.FlattenObservation(
        env
    )  # NOTE: Done to make sure the alg works with dict observation spaces
    if num_test_episodes != 0:
        test_env = env_fn()
        test_env = gym.wrappers.FlattenObservation(test_env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    rew_dim = (
        env.reward_range.shape[0] if isinstance(env.reward_range, gym.spaces.Box) else 1
    )

    logger_kwargs["verbose_vars"] = (
        logger_kwargs["verbose_vars"]
        if (
            "verbose_vars" in logger_kwargs.keys()
            and logger_kwargs["verbose_vars"] is not None
        )
        else STD_OUT_LOG_VARS_DEFAULT
    )  # NOTE: Done to ensure the std_out doesn't get cluttered.
    logger_kwargs["backend"] = "tf"  # NOTE: Use tensorflow tensorboard backend
    tb_low_log_freq = (
        logger_kwargs.pop("tb_log_freq").lower() == "low"
        if "tb_log_freq" in logger_kwargs.keys()
        else True
    )
    use_tensorboard = (
        logger_kwargs.pop("use_tensorboard")
        if "use_tensorboard" in logger_kwargs.keys()
        else False
    )
    logger = EpochLogger(**logger_kwargs)
    hyper_paramet_dict = {
        k: v for k, v in locals().items() if k not in ["logger"]
    }  # Retrieve hyperparameters (Ignore logger object)
    logger.save_config(hyper_paramet_dict)  # Write hyperparameters to logger

    # Retrieve max episode length
    if max_ep_len is None:
        max_ep_len = env.env._max_episode_steps
    else:
        if max_ep_len > env.env._max_episode_steps:
            logger.log(
                (
                    f"You defined your 'max_ep_len' to be {max_ep_len} "
                    "while the environment 'max_epsisode_steps' is "
                    f"{env.env._max_episode_steps}. As a result the environment "
                    f"'max_episode_steps' has been increased to {max_ep_len}"
                ),
                type="warning",
            )
            env.env._max_episode_steps = max_ep_len
            if num_test_episodes != 0:
                test_env.env._max_episode_steps = max_ep_len

    # Get default actor critic if no 'actor_critic' was supplied
    actor_critic = SoftActorCritic if actor_critic is None else actor_critic

    # Set random seed for reproducible results
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    policy = SAC(
        env,
        actor_critic,
        ac_kwargs,
        opt_type,
        alpha,
        gamma,
        polyak,
        target_entropy,
        adaptive_temperature,
        lr_a,
        lr_c,
        device,
    )

    # Create learning rate schedulers
    # NOTE: Alpha currently uses the same scheduler as the actor.
    lr_decay_ref_var = total_steps if lr_decay_ref.lower() == "steps" else epochs
    lr_a_scheduler = get_lr_scheduler(lr_decay_type, lr_a, lr_a_final, lr_decay_ref_var)
    lr_c_scheduler = get_lr_scheduler(lr_decay_type, lr_c, lr_c_final, lr_decay_ref_var)

    # Restore policy if supplied
    if start_policy is not None:
        logger.log(f"Restoring model from '{start_policy}'.", type="info")
        try:
            policy.restore(start_policy)
            logger.log("Model successfully restored.", type="info")
        except Exception as e:
            logger.log(
                "Shutting down training since {}.".format(
                    e.args[0].lower().rstrip(".")
                ),
                type="error",
            )
            sys.exit(0)

    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        rew_dim=rew_dim,
        size=replay_size,
    )

    # Count variables and print network structure
    var_counts = tuple(
        count_vars(module) for module in [policy.ac.pi, policy.ac.Q1, policy.ac.Q2]
    )
    logger.log(
        "Number of parameters: \t pi: %d, \t Q1: %d, \t Q2: %d\n" % var_counts,
        type="info",
    )
    logger.log("Network structure:\n", type="info")
    policy.summary()

    logger.setup_tf_saver(policy)

    # Setup diagnostics tb_write dict and store initial learning rates
    diag_tb_log_list = ["LossQ", "LossPi", "Alpha", "LossAlpha", "Entropy"]
    if use_tensorboard:
        logger.log_to_tb(
            "Lr_a",
            policy._pi_optimizer.lr.numpy(),
            tb_prefix="LearningRates",
            global_step=0,
        )
        logger.log_to_tb(
            "Lr_c",
            policy._c_optimizer.lr.numpy(),
            tb_prefix="LearningRates",
            global_step=0,
        )
        logger.log_to_tb(
            "Lr_alpha",
            policy._log_alpha_optimizer.lr.numpy(),
            tb_prefix="LearningRates",
            global_step=0,
        )

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = policy.get_action(o)
        else:
            a = env.action_space.sample()

        # Take step in the env
        o_, r, d, truncated, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        replay_buffer.store(o, a, r, o_, d)

        # Make sure to update most recent observation!
        o = o_

        # End of trajectory handling
        if d or truncated:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            ep_ret, ep_len = 0, 0

        # Update handling
        if (t + 1) >= update_after and ((t + 1) - update_after) % update_every == 0:
            # Step based learning rate decay
            if lr_decay_ref.lower() == "step":
                lr_a_now = max(
                    lr_a_scheduler(t + 1), lr_a_final
                )  # Make sure lr is bounded above final lr
                lr_c_now = max(
                    lr_c_scheduler(t + 1), lr_c_final
                )  # Make sure lr is bounded above final lr
                policy.set_learning_rates(
                    lr_a=lr_a_now, lr_c=lr_c_now, lr_alpha=lr_a_now
                )

            for _ in range(steps_per_update):
                batch = replay_buffer.sample_batch(batch_size)
                update_diagnostics = policy.update(data=batch)
                logger.store(**update_diagnostics)  # Log diagnostics

            # SGD batch tb logging
            if use_tensorboard and not tb_low_log_freq:
                logger.log_to_tb(keys=diag_tb_log_list, global_step=t)

        # End of epoch handling (Save model, test performance and log data)
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({"env": env}, itr=epoch)

            # Test the performance of the deterministic version of the agent
            if num_test_episodes != 0:
                eps_ret, eps_len = test_agent(
                    policy, test_env, num_test_episodes, max_ep_len=max_ep_len
                )
                logger.store(
                    TestEpRet=eps_ret,
                    TestEpLen=eps_len,
                    extend=True,
                )

            # Epoch based learning rate decay
            if lr_decay_ref.lower() != "step":
                lr_a_now = max(
                    lr_a_scheduler(epoch), lr_a_final
                )  # Make sure lr is bounded above final
                lr_c_now = max(
                    lr_c_scheduler(epoch), lr_c_final
                )  # Make sure lr is bounded above final
                policy.set_learning_rates(
                    lr_a=lr_a_now, lr_c=lr_c_now, lr_alpha=lr_a_now
                )

            # Log performance measure to ray tuning
            # NOTE: Only executed when the ray tuner invokes the script
            if hasattr(tune, "session") and tune.session._session is not None:
                mean_ep_ret = logger.get_stats("EpRet")
                mean_ep_len = logger.get_stats("EpLen")
                tune.report(
                    mean_ep_ret=mean_ep_ret[0], epoch=epoch, mean_ep_len=mean_ep_len[0]
                )

            # Log info about epoch
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular(
                "EpRet",
                with_min_and_max=True,
                tb_write=use_tensorboard,
            )
            if num_test_episodes != 0:
                logger.log_tabular(
                    "TestEpRet",
                    with_min_and_max=True,
                    tb_write=use_tensorboard,
                )
                logger.log_tabular("EpLen", average_only=True, tb_write=use_tensorboard)
                logger.log_tabular(
                    "TestEpLen",
                    average_only=True,
                    tb_write=use_tensorboard,
                )
            logger.log_tabular(
                "Lr_a",
                policy._pi_optimizer.lr.numpy(),
                tb_write=use_tensorboard,
                tb_prefix="LearningRates",
            )
            logger.log_tabular(
                "Lr_c",
                policy._c_optimizer.lr.numpy(),
                tb_write=use_tensorboard,
                tb_prefix="LearningRates",
            )
            logger.log_tabular(
                "Lr_alpha",
                policy._log_alpha_optimizer.lr.numpy(),
                tb_write=use_tensorboard,
                tb_prefix="LearningRates",
            )
            logger.log_tabular(
                "Alpha",
                average_only=True,
                tb_write=(use_tensorboard and tb_low_log_freq),
            )
            logger.log_tabular(
                "LossPi",
                average_only=True,
                tb_write=(use_tensorboard and tb_low_log_freq),
            )
            logger.log_tabular(
                "LossQ",
                average_only=True,
                tb_write=(use_tensorboard and tb_low_log_freq),
            )
            if adaptive_temperature:
                logger.log_tabular(
                    "LossAlpha",
                    average_only=True,
                    tb_write=(use_tensorboard and tb_low_log_freq),
                )
            logger.log_tabular("Q1Vals", with_min_and_max=True)
            logger.log_tabular("Q2Vals", with_min_and_max=True)
            logger.log_tabular("LogPi", with_min_and_max=True)
            logger.log_tabular(
                "Entropy",
                average_only=True,
                tb_write=(use_tensorboard and tb_low_log_freq),
            )
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular(global_step=t)

    # Export model to 'SavedModel'
    if export:
        policy.export(logger.output_dir)

    print("" if logger_kwargs["verbose"] else "\n")
    logger.log(
        "Training finished after {}s".format(time.time() - start_time),
        type="info",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a SAC agent in a given environment."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="stable_gym:Oscillator-v1",
        help="the gymnasium env (default: stable_gym:Oscillator-v1)",
    )  # NOTE: Ensure the environment is installed in the current python environment.
    parser.add_argument(
        "--hid_a",
        type=int,
        default=256,
        help="hidden layer size of the actor (default: 256)",
    )
    parser.add_argument(
        "--hid_c",
        type=int,
        default=256,
        help="hidden layer size of the (soft) critic (default: 256)",
    )
    parser.add_argument(
        "--l_a",
        type=int,
        default=2,
        help="number of hidden layer in the actor (default: 2)",
    )
    parser.add_argument(
        "--l_c",
        type=int,
        default=2,
        help="number of hidden layer in the critic (default: 2)",
    )
    parser.add_argument(
        "--act_a",
        type=str,
        default="nn.relu",
        help="the hidden layer activation function of the actor (default: nn.relu)",
    )
    parser.add_argument(
        "--act_c",
        type=str,
        default="nn.relu",
        help="the hidden layer activation function of the critic (default: nn.relu)",
    )
    parser.add_argument(
        "--act_out_a",
        type=str,
        default="nn.relu",
        help="the output activation function of the actor (default: nn.relu)",
    )
    parser.add_argument(
        "--act_out_c",
        type=str,
        default="None",
        help="the output activation function of the critic (default: None)",
    )
    parser.add_argument(
        "--opt_type",
        type=str,
        default="minimize",
        help="algorithm optimization type (default: minimize)",
    )
    parser.add_argument(
        "--max_ep_len",
        type=int,
        default=None,
        help="maximum episode length (default: None)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="the number of epochs (default: 50)"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=2048,
        help="the number of steps per epoch (default: 2048)",
    )
    parser.add_argument(
        "--start_steps",
        type=int,
        default=0,
        help="the number of random exploration steps (default: 0)",
    )
    parser.add_argument(
        "--update_every",
        type=int,
        default=100,
        help=(
            "the number of env interactions that should elapse between SGD updates "
            "(default: 100)"
        ),
    )
    parser.add_argument(
        "--update_after",
        type=int,
        default=1000,
        help="the number of steps before starting the SGD (default: 1000)",
    )
    parser.add_argument(
        "--steps_per_update",
        type=int,
        default=100,
        help=(
            "the number of gradient descent steps that are"
            "performed for each SGD update (default: 100)"
        ),
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=10,
        help=(
            "the number of episodes for the performance analysis (default: 10). When "
            "set to zero no test episodes will be performed"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="the entropy regularization coefficient (default: 0.99)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--polyak",
        type=float,
        default=0.995,
        help="the interpolation factor in polyak averaging (default: 0.995)",
    )
    parser.add_argument(
        "--target_entropy",
        type=float,
        default=None,
        help="the initial target entropy (default: -action_space)",
    )
    parser.add_argument(
        "--adaptive_temperature",
        type=bool,
        default=True,
        help="the boolean for enabling automating Entropy Adjustment (default: True)",
    )
    parser.add_argument(
        "--lr_a", type=float, default=1e-4, help="actor learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--lr_c", type=float, default=3e-4, help="critic learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--lr_a_final",
        type=float,
        default=1e-10,
        help="the finalactor learning rate (default: 1e-10)",
    )
    parser.add_argument(
        "--lr_c_final",
        type=float,
        default=1e-10,
        help="the finalcritic learning rate (default: 1e-10)",
    )
    parser.add_argument(
        "--lr_decay_type",
        type=str,
        default="linear",
        help="the learning rate decay type (default: linear)",
    )
    parser.add_argument(
        "--lr_decay_ref",
        type=str,
        default="epoch",
        help=(
            "the reference variable that is used for decaying the learning rate "
            "'epoch' or 'step' (default: 'epoch')"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=float,
        default=256,
        help="mini batch size of the SGD (default: 256)",
    )
    parser.add_argument(
        "--replay-size",
        type=int,
        default=int(1e6),
        help="replay buffer size (default: 1e6)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0, help="the random seed (default: 0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="The device the networks are placed on (default: gpu)",
    )
    parser.add_argument(
        "--start_policy",
        type=str,
        default=None,
        help=(
            "The policy which you want to use as the starting point for the training"
            " (default: None)"
        ),
    )
    parser.add_argument(
        "--export",
        type=str,
        default=False,
        help=(
            "Whether you want to export the model in the 'SavedModel' format "
            "such that it can be deployed to hardware (Default: False)"
        ),
    )

    # Parse logger related arguments
    parser.add_argument(
        "--exp_name",
        type=str,
        default="sac",
        help="the name of the experiment (default: sac)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=bool,
        default=True,
        help="log diagnostics to std out (default: True)",
    )
    parser.add_argument(
        "--verbose_fmt",
        type=str,
        default="line",
        help=(
            "log diagnostics std out format (options: 'table' or 'line', default: "
            "line)"
        ),
    )
    parser.add_argument(
        "--verbose_vars",
        nargs="+",
        default=STD_OUT_LOG_VARS_DEFAULT,
        help=("a space separated list of the values you want to show on the std out."),
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="how often (in epochs) the policy should be saved (default: 1)",
    )
    parser.add_argument(
        "--save_checkpoints",
        type=bool,
        default=False,
        help="use model checkpoints (default: False)",
    )
    parser.add_argument(
        "--use_tensorboard",
        type=bool,
        default=False,
        help="use tensorboard (default: False)",
    )
    parser.add_argument(
        "--tb_log_freq",
        type=str,
        default="low",
        help=(
            "the tensorboard log frequency. Options are 'low' (Recommended: logs at "
            "every epoch) and 'high' (logs at every SGD update batch). Default is 'low'"
            ""
        ),
    )
    args = parser.parse_args()

    # Setup actor critic arguments
    output_activation = {}
    output_activation["actor"] = safer_eval(args.act_out_a, backend="tf")
    output_activation["critic"] = safer_eval(args.act_out_c, backend="tf")
    ac_kwargs = dict(
        hidden_sizes={
            "actor": [args.hid_a] * args.l_a,
            "critic": [args.hid_c] * args.l_c,
        },
        activation={
            "actor": safer_eval(args.act_a, backend="tf"),
            "critic": safer_eval(args.act_c, backend="tf"),
        },
        output_activation=output_activation,
    )

    # Setup output dir for logger and return output kwargs
    logger_kwargs = setup_logger_kwargs(
        args.exp_name,
        args.seed,
        save_checkpoints=args.save_checkpoints,
        use_tensorboard=args.use_tensorboard,
        tb_log_freq=args.tb_log_freq,
        verbose=args.verbose,
        verbose_fmt=args.verbose_fmt,
        verbose_vars=args.verbose_vars,
    )
    logger_kwargs["output_dir"] = osp.abspath(
        osp.join(
            osp.dirname(osp.realpath(__file__)),
            f"../../../../../data/sac/{args.env.lower()}/runs/run_{int(time.time())}",
        )
    )

    sac(
        lambda: gym.make(args.env),
        actor_critic=SoftActorCritic,
        ac_kwargs=ac_kwargs,
        opt_type=args.opt_type,
        max_ep_len=args.max_ep_len,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        start_steps=args.start_steps,
        update_every=args.update_every,
        update_after=args.update_after,
        steps_per_update=args.steps_per_update,
        num_test_episodes=args.num_test_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        polyak=args.polyak,
        target_entropy=args.target_entropy,
        adaptive_temperature=args.adaptive_temperature,
        lr_a=args.lr_a,
        lr_c=args.lr_c,
        lr_a_final=args.lr_a_final,
        lr_c_final=args.lr_c_final,
        lr_decay_type=args.lr_decay_type,
        lr_decay_ref=args.lr_decay_ref,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        seed=args.seed,
        save_freq=args.save_freq,
        device=args.device,
        start_policy=args.start_policy,
        export=args.export,
        logger_kwargs=logger_kwargs,
    )