"""Lyapunov (soft) Actor-Critic (LAC) algorithm.

This module contains a TensorFlow 2.x implementation of the LAC algorithm of
`Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_.

.. note::
    Code Conventions:
        - We use a `_` suffix to distinguish the next state from the current state.
        - We use a `targ` suffix to distinguish actions/values coming from the target
          network.
"""

# noqa: E402
import argparse
import os
import os.path as osp
import sys
import time
from pathlib import Path, PurePath

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow.nn as nn
from gymnasium.utils import seeding
from tensorflow.keras.optimizers import Adam

from stable_learning_control.algos.common.buffers import (
    FiniteHorizonReplayBuffer,
    ReplayBuffer,
)
from stable_learning_control.algos.common.helpers import heuristic_target_entropy
from stable_learning_control.algos.tf2.common.get_lr_scheduler import get_lr_scheduler
from stable_learning_control.algos.tf2.common.helpers import (
    count_vars,
    full_model_summary,
    set_device,
)
from stable_learning_control.algos.tf2.policies.lyapunov_actor_critic import (
    LyapunovActorCritic,
)
from stable_learning_control.algos.tf2.policies.lyapunov_actor_twin_critic import (
    LyapunovActorTwinCritic,
)
from stable_learning_control.common.helpers import combine_shapes, get_env_id
from stable_learning_control.utils.eval_utils import test_agent
from stable_learning_control.utils.gym_utils import is_discrete_space, is_gym_env
from stable_learning_control.utils.log_utils.helpers import (
    log_to_std_out,
    setup_logger_kwargs,
)
from stable_learning_control.utils.log_utils.logx import EpochLogger
from stable_learning_control.utils.safer_eval_util import safer_eval
from stable_learning_control.utils.serialization_utils import save_to_json

# Script settings.
SCALE_LAMBDA_MIN_MAX = (
    0.0,
    1.0,
)  # Range of lambda Lagrance multiplier.
SCALE_ALPHA_MIN_MAX = (0.0, np.inf)  # Range of alpha Lagrance multiplier.
STD_OUT_LOG_VARS_DEFAULT = [
    "Epoch",
    "TotalEnvInteracts",
    "AverageEpRet",
    "AverageTestEpRet",
    "AverageTestEpLen",
    "AverageAlpha",
    "AverageLambda",
    "AverageLossAlpha",
    "AverageLossLambda",
    "AverageErrorL",
    "AverageLossPi",
    "AverageEntropy",
]

# tf.config.run_functions_eagerly(True)  # NOTE: Uncomment for debugging.


class LAC(tf.keras.Model):
    """The Lyapunov (soft) Actor-Critic (LAC) algorithm.

    Attributes:
        ac (tf.Module): The (lyapunov) actor critic module.
        ac_ (tf.Module): The (lyapunov) target actor critic module.
        log_alpha (tf.Variable): The temperature Lagrance multiplier.
        log_labda (tf.Variable): The Lyapunov Lagrance multiplier.
    """

    def __init__(
        self,
        env,
        actor_critic=None,
        ac_kwargs=dict(
            hidden_sizes={"actor": [256] * 2, "critic": [256] * 2},
            activation=nn.relu,
            output_activation={"actor": nn.relu},
        ),
        opt_type="minimize",
        alpha=0.99,
        alpha3=0.2,
        labda=0.99,
        gamma=0.99,
        polyak=0.995,
        target_entropy=None,
        adaptive_temperature=True,
        lr_a=1e-4,
        lr_c=3e-4,
        device="cpu",
        name="LAC",
    ):
        """Initialise the LAC algorithm.

        Args:
            env (:obj:`gym.env`): The gymnasium environment the LAC is training in. This is
                used to retrieve the activation and observation space dimensions. This
                is used while creating the network sizes. The environment must satisfy
                the gymnasium API.
            actor_critic (tf.Module, optional): The constructor method for a
                TensorFlow Module with an ``act`` method, a ``pi`` module and several
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
                :class:`~stable_learning_control.algos.tf2.policies.lyapunov_actor_critic.LyapunovActorCritic`
            ac_kwargs (dict, optional): Any kwargs appropriate for the ActorCritic
                object you provided to LAC. Defaults to:

                =======================  ============================================
                Kwarg                    Value
                =======================  ============================================
                ``hidden_sizes_actor``    ``256 x 2``
                ``hidden_sizes_critic``   ``256 x 2``
                ``activation``            :class:`tf.nn.relu`
                ``output_activation``     :class:`tf.nn.relu`
                =======================  ============================================
            opt_type (str, optional): The optimization type you want to use. Options
                ``maximize`` and ``minimize``. Defaults to ``maximize``.
            alpha (float, optional): Entropy regularization coefficient (Equivalent to
                inverse of reward scale in the original SAC paper). Defaults to
                ``0.99``.
            alpha3 (float, optional): The Lyapunov constraint error boundary. Defaults
                to ``0.2``.
            labda (float, optional): The Lyapunov Lagrance multiplier. Defaults to
                ``0.99``.
            gamma (float, optional): Discount factor. (Always between 0 and 1.).
                Defaults to ``0.99`` per Haarnoja et al. 2018, not ``0.995`` as in
                Han et al. 2020.
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
            lr_c (float, optional): Learning rate used for the (lyapunov) critic.
                Defaults to ``1e-4``.
            device (str, optional): The device the networks are placed on (``cpu``
                or ``gpu``). Defaults to ``cpu``.

        .. attention::
            This class will behave differently when the ``actor_critic`` argument
            is set to the :class:`~stable_learning_control.algos.pytorch.policies.lyapunov_actor_twin_critic.LyapunovActorTwinCritic`.
            For more information see the :ref:`LATC <latc>` documentation.
        """  # noqa: E501, D301
        self._device = set_device(
            device
        )  # NOTE: Needs to be called before super().__init__() call.

        super().__init__(name=name)
        self._setup_kwargs = {
            k: v for k, v in locals().items() if k not in ["self", "__class__", "env"]
        }

        # Validate gymnasium env.
        # NOTE: The current implementation only works with continuous spaces.
        if not is_gym_env(env):
            raise ValueError("Env must be a valid gymnasium environment.")
        if is_discrete_space(env.action_space) or is_discrete_space(
            env.observation_space
        ):
            raise NotImplementedError(
                "The LAC algorithm does not yet support discrete observation/action "
                "spaces. Please open a feature/pull request on "
                "https://github.com/rickstaa/stable-learning-control/issues if you "
                "need this."
            )

        # Print out some information about the environment and algorithm.
        if hasattr(env.unwrapped.spec, "id"):
            log_to_std_out(
                "You are using the '{}' environment.".format(get_env_id(env)),
                type="info",
            )
        else:
            log_to_std_out(
                "You are using the '{}' environment.".format(
                    type(env.unwrapped).__name__
                ),
                type="info",
            )
        log_to_std_out("You are using the LAC algorithm.", type="info")
        log_to_std_out(
            "This agent is {}.".format(
                "minimizing the cost"
                if opt_type.lower() == "minimize"
                else "maximizing the return"
            ),
            type="info",
        )

        # Store algorithm parameters.
        self._act_dim = env.action_space.shape
        self._obs_dim = env.observation_space.shape
        self._adaptive_temperature = adaptive_temperature
        self._opt_type = opt_type
        self._polyak = polyak
        self._gamma = gamma
        self._alpha3 = alpha3
        self._lr_a = tf.Variable(lr_a, name="Lr_a")
        if self._adaptive_temperature:
            self._lr_alpha = tf.Variable(lr_a, name="Lr_alpha")
        self._lr_lag = tf.Variable(lr_a, name="Lr_lag")
        self._lr_c = tf.Variable(lr_c, name="Lr_c")
        if not isinstance(target_entropy, (float, int)):
            self._target_entropy = heuristic_target_entropy(env.action_space)
        else:
            self._target_entropy = target_entropy
        self._use_twin_critic = False

        # Create variables for the Lagrance multipliers.
        # NOTE: Clip at 1e-37 to prevent log_alpha/log_lambda from becoming -np.inf
        self.log_alpha = tf.Variable(
            tf.math.log(1e-37 if alpha < 1e-37 else alpha), name="log_alpha"
        )
        self.log_labda = tf.Variable(
            tf.math.log(1e-37 if labda < 1e-37 else labda), name="log_lambda"
        )

        # Get default actor critic if no 'actor_critic' was supplied
        actor_critic = LyapunovActorCritic if actor_critic is None else actor_critic

        # Create actor-critic module and target networks
        self.ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        self.ac_targ = actor_critic(
            env.observation_space,
            env.action_space,
            name="lyapunov_actor_critic_target",
            **ac_kwargs,
        )

        self._init_targets()

        # Create optimizers.
        # NOTE: We here optimize for log_alpha and log_labda instead of alpha and labda
        # because it is more numerically stable (see:
        # https://github.com/rail-berkeley/softlearning/issues/136)
        self._pi_optimizer = Adam(learning_rate=self._lr_a)
        self._pi_params = self.ac.pi.trainable_variables
        if self._adaptive_temperature:
            self._log_alpha_optimizer = Adam(learning_rate=self._lr_alpha)
        self._log_labda_optimizer = Adam(learning_rate=self._lr_lag)
        self._c_params = self.ac.L.trainable_variables
        self._c_optimizer = Adam(learning_rate=self._lr_c)

        # ==== LATC code ===============================
        # NOTE: Added here to reduce code duplication.
        # Ensure parameters of both critics are used by the critic optimizer.
        if isinstance(self.ac, LyapunovActorTwinCritic):
            self._use_twin_critic = True
            self._c_params = (
                self.ac.L.trainable_variables + self.ac.L2.trainable_variables
            )  # Chain parameters of the two Q-critics
            self._c_optimizer = Adam(learning_rate=self._lr_c)
        # ==============================================

    @tf.function
    def call(self, s, deterministic=False):
        """Wrapper around the :meth:`get_action` method that enables users to also
        receive actions directly by invoking ``LAC(observations)``.

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
        # Make sure s is float32 TensorFlow tensor
        if not isinstance(s, tf.Tensor):
            s = tf.convert_to_tensor(s, dtype=tf.float32)
        elif s.dtype != tf.float32:
            s = tf.cast(s, dtype=tf.float32)

        return self.ac.act(s, deterministic)

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
        # Check if expected cumulative finite-horizon reward is supplied.
        if "horizon_rew" in data.keys():
            v = data["horizon_rew"]
        ################################################
        # Optimize (Lyapunov/Q) critic #################
        ################################################
        if self._opt_type.lower() == "maximize":
            raise NotImplementedError(
                "The LAC algorithm does not yet support maximization "
                "environments. Please open a feature/pull request on "
                "https://github.com/rickstaa/stable-learning-control/issues "
                "if you need this."
            )

        # Get target Lyapunov value (Bellman-backup)
        pi_targ_, _ = self.ac_targ.pi(
            o_
        )  # NOTE: Target actions come from *current* *target* policy.
        l_pi_targ = self.ac_targ.L([o_, pi_targ_])

        # ==== LATC code ===============================
        # Retrieve second target Lyapunov value.
        # NOTE: Added here to reduce code duplication.
        if self._use_twin_critic:
            l_pi_targ2 = self.ac_targ.L2([o_, pi_targ_])
            l_pi_targ = tf.math.maximum(
                l_pi_targ, l_pi_targ2
            )  # Use max clipping to prevent underestimation bias.
        # ==============================================

        # Calculate Lyapunov target.
        # NOTE: Calculation depends on the chosen Lyapunov candidate.
        if "horizon_rew" in data.keys():  # Finite-horizon candidate (Han eq. 9).
            l_backup = v
        else:  # Inifinite-horizon candidate (Han eq. 8).
            l_backup = r + self._gamma * (1 - d) * l_pi_targ

        # Compute Lyapunov Critic error gradients.
        with tf.GradientTape() as l_tape:
            # Get current Lyapunov value.
            l1 = self.ac.L([o, a])

            # ==== LATC code ===============================
            # NOTE: Added here to reduce code duplication.
            # Retrieve second Lyapunov value.
            if self._use_twin_critic:
                l2 = self.ac.L2([o, a])
            # ==============================================

            # Calculate L-critic MSE loss against Bellman backup.
            # NOTE: The 0.5 multiplication factor was added to make the derivation
            # cleaner and can be safely removed without influencing the
            # minimization. We kept it here for consistency.
            l_error = 0.5 * tf.reduce_mean((l1 - l_backup) ** 2)  # See Han eq. 7

            # ==== LATC code ===============================
            # NOTE: Added here to reduce code duplication.
            # Add second Lyapunov *CRITIC error*.
            if self._use_twin_critic:
                l_error += 0.5 * tf.reduce_mean((l2 - l_backup) ** 2)
            # ==============================================

        c_grads = l_tape.gradient(l_error, self._c_params)
        self._c_optimizer.apply_gradients(zip(c_grads, self._c_params))

        l_info = dict(LVals=l1)

        # ==== LATC code ===============================
        # NOTE: Added here to reduce code duplication.
        # Add second Lyapunov value to info dict.
        if self._use_twin_critic:
            l_info.update({"LVals2": l2})
        # ==============================================

        diagnostics.update({**l_info, "ErrorL": l_error})
        ################################################
        # Optimize Gaussian actor ######################
        ################################################
        # Compute actor loss gradients.
        with tf.GradientTape() as a_tape:
            # Retrieve log probabilities of batch observations based on *current* policy
            _, logp_pi = self.ac.pi(o)

            # Get target lyapunov value.
            pi_, _ = self.ac.pi(o_)  # NOTE: Target actions come from *current* policy
            lya_l_ = self.ac.L([o_, pi_])

            # ==== LATC code ===============================
            # NOTE: Added here to reduce code duplication.
            # Retrieve second target Lyapunov value.
            if self._use_twin_critic:
                lya_l2_ = self.ac.L2([o_, pi_])
                lya_l_ = tf.math.maximum(
                    lya_l_, lya_l2_
                )  # Use max clipping to prevent underestimation bias.
            # ==============================================

            # Compute Lyapunov Actor error.
            l_delta = tf.reduce_mean(
                lya_l_ - tf.stop_gradient(l1) + self._alpha3 * r
            )  # See Han eq. 11

            # Calculate entropy-regularized policy loss
            a_loss = tf.stop_gradient(self.labda) * l_delta + tf.stop_gradient(
                self.alpha
            ) * tf.reduce_mean(
                logp_pi
            )  # See Han eq. 12

        a_grads = a_tape.gradient(a_loss, self._pi_params)
        self._pi_optimizer.apply_gradients(zip(a_grads, self._pi_params))

        pi_info = dict(LogPi=logp_pi, Entropy=-tf.reduce_mean(logp_pi))
        diagnostics.update({**pi_info, "LossPi": a_loss})
        ################################################
        # Optimize alpha (Entropy temperature) #########
        ################################################
        if self._adaptive_temperature:
            # Compute alpha loss gradients.
            with tf.GradientTape() as alpha_tape:
                # Calculate alpha loss.
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
        # Optimize labda (Lyapunov temperature) ########
        ################################################

        # Compute labda loss gradients.
        with tf.GradientTape() as lambda_tape:
            # Calculate labda loss.
            # NOTE: Log_labda was used in the lambda_loss function because using
            # lambda caused the gradients to vanish. This is caused since we
            # restrict lambda within a 0-1.0 range using the clamp function
            # (see #34). Using log_lambda also is more numerically stable.
            labda_loss = -tf.reduce_mean(
                self.log_labda * tf.stop_gradient(l_delta)
            )  # See formulas under Han eq. 14

        lambda_grads = lambda_tape.gradient(labda_loss, [self.log_labda])
        self._log_labda_optimizer.apply_gradients(zip(lambda_grads, [self.log_labda]))

        labda_info = dict(Lambda=self.labda)
        diagnostics.update({**labda_info, "LossLambda": labda_loss})
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
            :tf2:`keras/Model#save_weights`). The model should therefore be restored
            using the :meth:`tf.keras.Model.load_weights` method (see
            :tf2:`keras/Model#load_weights`). If
            you want to deploy the full model use the :meth:`export` method instead.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_path = path.joinpath(f"policy/{checkpoint_name}")
        try:
            self.save_weights(save_path)
        except Exception as e:
            raise Exception("LAC model could not be saved.") from e

        # Save additional information.
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
                the Lagrance multipliers. By fault ``False``.

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

        # Store initial values in order to ignore them when loading the weights.
        lr_a = self._lr_a.value()
        lr_alpha = self._lr_alpha.value()
        lr_lag = self._lr_lag.value()
        lr_c = self._lr_c.value()
        if not restore_lagrance_multipliers:
            log_alpha_init = self.log_alpha.value()
            log_labda_init = self.log_labda.value()

        try:
            self.load_weights(latest)
        except Exception as e:
            raise Exception(
                f"Something went wrong when trying to load model '{latest}'."
            ) from e

        # Make sure learning rates (and Lagrance multipliers) are not restored.
        self._lr_a.assign(lr_a)
        self._lr_alpha.assign(lr_alpha)
        self._lr_lag.assign(lr_lag)
        self._lr_c.assign(lr_c)
        if not restore_lagrance_multipliers:
            self.log_alpha.assign(log_alpha_init)
            self.log_labda.assign(log_labda_init)
            log_to_std_out("Restoring Lagrance multipliers.", type="info")
        else:
            log_to_std_out(
                "Keeping Lagrance multipliers at their initial value.", type="info"
            )

    def export(self, path):
        """Can be used to export the model in the ``SavedModel`` format such that it can
        be deployed to hardware.

        Args:
            path (str): The path where you want to export the policy too.
        """
        if tf.config.functions_run_eagerly():
            log_to_std_out(
                "Exporting the TensorFlow model is not supported in eager mode.",
                type="error",
            )
        else:
            # NOTE: Currently we only export the actor as this is what is Useful when
            # deploying the algorithm.
            obs_dummy = tf.random.uniform(
                combine_shapes(1, self._obs_dim), dtype=tf.float32
            )
            self.ac.pi.get_action(obs_dummy)  # Make sure the full graph was traced.
            self.ac.pi.save(osp.join(path, "tf2_save"))

    def build(self):
        """Function that can be used to build the full model structure such that it can
        be visualized using the `tf.keras.Model.summary()`. This is done by calling the
        build method of the parent class with the correct input shape.

        .. note::
            This is done by calling the build methods of the submodules.
        """
        super().build(combine_shapes(None, self._obs_dim))

    def summary(self):
        """Small wrapper around the :meth:`tf.keras.Model.summary()` method used to
        apply a custom format to the model summary.
        """  # noqa: D402
        if not self.built:  # Ensure the model is built.
            self.build()
        super().summary()

    def full_summary(self):
        """Prints a full summary of all the layers of the TensorFlow model"""
        if not self.built:  # Ensure the model is built.
            self.build()
        full_model_summary(self)

    def set_learning_rates(self, lr_a=None, lr_c=None, lr_alpha=None, lr_labda=None):
        """Adjusts the learning rates of the optimizers.

        Args:
            lr_a (float, optional): The learning rate of the actor optimizer. Defaults
                to ``None``.
            lr_c (float, optional): The learning rate of the (Lyapunov) Critic. Defaults
                to ``None``.
            lr_alpha (float, optional): The learning rate of the temperature optimizer.
                Defaults to ``None``.
            lr_labda (float, optional): The learning rate of the Lyapunov Lagrance
                multiplier optimizer. Defaults to ``None``.
        """
        if lr_a:
            self._pi_optimizer.lr.assign(lr_a)
        if lr_c:
            self._c_optimizer.lr.assign(lr_c)
        if self._adaptive_temperature:
            if lr_alpha:
                self._log_alpha_optimizer.lr.assign(lr_alpha)
        if lr_labda:
            self._log_labda_optimizer.lr.assign(lr_labda)

    @tf.function
    def _init_targets(self):
        """Updates the target network weights to the main network weights."""
        for ac_main, ac_targ in zip(self.ac.variables, self.ac_targ.variables):
            ac_targ.assign(ac_main)

    @tf.function
    def _update_targets(self):
        """Updates the target networks based on a Exponential moving average
        (Polyak averaging).
        """
        for ac_main, ac_targ in zip(self.ac.variables, self.ac_targ.variables):
            ac_targ.assign(self._polyak * ac_targ + (1 - self._polyak) * ac_main)

    @property
    def alpha(self):
        """Property used to clip :attr:`alpha` to be equal or bigger than ``0.0`` to
        prevent it from becoming nan when :attr:`log_alpha` becomes ``-inf``. For
        :attr:`alpha` no upper bound is used.
        """
        # NOTE: Clamping isn't needed when alpha max is np.inf due to the exponential.
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
    def labda(self):
        """Property used to clip :attr:`lambda` to be equal or bigger than ``0.0`` in
        order to prevent it from becoming ``nan`` when log_labda becomes -inf. Further
        we clip it to be lower or equal than ``1.0`` in order to prevent lambda from
        exploding when the the hyperparameters are chosen badly.
        """
        return tf.clip_by_value(tf.exp(self.log_labda), *SCALE_LAMBDA_MIN_MAX)

    @labda.setter
    def labda(self, set_val):
        """Property used to make sure labda and log_labda are related."""
        self.log_labda.assign(
            tf.convert_to_tensor(
                np.log(1e-37 if set_val < 1e-37 else set_val),
                dtype=self.log_labda.dtype,
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
    """Checks if the input arguments have valid values.

    Raises:
        ValueError: If a value is invalid.
    """
    if kwargs["update_after"] > kwargs["steps_per_epoch"]:
        raise ValueError(
            "You can not set 'update_after' bigger than the 'steps_per_epoch'. Please "
            "change this and try again."
        )


def lac(
    env_fn,
    actor_critic=None,
    ac_kwargs=dict(
        hidden_sizes={"actor": [256] * 2, "critic": [256] * 2},
        activation=nn.relu,
        output_activation=nn.relu,
    ),
    opt_type="minimize",
    max_ep_len=None,
    epochs=100,
    steps_per_epoch=2048,
    start_steps=0,
    update_every=100,
    update_after=1000,
    steps_per_update=100,
    num_test_episodes=10,
    alpha=0.99,
    alpha3=0.2,
    labda=0.99,
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
    horizon_length=0,
    device="cpu",
    logger_kwargs=dict(),
    save_freq=1,
    start_policy=None,
    export=False,
):
    """Trains the LAC algorithm in a given environment.

    Args:
        env_fn: A function which creates a copy of the environment. The environment
            must satisfy the gymnasium API.
        actor_critic (tf.Module, optional): The constructor method for a
            TensorFlow Module with an ``act`` method, a ``pi`` module and several ``Q``
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
            :class:`~stable_learning_control.algos.tf2.policies.lyapunov_actor_critic.LyapunovActorCritic`
        ac_kwargs (dict, optional): Any kwargs appropriate for the ActorCritic
            object you provided to LAC. Defaults to:

            =======================  ============================================
            Kwarg                    Value
            =======================  ============================================
            ``hidden_sizes_actor``    ``256 x 2``
            ``hidden_sizes_critic``   ``256 x 2``
            ``activation``            :class:`tf.nn.ReLU`
            ``output_activation``     :class:`tf.nn.ReLU`
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
        alpha3 (float, optional): The Lyapunov constraint error boundary. Defaults
            to ``0.2``.
        labda (float, optional): The Lyapunov Lagrance multiplier. Defaults to
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
        lr_c (float, optional): Learning rate used for the (lyapunov) critic. Defaults
            to ``1e-4``.
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
        horizon_length (int, optional): The length of the finite-horizon used for the
            Lyapunov Critic target. Defaults to ``0`` meaning the infinite-horizon
            bellman backup is used.
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

    Returns:
        (tuple): tuple containing:

            -   policy (:class:`LAC`): The trained actor-critic policy.
            -   replay_buffer (union[:class:`~stable_learning_control.algos.common.buffers.ReplayBuffer`, :class:`~stable_learning_control.algos.common.buffers.FiniteHorizonReplayBuffer`]):
                The replay buffer used during training.
    """  # noqa: E501, D301
    validate_args(**locals())

    # Retrieve hyperparameters while filtering out the logger_kwargs.
    hyper_param_dict = {k: v for k, v in locals().items() if k not in ["logger_kwargs"]}

    # Setup algorithm parameters.
    total_steps = steps_per_epoch * epochs
    env = env_fn()
    hyper_param_dict["env"] = env  # Add env to hyperparameters.

    # Validate gymnasium env.
    # NOTE: The current implementation only works with continuous spaces.
    if not is_gym_env(env):
        raise ValueError("Env must be a valid gymnasium environment.")
    if is_discrete_space(env.action_space) or is_discrete_space(env.observation_space):
        raise NotImplementedError(
            "The LAC algorithm does not yet support discrete observation/action "
            "spaces. Please open a feature/pull request on "
            "https://github.com/rickstaa/stable-learning-control/issues if you "
            "need this."
        )

    # Create test environment.
    if num_test_episodes != 0:
        test_env = env_fn()

    # Flatten observation space and get observation, action and reward space dimensions.
    # NOTE: Done to ensure the algorithm works with GoalEnv environments. See
    # https://robotics.farama.org/content/multi-goal_api/#goalenv.
    env = gym.wrappers.FlattenObservation(env)
    test_env = gym.wrappers.FlattenObservation(test_env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Setup logger.
    logger_kwargs["quiet"] = (
        logger_kwargs["quiet"] if "quiet" in logger_kwargs.keys() else False
    )
    logger_kwargs["verbose_vars"] = (
        logger_kwargs["verbose_vars"]
        if (
            "verbose_vars" in logger_kwargs.keys()
            and logger_kwargs["verbose_vars"] is not None
        )
        else STD_OUT_LOG_VARS_DEFAULT
    )  # NOTE: Done to ensure the stdout doesn't get cluttered.
    logger_kwargs["backend"] = "tf2"  # NOTE: Use TensorFlow TensorBoard backend.
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
    if logger_kwargs.get("use_wandb") and not logger_kwargs.get("wandb_run_name"):
        # Create wandb_run_name if wandb is used and no name is provided.
        logger_kwargs["wandb_run_name"] = PurePath(logger_kwargs["output_dir"]).parts[
            -1
        ]
    logger = EpochLogger(**logger_kwargs)

    # Retrieve max episode length.
    if max_ep_len is None:
        max_ep_len = env.env._max_episode_steps
    else:
        if max_ep_len > env.env._max_episode_steps:
            logger.log(
                (
                    f"You defined your 'max_ep_len' to be {max_ep_len} "
                    "while the environment 'max_episode_steps' is "
                    f"{env.env._max_episode_steps}. As a result the environment "
                    f"'max_episode_steps' has been increased to {max_ep_len}"
                ),
                type="warning",
            )
            env.env._max_episode_steps = max_ep_len
            if num_test_episodes != 0:
                test_env.env._max_episode_steps = max_ep_len
    if hyper_param_dict["max_ep_len"] is None:  # Store in hyperparameter dict.
        hyper_param_dict["max_ep_len"] = max_ep_len

    # Save experiment config to logger.
    logger.save_config(hyper_param_dict)  # Write hyperparameters to logger.

    # Get default actor critic if no 'actor_critic' was supplied
    actor_critic = LyapunovActorCritic if actor_critic is None else actor_critic

    # Ensure the environment is correctly seeded.
    # NOTE: Done here since we donote:n't want to seed on every env.reset() call.
    if seed is not None:
        env.np_random, _ = seeding.np_random(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        test_env.np_random, _ = seeding.np_random(seed)
        test_env.action_space.seed(seed)
        test_env.observation_space.seed(seed)

    # Set random seed for reproducible results.
    # NOTE: see https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed.  # noqa: E501
    # and https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism.  # noqa: E501
    # NOTE: Please note that seeding will slow down the training.
    if seed is not None:
        tf.keras.utils.set_random_seed(
            seed
        )  # Sets all random seeds for the program (Python, NumPy, and TensorFlow).
        os.environ["PYTHONHASHSEED"] = str(
            seed
        )  # Ensure python hashing is deterministic.
        tf.config.experimental.enable_op_determinism()  # Make ops deterministic.
        # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable for reproducibility.

    policy = LAC(
        env,
        actor_critic,
        ac_kwargs,
        opt_type,
        alpha,
        alpha3,
        labda,
        gamma,
        polyak,
        target_entropy,
        adaptive_temperature,
        lr_a,
        lr_c,
        device,
    )

    # Create learning rate schedulers.
    # NOTE: Alpha and labda currently use the same scheduler as the actor.
    lr_decay_ref_var = total_steps if lr_decay_ref.lower() == "steps" else epochs
    lr_a_scheduler = get_lr_scheduler(lr_decay_type, lr_a, lr_a_final, lr_decay_ref_var)
    lr_c_scheduler = get_lr_scheduler(lr_decay_type, lr_c, lr_c_final, lr_decay_ref_var)

    # Restore policy if supplied.
    if start_policy is not None:
        logger.log(f"Restoring model from '{start_policy}'.", type="info")
        try:
            policy.restore(start_policy)
            logger.log("Model successfully restored.", type="info")
        except Exception as e:
            err_str = e.args[0].lower().rstrip(".")
            logger.log(
                f"Training process has been terminated. Unable to restore the "
                f"'start_policy' from '{start_policy}'. Please ensure the "
                f"'start_policy' is correct and try again. Error details: {err_str}.",
                type="error",
            )
            sys.exit(0)

    # Initialize replay buffer.
    if horizon_length != 0:  # Finite-horizon.
        replay_buffer = FiniteHorizonReplayBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=replay_size,
            horizon_length=horizon_length,
        )
    else:  # Infinite-horizon.
        replay_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=replay_size,
        )

    logger.setup_tf_saver(policy)

    # Log model to TensorBoard.
    # NOTE: Needs to be done before policy.summary because of tf bug (see
    # https://github.com/tensorflow/tensorboard/issues/1961#issuecomment-1127031827)
    if use_tensorboard:
        logger.log_model_to_tb(
            policy.ac,
            input_to_model=[
                tf.convert_to_tensor(
                    np.array([env.observation_space.sample()]), dtype=tf.float32
                ),
                tf.convert_to_tensor(
                    np.array([env.action_space.sample()]), dtype=tf.float32
                ),
            ],
        )

    # Count variables and print network structure.
    var_counts = tuple(count_vars(module) for module in [policy.ac.pi, policy.ac.L])
    param_count_log_str = "Number of parameters: \t pi: %d, \t L: %d" % var_counts

    # ==== LATC code ===============================
    # NOTE: Added here to reduce code duplication.
    # Extend param count log string.
    if isinstance(policy.ac, LyapunovActorTwinCritic):
        param_count_log_str += ", \t L2: %d" % count_vars(policy.ac.L2)
    # ===============================================

    # Print network structure.
    logger.log(param_count_log_str, type="info")
    logger.log("Network structure:\n", type="info")
    policy.full_summary()

    # Setup diagnostics tb_write dict and store initial learning rates.
    diag_tb_log_list = [
        "ErrorL",
        "LossPi",
        "Alpha",
        "LossAlpha",
        "Lambda",
        "LossLambda",
        "Entropy",
    ]
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
        logger.log_to_tb(
            "Lr_labda",
            policy._log_labda_optimizer.lr.numpy(),
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
        if t > start_steps or start_steps == 0:
            a = policy.get_action(o)
        else:
            a = tf.convert_to_tensor(env.action_space.sample(), dtype=tf.float32)

        # Take step in the env.
        o_, r, d, truncated, _ = env.step(a.numpy())
        ep_ret += r
        ep_len += 1

        # Store experience to replay buffer.
        if isinstance(replay_buffer, FiniteHorizonReplayBuffer):
            replay_buffer.store(o, a, r, o_, d, truncated)
        else:
            replay_buffer.store(o, a, r, o_, d)

        # Make sure to update most recent observation!
        o = o_

        # End of trajectory handling.
        if d or truncated:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            ep_ret, ep_len = 0, 0

        # Update handling.
        # NOTE: Improved compared to Han et al. 2020. Previously, updates were based on
        # memory size, which only changed at terminal states.
        if (t + 1) >= update_after and ((t + 1) - update_after) % update_every == 0:
            for _ in range(steps_per_update):
                batch = replay_buffer.sample_batch(batch_size)
                update_diagnostics = policy.update(data=batch)
                logger.store(**update_diagnostics)  # Log diagnostics.

            # Step based learning rate decay.
            if lr_decay_ref.lower() == "step":
                lr_a_now = max(
                    lr_a_scheduler(t + 1), lr_a_final
                )  # Make sure lr is bounded above final lr.
                lr_c_now = max(
                    lr_c_scheduler(t + 1), lr_c_final
                )  # Make sure lr is bounded above final lr.
                policy.set_learning_rates(
                    lr_a=lr_a_now, lr_c=lr_c_now, lr_alpha=lr_a_now, lr_labda=lr_a_now
                )

            # SGD batch tb logging.
            if use_tensorboard and not tb_low_log_freq:
                logger.log_to_tb(keys=diag_tb_log_list, global_step=t)

        # End of epoch handling (Save model, test performance and log data)
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model.
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({"env": env}, itr=epoch)

            # Test the performance of the deterministic version of the agent.
            if num_test_episodes != 0:
                eps_ret, eps_len = test_agent(policy, test_env, num_test_episodes)
                logger.store(
                    TestEpRet=eps_ret,
                    TestEpLen=eps_len,
                    extend=True,
                )

            # Epoch based learning rate decay.
            if lr_decay_ref.lower() != "step":
                lr_a_now = max(
                    lr_a_scheduler(epoch), lr_a_final
                )  # Make sure lr is bounded above final.
                lr_c_now = max(
                    lr_c_scheduler(epoch), lr_c_final
                )  # Make sure lr is bounded above final.
                policy.set_learning_rates(
                    lr_a=lr_a_now, lr_c=lr_c_now, lr_alpha=lr_a_now, lr_labda=lr_a_now
                )

            # Log info about epoch.
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
                "Lr_labda",
                policy._log_labda_optimizer.lr.numpy(),
                tb_write=use_tensorboard,
                tb_prefix="LearningRates",
            )
            logger.log_tabular(
                "Alpha",
                average_only=True,
                tb_write=(use_tensorboard and tb_low_log_freq),
            )
            logger.log_tabular(
                "Lambda",
                average_only=True,
                tb_write=(use_tensorboard and tb_low_log_freq),
            )
            logger.log_tabular(
                "LossPi",
                average_only=True,
                tb_write=(use_tensorboard and tb_low_log_freq),
            )
            logger.log_tabular(
                "ErrorL",
                average_only=True,
                tb_write=(use_tensorboard and tb_low_log_freq),
            )
            if adaptive_temperature:
                logger.log_tabular(
                    "LossAlpha",
                    average_only=True,
                    tb_write=(use_tensorboard and tb_low_log_freq),
                )
            logger.log_tabular(
                "LossLambda",
                average_only=True,
                tb_write=(use_tensorboard and tb_low_log_freq),
            )
            logger.log_tabular("LVals", with_min_and_max=True)
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

    print("" if not logger_kwargs["quiet"] else "\n")
    logger.log(
        "Training finished after {}s".format(time.time() - start_time),
        type="info",
    )

    # Close environment and return policy and replay buffer.
    env.close()
    return policy, replay_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a LAC agent in a given environment."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="stable_gym:Oscillator-v1",
        help="the gymnasium env (default: stable_gym:Oscillator-v1)",
    )  # NOTE: Environment found in https://rickstaa.dev/stable-gym.
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
        help="hidden layer size of the lyapunov critic (default: 256)",
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
        "--alpha3",
        type=float,
        default=0.2,
        help="the Lyapunov constraint error boundary (default: 0.2)",
    )
    parser.add_argument(
        "--labda",
        type=float,
        default=0.99,
        help="the Lyapunov Lagrance multiplier (default: 0.99)",
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
            "'epoch' or 'step' (default: epoch)"
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
        "--horizon_length",
        type=int,
        default=0,
        help=(
            "length of the finite-horizon used for the Lyapunov Critic target ( "
            "Default: 0, meaning the infinite-horizon bellman backup is used)."
        ),
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0, help="the random seed (default: 0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "The device the networks are placed on: 'cpu' or 'gpu' (options: "
            "default: cpu)"
        ),
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

    # Parse logger related arguments.
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lac",
        help="the name of the experiment (default: lac)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress logging of diagnostics to stdout (default: False)",
    )
    parser.add_argument(
        "--verbose_fmt",
        type=str,
        default="line",
        help=(
            "log diagnostics stdout format (options: 'table' or 'line', default: "
            "line)"
        ),
    )
    parser.add_argument(
        "--verbose_vars",
        nargs="+",
        default=STD_OUT_LOG_VARS_DEFAULT,
        help=("a space separated list of the values you want to show on the stdout."),
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="how often (in epochs) the policy should be saved (default: 1)",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="use model checkpoints (default: False)",
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="use TensorBoard (default: False)",
    )
    parser.add_argument(
        "--tb_log_freq",
        type=str,
        default="low",
        help=(
            "the TensorBoard log frequency. Options are 'low' (Recommended: logs at "
            "every epoch) and 'high' (logs at every SGD update batch). Default is 'low'"
        ),
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="use Weights & Biases (default: False)",
    )
    parser.add_argument(
        "--wandb_job_type",
        type=str,
        default="train",
        help="the Weights & Biases job type (default: train)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stable-learning-control",
        help="the name of the wandb project (default: stable-learning-control)",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help=(
            "the name of the Weights & Biases group you want to assign the run to "
            "(defaults: None)"
        ),
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help=(
            "the name of the Weights & Biases run (defaults: None, which will be "
            "set to the experiment name)"
        ),
    )
    args = parser.parse_args()

    # Setup actor critic arguments.
    output_activation = {}
    output_activation["actor"] = safer_eval(args.act_out_a, backend="tf2")
    ac_kwargs = dict(
        hidden_sizes={
            "actor": [args.hid_a] * args.l_a,
            "critic": [args.hid_c] * args.l_c,
        },
        activation={
            "actor": safer_eval(args.act_a, backend="tf2"),
            "critic": safer_eval(args.act_c, backend="tf2"),
        },
        output_activation=output_activation,
    )

    # Setup output dir for logger and return output kwargs.
    logger_kwargs = setup_logger_kwargs(
        args.exp_name,
        args.seed,
        save_checkpoints=args.save_checkpoints,
        use_tensorboard=args.use_tensorboard,
        tb_log_freq=args.tb_log_freq,
        use_wandb=args.use_wandb,
        wandb_job_type=args.wandb_job_type,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        quiet=args.quiet,
        verbose_fmt=args.verbose_fmt,
        verbose_vars=args.verbose_vars,
    )
    logger_kwargs["output_dir"] = osp.abspath(
        osp.join(
            osp.dirname(osp.realpath(__file__)),
            f"../../../../../data/lac/{args.env.lower()}/runs/run_{int(time.time())}",
        )
    )

    lac(
        lambda: gym.make(args.env),
        actor_critic=LyapunovActorCritic,
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
        alpha3=args.alpha3,
        labda=args.labda,
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
        horizon_length=args.horizon_length,
        seed=args.seed,
        save_freq=args.save_freq,
        device=args.device,
        start_policy=args.start_policy,
        export=args.export,
        logger_kwargs=logger_kwargs,
    )
