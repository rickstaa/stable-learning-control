"""Soft Actor-Critic (SAC) algorithm.

This module contains the Pytorch implementation of the SAC algorithm of
`Haarnoja et al. 2019 <https://arxiv.org/abs/1812.05905>`_.

.. note::
    Code Conventions:
        - We use a `_` suffix to distinguish the next state from the current state.
        - We use a `targ` suffix to distinguish actions/values coming from the target
          network.
"""

import argparse
import glob
import itertools
import os
import os.path as osp
import random
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path, PurePath

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.utils import seeding
from torch.optim import Adam

from stable_learning_control.algos.common.helpers import heuristic_target_entropy
from stable_learning_control.algos.pytorch.common.buffers import ReplayBuffer
from stable_learning_control.algos.pytorch.common.get_lr_scheduler import (
    get_lr_scheduler,
    estimate_step_learning_rate,
)
from stable_learning_control.algos.pytorch.common.helpers import (
    count_vars,
    retrieve_device,
)
from stable_learning_control.algos.pytorch.policies.soft_actor_critic import (
    SoftActorCritic,
)
from stable_learning_control.common.helpers import friendly_err, get_env_id
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
    "AverageLossAlpha",
    "AverageQ1Vals",
    "AverageQ2Vals",
    "AverageLossPi",
    "AverageEntropy",
]
VALID_DECAY_TYPES = ["linear", "exponential", "constant"]
VALID_DECAY_REFERENCES = ["step", "epoch"]
DEFAULT_DECAY_TYPE = "linear"
DEFAULT_DECAY_REFERENCE = "epoch"


class SAC(nn.Module):
    """The Soft Actor Critic algorithm.

    Attributes:
        ac (torch.nn.Module): The soft actor critic module.
        ac_ (torch.nn.Module): The target soft actor critic module.
        log_alpha (torch.Tensor): The temperature Lagrance multiplier.
    """

    def __init__(
        self,
        env,
        actor_critic=None,
        ac_kwargs=dict(
            hidden_sizes={"actor": [256] * 2, "critic": [256] * 2},
            activation={"actor": nn.ReLU, "critic": nn.ReLU},
            output_activation={"actor": nn.ReLU, "critic": nn.Identity},
        ),
        opt_type="maximize",
        alpha=0.99,
        gamma=0.99,
        polyak=0.995,
        target_entropy=None,
        adaptive_temperature=True,
        lr_a=1e-4,
        lr_c=3e-4,
        lr_alpha=1e-4,
        device="cpu",
    ):
        """Initialise the SAC algorithm.

        Args:
            env (:obj:`gym.env`): The gymnasium environment the SAC is training in. This is
                used to retrieve the activation and observation space dimensions. This
                is used while creating the network sizes. The environment must satisfy
                the gymnasium API.
            actor_critic (torch.nn.Module, optional): The constructor method for a
                Torch Module with an ``act`` method, a ``pi`` module and several
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
                :class:`~stable_learning_control.algos.pytorch.policies.soft_actor_critic.SoftActorCritic`
            ac_kwargs (dict, optional): Any kwargs appropriate for the ActorCritic
                object you provided to SAC. Defaults to:

                =======================  ============================================
                Kwarg                    Value
                =======================  ============================================
                ``hidden_sizes_actor``    ``64 x 2``
                ``hidden_sizes_critic``   ``128 x 2``
                ``activation``            :class:`torch.nn.ReLU`
                ``output_activation``     :class:`torch.nn.ReLU`
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

                where :math:`\\rho` is polyak (Always between 0 and 1, usually close to
                1.). In some papers :math:`\\rho` is defined as (1 - :math:`\\tau`)
                where :math:`\\tau` is the soft replacement factor. Defaults to
                ``0.995``.
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
            lr_c (float, optional): Learning rate used for the (Soft) critic.
                Defaults to ``1e-4``.
            lr_alpha (float, optional): Learning rate used for the entropy temperature.
                Defaults to ``1e-4``.
            device (str, optional): The device the networks are placed on (options:
                ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1``, etc.). Defaults to ``cpu``.
        """  # noqa: E501, D301
        super().__init__()
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
                "The SAC algorithm does not yet support discrete observation/action "
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
        log_to_std_out("You are using the SAC algorithm.", type="info")
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
        self._device = retrieve_device(device)
        self._adaptive_temperature = adaptive_temperature
        self._opt_type = opt_type
        self._polyak = polyak
        self._gamma = gamma
        self._lr_a = lr_a
        if self._adaptive_temperature:
            self._lr_alpha = lr_alpha
        self._lr_c = lr_c
        if not isinstance(target_entropy, (float, int)):
            self._target_entropy = heuristic_target_entropy(env.action_space)
        else:
            self._target_entropy = target_entropy

        # Create variables for the Lagrance multipliers.
        # NOTE: Clip at 1e-37 to prevent log_alpha/log_lambda from becoming -np.inf
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(1e-37 if alpha < 1e-37 else alpha), requires_grad=True)
        )

        # Get default actor critic if no 'actor_critic' was supplied
        actor_critic = SoftActorCritic if actor_critic is None else actor_critic

        # Create actor-critic module and target networks
        # NOTE: Pytorch currently uses kaiming initialization for the baises in the
        # future this will change to zero initialization
        # (https://github.com/pytorch/pytorch/issues/18182). This however does not
        # influence the results.
        self.ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(
            self._device
        )
        self.ac_targ = deepcopy(self.ac).to(self._device)

        # Freeze target networks with respect to optimizers (updates via polyak avg.)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Create optimizers.
        # NOTE: We here optimize for log_alpha instead of alpha because it is more
        # numerically stable (see:
        # https://github.com/rail-berkeley/softlearning/issues/136)
        # NOTE: The parameters() method returns a generator. This generator becomes
        # empty after you looped through all values. As a result, below we use a
        # lambda function to keep referencing the actual model parameters.
        self._pi_optimizer = Adam(self.ac.pi.parameters(), lr=self._lr_a)
        self._pi_params = lambda: self.ac.pi.parameters()
        if self._adaptive_temperature:
            self._log_alpha_optimizer = Adam([self.log_alpha], lr=self._lr_alpha)
        # List of parameters for both Q-networks (save this for convenience)
        self._c_params = lambda: itertools.chain(
            *[gen() for gen in [self.ac.Q1.parameters, self.ac.Q2.parameters]]
        )  # Chain parameters of the two Q-critics
        self._c_optimizer = Adam(self._c_params(), lr=self._lr_c)

    def forward(self, s, deterministic=False):
        """Wrapper around the :meth:`get_action` method that enables users to also
        receive actions directly by invoking ``SAC(observations)``.

        Args:
            s (numpy.ndarray): The current state.
            deterministic (bool, optional): Whether to return a deterministic action.
                Defaults to ``False``.

        Returns:
            numpy.ndarray: The current action.
        """
        return self.get_action(s, deterministic=deterministic)

    def get_action(self, s, deterministic=False):
        """Returns the current action of the policy.

        Args:
            s (numpy.ndarray): The current state.
            deterministic (bool, optional): Whether to return a deterministic action.
                Defaults to ``False``.

        Returns:
            numpy.ndarray: The current action.
        """
        return self.ac.act(
            torch.as_tensor(s, dtype=torch.float32, device=self._device), deterministic
        )

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
        self._c_optimizer.zero_grad()

        # Get target Q values (Bellman-backup)
        with torch.no_grad():
            pi_, logp_pi_ = self.ac.pi(
                o_
            )  # NOTE: Target actions coming from *current* policy

            # Get target Q values based on optimization type.
            q1_pi_targ = self.ac_targ.Q1(o_, pi_)
            q2_pi_targ = self.ac_targ.Q2(o_, pi_)
            if self._opt_type.lower() == "minimize":
                q_pi_targ = torch.max(
                    q1_pi_targ,
                    q2_pi_targ,
                )  # Use max clipping to prevent underestimation bias.
            else:
                q_pi_targ = torch.min(
                    q1_pi_targ, q2_pi_targ
                )  # Use min clipping to prevent overestimation bias.
            q_backup = r + self._gamma * (1 - d) * (q_pi_targ - self.alpha * logp_pi_)

        # Retrieve the current Q values.
        q1 = self.ac.Q1(o, a)
        q2 = self.ac.Q2(o, a)

        # Calculate Q-critic MSE loss against Bellman backup.
        loss_q1 = 0.5 * ((q1 - q_backup) ** 2).mean()  # See Haarnoja eq. 5
        loss_q2 = 0.5 * ((q2 - q_backup) ** 2).mean()
        q_loss = loss_q1 + loss_q2

        q_loss.backward()
        self._c_optimizer.step()

        q_info = dict(
            Q1Vals=q1.cpu().detach().numpy(), Q2Vals=q2.cpu().detach().numpy()
        )
        diagnostics.update({**q_info, "LossQ": q_loss.cpu().detach().numpy()})
        ################################################
        # Optimize Gaussian actor ######################
        ################################################
        self._pi_optimizer.zero_grad()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self._c_params():
            p.requires_grad = False

        # Retrieve log probabilities of batch observations based on *current* policy
        pi, logp_pi = self.ac.pi(o)

        # Retrieve current Q values.
        # NOTE: Actions come from *current* policy
        q1_pi = self.ac.Q1(o, pi)
        q2_pi = self.ac.Q2(o, pi)
        if self._opt_type.lower() == "minimize":
            q_pi = torch.max(
                q1_pi, q2_pi
            )  # Use max clipping to prevent underestimation bias.
        else:
            q_pi = torch.min(
                q1_pi, q2_pi
            )  # Use min clipping to prevent overestimation bias.

        # Calculate entropy-regularized policy loss
        if self._opt_type.lower() == "minimize":
            a_loss = (
                self.alpha.detach() * logp_pi + q_pi
            ).mean()  # Minimization version of Haarnoja eq. 7
        else:
            a_loss = (self.alpha.detach() * logp_pi - q_pi).mean()  # See Haarnoja eq. 7

        a_loss.backward()
        self._pi_optimizer.step()

        pi_info = dict(
            LogPi=logp_pi.cpu().detach().numpy(),
            Entropy=-torch.mean(logp_pi).cpu().detach().numpy(),
        )
        diagnostics.update({**pi_info, "LossPi": a_loss.cpu().detach().numpy()})

        # Q networks so you can optimize it at next SGD step.
        for p in self._c_params():
            p.requires_grad = True
        ################################################
        # Optimize alpha (Entropy temperature) #########
        ################################################
        if self._adaptive_temperature:
            self._log_alpha_optimizer.zero_grad()

            # Calculate alpha loss.
            alpha_loss = -(
                self.alpha * (logp_pi.detach() + self.target_entropy)
            ).mean()  # See Haarnoja eq. 17

            alpha_loss.backward()
            self._log_alpha_optimizer.step()

            alpha_info = dict(Alpha=self.alpha.cpu().detach().numpy())
            diagnostics.update(
                {**alpha_info, "LossAlpha": alpha_loss.cpu().detach().numpy()}
            )

        ################################################
        # Update target networks and return ############
        # diagnostics. #################################
        ################################################
        self._update_targets()
        return diagnostics

    def save(self, path):
        """Can be used to save the current model state.

        Args:
            path (str): The path where you want to save the policy.

        Raises:
            Exception: Raises an exception if something goes wrong during saving.
        """
        model_state_dict = self.state_dict()
        path = Path(path)

        save_path = path.joinpath("policy/model.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(model_state_dict, save_path)
        except Exception as e:
            raise Exception("SAC model could not be saved.") from e

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
        if osp.basename(path) != ["model.pt", "model.pth"]:
            load_path = glob.glob(
                osp.join(path, "**", "model_state.pt*"), recursive=True
            )
            if len(load_path) == 0:
                model_path = glob.glob(
                    osp.join(path, "**", "model.pt*"), recursive=True
                )
                if len(model_path) > 1:
                    raise Exception(
                        f"Only whole pickled models found in '{path}'. Using whole "
                        "pickled models as a starting point is currently not supported "
                        "as this method of loading is discouraged by the pytorch "
                        "documentation. Please supply a path that contains a "
                        "'model_state' dictionary and try again."
                    )
                else:
                    raise Exception(
                        f"No models found in '{path}'. Please check your policy restore"
                        "path and try again."
                    )
            elif len(load_path) > 1:
                raise Exception(
                    f"Multiple models found in path '{path}'. Please check your policy "
                    "restore path and try again."
                )
            load_path = load_path[0]

        restored_model_state_dict = torch.load(load_path, map_location=self._device)
        self.load_state_dict(
            restored_model_state_dict,
            restore_lagrance_multipliers,
        )
        self.ac.to(self._device)
        self.ac_targ.to(self._device)

    def export(self, path):
        """Can be used to export the model as a ``TorchScript`` such that it can be
        deployed to hardware.

        Args:
            path (str): The path where you want to export the policy too.

        Raises:
            NotImplementedError: Raised until the feature is fixed on the upstream.
        """
        # IMPROVE: Replace with TorchScript and Onyx versions when
        # https://github.com/pytorch/pytorch/issues/29843 and
        # https://github.com/onnx/onnx/issues/3033 are solved (see
        # https://pytorch.org/tutorials/advanced/cpp_export.html and
        # https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html)
        raise NotImplementedError(
            "The SAC Pytorch module could not be exported as a 'TorchScript' since the "
            "'torch.distributions.normal' method is not yet 'TorchScript' compatible. "
            "The feature will be implemented when this is added to the upstream "
            "see https://github.com/pytorch/pytorch/issues/29843 to follow progress on "
            "this."
        )

    def load_state_dict(self, state_dict, restore_lagrance_multipliers=True):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            restore_lagrance_multipliers (bool, optional): Whether you want to restore
                the Lagrance multipliers. By fault ``True``.
        """
        if (
            "alg_name" in self.state_dict()
            and "alg_name" in state_dict.keys()
            and self.state_dict()["alg_name"] != state_dict["alg_name"]
        ):
            raise ValueError(
                friendly_err(
                    "The supplied 'state_dict' could not be loaded onto the {} ".format(
                        self.state_dict()["alg_name"]
                    )
                    + "agent as it belongs to a {} agent. Please supply a {} ".format(
                        state_dict["alg_name"], self.state_dict()["alg_name"]
                    )
                    + "compatible 'state_dict' and try again."
                )
            )
        if self.state_dict().keys() != state_dict.keys():
            raise ValueError(
                friendly_err(
                    "The 'state_dict' you tried to load does not seem to be right. It "
                    "contains keys: \n\n{}\n\n while keys: \n\n{}\n\n keys ".format(
                        list(state_dict.keys()), list(self.state_dict().keys())
                    )
                    + "are expected for the '{}' model.".format(
                        self.state_dict()["alg_name"]
                    )
                )
            )
        if not restore_lagrance_multipliers:
            log_to_std_out(
                "Keeping Lagrance multipliers at their initial value.", type="info"
            )
            try:
                del state_dict["log_alpha"]
            except KeyError:
                pass
        else:
            log_to_std_out("Restoring Lagrance multipliers.", type="info")

        try:
            super().load_state_dict(state_dict, strict=False)
        except (AttributeError, RuntimeError) as e:
            raise type(e)(
                "The 'state_dict' could not be loaded successfully.",
            ) from e

    def state_dict(self):
        """Simple wrapper around the :meth:`torch.nn.Module.state_dict` method that
        saves the current class name. This is used to enable easy loading of the model.
        """
        state_dict = super().state_dict()
        state_dict["alg_name"] = (
            self.__class__.__name__
        )  # Save algorithm name state dict.
        return state_dict

    def bound_lr(self, lr_a_final=None, lr_c_final=None, lr_alpha_final=None):
        """Function that can be used to make sure the learning rate doesn't go beyond
        a lower bound.

        Args:
            lr_a_final (float, optional): The lower bound for the actor learning rate.
                Defaults to ``None``.
            lr_c_final (float, optional): The lower bound for the critic learning rate.
                Defaults to ``None``.
            lr_alpha_final (float, optional): The lower bound for the alpha Lagrance
                multiplier learning rate. Defaults to ``None``.
        """
        if lr_a_final is not None:
            if self._pi_optimizer.param_groups[0]["lr"] < lr_a_final:
                self._pi_optimizer.param_groups[0]["lr"] = lr_a_final
        if lr_c_final is not None:
            if self._c_optimizer.param_groups[0]["lr"] < lr_c_final:
                self._c_optimizer.param_groups[0]["lr"] = lr_c_final
        if lr_alpha_final is not None:
            if self._log_alpha_optimizer.param_groups[0]["lr"] < lr_alpha_final:
                self._log_alpha_optimizer.param_groups[0]["lr"] = lr_alpha_final

    def _update_targets(self):
        """Updates the target networks based on a Exponential moving average
        (Polyak averaging).
        """
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NOTE: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make
                # new tensors.
                p_targ.data.mul_(self._polyak)
                p_targ.data.add_((1 - self._polyak) * p.data)

    def _set_learning_rates(self, lr_a=None, lr_c=None, lr_alpha=None):
        """Can be used to manually adjusts the learning rates of the optimizers.

        Args:
            lr_a (float, optional): The learning rate of the actor optimizer. Defaults
                to ``None``.
            lr_c (float, optional): The learning rate of the (soft) Critic. Defaults
                to ``None``.
            lr_alpha (float, optional): The learning rate of the temperature optimizer.
                Defaults to ``None``.
        """
        if lr_a:
            self._pi_optimizer.param_groups[0]["lr"] = lr_a
        if lr_c:
            self._c_optimizer.param_groups[0]["lr"] = lr_c
        if self._adaptive_temperature:
            if lr_alpha:
                self._log_alpha_optimizer.param_groups[0]["lr"] = lr_alpha

    @property
    def alpha(self):
        """Property used to clip :attr:`alpha` to be equal or bigger than ``0.0`` to
        prevent it from becoming nan when :attr:`log_alpha` becomes ``-inf``. For
        :attr:`alpha` no upper bound is used.
        """
        # NOTE: Clamping isn't needed when alpha max is np.inf due to the exponential.
        return torch.clamp(self.log_alpha.exp(), *SCALE_ALPHA_MIN_MAX)

    @alpha.setter
    def alpha(self, set_val):
        """Property used to ensure :attr:`alpha` and :attr:`log_alpha` are related."""
        self.log_alpha.data = torch.as_tensor(
            np.log(1e-37 if set_val < 1e-37 else set_val),
            dtype=self.log_alpha.dtype,
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
        """The device the networks are placed on (options: ``cpu``, ``gpu``, ``gpu:0``,
        ``gpu:1``, etc.).
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


def sac(
    env_fn,
    actor_critic=None,
    ac_kwargs=dict(
        hidden_sizes={"actor": [256] * 2, "critic": [256] * 2},
        activation={"actor": nn.ReLU, "critic": nn.ReLU},
        output_activation={"actor": nn.ReLU, "critic": nn.Identity},
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
    lr_alpha=1e-4,
    lr_a_final=1e-10,
    lr_c_final=1e-10,
    lr_alpha_final=1e-10,
    lr_decay_type=DEFAULT_DECAY_TYPE,
    lr_a_decay_type=None,
    lr_c_decay_type=None,
    lr_alpha_decay_type=None,
    lr_decay_ref=DEFAULT_DECAY_REFERENCE,
    batch_size=256,
    replay_size=int(1e6),
    seed=None,
    device="cpu",
    logger_kwargs=dict(),
    save_freq=1,
    start_policy=None,
    export=False,
):
    """Trains the SAC algorithm in a given environment.

    Args:
        env_fn: A function which creates a copy of the environment. The environment
            must satisfy the gymnasium API.
        actor_critic (torch.nn.Module, optional): The constructor method for a
            Torch Module with an ``act`` method, a ``pi`` module and several
            ``Q`` or ``L`` modules. The ``act`` method and ``pi`` module should
            accept batches of observations as inputs, and the ``Q*`` and ``L``
            modules should accept a batch of observations and a batch of actions as
            inputs. When called, these modules should return:

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
            :class:`~stable_learning_control.algos.pytorch.policies.soft_actor_critic.SoftActorCritic`
        ac_kwargs (dict, optional): Any kwargs appropriate for the ActorCritic
            object you provided to SAC. Defaults to:

            =======================  ============================================
            Kwarg                    Value
            =======================  ============================================
            ``hidden_sizes_actor``    ``64 x 2``
            ``hidden_sizes_critic``   ``128 x 2``
            ``activation``            :class:`torch.nn.ReLU`
            ``output_activation``     :class:`torch.nn.ReLU`
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

            where :math:`\\rho` is polyak (Always between 0 and 1, usually close to 1.).
            In some papers :math:`\\rho` is defined as (1 - :math:`\\tau`) where
            :math:`\\tau` is the soft replacement factor. Defaults to ``0.995``.
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
        lr_alpha (float, optional): Learning rate used for the entropy temperature.
            Defaults to ``1e-4``.
        lr_a_final(float, optional): The final actor learning rate that is achieved
            at the end of the training. Defaults to ``1e-10``.
        lr_c_final(float, optional): The final critic learning rate that is achieved
            at the end of the training. Defaults to ``1e-10``.
        lr_alpha_final(float, optional): The final alpha learning rate that is
            achieved at the end of the training. Defaults to ``1e-10``.
        lr_decay_type (str, optional): The learning rate decay type that is used (options
            are: ``linear`` and ``exponential`` and ``constant``). Defaults to
            ``linear``. Can be overridden by the specific learning rate decay types.
        lr_a_decay_type (str, optional): The learning rate decay type that is used for
            the actor learning rate (options are: ``linear`` and ``exponential`` and
            ``constant``). If not specified, the general learning rate decay type is used.
        lr_c_decay_type (str, optional): The learning rate decay type that is used for
            the critic learning rate (options are: ``linear`` and ``exponential`` and
            ``constant``). If not specified, the general learning rate decay type is used.
        lr_alpha_decay_type (str, optional): The learning rate decay type that is used
            for the alpha learning rate (options are: ``linear`` and ``exponential``
            and ``constant``). If not specified, the general learning rate decay type is used.
        lr_decay_ref (str, optional): The reference variable that is used for decaying
            the learning rate (options: ``epoch`` and ``step``). Defaults to ``epoch``.
        batch_size (int, optional): Minibatch size for SGD. Defaults to ``256``.
        replay_size (int, optional): Maximum length of replay buffer. Defaults to
            ``1e6``.
        seed (int): Seed for random number generators. Defaults to ``None``.
        device (str, optional): The device the networks are placed on (options: ``cpu``,
            ``gpu``, ``gpu:0``, ``gpu:1``, etc.). Defaults to ``cpu``.
        logger_kwargs (dict, optional): Keyword args for EpochLogger.
        save_freq (int, optional): How often (in terms of gap between epochs) to save
            the current policy and value function.
        start_policy (str): Path of a already trained policy to use as the starting
            point for the training. By default a new policy is created.
        export (bool): Whether you want to export the model as a ``TorchScript`` such
            that it can be deployed on hardware. By default ``False``.

    Returns:
        (tuple): tuple containing:

            -   policy (:class:`SAC`): The trained actor-critic policy.
            -   replay_buffer (union[:class:`~stable_learning_control.algos.common.buffers.ReplayBuffer`, :class:`~stable_learning_control.algos.common.buffers.FiniteHorizonReplayBuffer`]):
                The replay buffer used during training.
    """  # noqa: E501, D301
    update_after = max(1, update_after)  # You can not update before the first step.
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
            "The SAC algorithm does not yet support discrete observation/action "
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
    use_wandb = logger_kwargs.get("use_wandb")
    if use_wandb and not logger_kwargs.get("wandb_run_name"):
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
    logger.save_config(hyper_param_dict)

    # Get default actor critic if no 'actor_critic' was supplied
    actor_critic = SoftActorCritic if actor_critic is None else actor_critic

    # Ensure the environment is correctly seeded.
    # NOTE: Done here since we don't want to seed on every env.reset() call.
    if seed is not None:
        env.np_random, _ = seeding.np_random(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        test_env.np_random, _ = seeding.np_random(seed)
        test_env.action_space.seed(seed)
        test_env.observation_space.seed(seed)

    # Set other random seed for reproducible policy results.
    if seed is not None:
        np.random.seed(seed)  # Ensure numpy is deterministic.
        torch.manual_seed(seed)  # Ensure pytorch is deterministic.
        random.seed(seed)  # Ensure python is deterministic.
        os.environ["PYTHONHASHSEED"] = str(
            seed
        )  # Ensure python hashing is deterministic.
        # torch.use_deterministic_algorithms(True)  # Disable for reproducibility.
        # torch.backends.cudnn.benchmark = False  # Disable for reproducibility.

    policy = SAC(
        env=env,
        actor_critic=actor_critic,
        ac_kwargs=ac_kwargs,
        opt_type=opt_type,
        alpha=alpha,
        gamma=gamma,
        polyak=polyak,
        target_entropy=target_entropy,
        adaptive_temperature=adaptive_temperature,
        lr_a=lr_a,
        lr_c=lr_c,
        lr_alpha=lr_alpha,
        device=device,
    )

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

    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=replay_size,
        device=policy.device,
    )

    # Count variables and print network structure.
    var_counts = tuple(
        count_vars(module) for module in [policy.ac.pi, policy.ac.Q1, policy.ac.Q2]
    )
    logger.log(
        "Number of parameters: \t pi: %d, \t Q1: %d, \t Q2: %d\n" % var_counts,
        type="info",
    )
    logger.log("Network structure:\n", type="info")
    logger.log(policy.ac, end="\n\n")

    # Parse learning rate decay reference.
    lr_decay_ref = lr_decay_ref.lower()
    if lr_decay_ref not in VALID_DECAY_REFERENCES:
        options = [f"'{option}'" for option in VALID_DECAY_REFERENCES]
        logger.log(
            f"The learning rate decay reference variable was set to '{lr_decay_ref}', "
            "which is not a valid option. Valid options are "
            f"{', '.join(options)}. The learning rate decay reference "
            f"variable has been set to '{DEFAULT_DECAY_REFERENCE}'.",
            type="warning",
        )
        lr_decay_ref = DEFAULT_DECAY_REFERENCE

    # Parse learning rate decay types.
    lr_decay_type = lr_decay_type.lower()
    if lr_decay_type not in VALID_DECAY_TYPES:
        options = [f"'{option}'" for option in VALID_DECAY_TYPES]
        logger.log(
            f"The learning rate decay type was set to '{lr_decay_type}', which is not "
            "a valid option. Valid options are "
            f"{', '.join(options)}. The learning rate decay type has been set to "
            f"'{DEFAULT_DECAY_TYPE}'.",
            type="warning",
        )
        lr_decay_type = DEFAULT_DECAY_TYPE
    decay_types = {
        "actor": lr_a_decay_type.lower() if lr_a_decay_type else None,
        "critic": lr_c_decay_type.lower() if lr_c_decay_type else None,
        "alpha": lr_alpha_decay_type.lower() if lr_alpha_decay_type else None,
    }
    for name, decay_type in decay_types.items():
        if decay_type is None:
            decay_types[name] = lr_decay_type
        else:
            if decay_type not in VALID_DECAY_TYPES:
                logger.log(
                    f"Invalid {name} learning rate decay type: '{decay_type}'. Using "
                    f"global learning rate decay type: '{lr_decay_type}' instead.",
                    type="warning",
                )
                decay_types[name] = lr_decay_type
    lr_a_decay_type, lr_c_decay_type, lr_alpha_decay_type = decay_types.values()

    # Calculate the number of learning rate scheduler steps.
    if lr_decay_ref == "step":
        # NOTE: Decay applied at policy update to improve performance.
        lr_decay_steps = (
            total_steps - update_after
        ) / update_every + 1  # NOTE: +1 since we start at the initial learning rate.
    else:
        lr_decay_steps = epochs

    # Setup learning rate schedulers.
    lr_a_init, lr_c_init, lr_alpha_init = lr_a, lr_c, lr_alpha
    opt_schedulers = {
        "pi": get_lr_scheduler(
            policy._pi_optimizer,
            lr_a_decay_type,
            lr_a_init,
            lr_a_final,
            lr_decay_steps,
        ),
        "c": get_lr_scheduler(
            policy._c_optimizer,
            lr_c_decay_type,
            lr_c_init,
            lr_c_final,
            lr_decay_steps,
        ),
        "alpha": get_lr_scheduler(
            policy._log_alpha_optimizer,
            lr_alpha_decay_type,
            lr_alpha_init,
            lr_alpha_final,
            lr_decay_steps,
        ),
    }

    logger.setup_pytorch_saver(policy)

    # Log model to TensorBoard.
    if use_tensorboard:
        with warnings.catch_warnings():
            # NOTE: Suppress TracerWarning because our policy is non-deterministic.
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            logger.log_model_to_tb(
                policy.ac,
                input_to_model=(
                    torch.as_tensor(
                        env.observation_space.sample(), dtype=torch.float32
                    ),
                    torch.as_tensor(env.action_space.sample(), dtype=torch.float32),
                ),
            )

    # Log model to Weight & Biases.
    if use_wandb:
        logger.watch_model_in_wandb(policy.ac)

    # Setup diagnostics tb_write dict and store initial learning rates.
    diag_tb_log_list = ["LossQ", "LossPi", "Alpha", "LossAlpha", "Entropy"]
    if use_tensorboard:
        # NOTE: TensorBoard counts from 0.
        logger.log_to_tb(
            "Lr_a",
            policy._pi_optimizer.param_groups[0]["lr"],
            tb_prefix="LearningRates",
            global_step=0,
        )
        logger.log_to_tb(
            "Lr_c",
            policy._c_optimizer.param_groups[0]["lr"],
            tb_prefix="LearningRates",
            global_step=0,
        )
        logger.log_to_tb(
            "Lr_alpha",
            policy._log_alpha_optimizer.param_groups[0]["lr"],
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
            a = env.action_space.sample()

        # Take step in the env.
        o_, r, d, truncated, _ = env.step(a)
        ep_ret += r
        ep_len += 1

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
            if lr_decay_ref == "step":
                for scheduler in opt_schedulers.values():
                    scheduler.step()
                policy.bound_lr(
                    lr_a_final, lr_c_final, lr_alpha_final
                )  # Make sure lr is bounded above the final lr.

            # SGD batch tb logging.
            if use_tensorboard and not tb_low_log_freq:
                logger.log_to_tb(
                    keys=diag_tb_log_list, global_step=t
                )  # NOTE: TensorBoard counts from 0.

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

            # Retrieve current learning rates.
            if lr_decay_ref == "step":
                # NOTE: Estimate since 'step' decay is applied at policy update.
                lr_actor = estimate_step_learning_rate(
                    opt_schedulers["pi"],
                    lr_a_init,
                    lr_a_final,
                    update_after,
                    total_steps,
                    t + 1,
                )
                lr_critic = estimate_step_learning_rate(
                    opt_schedulers["c"],
                    lr_c_init,
                    lr_c_final,
                    update_after,
                    total_steps,
                    t + 1,
                )
                lr_alpha = estimate_step_learning_rate(
                    opt_schedulers["alpha"],
                    lr_alpha_init,
                    lr_alpha_final,
                    update_after,
                    total_steps,
                    t + 1,
                )
            else:
                lr_actor = policy._pi_optimizer.param_groups[0]["lr"]
                lr_critic = policy._c_optimizer.param_groups[0]["lr"]
                lr_alpha = policy._log_alpha_optimizer.param_groups[0]["lr"]

            # Log info about epoch.
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("TotalEnvInteracts", t + 1)
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
                lr_actor,
                tb_write=use_tensorboard,
                tb_prefix="LearningRates",
            )
            logger.log_tabular(
                "Lr_c",
                lr_critic,
                tb_write=use_tensorboard,
                tb_prefix="LearningRates",
            )
            logger.log_tabular(
                "Lr_alpha",
                lr_alpha,
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
            logger.dump_tabular(global_step=t)  # NOTE: TensorBoard counts from 0.

            # Epoch based learning rate decay.
            if lr_decay_ref != "step":
                for scheduler in opt_schedulers.values():
                    scheduler.step()
                policy.bound_lr(
                    lr_a_final, lr_c_final, lr_alpha_final
                )  # Make sure lr is bounded above the final lr.

    # Export model to 'TorchScript'
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
        description="Trains a SAC agent in a given environment."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Pendulum-v1",
        help="the gymnasium env (default: Pendulum-v1)",
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
        default="nn.ReLU",
        help="the hidden layer activation function of the actor (default: nn.ReLU)",
    )
    parser.add_argument(
        "--act_c",
        type=str,
        default="nn.ReLU",
        help="the hidden layer activation function of the critic (default: nn.ReLU)",
    )
    parser.add_argument(
        "--act_out_a",
        type=str,
        default="nn.ReLU",
        help="the output activation function of the actor (default: nn.ReLU)",
    )
    parser.add_argument(
        "--act_out_c",
        type=str,
        default="nn.Identity",
        help="the output activation function of the critic (default: nn.Identity)",
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
        "--lr_alpha",
        type=float,
        default=1e-4,
        help="entropy temperature learning rate (default: 1e-4)",
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
        "--lr_alpha_final",
        type=float,
        default=1e-10,
        help="the final entropy temperature learning rate (default: 1e-10)",
    )
    parser.add_argument(
        "--lr_decay_type",
        type=str,
        default="linear",
        help="the learning rate decay type (default: linear)",
    )
    parser.add_argument(
        "--lr_a_decay_type",
        type=str,
        default=None,
        help=(
            "the learning rate decay type that is used for the actor learning rate. "
            "If not specified, the general learning rate decay type is used."
        ),
    )
    parser.add_argument(
        "--lr_c_decay_type",
        type=str,
        default=None,
        help=(
            "the learning rate decay type that is used for the critic learning rate. "
            "If not specified, the general learning rate decay type is used."
        ),
    )
    parser.add_argument(
        "--lr_alpha_decay_type",
        type=str,
        default=None,
        help=(
            "the learning rate decay type that is used for the entropy temperature "
            "learning rate. If not specified, the general learning rate decay type is "
            "used."
        ),
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
        "--seed", "-s", type=int, default=0, help="the random seed (default: 0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "The device the networks are placed on. Options: 'cpu', 'gpu', 'gpu:0', "
            "'gpu:1', etc. Defaults to 'cpu'."
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
            "Whether you want to export the model as a 'TorchScript' such that "
            "it can be deployed on hardware (default: False)"
        ),
    )

    # Parse logger related arguments.
    parser.add_argument(
        "--exp_name",
        type=str,
        default="sac",
        help="the name of the experiment (default: sac)",
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
    output_activation["actor"] = safer_eval(args.act_out_a, backend="torch")
    output_activation["critic"] = safer_eval(args.act_out_c, backend="torch")
    ac_kwargs = dict(
        hidden_sizes={
            "actor": [args.hid_a] * args.l_a,
            "critic": [args.hid_c] * args.l_c,
        },
        activation={
            "actor": safer_eval(args.act_a, backend="torch"),
            "critic": safer_eval(args.act_c, backend="torch"),
        },
        output_activation=output_activation,
    )

    # Setup output dir for logger and return output kwargs.
    logger_kwargs = setup_logger_kwargs(
        args.exp_name,
        seed=args.seed,
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
            f"../../../../../data/sac/{args.env.lower()}/runs/run_{int(time.time())}",
        )
    )
    torch.set_num_threads(torch.get_num_threads())

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
        lr_alpha=args.lr_alpha,
        lr_a_final=args.lr_a_final,
        lr_c_final=args.lr_c_final,
        lr_alpha_final=args.lr_a_final,
        lr_decay_type=args.lr_decay_type,
        lr_a_decay_type=args.lr_a_decay_type,
        lr_c_decay_type=args.lr_c_decay_type,
        lr_alpha_decay_type=args.lr_alpha_decay_type,
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
