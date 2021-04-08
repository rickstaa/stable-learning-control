# NOTE: Version in which the Lyapunov constraint has been added to the critic loss
"""Lyapunov Actor-Critic algorithm (Version 2)

This module contains a improved version of the pytorch implementation of the LAC algorithm of
`Han et al. 2020 <http://arxiv.org/abs/2004.14288>`_.

.. note::
    Code Conventions:
        - We use a `_` suffix to distinguish the next state from the current state.
        - We use a `targ` suffix to distinguish actions/values coming from the target network.
"""  # noqa

import argparse
import glob
import os
import os.path as osp
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn as nn
from bayesian_learning_control.control.algos.pytorch.common import get_lr_scheduler
from bayesian_learning_control.control.algos.pytorch.common.buffers import ReplayBuffer
from bayesian_learning_control.control.algos.pytorch.common.helpers import (
    count_vars,
    retrieve_device,
)

# fmt: off
from bayesian_learning_control.control.algos.pytorch.policies.lyapunov_actor_critic import \
    LyapunovActorCritic  # noqa: E501
# fmt: on
from bayesian_learning_control.control.common.helpers import heuristic_target_entropy
from bayesian_learning_control.control.utils import safer_eval
from bayesian_learning_control.control.utils.eval_utils import test_agent
from bayesian_learning_control.control.utils.gym_utils import (
    is_discrete_space,
    is_gym_env,
)
from bayesian_learning_control.utils.import_utils import lazy_importer
from bayesian_learning_control.utils.log_utils import (
    EpochLogger,
    friendly_err,
    log_to_std_out,
    setup_logger_kwargs,
)
from bayesian_learning_control.utils.serialization_utils import (
    save_to_json,
)
from torch.optim import Adam

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
    "AverageEpLen",
    "AverageTestEpLen",
    "AverageAlpha",
    "AverageLambda",
    "AverageLossAlpha",
    "AverageLossLambda",
    "AverageLossPi",
    "AverageEntropy",
]


class LAC(nn.Module):
    """The Lyapunov actor critic algorithm.

    Attributes:
        ac (torch.nn.Module): The (lyapunov) actor critic module.
        ac_ (torch.nn.Module): The (lyapunov) target actor critic module.
        log_alpha (torch.Tensor): The temperature lagrance multiplier.
        log_labda (torch.Tensor): The Lyapunov lagrance multiplier.
        target_entropy (int): The target entropy.
        device (str): The device the networks are placed on (CPU or GPU).
    """

    def __init__(  # noqa: C901
        self,
        env,
        actor_critic=None,
        ac_kwargs=dict(
            hidden_sizes={"actor": [64] * 2, "critic": [128] * 2},
            activation=nn.ReLU,
            output_activation={"actor": nn.ReLU},
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
    ):
        """Lyapunov (soft) Actor-Critic (LAC)

        Args:
            env (:obj:`gym.env`): The gym environment the LAC is training in. This is
                used to retrieve the activation and observation space dimensions. This
                is used while creating the network sizes. The environment must satisfy
                the OpenAI Gym API.
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
                :class:`~bayesian_learning_control.control.algos.pytorch.policies.lyapunov_actor_critic.LyapunovActorCritic`
            ac_kwargs (dict, optional): Any kwargs appropriate for the ActorCritic
                object you provided to LAC. Defaults to:

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
            alpha3 (float, optional): The Lyapunov constraint error boundary. Defaults
                to ``0.2``.
            labda (float, optional): The Lyapunov lagrance multiplier. Defaults to
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
            lr_c (float, optional): Learning rate used for the (lyapunov) critic.
                Defaults to ``1e-4``.
            device (str, optional): The device the networks are placed on (``cpu``
                or ``gpu``). Defaults to ``cpu``.
        """  # noqa: E501
        super().__init__()
        self._setup_kwargs = {
            k: v for k, v in locals().items() if k not in ["self", "__class__", "env"]
        }

        # Validate gym env
        # NOTE: The current implementation only works with continuous spaces.
        if not is_gym_env(env):
            raise ValueError("Env must be a valid Gym environment.")
        if is_discrete_space(env.action_space) or is_discrete_space(
            env.observation_space
        ):
            raise NotImplementedError(
                "The LAC algorithm does not yet support discrete observation/action "
                "spaces. Please open a feature/pull request on "
                "https://github.com/rickstaa/bayesian-learning-control/issues if you "
                "need this."
            )

        log_to_std_out(
            "You are using the {} environment.".format(env.unwrapped.spec.id),
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

        # Store algorithm parameters
        self._act_dim = env.action_space.shape
        self._obs_dim = env.observation_space.shape
        self._device = retrieve_device(device)
        self._adaptive_temperature = adaptive_temperature
        self._opt_type = opt_type
        self._polyak = polyak
        self._gamma = gamma
        self._alpha3 = alpha3
        self._lr_a = lr_a
        if self._adaptive_temperature:
            self._lr_alpha = lr_a
        self._lr_lag = lr_a
        self._lr_c = lr_c
        if not isinstance(target_entropy, (float, int)):
            self._target_entropy = heuristic_target_entropy(env.action_space)
        else:
            self._target_entropy = target_entropy

        # Create variables for the Lagrance multipliers
        # NOTE: Clip at 1e-37 to prevent log_alpha/log_lambda from becoming -np.inf
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(1e-37 if alpha < 1e-37 else alpha), requires_grad=True)
        )
        self.log_labda = nn.Parameter(
            torch.tensor(np.log(1e-37 if labda < 1e-37 else labda), requires_grad=True)
        )

        # Get default actor critic if no 'actor_critic' was supplied
        actor_critic = LyapunovActorCritic if actor_critic is None else actor_critic

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

        # Create optimizers
        # NOTE: We here optimize for log_alpha and log_labda instead of alpha and labda
        # because it is more numerically stable (see:
        # https://github.com/rail-berkeley/softlearning/issues/136)
        # NOTE: The parameters() method returns a generator. This generator becomes
        # empty after you looped through all values. As a result, below we use a
        # lambda function to keep referencing the actual model parameters.
        self._pi_optimizer = Adam(self.ac.pi.parameters(), lr=self._lr_a)
        self._pi_params = lambda: self.ac.pi.parameters()
        if self._adaptive_temperature:
            self._log_alpha_optimizer = Adam([self.log_alpha], lr=self._lr_alpha)
        self._log_labda_optimizer = Adam([self.log_labda], lr=self._lr_lag)
        self._c_optimizer = Adam(self.ac.L.parameters(), lr=self._lr_c)
        self._c_params = lambda: self.ac.L.parameters()

    def forward(self, s, deterministic=False):
        """Wrapper around the :meth:`.get_action` method that enables users to also
        receive actions directly by invoking ``LAC(observations)``.

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
            torch.as_tensor(s, dtype=torch.float32).to(self._device), deterministic
        )

    def update(self, data):  # noqa: C901
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
        # Optimize (Lyapunov/Q) critic #################
        ################################################
        self._c_optimizer.zero_grad()
        if self._opt_type.lower() == "maximize":
            raise NotImplementedError(
                "The LAC algorithm does not yet support maximization "
                "environments. Please open a feature/pull request on "
                "https://github.com/rickstaa/bayesian-learning-control/issues "
                "if you need this."
            )

        # Get target Lyapunov value (Bellman-backup)
        with torch.no_grad():
            pi_targ_, _ = self.ac_targ.pi(
                o_
            )  # NOTE: Target actions come from *current* *target* policy
            l_pi_targ = self.ac_targ.L(o_, pi_targ_)
            l_backup = r + self._gamma * (1 - d) * l_pi_targ  # The Lyapunov candidate

        # Get current Lyapunov value
        l1 = self.ac.L(o, a)

        # Get target lyapunov value
        pi_, _ = self.ac.pi(o_)  # NOTE: Target actions come from *current* policy
        lya_l_ = self.ac.L(o_, pi_)

        # Compute Lyapunov Actor error
        l_delta = torch.mean(lya_l_ - l1.detach() + self._alpha3 * r)  # See Han eq. 11

        # Calculate Lyapunov *CRITIC* error
        # NOTE: The 0.5 multiplication factor was added to make the derivation
        # cleaner and can be safely removed without influencing the minimization. We
        # kept it here for consistency.
        # NOTE: We currently use a manual implementation instead of using F.mse_loss
        # as this is 2 times faster. This can be changed back to F.mse_loss if
        # Torchscript is used.
        # l_error = (
        #     0.5 * ((l1 - l_backup) ** 2 - self.labda.detach() * l_delta).mean()
        # )  # See Han eq. 7
        l_error = (
            0.5 * ((l1 - l_backup) ** 2 + self.labda.detach() * l_delta).mean()
        )  # See Han eq. 7

        l_error.backward()
        self._c_optimizer.step()

        l_info = dict(LVals=l1.cpu().detach().numpy())
        diagnostics.update({**l_info, "ErrorL": l_error.cpu().detach().numpy()})
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

        if self._opt_type.lower() == "maximize":
            raise NotImplementedError(
                "The LAC algorithm does not yet support maximization "
                "environments. Please open a feature/pull request on "
                "https://github.com/rickstaa/bayesian-learning-control/issues "
                "if you need this."
            )

        # Retrieve current L value
        # NOTE: Actions come from *current* policy
        l_pi = self.ac.L(o, pi)

        # Calculate entropy-regularized policy loss
        a_loss = (self.alpha.detach() * logp_pi + l_pi).mean()

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

            # Calculate alpha loss
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
        # Optimize labda (Lyapunov temperature) ########
        ################################################
        self._log_labda_optimizer.zero_grad()

        # Calculate labda loss
        # NOTE: Log_labda was used in the lambda_loss function because using lambda
        # caused the gradients to vanish. This is caused since we restrict lambda
        # within a 0-1.0 range using the clamp function (see #38). Using log_lambda
        # also is more numerically stable.
        labda_loss = -(
            self.log_labda * l_delta.detach()
        ).mean()  # See formulas under Han eq. 14

        labda_loss.backward()
        self._log_labda_optimizer.step()

        labda_info = dict(Lambda=self.labda.cpu().detach().numpy())
        diagnostics.update(
            {**labda_info, "LossLambda": labda_loss.cpu().detach().numpy()}
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
            raise Exception("LAC model could not be saved.") from e

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
            "The LAC Pytorch module could not be exported as a 'TorchScript' since the "
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
                the lagrance multipliers. By fault ``True``.
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
                "Keeping lagrance multipliers at their initial value.", type="info"
            )
            try:
                del state_dict["log_alpha"], state_dict["log_labda"]
            except KeyError:
                pass
        else:
            log_to_std_out("Restoring lagrance multipliers.", type="info")

        try:
            super().load_state_dict(state_dict, strict=False)
        except (AttributeError, RuntimeError) as e:
            raise type(e)(
                "The 'state_dict' could not be loaded successfully.",
            ) from e

    def state_dict(self):
        """Simple wrapper around the :meth:`torch.nn.Module.state_dict` method that saves
        the current class name. This is used to enable easy loading of the model.
        """
        state_dict = super().state_dict()
        state_dict[
            "alg_name"
        ] = self.__class__.__name__  # Save algorithm name state dict
        return state_dict

    def bound_lr(
        self, lr_a_final=None, lr_c_final=None, lr_alpha_final=None, lr_labda_final=None
    ):
        """Function that can be used to make sure the learning rate doesn't go beyond
        a lower bound.

        Args:
            lr_a_final (float, optional): The lower bound for the actor learning rate.
                Defaults to None.
            lr_c_final (float, optional): The lower bound for the critic learning rate.
                Defaults to None.
            lr_alpha_final (float, optional): The lower bound for the alpha Lagrance
                multiplier learning rate. Defaults to None.
            lr_labda_final (float, optional): The lower bound for the labda Lagrance
                multiplier learning rate. Defaults to None.
        """
        if lr_a_final is not None:
            if self._pi_optimizer.param_groups[0]["lr"] < lr_a_final:
                self._pi_optimizer.param_groups[0]["lr"] = lr_a_final
        if lr_c_final is not None:
            if self._c_optimizer.param_groups[0]["lr"] < lr_c_final:
                self._c_optimizer.param_groups[0]["lr"] = lr_c_final
        if lr_alpha_final is not None:
            if self._log_alpha_optimizer.param_groups[0]["lr"] < lr_a_final:
                self._log_alpha_optimizer.param_groups[0]["lr"] = lr_a_final
        if lr_labda_final is not None:
            if self._log_labda_optimizer.param_groups[0]["lr"] < lr_labda_final:
                self._log_labda_optimizer.param_groups[0]["lr"] = lr_labda_final

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

    def _set_learning_rates(self, lr_a=None, lr_c=None, lr_alpha=None, lr_labda=None):
        """Can be used to manually adjusts the learning rates of the optimizers.

        Args:
            lr_a (float, optional): The learning rate of the actor optimizer. Defaults
                to None.
            lr_c (float, optional): The learning rate of the (Lyapunov) Critic. Defaults
                to None.
            lr_alpha (float, optional): The learning rate of the temperature optimizer.
                Defaults to None.
            lr_labda (float, optional): The learning rate of the Lyapunov Lagrance
                multiplier optimizer. Defaults to None.
        """
        if lr_a:
            self._pi_optimizer.param_groups[0]["lr"] = lr_a
        if lr_c:
            self._c_optimizer.param_groups[0]["lr"] = lr_c
        if self._adaptive_temperature:
            if lr_alpha:
                self._log_alpha_optimizer.param_groups[0]["lr"] = lr_alpha
        if lr_labda:
            self._log_labda_optimizer.param_groups[0]["lr"] = lr_labda

    @property
    def alpha(self):
        """Property used to clip :attr:`alpha` to be equal or bigger than ``0.0`` to
        prevent it from becoming nan when :attr:`log_alpha` becomes ``-inf``. For
        :attr:`alpha` no upper bound is used.
        """
        return torch.clamp(self.log_alpha.exp(), *SCALE_ALPHA_MIN_MAX)

    @alpha.setter
    def alpha(self, set_val):
        """Property used to ensure :attr:`alpha` and :attr:`log_alpha` are related."""
        self.log_alpha.data = torch.as_tensor(
            np.log(1e-37 if set_val < 1e-37 else set_val),
            dtype=self.log_alpha.dtype,
        )

    @property
    def labda(self):
        """Property used to clip :attr:`lambda` to be equal or bigger than ``0.0`` in
        order to prevent it from becoming ``nan`` when log_labda becomes -inf. Further
        we clip it to be lower or equal than ``1.0`` in order to prevent lambda from
        exploding when the the hyperparameters are chosen badly.
        """
        return torch.clamp(self.log_labda.exp(), *SCALE_LAMBDA_MIN_MAX)

    @labda.setter
    def labda(self, set_val):
        """Property used to make sure labda and log_labda are related."""
        self.log_labda.data = torch.as_tensor(
            np.log(1e-37 if set_val < 1e-37 else set_val),
            dtype=self.log_labda.dtype,
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
            "https://github.com/rickstaa/bayesian-learning-control/issues if you need "
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


def lac4(  # noqa: C901
    env_fn,
    actor_critic=None,
    ac_kwargs=dict(
        hidden_sizes={"actor": [64] * 2, "critic": [128] * 2},
        activation=nn.ReLU,
        output_activation=nn.ReLU,
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
    device="cpu",
    logger_kwargs=dict(),
    save_freq=1,
    start_policy=None,
    export=False,
):
    """Trains the lac algorithm in a given environment.

    Args:
        env_fn: A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic (torch.nn.Module, optional): The constructor method for a
            Torch Module with an ``act`` method, a ``pi`` module and several
            ``Q`` or ``L`` modules. The ``act`` method and ``pi`` module should
            accept batches of observations as inputs, and the ``Q*`` and  ``L``
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
            :class:`~bayesian_learning_control.control.algos.pytorch.policies.lyapunov_actor_critic.LyapunovActorCritic`
        ac_kwargs (dict, optional): Any kwargs appropriate for the ActorCritic
            object you provided to LAC. Defaults to:

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
            performed for each gradient descent update. NOTE: This determins the
            ratio of env steps to gradient steps. Defaults to ``100`` which is
            equivalent to a env_step/grad_step ratio of 1.
        num_test_episodes (int, optional): Number of episodes used to test the
            deterministic policy at the end of each epoch. This is used for logging
            the performance. Defaults to ``10``.
        alpha (float, optional): Entropy regularization coefficient (Equivalent to
            inverse of reward scale in the original SAC paper). Defaults to
            ``0.99``.
        alpha3 (float, optional): The Lyapunov constraint error boundary. Defaults
            to ``0.2``.
        labda (float, optional): The Lyapunov lagrance multiplier. Defaults to
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
        lr_c (float, optional): Learning rate used for the (lyapunov) critic.
            Defaults to ``1e-4``.
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
        export (bool): Whether you want to export the model as a ``TorchScript`` such
            that it can be deployed on hardware. By default ``False``.
    """  # noqa: E501
    total_steps = steps_per_epoch * epochs

    validate_args(**locals())

    logger_kwargs["verbose_vars"] = (
        logger_kwargs["verbose_vars"]
        if (
            "verbose_vars" in logger_kwargs.keys()
            and logger_kwargs["verbose_vars"] is not None
        )
        else STD_OUT_LOG_VARS_DEFAULT
    )  # NOTE: Done to ensure the std_out doesn't get cluttered.
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

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    rew_dim = (
        env.reward_range.shape[0] if isinstance(env.reward_range, gym.spaces.Box) else 1
    )

    # Retrieve max episode length
    if max_ep_len is None:
        max_ep_len = env._max_episode_steps
    else:
        if max_ep_len > env._max_episode_steps:
            logger.log(
                (
                    f"You defined your 'max_ep_len' to be {max_ep_len} "
                    "while the environment 'max_epsiode_steps' is "
                    f"{env._max_episode_steps}. As a result the environment "
                    f"'max_episode_steps' has been increased to {max_ep_len}"
                ),
                type="warning",
            )
            env._max_episode_steps = max_ep_len

    # Get default actor critic if no 'actor_critic' was supplied
    actor_critic = LyapunovActorCritic if actor_critic is None else actor_critic

    # Set random seed for reproducible results
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)
        test_env.seed(seed)

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

    # Restore policy if supplied
    if start_policy is not None:
        logger.log(f"Restoring model from '{start_policy}'.", type="info")
        try:
            policy.restore(start_policy)
            logger.log("Model successfully restored.", type="info")
        except Exception as e:
            logger.log(
                (
                    "Shutting down training since {}.".format(
                        e.args[0].lower().rstrip(".")
                    )
                ),
                type="error",
            )
            sys.exit(0)

    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        rew_dim=rew_dim,
        size=replay_size,
        device=policy.device,
    )

    # Count variables and print network structure
    var_counts = tuple(count_vars(module) for module in [policy.ac.pi, policy.ac.L])
    logger.log("Number of parameters: \t pi: %d, \t L: %d\n" % var_counts, type="info")
    logger.log("Network structure:\n", type="info")
    logger.log(policy.ac, end="\n\n")

    # Create learning rate schedulers
    opt_schedulers = []
    lr_decay_ref_var = total_steps if lr_decay_ref.lower() == "steps" else epochs
    pi_opt_scheduler = get_lr_scheduler(
        policy._pi_optimizer, lr_decay_type, lr_a, lr_a_final, lr_decay_ref_var
    )
    opt_schedulers.append(pi_opt_scheduler)
    alpha_opt_scheduler = get_lr_scheduler(
        policy._log_alpha_optimizer, lr_decay_type, lr_a, lr_a_final, lr_decay_ref_var
    )
    opt_schedulers.append(alpha_opt_scheduler)
    c_opt_scheduler = get_lr_scheduler(
        policy._c_optimizer, lr_decay_type, lr_c, lr_c_final, lr_decay_ref_var
    )
    opt_schedulers.append(c_opt_scheduler)
    labda_opt_scheduler = get_lr_scheduler(
        policy._log_labda_optimizer,
        lr_decay_type,
        lr_a,
        lr_a_final,
        lr_decay_ref_var,
    )
    opt_schedulers.append(labda_opt_scheduler)

    logger.setup_pytorch_saver(policy)

    # Setup diagnostics tb_write dict and store initial learning rates
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
        logger.log_to_tb(
            "Lr_labda",
            policy._log_labda_optimizer.param_groups[0]["lr"],
            tb_prefix="LearningRates",
            global_step=0,
        )

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = policy.get_action(o)
        else:
            a = env.action_space.sample()

        # Take step in the env
        o_, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        replay_buffer.store(o, a, r, o_, d)

        # Make sure to update most recent observation!
        o = o_

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if (t + 1) >= update_after and ((t + 1) - update_after) % update_every == 0:

            # Step based learning rate decay
            if lr_decay_ref.lower() == "step":
                for scheduler in opt_schedulers:
                    scheduler.step()
                policy.bound_lr(
                    lr_a_final, lr_c_final, lr_a_final, lr_a_final
                )  # Make sure lr is bounded above the final lr

            for _ in range(steps_per_update):
                batch = replay_buffer.sample_batch(batch_size)
                update_diagnostics = policy.update(data=batch)

                # Log diagnostics
                logger.store(**update_diagnostics)

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
                for scheduler in opt_schedulers:
                    scheduler.step()
                policy.bound_lr(
                    lr_a_final, lr_c_final, lr_a_final, lr_a_final
                )  # Make sure lr is bounded above the final lr

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
                policy._pi_optimizer.param_groups[0]["lr"],
                tb_write=use_tensorboard,
                tb_prefix="LearningRates",
            )
            logger.log_tabular(
                "Lr_c",
                policy._c_optimizer.param_groups[0]["lr"],
                tb_write=use_tensorboard,
                tb_prefix="LearningRates",
            )
            logger.log_tabular(
                "Lr_alpha",
                policy._log_alpha_optimizer.param_groups[0]["lr"],
                tb_write=use_tensorboard,
                tb_prefix="LearningRates",
            )
            logger.log_tabular(
                "Lr_labda",
                policy._log_labda_optimizer.param_groups[0]["lr"],
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

    # Export model to 'TorchScript'
    if export:
        policy.export(logger.output_dir)

    print("" if logger_kwargs["verbose"] else "\n")
    logger.log(
        "Training finished after {}s".format(time.time() - start_time),
        type="info",
    )


if __name__ == "__main__":

    # Import gym environments
    import bayesian_learning_control.simzoo.simzoo.envs  # noqa: F401

    parser = argparse.ArgumentParser(
        description="Trains a LAC agent in a given environment."
    )
    parser.add_argument(
        "--env",
        type=str,
        # default="Oscillator-v1",
        default="CartPoleCost-v0",  # DEBUG
        help="the gym env (default: Oscillator-v1)",
    )
    parser.add_argument(
        "--hid_a",
        type=int,
        default=64,
        help="hidden layer size of the actor (default: 64)",
    )
    parser.add_argument(
        "--hid_c",
        type=int,
        default=128,
        help="hidden layer size of the lyapunov critic (default: 128)",
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
        help="the number of steps for each SGD update (default: 100)",
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
        help="the number of episodes for the performance analysis (default: 10)",
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
        help="the Lyapunov lagrance multiplier (default: 0.99)",
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
        default="cpu",
        help="The device the networks are placed on (default: cpu)",
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

    # Parse logger related arguments
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lac",
        help="the name of the experiment (default: lac)",
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
        help=("a space seperated list of the values you want to show on the std out."),
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
    output_activation["actor"] = safer_eval(args.act_out_a, backend="torch")
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
            f"../../../../../data/lac/{args.env.lower()}/runs/run_{int(time.time())}",
        )
    )
    torch.set_num_threads(torch.get_num_threads())

    lac4(
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
        seed=args.seed,
        save_freq=args.save_freq,
        device=args.device,
        start_policy=args.start_policy,
        export=args.export,
        logger_kwargs=logger_kwargs,
    )
