"""Lyapunov (soft) actor twin critic policy.

This module contains a modified Pytorch implementation of the Lyapunov Actor-Critic
policy of `Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_. Like the original SAC
algorithm, this LAC variant uses two critics instead of one to mitigate a possible
underestimation bias, while the original LAC only uses one critic.
"""
import torch
import torch.nn as nn

# fmt: off
from stable_learning_control.algos.pytorch.policies.actors.squashed_gaussian_actor import \
    SquashedGaussianActor  # noqa: E501
# fmt: on
from stable_learning_control.algos.pytorch.policies.critics.L_critic import LCritic
from stable_learning_control.common.helpers import strict_dict_update
from stable_learning_control.utils.log_utils.helpers import log_to_std_out

HIDDEN_SIZES_DEFAULT = {"actor": (64, 64), "critic": (128, 128)}
ACTIVATION_DEFAULT = {"actor": nn.ReLU, "critic": nn.ReLU}
OUTPUT_ACTIVATION_DEFAULT = {
    "actor": nn.ReLU,
}


class LyapunovActorTwinCritic(nn.Module):
    """Lyapunov (soft) Actor-(twin Critic) network.

    Attributes:
        self.pi (:class:`~stable_learning_control.algos.pytorch.policies.actors.squashed_gaussian_actor.SquashedGaussianActor`):
            The squashed gaussian policy network (actor).
        self.L (:obj:`~stable_learning_control.algos.pytorch.policies.critics.L_critic.LCritic`):
            The first soft L-network (critic).
        self.L2 (:obj:`~stable_learning_control.algos.pytorch.policies.critics.L_critic.LCritic`):
            The second soft L-network (critic).
    """  # noqa: E501

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=HIDDEN_SIZES_DEFAULT,
        activation=ACTIVATION_DEFAULT,
        output_activation=OUTPUT_ACTIVATION_DEFAULT,
    ):
        """Initialise the LyapunovActorTwinCritic object.

        Args:
            observation_space (:obj:`gym.space.box.Box`): A gymnasium observation space.
            action_space (:obj:`gym.space.box.Box`): A gymnasium action space.
            hidden_sizes (Union[dict, tuple, list], optional): Sizes of the hidden
                layers for the actor. Defaults to ``(256, 256)``.
            activation (Union[:obj:`dict`, :obj:`torch.nn.modules.activation`], optional):
                The (actor and critic) hidden layers activation function. Defaults to
                :class:`torch.nn.ReLU`.
            output_activation (Union[:obj:`dict`, :obj:`torch.nn.modules.activation`], optional):
                The (actor and critic) output activation function. Defaults to
                :class:`torch.nn.ReLU` for the actor and nn.Identity for the critic.

        .. note::
            It is currently not possible to set the critic output activation function
            when using the LyapunovActorTwinCritic. This is since it by design requires the
            critic output activation to by of type :meth:`torch.square`.
        """  # noqa: E501
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # Parse hidden sizes, activation inputs arguments and action_limits
        hidden_sizes, _ = strict_dict_update(HIDDEN_SIZES_DEFAULT, hidden_sizes)
        activation, _ = strict_dict_update(ACTIVATION_DEFAULT, activation)
        output_activation, ignored = strict_dict_update(
            OUTPUT_ACTIVATION_DEFAULT, output_activation
        )
        act_limits = {"low": action_space.low, "high": action_space.high}

        if "critic" in ignored:
            log_to_std_out(
                (
                    "The critic output activation function was ignored since it can "
                    "not be set using the LyapunovActorTwinCritic architecture. This is "
                    "since it, by design, uses the 'torch.square' output activation "
                    "function."
                ),
                type="warning",
            )

        self.pi = SquashedGaussianActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes["actor"],
            activation=activation["actor"],
            output_activation=output_activation["actor"],
            act_limits=act_limits,
        )
        self.L = LCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes["critic"],
            activation=activation["critic"],
        )
        self.L2 = LCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes["critic"],
            activation=activation["critic"],
        )

    def forward(self, obs, act, deterministic=False, with_logprob=True):
        """Performs a forward pass through all the networks (Actor and both L critics).

        Args:
            obs (torch.Tensor): The tensor of observations.
            act (torch.Tensor): The tensor of actions.
            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If false the action is sampled from the stochastic
                policy. Defaults to ``False``.
            with_logprob (bool, optional): Whether we want to return the log probability
                of an action. Defaults to ``True``.

        Returns:
            (tuple): tuple containing:

                - pi_action (:obj:`torch.Tensor`): The actions given by the policy.
                - logp_pi (:obj:`torch.Tensor`): The log probabilities of each of these actions.
                - L (:obj:`torch.Tensor`): First critic L values.
                - L2 (:obj:`torch.Tensor`): Second critic L values.

        .. note::
            Useful for when you want to print out the full network graph using
            TensorBoard.
        """  # noqa: E501
        pi_action, logp_pi = self.pi(
            obs, deterministic=deterministic, with_logprob=with_logprob
        )
        L = self.L(obs, act)
        L2 = self.L2(obs, act)
        return pi_action, logp_pi, L, L2

    def act(self, obs, deterministic=False):
        """Returns the action from the current state given the current policy.

        Args:
            obs (torch.Tensor): The current observation (state).
            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If ``False`` the action is sampled from the
                stochastic policy. Defaults to ``False``.

        Returns:
            numpy.ndarray: The action from the current state given the current policy.
        """
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
