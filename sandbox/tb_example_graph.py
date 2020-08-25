from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

import gym


def mlp(sizes, activation, output_activation=nn.Identity):
    """Create a multi-layered perceptron using pytorch."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Network1(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class Network2(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class Wrapper(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # build policy and value functions
        self.net1 = Network1(obs_dim, act_dim, hidden_sizes, activation)
        self.net2 = Network1(obs_dim, act_dim, hidden_sizes, activation)

    def forward(self, obs, act):

        # Perform a forward pass through all the networks and return the result
        q1 = self.net1(obs, act)
        q2 = self.net2(obs, act)
        return q1, q2


if __name__ == "__main__":

    # Create tensorboard writer
    writer = SummaryWriter("runs/exp-1")

    # Create environment
    env = gym.make("Pendulum-v0")

    # Create a wrapper network
    wrapper = Wrapper(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    )

    # Add combined graph to tensorboard
    writer.add_graph(
        wrapper,
        (
            torch.Tensor(env.observation_space.sample()),
            torch.Tensor(env.action_space.sample()),
        ),
    )

    # Close tensorboard writer
    writer.close()
