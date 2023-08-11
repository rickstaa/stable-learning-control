"""Checks if the Torch algorithms in this package are still working as expected when run
on GPU.

This is done by comparing the current output of the algorithm with a previously saved
snapshot. We do this so that we can be sure that the algorithm still gives a
deterministic result when seeded after a change in the code.
"""
import importlib

import gymnasium as gym  # noqa: F401
import pytest
from gymnasium.utils import seeding

ALGOS = [
    "stable_learning_control.algos.pytorch.lac.lac",
    "stable_learning_control.algos.pytorch.latc.latc",
    "stable_learning_control.algos.pytorch.sac.sac",
]


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("device", ["gpu"])
class TestTorchAlgosGPU:
    @pytest.fixture
    def env(self):
        """Create Pendulum environment."""
        env = gym.make("Pendulum-v1")  # Used because it is a simple environment.

        # Seed the environment.
        env.np_random, _ = seeding.np_random(0)
        env.action_space.seed(0)
        env.observation_space.seed(0)

        return env

    def test_reproducibility(self, algo, device, snapshot, env):
        """Checks if the algorithm is still working as expected."""
        # Import the algorithm run function.
        run = getattr(importlib.import_module(algo), algo.split(".")[-1])

        # Run the algorithm.
        trained_policy, replay_buffer = run(
            lambda: env,
            seed=0,
            epochs=1,
            update_after=400,
            steps_per_epoch=800,
            logger_kwargs=dict(quiet=True),
            device=device,
        )

        # Test if the replay buffer is the same.
        assert replay_buffer == snapshot

        # Test if the actions returned by the policy are the same.
        for _ in range(5):
            action = trained_policy.get_action(env.observation_space.sample())
            assert action == snapshot
