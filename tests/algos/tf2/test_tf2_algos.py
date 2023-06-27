"""Checks if the Tensorflow algorithms in this package are still working as expected
when run on the CPU.

This is done by comparing the current output of the algorithm with a previously saved
snapshot. We do this so that we can be sure that the algorithm still gives a
deterministic result when seeded after a change in the code.
"""
import importlib

import gymnasium as gym  # noqa: F401
import pytest
from gymnasium.utils import seeding

ALGOS = [
    "stable_learning_control.control.algos.tf2.lac.lac",
    "stable_learning_control.control.algos.tf2.sac.sac",
]


@pytest.mark.parametrize("algo", ALGOS)
class TestTF2Algos:
    env = gym.make("Pendulum-v1")  # Used because it is a simple environment.

    # Seed the environment.
    env.np_random, seed = seeding.np_random(0)
    env.action_space.seed(0)
    env.observation_space.seed(0)

    def test_reproducibility(self, algo, snapshot):
        """Checks if the algorithm is still working as expected."""
        # Check if tensorflow is available.
        if not importlib.util.find_spec("tensorflow"):
            pytest.skip(
                "Tensorflow not available. Please install it using `pip install .[tf]`."
            )

        # Import the algorithm run function.
        run = getattr(importlib.import_module(algo), algo.split(".")[-1])

        # Run the algorithm.
        trained_policy, replay_buffer = run(
            lambda: self.env,
            seed=0,
            epochs=1,
            update_after=400,
            steps_per_epoch=800,
            logger_kwargs=dict(verbose=False),
            device="cpu",
        )

        # Test if the replay buffer is the same.
        assert replay_buffer == snapshot

        # Test if the actions returned by the policy are the same.
        for _ in range(5):
            action = trained_policy.get_action(self.env.observation_space.sample())
            assert action.numpy() == snapshot
