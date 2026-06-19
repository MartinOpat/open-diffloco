"""Common environment protocol."""

from typing import Protocol


class Environment(Protocol):
    """Minimal interface expected by training algorithms."""

    obs_dim: int
    action_dim: int

    def reset(self, rng, difficulty=0.0):
        """Reset the environment."""
        ...

    def step(self, state, action):
        """Advance the environment by one control step."""
        ...
