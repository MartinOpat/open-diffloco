"""Shared neural network modules."""

from typing import Sequence

import jax.numpy as jp
import flax.linen as nn


class Actor(nn.Module):
    """Policy MLP with zero-initialized action head."""

    action_dim: int
    hidden: Sequence[int] = (512, 256, 128)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        x = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)

        return nn.tanh(x)


class Critic(nn.Module):
    """Value MLP."""

    hidden: Sequence[int] = (512, 256, 128)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        return nn.Dense(1)(x)


class LearnedDynamicsModel(nn.Module):
    """Predict normalized observation residual from obs/action."""

    obs_dim: int
    hidden: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs_norm, action):
        x = jp.concatenate([obs_norm, action], axis=-1)
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        return nn.Dense(self.obs_dim)(x)
