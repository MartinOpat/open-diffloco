"""Shared training data structures."""

from typing import Any

import jax
import jax.numpy as jp
import flax
import optax
from mujoco import mjx


@flax.struct.dataclass
class NormState:
    """Running statistics for observation normalization."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array


class Normalizer:
    """Welford online normalizer for observations."""

    def __init__(self, size: int, eps: float = 1e-4):
        self.size = size
        self.eps = eps

    def init(self) -> NormState:
        """Initialize normalizer state."""
        return NormState(
            mean=jp.zeros(self.size), var=jp.ones(self.size), count=jp.array(1e-4)
        )

    def update(self, state: NormState, x: jax.Array) -> NormState:
        """Update running statistics from a batch."""
        batch_mean = jp.mean(x, axis=0)
        batch_var = jp.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - state.mean
        total = state.count + batch_count
        new_mean = state.mean + delta * batch_count / total

        m_a = state.var * state.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jp.square(delta) * state.count * batch_count / total

        return state.replace(mean=new_mean, var=m2 / total, count=total)

    def normalize(self, state: NormState, x: jax.Array) -> jax.Array:
        """Normalize observations using running statistics."""
        return (x - state.mean) / jp.sqrt(state.var + self.eps)


@flax.struct.dataclass
class EnvState:
    """Batched environment state."""

    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict
    metrics: dict


@flax.struct.dataclass
class ReplayBuffer:
    """Fixed-capacity circular replay buffer."""

    obs: jax.Array
    action: jax.Array
    next_obs: jax.Array
    done: jax.Array
    ptr: jax.Array
    size: jax.Array


def init_replay_buffer(capacity: int, obs_dim: int, action_dim: int) -> ReplayBuffer:
    """Create an empty replay buffer with pre-allocated arrays."""
    return ReplayBuffer(
        obs=jp.zeros((capacity, obs_dim), dtype=jp.float32),
        action=jp.zeros((capacity, action_dim), dtype=jp.float32),
        next_obs=jp.zeros((capacity, obs_dim), dtype=jp.float32),
        done=jp.zeros((capacity,), dtype=jp.float32),
        ptr=jp.array(0, dtype=jp.int32),
        size=jp.array(0, dtype=jp.int32),
    )


def add_to_replay_buffer(
    buf: ReplayBuffer,
    obs: jax.Array,
    action: jax.Array,
    next_obs: jax.Array,
    done: jax.Array,
    capacity: int,
) -> ReplayBuffer:
    """Add a flat batch of transitions."""
    batch_size = obs.shape[0]
    indices = (buf.ptr + jp.arange(batch_size)) % capacity

    return buf.replace(
        obs=buf.obs.at[indices].set(obs.astype(jp.float32)),
        action=buf.action.at[indices].set(action.astype(jp.float32)),
        next_obs=buf.next_obs.at[indices].set(next_obs.astype(jp.float32)),
        done=buf.done.at[indices].set(done.astype(jp.float32)),
        ptr=(buf.ptr + batch_size) % capacity,
        size=jp.minimum(buf.size + batch_size, capacity),
    )


def sample_replay_buffer(buf: ReplayBuffer, key: jax.Array, batch_size: int) -> tuple:
    """Sample a uniform random batch."""
    indices = jax.random.randint(key, (batch_size,), 0, buf.size)
    return (
        buf.obs[indices],
        buf.action[indices],
        buf.next_obs[indices],
        buf.done[indices],
    )


@flax.struct.dataclass
class TrainState:
    """Training state shared by SHAC and JAVE."""

    key: jax.Array
    env_state: EnvState
    actor_params: Any
    critic_params: Any
    target_critic_params: Any
    normalizer: NormState
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    step: int
    critic_normalizer: Any = None
    ldm_params: Any = None
    ldm_opt: Any = None
    replay_buffer: Any = None
