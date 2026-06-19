"""Implicit terrain helpers for Go2 training."""

import jax
import jax.numpy as jp


def sample_slope_gravity(
    rng,
    *,
    difficulty,
    slope_max_deg: float,
    flat_prob: float,
    g: float = 9.81,
):
    """Sample a per-episode tilted gravity vector."""
    k_az, k_flat = jax.random.split(rng, 2)
    theta = slope_max_deg * jp.pi / 180.0 * difficulty
    azimuth = jax.random.uniform(k_az, (), minval=0.0, maxval=2 * jp.pi)

    sin_t = jp.sin(theta)
    gravity = g * jp.array(
        [
            -sin_t * jp.cos(azimuth),
            -sin_t * jp.sin(azimuth),
            -jp.cos(theta),
        ]
    )

    flat = jax.random.uniform(k_flat) < flat_prob
    return jp.where(flat, jp.array([0.0, 0.0, -g]), gravity)


def differentiated_ou_foot_forces(
    foot_bump_ou,
    innovations,
    normal_forces,
    *,
    difficulty,
    std,
    decay,
    robot_weight,
):
    """Update dOU terrain bumps and return contact-scaled foot forces."""
    effective_std = std * difficulty
    next_ou = (1.0 - decay) * foot_bump_ou + effective_std * innovations.astype(
        jp.float64
    )
    delta = next_ou - foot_bump_ou
    delta = delta.at[:, 2].set(jp.maximum(delta[:, 2], 0.0))
    force_scale = normal_forces / jp.maximum(robot_weight, 1e-6)
    return next_ou, delta * force_scale[:, None]
