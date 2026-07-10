"""
Export trained Go2 policies to deployment .npz files.

Usage:
    python -m src.deploy.export_policy training_runs/<run>/policy_best.pkl

Output:
    training_runs/<run>/policy_best_deploy.npz

The exported .npz is consumed by the per-variant C++ deployment executables
(src/envs/go2/variants/<variant>/deploy_cpp). The environment variant is read
from the run's hparams.json, so variant-specific data (e.g. the kinematic
gait reference) is exported automatically.
"""

import sys
import os
import pickle
import json

import numpy as np

from src.envs.go2.environment import Go2Env


def export(policy_path: str):
    """Export policy checkpoint to portable .npz format."""

    print(f"Loading checkpoint: {policy_path}")
    with open(policy_path, "rb") as f:
        state = pickle.load(f)

    # Extract actor network weights
    params = state.actor_params
    p = params["params"]

    weights = {}
    n_hidden = 0
    while f"Dense_{n_hidden}" in p and f"LayerNorm_{n_hidden}" in p:
        weights[f"dense_{n_hidden}_kernel"] = np.array(p[f"Dense_{n_hidden}"]["kernel"])
        weights[f"dense_{n_hidden}_bias"] = np.array(p[f"Dense_{n_hidden}"]["bias"])
        weights[f"ln_{n_hidden}_scale"] = np.array(p[f"LayerNorm_{n_hidden}"]["scale"])
        weights[f"ln_{n_hidden}_bias"] = np.array(p[f"LayerNorm_{n_hidden}"]["bias"])
        n_hidden += 1

    # Output layer
    out_key = f"Dense_{n_hidden}"
    weights[f"dense_{n_hidden}_kernel"] = np.array(p[out_key]["kernel"])
    weights[f"dense_{n_hidden}_bias"] = np.array(p[out_key]["bias"])

    in_dim = weights["dense_0_kernel"].shape[0]
    out_dim = weights[f"dense_{n_hidden}_kernel"].shape[1]
    hidden_str = " -> ".join(
        str(weights[f"dense_{i}_kernel"].shape[1]) for i in range(n_hidden)
    )
    print(f"  Actor: {in_dim} -> {hidden_str} -> {out_dim}")
    print(f"  Hidden layers: {n_hidden} (each Dense + LayerNorm + ELU)")

    # Extract normalizer statistics
    norm = state.normalizer
    weights["norm_mean"] = np.array(norm.mean)
    weights["norm_var"] = np.array(norm.var)
    inferred_history_len = in_dim // len(weights["norm_mean"])
    print(
        f"  Normalizer: dim={len(weights['norm_mean'])}, count={float(norm.count):.0f}"
    )

    # Extract environment config from hparams.json
    hparams_path = os.path.join(os.path.dirname(policy_path), "hparams.json")
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hp = json.load(f)
        print(f"  Loaded hparams from {hparams_path}")
    else:
        print(f"  WARNING: No hparams.json found, using defaults")
        hp = {}

    env_kwargs = {}
    for key in [
        "action_scale",
        "cmd_vel_x_range",
        "cmd_vel_y_range",
        "cmd_yaw_rate_range",
    ]:
        if key in hp:
            env_kwargs[key] = hp[key]
    if "xml_path" in hp:
        env_kwargs["xml_path"] = hp["xml_path"]
    if "env_variant" in hp:
        env_kwargs["variant"] = hp["env_variant"]
    env_kwargs["actor_history_len"] = hp.get(
        "actor_history_len", inferred_history_len
    )

    variant = env_kwargs.get("variant", "blind_nolinvel_nokinref")
    print(f"  Variant: {variant}")

    env = Go2Env(**env_kwargs)
    if (
        len(weights["norm_mean"]) != env.actor_frame_obs_dim
        or in_dim != env.actor_obs_dim
    ):
        raise ValueError(
            "Checkpoint actor observations are incompatible with the "
            f"'{variant}' environment "
            f"(expected {env.actor_history_len}x{env.actor_frame_obs_dim}, "
            f"got {inferred_history_len}x{len(weights['norm_mean'])})"
        )

    weights["default_joints"] = np.array(env.default_joints)
    weights["action_scale"] = np.array(env.action_scale)
    weights["cmd_vel_x_range"] = np.array(env.cmd_vel_x_range)
    weights["cmd_vel_y_range"] = np.array(env.cmd_vel_y_range)
    weights["cmd_yaw_rate_range"] = np.array(env.cmd_yaw_rate_range)
    weights["dt"] = np.array(float(env.dt))
    weights["n_hidden"] = np.array(n_hidden)
    weights["actor_history_len"] = np.array(env.actor_history_len)
    weights["actor_frame_obs_dim"] = np.array(env.actor_frame_obs_dim)

    # Variant-specific: kinematic gait reference (blind_linvel_kinref)
    if hasattr(env, "gait_ref"):
        weights["gait_ref"] = np.array(env.gait_ref)
        print(
            f"  Gait reference: cycle_len={weights['gait_ref'].shape[0]}, "
            f"scale={float(getattr(env, 'gait_scale', float('nan')))}"
        )

    # Extract actuator gains (kp/kd) from MuJoCo model
    mj_model = env.mj_model
    n_act = mj_model.nu  # should be 12
    assert n_act == 12, f"Expected 12 actuators, got {n_act}"

    # gainprm[:,0] is the position gain for each actuator
    act_kp = np.array(mj_model.actuator_gainprm[:n_act, 0])
    # biasprm[:,2] is the velocity bias
    act_kd = np.abs(np.array(mj_model.actuator_biasprm[:n_act, 2]))

    # joint-level damping
    joint_damping = np.array(mj_model.dof_damping[6 : 6 + n_act])

    weights["actuator_kp"] = act_kp
    weights["actuator_kd"] = act_kd
    weights["joint_damping"] = joint_damping

    # Export kp/kd training ranges if available
    if "kp_range" in hp:
        weights["kp_range"] = np.array(hp["kp_range"])
        weights["kd_range"] = np.array(hp["kd_range"])
        print(f"\n  Gain randomization ranges (from training):")
        print(f"    kp_range = {hp['kp_range']}")
        print(f"    kd_range = {hp['kd_range']}")

    print(f"\n  Environment config:")
    print(f"    default_joints = {weights['default_joints']}")
    print(f"    action_scale   = {float(weights['action_scale'])}")
    print(f"    cmd_vel_x      = {weights['cmd_vel_x_range']}")
    print(f"    cmd_vel_y      = {weights['cmd_vel_y_range']}")
    print(f"    cmd_yaw_rate   = {weights['cmd_yaw_rate_range']}")
    print(f"    dt             = {weights['dt']}s ({1.0 / weights['dt']:.0f} Hz)")
    print(f"\n  Actuator gains (CRITICAL for deployment):")
    print(f"    actuator kp    = {act_kp}")
    print(f"    actuator kd    = {act_kd}")
    print(f"    joint damping  = {joint_damping}")
    print(
        f"    total damping  = {(act_kd + joint_damping)[0]:.1f} "
        f"(actuator kd + joint_damping)"
    )

    print(
        f"\n  WARNING: The C++ deployment clamps velocity commands to ranges "
        f"hard-coded in policy.hpp (defaults: vx +/-1.5, vy +/-1.0, yaw +/-1.5).\n"
        f"  Make sure they match the training ranges printed above before "
        f"deploying this policy."
    )

    # Save
    out_path = policy_path.replace(".pkl", "_deploy.npz")
    np.savez_compressed(out_path, **weights)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n  Saved: {out_path} ({size_kb:.1f} KB)")
    print(f"\n  Deploy with:")
    print(
        f"    ./build/src/envs/go2/variants/{variant}/deploy_cpp/deploy_{variant} \\\n"
        f"      --policy {out_path} --interface lo --domain-id 1   # sim"
    )
    print(
        f"    ./build/src/envs/go2/variants/{variant}/deploy_cpp/deploy_{variant} \\\n"
        f"      --policy {out_path} --interface eth0               # real robot"
    )

    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.deploy.export_policy <policy.pkl>")
        sys.exit(1)
    export(sys.argv[1])
