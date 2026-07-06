"""Visualization and evaluation utilities for trained Go2 policies."""

import os
import pickle
import json
import subprocess
import sys

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer

from src.core.data_structures import Normalizer
from src.core.networks import Actor
from src.core.utils import quat_inv, quat_rotate
from src.envs.go2.environment import Go2Env


def _load_env_kwargs(policy_path: str) -> dict:
    """Load Go2Env constructor kwargs from the policy run directory."""
    hparams_path = os.path.join(os.path.dirname(policy_path), "hparams.json")
    kwargs = {}

    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparams = json.load(f)

        if "xml_path" in hparams:
            kwargs["xml_path"] = hparams["xml_path"]
        if "env_variant" in hparams:
            kwargs["variant"] = hparams["env_variant"]
        kwargs["action_scale"] = hparams.get("action_scale", 1.0)
        kwargs["actor_history_len"] = hparams.get("actor_history_len", 10)

        for key in [
            "cmd_vel_x_range",
            "cmd_vel_y_range",
            "cmd_yaw_rate_range",
            "cmd_zero_prob",
        ]:
            if key in hparams:
                kwargs[key] = tuple(hparams[key])
        if "cmd_ctrl_interval_range" in hparams:
            kwargs["cmd_ctrl_interval_range"] = tuple(
                hparams["cmd_ctrl_interval_range"]
            )

        print(f"  Loaded hparams: action_scale={kwargs.get('action_scale')}")
        print(
            f"  Commands: vx={kwargs.get('cmd_vel_x_range', 'default')}, "
            f"vy={kwargs.get('cmd_vel_y_range', 'default')}, "
            f"yaw={kwargs.get('cmd_yaw_rate_range', 'default')}"
        )
    else:
        print(f"  WARNING: No hparams.json found at {hparams_path}, using defaults")

    return kwargs


def _quat_to_yaw(quat):
    """Extract yaw angle from quaternion [w, x, y, z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _draw_arrow(scene, pos, direction, rgba, shaft_width=0.008):
    """Add an arrow geom to a MjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return

    d = np.array(direction, dtype=np.float64)
    length = np.linalg.norm(d)
    if length < 1e-4:
        return
    d_hat = d / length

    if abs(d_hat[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, d_hat)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(d_hat, x_axis)

    mat = np.array(
        [
            x_axis[0],
            y_axis[0],
            d_hat[0],
            x_axis[1],
            y_axis[1],
            d_hat[1],
            x_axis[2],
            y_axis[2],
            d_hat[2],
        ],
        dtype=np.float64,
    )

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([shaft_width, shaft_width * 2.0, length], dtype=np.float64),
        np.array(pos, dtype=np.float64),
        mat,
        np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _draw_yaw_arrow(scene, pos, yaw_heading, yaw_rate_cmd, rgba):
    """Draw a tangential yaw-rate command arrow."""
    if abs(yaw_rate_cmd) < 0.05:
        return
    offset_r = 0.20
    scale = min(abs(yaw_rate_cmd) * 0.8, 0.6)

    fwd = np.array([np.cos(yaw_heading), np.sin(yaw_heading), 0.0])
    left = np.array([-np.sin(yaw_heading), np.cos(yaw_heading), 0.0])
    sign = 1.0 if yaw_rate_cmd > 0 else -1.0

    arrow_base = np.array(pos) + fwd * offset_r
    arrow_dir = left * scale * sign
    _draw_arrow(scene, arrow_base, arrow_dir, rgba, shaft_width=0.015)


def _draw_velocity_arrows(scene, qpos, cmd, local_linvel, local_angvel):
    """Draw command and measured velocity arrows on the robot."""
    pos = np.array(qpos[:3])
    quat = np.array(qpos[3:7])
    yaw = _quat_to_yaw(quat)

    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, -sy], [sy, cy]])
    arrow_z = pos[2] + 0.4
    vis_scale = 2.0

    cmd_xy_world = R @ np.array([cmd[0], cmd[1]])
    _draw_arrow(
        scene,
        [pos[0], pos[1], arrow_z],
        [cmd_xy_world[0] * vis_scale, cmd_xy_world[1] * vis_scale, 0.0],
        [0.2, 0.9, 0.2, 0.8],
        shaft_width=0.01,
    )

    vel_xy_world = R @ np.array([local_linvel[0], local_linvel[1]])
    _draw_arrow(
        scene,
        [pos[0], pos[1], arrow_z + 0.05],
        [vel_xy_world[0] * vis_scale, vel_xy_world[1] * vis_scale, 0.0],
        [0.2, 0.4, 1.0, 0.8],
        shaft_width=0.01,
    )

    _draw_yaw_arrow(
        scene, [pos[0], pos[1], arrow_z + 0.10], yaw, cmd[2], [1.0, 0.3, 0.2, 0.8]
    )


def _extract_local_vel(env_state):
    """Extract local-frame velocities from simulator state."""
    quat = env_state.data.qpos[3:7]
    inv_q = quat_inv(quat)
    return (
        quat_rotate(env_state.data.qvel[:3], inv_q),
        quat_rotate(env_state.data.qvel[3:6], inv_q),
    )


def _override_cmd(env_state, new_cmd):
    """Override the velocity command in an env state."""
    new_info = dict(env_state.info)
    new_info["cmd"] = jp.array(new_cmd)
    history = new_info["actor_obs_history"]
    cmd_start = 9 if history.shape[-1] > 45 else 6
    history = history.at[-1, cmd_start : cmd_start + 3].set(jp.array(new_cmd))
    new_info["actor_obs_history"] = history
    return env_state.replace(obs=history.reshape(-1), info=new_info)


def visualize_interactive(policy_path: str):
    """Open the interactive MuJoCo viewer."""
    print(f"Loading policy from {policy_path}")
    with open(policy_path, "rb") as f:
        state = pickle.load(f)

    env_kwargs = _load_env_kwargs(policy_path)
    env = Go2Env(**env_kwargs, max_episode_length=int(1e9))
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    actor = Actor(env.action_dim)
    norm = Normalizer(env.actor_frame_obs_dim)

    @jax.jit
    def get_action(params, norm_state, obs):
        obs_n = env.normalize_actor_obs(norm, norm_state, obs).astype(jp.float32)
        return actor.apply(params, obs_n).astype(jp.float64)

    @jax.jit
    def step_env(env_state, action):
        return env.step(env_state, action)

    jit_override = jax.jit(_override_cmd)

    @jax.jit
    def inject_xfrc(env_state, xfrc_applied):
        """Copy viewer perturbation forces into the JAX env state."""
        new_data = env_state.data.replace(xfrc_applied=xfrc_applied)
        return env_state.replace(data=new_data)

    rng = jax.random.PRNGKey(42)
    env_state = env.reset(rng)

    mj_data.qpos[:] = np.array(env_state.data.qpos)
    mj_data.qvel[:] = np.array(env_state.data.qvel)
    mujoco.mj_forward(mj_model, mj_data)

    cmd_step = 0.1
    vx_range = env.cmd_vel_x_range
    vy_range = env.cmd_vel_y_range
    yaw_range = env.cmd_yaw_rate_range

    viewer_state = {
        "env_state": env_state,
        "paused": False,
        "step_count": 0,
        "user_cmd": np.array([0.0, 0.0, 0.0]),
        "history": [],
        "vel_ema": np.array([0.0, 0.0, 0.0]),
    }

    def reset_env():
        """Reset environment to initial state."""
        nonlocal rng
        rng, key = jax.random.split(rng)
        viewer_state["env_state"] = env.reset(key)
        viewer_state["step_count"] = 0
        viewer_state["history"] = []
        viewer_state["vel_ema"][:] = 0.0
        mj_data.qpos[:] = np.array(viewer_state["env_state"].data.qpos)
        mj_data.qvel[:] = np.array(viewer_state["env_state"].data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        print("Environment reset")

    def key_callback(keycode):
        cmd = viewer_state["user_cmd"]
        if keycode == 32:  # Space
            viewer_state["paused"] = not viewer_state["paused"]
            print("Paused" if viewer_state["paused"] else "Running")
        elif keycode == 8:  # Backspace
            reset_env()
        elif keycode == 265:  # Up arrow
            cmd[0] = np.clip(cmd[0] + cmd_step, vx_range[0], vx_range[1])
        elif keycode == 264:  # Down arrow
            cmd[0] = np.clip(cmd[0] - cmd_step, vx_range[0], vx_range[1])
        elif keycode == 263:  # Left arrow
            cmd[1] = np.clip(cmd[1] + cmd_step, vy_range[0], vy_range[1])
        elif keycode == 262:  # Right arrow
            cmd[1] = np.clip(cmd[1] - cmd_step, vy_range[0], vy_range[1])
        elif keycode == 81:  # Q
            cmd[2] = np.clip(cmd[2] + cmd_step, yaw_range[0], yaw_range[1])
        elif keycode == 69:  # E
            cmd[2] = np.clip(cmd[2] - cmd_step, yaw_range[0], yaw_range[1])
        elif keycode == 88:  # X
            cmd[:] = 0.0

        if keycode in (265, 264, 263, 262, 81, 69, 88):
            print(f"  cmd: vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  yaw={cmd[2]:+.2f}")

    print("\n=== Interactive Omni-directional Viewer ===")
    print("Velocity command controls:")
    print("  ↑/↓   - Forward/backward (vx)")
    print("  ←/→   - Left/right (vy)")
    print("  Q/E   - Yaw left/right")
    print("  X     - Zero all commands")
    print("  Space - Pause | Backspace - Reset | Esc - Quit")
    print("  Ctrl + Left-click  - Apply force to body")
    print("  Ctrl + Right-click - Apply torque to body")
    print(f"\nCommand step: {cmd_step}")
    print(f"Ranges: vx={vx_range}, vy={vy_range}, yaw={yaw_range}")

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback
    ) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0, 0, 0.3]

        while viewer.is_running():
            if not viewer_state["paused"]:
                es = viewer_state["env_state"]
                es = jit_override(es, viewer_state["user_cmd"])

                xfrc_np = mj_data.xfrc_applied.copy()
                es = inject_xfrc(es, jp.array(xfrc_np))

                action = get_action(state.actor_params, state.normalizer, es.obs)
                es = step_env(es, action)
                viewer_state["env_state"] = es
                viewer_state["step_count"] += 1

                local_linvel, local_angvel = _extract_local_vel(es)
                cmd = viewer_state["user_cmd"]

                ema = viewer_state["vel_ema"]
                ema_alpha = 0.05
                ema[0] = ema_alpha * local_linvel[0] + (1 - ema_alpha) * ema[0]
                ema[1] = ema_alpha * local_linvel[1] + (1 - ema_alpha) * ema[1]
                ema[2] = ema_alpha * local_angvel[2] + (1 - ema_alpha) * ema[2]

                viewer_state["history"].append(
                    (
                        float(local_linvel[0]),
                        float(local_linvel[1]),
                        float(local_angvel[2]),
                        float(cmd[0]),
                        float(cmd[1]),
                        float(cmd[2]),
                    )
                )

                mj_data.qpos[:] = np.array(es.data.qpos)
                mj_data.qvel[:] = np.array(es.data.qvel)
                mujoco.mj_forward(mj_model, mj_data)

                try:
                    scn = viewer.user_scn
                    scn.ngeom = 0
                    _draw_velocity_arrows(scn, np.array(es.data.qpos), cmd, ema, ema)
                except (AttributeError, Exception):
                    pass

                if viewer_state["step_count"] % 100 == 0:
                    h = viewer_state["history"][-100:]
                    avg_vx = np.mean([r[0] for r in h])
                    avg_vy = np.mean([r[1] for r in h])
                    avg_yaw = np.mean([r[2] for r in h])
                    print(
                        f"Step {viewer_state['step_count']:5d}  "
                        f"vel=({avg_vx:+.2f}, {avg_vy:+.2f}, {avg_yaw:+.2f})  "
                        f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})"
                    )

                if es.done > 0.5:
                    print(f"  Episode terminated at step {viewer_state['step_count']}")

            viewer.sync()

    if viewer_state["history"]:
        h = np.array(viewer_state["history"])
        print(f"\n=== Session Summary ({viewer_state['step_count']} steps) ===")
        print(
            f"Mean actual:  vx={np.mean(h[:, 0]):+.3f}  vy={np.mean(h[:, 1]):+.3f}  yaw={np.mean(h[:, 2]):+.3f}"
        )
        print(
            f"Mean command: vx={np.mean(h[:, 3]):+.3f}  vy={np.mean(h[:, 4]):+.3f}  yaw={np.mean(h[:, 5]):+.3f}"
        )
        print(
            f"Mean error:   vx={np.mean(np.abs(h[:, 0] - h[:, 3])):.3f}  "
            f"vy={np.mean(np.abs(h[:, 1] - h[:, 4])):.3f}  "
            f"yaw={np.mean(np.abs(h[:, 2] - h[:, 5])):.3f}"
        )


def visualize(policy_path: str, num_steps: int = 1750, save_path: str = None):
    """Generate an evaluation video and tracking plot."""
    try:
        import mediapy as media
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mediapy"])
        import mediapy as media
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "matplotlib"]
        )
        import matplotlib.pyplot as plt

    print(f"Loading policy from {policy_path}")
    with open(policy_path, "rb") as f:
        state = pickle.load(f)

    env_kwargs = _load_env_kwargs(policy_path)
    env = Go2Env(**env_kwargs, max_episode_length=int(1e9))
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    actor = Actor(env.action_dim)
    norm = Normalizer(env.actor_frame_obs_dim)

    @jax.jit
    def get_action(params, norm_state, obs):
        obs_n = env.normalize_actor_obs(norm, norm_state, obs).astype(jp.float32)
        return actor.apply(params, obs_n).astype(jp.float64)

    @jax.jit
    def step_env(env_state, action):
        return env.step(env_state, action)

    jit_override = jax.jit(_override_cmd)

    n_segments = 7
    seg_len = num_steps // n_segments
    rng = jax.random.PRNGKey(7)

    commands = [
        np.array([0.0, 0.0, 0.0]),  # 1. Standing
        np.array([1.0, 0.0, 0.0]),  # 2. Forward only
        np.array([0.0, 0.3, 0.0]),  # 3. Lateral only
        np.array([0.0, 0.0, 0.8]),  # 4. Yaw only
        np.array([-0.3, 0.0, 0.0]),  # 5. Backward
        np.array([0.5, 0.0, -0.4]),  # 6. 'Random' combo
        np.array([0.0, 0.1, 0.3]),  # 7. 'Random' combo
    ]
    seg_boundaries = [i * seg_len for i in range(n_segments)] + [num_steps]

    labels = [
        "standing",
        "vx only",
        "vy only",
        "yaw only",
        "backwards",
        "random",
        "random",
    ]
    print(
        f"\nEvaluation rollout: {num_steps} steps = {num_steps * env.dt:.1f}s "
        f"({n_segments} segments, {seg_len * env.dt:.1f}s each)"
    )
    for i, c in enumerate(commands):
        t0 = seg_boundaries[i] * env.dt
        t1 = seg_boundaries[i + 1] * env.dt
        print(
            f"  [{t0:5.1f}s - {t1:5.1f}s] {labels[i]:>9s}: "
            f"vx={c[0]:+.2f}  vy={c[1]:+.2f}  yaw={c[2]:+.2f}"
        )

    rng, reset_key = jax.random.split(rng)
    env_state = env.reset(reset_key)
    frames = []
    tracking = {
        k: [] for k in ["time", "vx", "vy", "yaw", "cmd_vx", "cmd_vy", "cmd_yaw"]
    }
    vel_ema = np.array([0.0, 0.0, 0.0])

    for i in range(num_steps):
        seg_idx = min(i // seg_len, n_segments - 1)
        cmd = commands[seg_idx]

        env_state = jit_override(env_state, cmd)
        action = get_action(state.actor_params, state.normalizer, env_state.obs)
        env_state = step_env(env_state, action)

        local_linvel, local_angvel = _extract_local_vel(env_state)

        ema_alpha = 0.05
        vel_ema[0] = ema_alpha * local_linvel[0] + (1 - ema_alpha) * vel_ema[0]
        vel_ema[1] = ema_alpha * local_linvel[1] + (1 - ema_alpha) * vel_ema[1]
        vel_ema[2] = ema_alpha * local_angvel[2] + (1 - ema_alpha) * vel_ema[2]

        tracking["time"].append(i * env.dt)
        tracking["vx"].append(float(local_linvel[0]))
        tracking["vy"].append(float(local_linvel[1]))
        tracking["yaw"].append(float(local_angvel[2]))
        tracking["cmd_vx"].append(float(cmd[0]))
        tracking["cmd_vy"].append(float(cmd[1]))
        tracking["cmd_yaw"].append(float(cmd[2]))

        mj_data.qpos[:] = np.array(env_state.data.qpos)
        mj_data.qvel[:] = np.array(env_state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera="track")

        try:
            scn = renderer.scene
            _draw_velocity_arrows(
                scn, np.array(env_state.data.qpos), cmd, vel_ema, vel_ema
            )
        except (AttributeError, Exception):
            pass

        frames.append(renderer.render())

        if (i + 1) % seg_len == 0 or i == num_steps - 1:
            s = slice(max(0, i - seg_len + 1), i + 1)
            vx_e = np.mean(
                np.abs(np.array(tracking["vx"][s]) - np.array(tracking["cmd_vx"][s]))
            )
            vy_e = np.mean(
                np.abs(np.array(tracking["vy"][s]) - np.array(tracking["cmd_vy"][s]))
            )
            ya_e = np.mean(
                np.abs(np.array(tracking["yaw"][s]) - np.array(tracking["cmd_yaw"][s]))
            )
            print(
                f"  t={(i + 1) * env.dt:5.1f}s  MAE: vx={vx_e:.3f}  vy={vy_e:.3f}  yaw={ya_e:.3f}"
            )

    renderer.close()

    for k in tracking:
        tracking[k] = np.array(tracking[k])

    if save_path is None:
        save_path = policy_path.replace(".pkl", "_eval.mp4")

    media.write_video(save_path, frames, fps=int(1.0 / env.dt))
    print(f"\nVideo saved to {save_path}")

    _plot_eval_tracking(
        tracking, seg_boundaries, env.dt, save_path.replace(".mp4", "_tracking.png")
    )
    eval_summary_text = _print_eval_summary(tracking, n_segments=n_segments)

    summary_path = os.path.splitext(save_path)[0] + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(eval_summary_text.lstrip("\n") + "\n")
    print(f"Evaluation summary saved to {summary_path}")

    hparams_path = os.path.join(os.path.dirname(policy_path), "hparams.json")
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparams = json.load(f)
        hparams["evaluation_summary"] = eval_summary_text.lstrip("\n")
        hparams["evaluation_summary_path"] = summary_path
        with open(hparams_path, "w") as f:
            json.dump(hparams, f, indent=2)

    return tracking


def _plot_eval_tracking(tracking, seg_boundaries, dt, save_path):
    """Generate evaluation tracking plot: cmd vs actual for all axes."""
    import matplotlib.pyplot as plt

    t = tracking["time"]
    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)

    labels = ["stand", "vx", "vy", "yaw", "rnd", "rnd", "rnd"]

    configs = [
        ("vx", "cmd_vx", "Forward Velocity (m/s)"),
        ("vy", "cmd_vy", "Lateral Velocity (m/s)"),
        ("yaw", "cmd_yaw", "Yaw Rate (rad/s)"),
    ]

    for ax, (actual_key, cmd_key, title) in zip(axes, configs):
        actual = tracking[actual_key]
        cmd = tracking[cmd_key]

        ax.plot(t, cmd, "r--", lw=2, label="Command", alpha=0.8)
        ax.plot(t, actual, "b-", lw=1.5, label="Actual", alpha=0.8)
        ax.fill_between(t, cmd, actual, alpha=0.15, color="red")

        for b in seg_boundaries[1:-1]:
            ax.axvline(x=b * dt, color="gray", linestyle=":", alpha=0.5)

        ax.set_ylabel(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", alpha=0.2)

    for i, label in enumerate(labels[: len(seg_boundaries) - 1]):
        mid = (
            (seg_boundaries[i] + seg_boundaries[min(i + 1, len(seg_boundaries) - 1)])
            / 2
            * dt
        )
        axes[0].text(
            mid,
            axes[0].get_ylim()[1],
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
            fontstyle="italic",
        )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Omni-directional Velocity Tracking Evaluation", fontweight="bold", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Tracking plot saved to {save_path}")
    plt.close()


def _format_eval_summary(tracking, n_segments=7):
    """Format evaluation summary statistics as a multiline string."""
    vx_err = np.abs(tracking["vx"] - tracking["cmd_vx"])
    vy_err = np.abs(tracking["vy"] - tracking["cmd_vy"])
    yaw_err = np.abs(tracking["yaw"] - tracking["cmd_yaw"])
    total_err = np.sqrt(vx_err**2 + vy_err**2 + yaw_err**2)

    lines = [
        "",
        "=== Evaluation Summary ===",
        f"  Duration: {tracking['time'][-1]:.1f}s ({len(tracking['time'])} steps)",
        "  Mean Absolute Error:",
        f"    vx:   {np.mean(vx_err):.3f} m/s  (std {np.std(vx_err):.3f})",
        f"    vy:   {np.mean(vy_err):.3f} m/s  (std {np.std(vy_err):.3f})",
        f"    yaw:  {np.mean(yaw_err):.3f} rad/s (std {np.std(yaw_err):.3f})",
        f"    L2:   {np.mean(total_err):.3f}       (std {np.std(total_err):.3f})",
    ]

    labels = [
        "standing",
        "vx only",
        "vy only",
        "yaw only",
        "backwards",
        "random",
        "random",
    ]
    n = len(tracking["time"])
    seg_len = n // n_segments
    dt = tracking["time"][1] - tracking["time"][0] if n > 1 else 0.02
    for seg in range(n_segments):
        s = slice(seg * seg_len, min((seg + 1) * seg_len, n))
        seg_vx = np.mean(np.abs(tracking["vx"][s] - tracking["cmd_vx"][s]))
        seg_vy = np.mean(np.abs(tracking["vy"][s] - tracking["cmd_vy"][s]))
        seg_yaw = np.mean(np.abs(tracking["yaw"][s] - tracking["cmd_yaw"][s]))
        t0 = seg * seg_len * dt
        idx = seg * seg_len
        label = labels[seg] if seg < len(labels) else "random"
        lines.append(
            f"  Seg {seg + 1} ({label:>9s}, t={t0:5.1f}s): "
            f"cmd=({tracking['cmd_vx'][idx]:+.2f}, "
            f"{tracking['cmd_vy'][idx]:+.2f}, "
            f"{tracking['cmd_yaw'][idx]:+.2f})  "
            f"MAE=({seg_vx:.3f}, {seg_vy:.3f}, {seg_yaw:.3f})"
        )

    return "\n".join(lines)


def _print_eval_summary(tracking, n_segments=7):
    """Print evaluation summary statistics."""
    summary_text = _format_eval_summary(tracking, n_segments=n_segments)
    print(summary_text)
    return summary_text


def plot_training(log_path: str, save_path: str = None):
    """Plot training curves for velocity tracking."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "matplotlib"]
        )
        import matplotlib.pyplot as plt

    data = np.load(log_path)
    if data.ndim < 2 or data.shape[0] == 0:
        print(
            f"No training data to plot (log has {data.shape[0] if data.ndim >= 1 else 0} entries). Training was likely too short!"
        )
        return

    ncols = data.shape[1]
    print(f"Log has {data.shape[0]} entries, {ncols} columns")

    steps = data[:, 0]
    rewards = data[:, 1]
    vel_x = data[:, 2]
    vel_y = data[:, 3] if ncols > 3 else np.zeros_like(steps)
    yaw_rate = data[:, 4] if ncols > 4 else np.zeros_like(steps)
    cmd_x = data[:, 5] if ncols > 5 else np.zeros_like(steps)
    cmd_y = data[:, 6] if ncols > 6 else np.zeros_like(steps)
    cmd_yaw = data[:, 7] if ncols > 7 else np.zeros_like(steps)
    actor_loss = data[:, 8] if ncols > 8 else np.zeros_like(steps)
    contacts = data[:, 9] if ncols > 9 else np.zeros_like(steps)
    actor_grad = data[:, 10] if ncols > 10 else np.zeros_like(steps)
    critic_loss = data[:, 11] if ncols > 11 else np.zeros_like(steps)
    vg_loss = data[:, 17] if ncols > 17 else np.zeros_like(steps)

    if ncols > 14:
        track_vx = data[:, 12]
        track_vy = data[:, 13]
        track_yaw = data[:, 14]
    else:
        track_vx = (vel_x - cmd_x) ** 2
        track_vy = (vel_y - cmd_y) ** 2
        track_yaw = (yaw_rate - cmd_yaw) ** 2

    steps_k = steps / 1000

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    ax = axes[0, 0]
    ax.plot(steps_k, rewards, "g-", lw=1.5)
    ax.set_title("Reward", fontweight="bold")
    ax.grid(True, alpha=0.3)

    for idx, (squared_error, title) in enumerate(
        [
            (track_vx, "VX Squared Tracking Error"),
            (track_vy, "VY Squared Tracking Error"),
            (track_yaw, "Yaw Squared Tracking Error"),
        ]
    ):
        ax = axes[0, idx + 1]
        ax.plot(steps_k, squared_error, "b-", lw=1.5)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps_k, track_vx, "-", lw=1, label="vx err^2", alpha=0.7)
    ax.plot(steps_k, track_vy, "-", lw=1, label="vy err^2", alpha=0.7)
    ax.plot(steps_k, track_yaw, "-", lw=1, label="yaw err^2", alpha=0.7)
    l2_err = np.sqrt(track_vx + track_vy + track_yaw)
    ax.plot(steps_k, l2_err, "k-", lw=1.5, label="Combined RMSE", alpha=0.8)
    ax.set_title("Per-env Squared Tracking Error", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps_k, vg_loss, "purple", lw=1.5)
    ax.set_title("VG Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(steps_k, actor_grad, "red", lw=1.5)
    ax.set_title("Actor Gradient Norm", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 3]
    ax.plot(steps_k, critic_loss, "blue", lw=1.5)
    ax.set_title("Critic Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Steps (k)")

    plt.tight_layout()

    if save_path is None:
        save_path = log_path.replace(".npy", "_plot.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()

    last = min(20, len(steps))
    print(f"\n=== Training Summary ===")
    print(f"Steps: {int(steps[0])} -> {int(steps[-1])}")
    print(f"Final reward: {rewards[-1]:.3f} (best: {np.max(rewards):.3f})")
    print(f"Final per-env squared tracking error (last {last} logs):")
    print(f"  vx:  {np.mean(track_vx[-last:]):.3f}")
    print(f"  vy:  {np.mean(track_vy[-last:]):.3f}")
    print(f"  yaw: {np.mean(track_yaw[-last:]):.3f}")
    print(f"  Combined RMSE: {np.mean(l2_err[-last:]):.3f}")
    print(f"Final VG loss: {vg_loss[-1]:.4f}")
    print(f"Final actor grad:  {actor_grad[-1]:.2f}")
    print(f"Final critic loss: {critic_loss[-1]:.4f}")
