"""Go2 MJX environment with command tracking and domain randomization."""

import functools

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from src.core.utils import (
    quat_inv,
    quat_rotate,
    axis_angle_to_quat,
)
from src.core.data_structures import EnvState
from src.envs.go2.terrain import sample_slope_gravity


class _RewardData:
    """Minimal data view needed by Go2Env._compute_rewards."""

    def __init__(self, qpos):
        self.qpos = qpos


class Go2Env:
    """Go2 locomotion environment used by the training algorithms."""

    def __init__(
        self,
        xml_path: str = "src/envs/go2/models/scene_mjx.xml",
        action_scale: float = 0.5,
        # Commands
        cmd_vel_x_range: tuple = (-1.5, 1.5),
        cmd_vel_y_range: tuple = (-1.0, 1.0),
        cmd_yaw_rate_range: tuple = (-1.5, 1.5),
        cmd_zero_prob: tuple = (0.1, 0.1, 0.1),
        cmd_ctrl_interval_range: tuple = (60, 140),
        # Randomization
        friction_range: tuple = (0.5, 2.0),
        mass_range: tuple = (0.85, 1.15),
        kp_range: tuple = (25.0, 45.0),
        kd_range: tuple = (0.3, 0.7),
        com_offset_range: tuple = (0.05, 0.05, 0.04),
        terrain_flat_prob: float = 0.2,
        terrain_slope_max: float = 5.0,
        max_episode_length: int = 500,
        actor_history_len: int = 10,
    ):
        """
        Initialize Go2 environment.

        Args:
            xml_path: Path to MuJoCo XML model file
            action_scale: Scale factor for learned action commands
            cmd_vel_x_range: (min, max) for forward velocity command (m/s)
            cmd_vel_y_range: (min, max) for lateral velocity command (m/s)
            cmd_yaw_rate_range: (min, max) for yaw rate command (rad/s)
            cmd_zero_prob: Per-component probability of zeroing (vx, vy, yaw)
            cmd_ctrl_interval_range: (min, max) steps to hold each random command
            friction_range: (lo, hi) multiplicative factor for geom_friction
            mass_range: (lo, hi) multiplicative factor for body_mass & body_inertia
            kp_range: (lo, hi) absolute range for actuator position gain (per-episode)
            kd_range: (lo, hi) absolute range for actuator velocity gain (per-episode)
            com_offset_range: (x, y, z) half-ranges in meters for per-episode
            center-of-mass offset randomization on the torso body.
            terrain_flat_prob: Fraction of episodes that use nominal gravity.
            terrain_slope_max: Maximum implicit slope angle in degrees.
            max_episode_length: Maximum steps per episode before forced reset.
            actor_history_len: Number of chronological actor observation frames.
        """
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)

        # PD gains (override XML defaults with nominal values).
        kp, kd = (kp_range[0] + kp_range[1]) / 2, (kd_range[0] + kd_range[1]) / 2
        self.mj_model.actuator_gainprm[:, 0] = kp
        self.mj_model.actuator_biasprm[:, 1] = -kp
        self.mj_model.actuator_biasprm[:, 2] = -kd
        _xml_dof_damping = self.mj_model.dof_damping[6:]
        print(f"  XML dof_damping: {[f'{v:.4f}' for v in _xml_dof_damping]}")
        print(
            f"  Nominal actuator gains: kp={kp}, kd={kd} "
            f"(total damping per joint = kd + dof_damping = {kd + float(_xml_dof_damping[0]):.1f})"
        )

        self.mj_model.opt.timestep = 0.004

        # MJX does not support geom margins.
        self.mj_model.geom_margin[:] = 0.0

        self.mjx_model = mjx.put_model(self.mj_model)

        self.init_qpos = jp.array(self.mj_model.keyframe("home").qpos)
        self.default_joints = self.init_qpos[7:]

        self.n_frames = 5  # Physics substeps per control step
        self.dt = self.mj_model.opt.timestep * self.n_frames

        self.action_dim = 12
        self.action_scale = action_scale  # scalar, broadcast in step()
        # NOTE: scalar is faster but only works if all 12 joints share the same scale

        # Command ranges for curriculum
        self.cmd_vel_x_range = cmd_vel_x_range
        self.cmd_vel_y_range = cmd_vel_y_range
        self.cmd_yaw_rate_range = cmd_yaw_rate_range
        self.cmd_zero_prob = jp.array(cmd_zero_prob)
        self.cmd_ctrl_interval_range = cmd_ctrl_interval_range

        self.termination_height = 0.18
        self.max_episode_length = max_episode_length
        if actor_history_len < 1:
            raise ValueError("actor_history_len must be at least 1")
        self.actor_history_len = actor_history_len

        # Domain randomization config
        self.friction_range = friction_range
        self.mass_range = mass_range
        self.kp_range = kp_range
        self.kd_range = kd_range
        self.terrain_flat_prob = terrain_flat_prob
        self.terrain_slope_max = terrain_slope_max

        # Nominal gains (center of range) - used for curriculum difficulty
        self.kp_nominal = 0.5 * (kp_range[0] + kp_range[1])
        self.kd_nominal = 0.5 * (kd_range[0] + kd_range[1])

        # Store nominal model values for randomization
        self.base_friction = self.mjx_model.geom_friction  # (ngeom, 3)
        self.base_mass = self.mjx_model.body_mass  # (nbody,)
        self.base_inertia = self.mjx_model.body_inertia  # (nbody, 3)
        self.base_gainprm = self.mjx_model.actuator_gainprm  # (nu, 10)
        self.base_biasprm = self.mjx_model.actuator_biasprm  # (nu, 10)
        self.base_gravity = self.mjx_model.opt.gravity

        # Nominal mass and gravity.
        self.nominal_total_mass = float(jp.sum(self.base_mass))
        self.base_gravity_mag = float(jp.linalg.norm(self.base_gravity))

        # COM offset randomization
        self.com_offset_range = jp.array(com_offset_range)  # (3,) half-ranges
        self.use_com_offset = any(r > 0 for r in com_offset_range)
        self.base_ipos = self.mjx_model.body_ipos  # (nbody, 3)

        self.torso_body_id = self._find_torso_body_id()
        _slope_accel = self.base_gravity_mag * float(
            jp.sin(jp.array(terrain_slope_max * jp.pi / 180.0))
        )
        print(
            f"  Domain randomization: friction={friction_range}, mass={mass_range}, "
            f"kp={kp_range}, kd={kd_range}, "
            f"com_offset=+/-{com_offset_range}"
        )
        print(
            f"  Implicit terrain: max_slope={terrain_slope_max} deg "
            f"(~{_slope_accel:.2f} m/s^2 along-slope), "
            f"flat_prob={terrain_flat_prob}"
        )

        # Discover foot sites/bodies for contact and terrain features.
        self._foot_site_ids = self._find_foot_ids()
        self._foot_geom_ids = self._find_foot_geom_ids()
        self.foot_radius = jp.array(
            self.mj_model.geom_size[int(self._foot_geom_ids[0]), 0]
        )
        self.soft_contact_height = 0.01
        self.soft_contact_margin = 0.02
        self._foot_body_ids = jp.array(
            [int(self.mj_model.site_bodyid[sid]) for sid in self._foot_site_ids]
        )
        standing_data_cpu = mujoco.MjData(self.mj_model)
        standing_data_cpu.qpos[:] = self.mj_model.keyframe("home").qpos
        mujoco.mj_forward(self.mj_model, standing_data_cpu)
        foot_ids_np = [int(i) for i in self._foot_site_ids]
        foot_z_standing = jp.array(
            [standing_data_cpu.site_xpos[i, 2] for i in foot_ids_np]
        )
        self.foot_z_ground = jp.mean(foot_z_standing)
        print(
            f"  Foot ground reference z={float(self.foot_z_ground):.4f}m "
            f"(per-foot: {[f'{float(z):.4f}' for z in foot_z_standing]})"
        )

        # Compute obs_dim
        dummy = mjx.make_data(self.mjx_model)
        dummy_info = {
            "step": jp.array(0),
            "cmd_step": jp.array(0, dtype=jp.int32),
            "last_act": jp.zeros(12),
            "cmd": jp.zeros(3),
            "cmd_ctrl_interval": jp.array(cmd_ctrl_interval_range[0], dtype=jp.int32),
            "rng": jax.random.PRNGKey(0),
            "friction_scale": jp.array(1.0),
            "mass_scale": jp.array(1.0),
            "com_offset": jp.zeros(3),
            "difficulty": jp.array(0.0),
            "slope_gravity": self.base_gravity,
            "foot_bump_ou": jp.zeros((4, 3)),
            "foot_normal_forces": jp.zeros(4),
            "kp": jp.array(float(kp)),
            "kd": jp.array(float(kd)),
            "ground_z": jp.array(0.0),
            "episode_max_len": jp.array(max_episode_length, dtype=jp.int32),
            "terminal": jp.array(0.0),
        }
        self.obs_dim = self._get_obs(dummy, dummy_info).shape[0]
        self.actor_frame_obs_dim = self._get_actor_obs(dummy, dummy_info).shape[0]
        self.actor_obs_dim = self.actor_frame_obs_dim * self.actor_history_len
        self.critic_obs_dim = self._get_critic_obs(dummy, dummy_info).shape[0]

        mean_interval = (cmd_ctrl_interval_range[0] + cmd_ctrl_interval_range[1]) / 2
        command_dur = mean_interval * self.dt
        print(f"Go2Env initialized:")
        print(
            f"  actor_obs={self.actor_history_len}x{self.actor_frame_obs_dim}"
            f"={self.actor_obs_dim}, critic_obs_dim={self.critic_obs_dim}"
        )
        print(
            f"  cmd_vel_x={cmd_vel_x_range}, cmd_vel_y={cmd_vel_y_range}, "
            f"cmd_yaw_rate={cmd_yaw_rate_range}"
        )
        print(
            f"  cmd_zero_prob={cmd_zero_prob}, "
            f"random command interval={cmd_ctrl_interval_range} steps "
            f"(~{command_dur:.1f}s mean)"
        )
        print(f"  action_scale={action_scale}")
        print(
            f"  max_episode_length={max_episode_length} ({max_episode_length * self.dt:.1f}s)"
        )

    def _get_randomized_model(self, info):
        """
        Build a model variant with per-environment friction, mass, gains, and COM offset.

        Called inside step() - each env gets its own model.

        Friction: scales all three friction components

        Mass: scales body_mass and body_inertia uniformly.

        COM offset: shifts the torso body's inertial frame position (body_ipos)
        """
        friction = self.base_friction * info["friction_scale"]
        mass = self.base_mass * info["mass_scale"]
        inertia = self.base_inertia * info["mass_scale"]

        # Per-episode actuator gain randomization
        gainprm = self.base_gainprm.at[:, 0].set(info["kp"])
        biasprm = self.base_biasprm.at[:, 1].set(-info["kp"])
        biasprm = biasprm.at[:, 2].set(-info["kd"])

        model = self.mjx_model.replace(
            geom_friction=friction,
            body_mass=mass,
            body_inertia=inertia,
            actuator_gainprm=gainprm,
            actuator_biasprm=biasprm,
        )

        model = model.replace(opt=model.opt.replace(gravity=info["slope_gravity"]))

        # Per-episode COM offset on torso body
        if self.use_com_offset:
            body_ipos = self.base_ipos.at[self.torso_body_id].add(info["com_offset"])
            model = model.replace(body_ipos=body_ipos)

        return model

    def _find_torso_body_id(self) -> int:
        """
        Find the torso/base body ID for external force application.
        """
        for name in ["base", "trunk", "base_link", "torso"]:
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                print(f"  Torso body: '{name}' -> id {bid}")
                return bid

        # Fallback: first non-world body
        print(f"  Torso body: fallback to body 1")
        return 1

    def _find_foot_ids(self) -> jax.Array:
        """
        Find foot site IDs in the MuJoCo model.

        Order: [FL, FR, RL, RR].
        """
        foot_order = ["FL", "FR", "RL", "RR"]

        for suffix in ["_foot", ""]:
            names = [f"{leg}{suffix}" for leg in foot_order]
            ids = [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, n)
                for n in names
            ]
            if all(i >= 0 for i in ids):
                print(f"  Foot sites: {names} -> ids {ids}")
                return jp.array(ids)
        available = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            for i in range(self.mj_model.nsite)
        ]
        raise ValueError(f"Could not find foot sites. Available: {available}")

    def _find_foot_geom_ids(self) -> jax.Array:
        """
        Find the contact geom IDs for each foot in [FL, FR, RL, RR] order.

        Contacts reference geom IDs, so we want the named foot contact geoms
        rather than the first geom attached to each calf body.
        """
        foot_order = ["FL", "FR", "RL", "RR"]
        foot_geom_ids = []

        for foot_name in foot_order:
            gid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            if gid < 0:
                raise ValueError(f"Could not find foot geom named '{foot_name}'")
            foot_geom_ids.append(gid)

        names = [self.mj_model.geom(gi).name for gi in foot_geom_ids]
        print(f"  Foot geoms: {names} -> ids {foot_geom_ids}")
        return jp.array(foot_geom_ids)

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample a piecewise-constant random velocity command."""
        rng, k_vals, k_zero = jax.random.split(rng, 3)

        ranges_lo = jp.array(
            [
                self.cmd_vel_x_range[0],
                self.cmd_vel_y_range[0],
                self.cmd_yaw_rate_range[0],
            ]
        )
        ranges_hi = jp.array(
            [
                self.cmd_vel_x_range[1],
                self.cmd_vel_y_range[1],
                self.cmd_yaw_rate_range[1],
            ]
        )

        raw = jax.random.uniform(k_vals, (3,), minval=ranges_lo, maxval=ranges_hi)

        zero_mask = jax.random.uniform(k_zero, (3,)) < self.cmd_zero_prob
        return jp.where(zero_mask, 0.0, raw)

    def _sample_ctrl_interval(self, rng: jax.Array) -> jax.Array:
        """Sample a random control interval from the configured range."""
        return jax.random.randint(
            rng,
            (),
            self.cmd_ctrl_interval_range[0],
            self.cmd_ctrl_interval_range[1] + 1,
        )

    def reset(self, rng: jax.Array, difficulty: jax.Array = jp.array(0.0)) -> EnvState:
        """Reset environment to initial state with small random perturbations
        and freshly sampled domain randomization."""
        rng, *keys = jax.random.split(rng, 13)

        qpos = self.init_qpos.copy()
        qvel = jp.zeros(self.mj_model.nv)

        qpos = qpos.at[:2].add(0.02 * (jax.random.uniform(keys[0], (2,)) - 0.5))
        angle = (jp.pi / 20) * (jax.random.uniform(keys[1], ()) - 0.5)
        axis = jp.array([0.0, 0.0, 1.0])
        qpos = qpos.at[3:7].set(axis_angle_to_quat(axis, angle))
        qpos = qpos.at[7:19].add(0.03 * (jax.random.uniform(keys[2], (12,)) - 0.5))

        friction_scale = jax.random.uniform(
            keys[6], (), minval=self.friction_range[0], maxval=self.friction_range[1]
        )
        mass_scale = jax.random.uniform(
            keys[7], (), minval=self.mass_range[0], maxval=self.mass_range[1]
        )
        kp_raw = jax.random.uniform(
            keys[9], (), minval=self.kp_range[0], maxval=self.kp_range[1]
        )
        kd_raw = jax.random.uniform(
            keys[10], (), minval=self.kd_range[0], maxval=self.kd_range[1]
        )
        # Curriculum gain randomization with difficulty:
        #   difficulty=0 --> kp=nominal
        #   difficulty=1 --> kp=full random range
        kp = self.kp_nominal + difficulty * (kp_raw - self.kp_nominal)
        kd = self.kd_nominal + difficulty * (kd_raw - self.kd_nominal)

        # Generate per-episode COM offset and slope gravity.
        k_com, k_terrain = jax.random.split(keys[8])

        # COM offset: uniform in [-range, +range], set by curriculum
        if self.use_com_offset:
            com_offset_raw = self.com_offset_range * (
                2.0 * jax.random.uniform(k_com, (3,)) - 1.0
            )
            com_offset = difficulty * com_offset_raw
        else:
            com_offset = jp.zeros(3)

        slope_gravity = sample_slope_gravity(
            k_terrain,
            difficulty=difficulty,
            slope_max_deg=self.terrain_slope_max,
            flat_prob=self.terrain_flat_prob,
            g=self.base_gravity_mag,
        )

        cmd = self._sample_command(keys[3])
        ctrl_interval = self._sample_ctrl_interval(keys[4])

        # Randomize episode length uniformly in +/-25% of max_episode_length.
        episode_max_len = jax.random.randint(
            keys[11],
            (),
            self.max_episode_length * 3 // 4,
            self.max_episode_length * 5 // 4 + 1,
        )

        info = {
            "step": jp.array(0, dtype=jp.int32),
            "cmd_step": jp.array(0, dtype=jp.int32),
            "last_act": jp.zeros(12),
            "cmd": cmd,
            "cmd_ctrl_interval": ctrl_interval,
            "rng": rng,
            "friction_scale": friction_scale,
            "mass_scale": mass_scale,
            "com_offset": com_offset,
            "difficulty": difficulty,
            "slope_gravity": slope_gravity,
            "foot_bump_ou": jp.zeros((4, 3)),
            "foot_normal_forces": jp.zeros(4),
            "kp": kp,
            "kd": kd,
            "ground_z": jp.array(0.0),
            "episode_max_len": episode_max_len,
            "terminal": jp.array(0.0),
        }

        # Use randomized model for penetration check
        model = self._get_randomized_model(info)
        data = mjx.make_data(model)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(model, data)

        pen = jp.minimum(jp.min(data._impl.contact.dist), 0.0)
        qpos = qpos.at[2].add(-pen + 0.01)
        data = data.replace(qpos=qpos)
        data = mjx.forward(model, data)

        actor_frame = self._get_actor_obs(data, info)
        actor_obs_history = jp.repeat(
            actor_frame[None, :], self.actor_history_len, axis=0
        )
        obs = actor_obs_history.reshape(-1)
        critic_obs = self._get_critic_obs(data, info)
        info = {
            **info,
            "actor_obs_history": actor_obs_history,
            "bootstrap_obs": obs,
            "bootstrap_critic_obs": critic_obs,
        }

        return EnvState(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            info=info,
            metrics=self._init_metrics(),
        )

    def _init_metrics(self) -> dict:
        """Initialize metrics dictionary with zeros (f32, not in gradient path)."""
        _z = jp.float32(0.0)
        return {
            "vel_x": _z,
            "vel_y": _z,
            "yaw_rate": _z,
            "cmd_x": _z,
            "cmd_y": _z,
            "cmd_yaw": _z,
            "contact_force": _z,
            "rew_vel_x": _z,
            "rew_vel_y": _z,
            "rew_yaw": _z,
            "rew_height": _z,
            "rew_vz": _z,
            "rew_upright": _z,
            "pen_rate": _z,
            "pen_act": _z,
            "pen_joint": _z,
            "height": jp.float32(0.27),
            "tilt": jp.float32(-1.0),
            "foot_normal_FL": _z,
            "foot_normal_FR": _z,
            "foot_normal_RL": _z,
            "foot_normal_RR": _z,
        }

    @functools.partial(jax.checkpoint, static_argnums=(0,))
    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        """
        Take one environment step with domain-randomized physics.

        NOTE: External disturbances are applied by the training rollout.
        """
        action = jp.clip(action, -1, 1).astype(jp.float64)
        ctrl = self.default_joints + action * self.action_scale

        model = self._get_randomized_model(state.info)

        # Physics simulation
        def physics_step(data, _):
            return mjx.step(model, data.replace(ctrl=ctrl)), None

        data, _ = jax.lax.scan(physics_step, state.data, None, length=self.n_frames)

        # Extract state info
        pos = data.qpos[:3]
        quat = data.qpos[3:7]
        inv_q = quat_inv(quat)
        local_linvel = quat_rotate(data.qvel[:3], inv_q)
        local_angvel = quat_rotate(data.qvel[3:6], inv_q)
        gravity_proj = quat_rotate(jp.array([0.0, 0.0, -1.0]), inv_q)
        cmd = state.info["cmd"]

        # Ground height (explicit, flat)
        ground_z = jp.float64(0.0)
        height_above_ground = pos[2]

        # Per-foot normal forces from constraint solver.
        foot_normal_forces = self._get_foot_normal_forces(data)

        # Compute rewards (using height above local ground)
        rewards = self._compute_rewards(
            data,
            action,
            state.info["last_act"],
            local_linvel,
            local_angvel,
            height_above_ground,
            gravity_proj,
            cmd,
        )
        total_reward = sum(rewards.values())

        contact_force = jp.sum(jp.abs(data.qfrc_constraint[:6]))

        # Termination conditions (using height above local ground)
        done, terminal = self._compute_done(
            height_above_ground,
            gravity_proj,
            data.qpos,
            state.info["step"],
            state.info["episode_max_len"],
        )

        pre_reset_info = {
            **state.info,
            "step": state.info["step"] + 1,
            "cmd_step": state.info["cmd_step"] + 1,
            "last_act": action,
            "ground_z": ground_z,
            "terminal": terminal,
        }
        bootstrap_actor_frame = self._get_actor_obs(data, pre_reset_info)
        bootstrap_history = jp.concatenate(
            [state.info["actor_obs_history"][1:], bootstrap_actor_frame[None, :]],
            axis=0,
        )
        bootstrap_obs = bootstrap_history.reshape(-1)
        bootstrap_critic_obs = self._get_critic_obs(data, pre_reset_info)

        # Done, reset + cmd resample
        rng, reset_key, resample_key, interval_key = jax.random.split(
            state.info["rng"], 4
        )
        reset_state = self.reset(reset_key, state.info["difficulty"])

        # Apply done reset (swap in fresh episode state)
        next_data = jax.tree_util.tree_map(
            lambda a, b: jp.where(done, b, a), data, reset_state.data
        )

        # Step counters
        next_step = jp.where(done, jp.array(0, dtype=jp.int32), state.info["step"] + 1)
        next_cmd_step = jp.where(
            done, jp.array(0, dtype=jp.int32), state.info["cmd_step"] + 1
        )

        # On done: take fresh interval and rng from reset
        next_interval = jp.where(
            done, reset_state.info["cmd_ctrl_interval"], state.info["cmd_ctrl_interval"]
        )
        next_rng = jax.tree_util.tree_map(
            lambda a, b: jp.where(done, b, a), rng, reset_state.info["rng"]
        )

        # Per-episode DR: swap on done
        next_friction = jp.where(
            done, reset_state.info["friction_scale"], state.info["friction_scale"]
        )
        next_mass = jp.where(
            done, reset_state.info["mass_scale"], state.info["mass_scale"]
        )
        next_foot_normal_forces = jp.where(done, jp.zeros(4), foot_normal_forces)
        next_foot_bump_ou = jp.where(done, jp.zeros((4, 3)), state.info["foot_bump_ou"])
        next_slope_gravity = jp.where(
            done, reset_state.info["slope_gravity"], state.info["slope_gravity"]
        )
        next_kp = jp.where(done, reset_state.info["kp"], state.info["kp"])
        next_kd = jp.where(done, reset_state.info["kd"], state.info["kd"])
        next_episode_max_len = jp.where(
            done, reset_state.info["episode_max_len"], state.info["episode_max_len"]
        )

        if self.use_com_offset:
            next_com_offset = jp.where(
                done, reset_state.info["com_offset"], state.info["com_offset"]
            )
        else:
            next_com_offset = jp.zeros(3)

        # Command update: hold a random command for a sampled interval, then resample.
        done_sg = jax.lax.stop_gradient(done)
        command_exhausted = (next_cmd_step >= next_interval) & (done_sg < 0.5)
        new_cmd = self._sample_command(resample_key)
        new_interval = self._sample_ctrl_interval(interval_key)
        next_cmd_step = jp.where(
            command_exhausted, jp.array(0, dtype=jp.int32), next_cmd_step
        )
        next_interval = jp.where(command_exhausted, new_interval, next_interval)
        next_cmd = jp.where(command_exhausted, new_cmd, state.info["cmd"])
        next_cmd = jp.where(done, reset_state.info["cmd"], next_cmd)

        next_info = {
            "step": next_step,
            "cmd_step": next_cmd_step,
            "last_act": jp.where(done, jp.zeros(12), action),
            "cmd": next_cmd,
            "cmd_ctrl_interval": next_interval,
            "rng": next_rng,
            "friction_scale": next_friction,
            "mass_scale": next_mass,
            "com_offset": next_com_offset,
            "difficulty": state.info["difficulty"],
            "slope_gravity": next_slope_gravity,
            "foot_bump_ou": next_foot_bump_ou,
            "foot_normal_forces": next_foot_normal_forces,
            "kp": next_kp,
            "kd": next_kd,
            "ground_z": jp.where(done, reset_state.info["ground_z"], ground_z),
            "episode_max_len": next_episode_max_len,
            "terminal": terminal,  # fall/NaN only
            "bootstrap_obs": bootstrap_obs,
            "bootstrap_critic_obs": bootstrap_critic_obs,
        }
        next_actor_frame = self._get_actor_obs(next_data, next_info)
        continued_history = jp.concatenate(
            [state.info["actor_obs_history"][1:], next_actor_frame[None, :]], axis=0
        )
        next_history = jp.where(
            done, reset_state.info["actor_obs_history"], continued_history
        )
        next_info = {**next_info, "actor_obs_history": next_history}

        metrics = {
            "vel_x": local_linvel[0],
            "vel_y": local_linvel[1],
            "yaw_rate": local_angvel[2],
            "cmd_x": cmd[0],
            "cmd_y": cmd[1],
            "cmd_yaw": cmd[2],
            "contact_force": contact_force,
            "rew_vel_x": rewards["lin"],  # Linear tracking is a single combined reward in this variant
            "rew_vel_y": rewards["lin"],
            "rew_yaw": rewards["yaw"],
            "rew_height": rewards["height"],
            "rew_vz": rewards["vz"],
            "rew_upright": rewards["upright"],
            "pen_rate": rewards["pen_rate"],
            "pen_act": rewards["pen_act"],
            "pen_joint": rewards["pen_joint"],
            "height": height_above_ground,
            "tilt": gravity_proj[2],
            "foot_normal_FL": foot_normal_forces[0],
            "foot_normal_FR": foot_normal_forces[1],
            "foot_normal_RL": foot_normal_forces[2],
            "foot_normal_RR": foot_normal_forces[3],
        }
        # Metrics are not in the gradient path, so cast to f32
        metrics = jax.tree_util.tree_map(
            lambda x: x.astype(jp.float32) if hasattr(x, "astype") else x, metrics
        )

        return EnvState(
            data=next_data,
            obs=next_history.reshape(-1),
            reward=total_reward,
            done=done,
            info=next_info,
            metrics=metrics,
        )

    def _get_foot_normal_forces(self, data) -> jax.Array:
        """
        Extract normal contact force for each foot from MJX solver output.

        Returns: shape (4,) array of normal forces [FL, FR, RL, RR].
        Positive = foot pushing into ground.
        """
        # efc_address maps each contact to its row in the constraint force array.
        # The normal constraint is at efc_address[i]; friction rows follow after.
        # Inactive contacts have efc_address == -1.
        efc_addr = data.contact.efc_address  # (ncon,)
        active = efc_addr >= 0

        safe_addr = jp.clip(efc_addr, 0, data.efc_force.shape[0] - 1)
        normal_forces = data.efc_force[safe_addr]  # (ncon,)
        normal_forces = jp.where(active, normal_forces, 0.0)

        # Match contacts to feet: contact.geom is (ncon, 2) geom-ID pairs
        geom1 = data.contact.geom[:, 0]
        geom2 = data.contact.geom[:, 1]

        # Broadcast: (4, ncon) mask of which contacts involve each foot
        foot_ids = self._foot_geom_ids[:, None]  # (4, 1)
        is_foot = (geom1[None, :] == foot_ids) | (geom2[None, :] == foot_ids)
        is_foot = is_foot & active[None, :]

        # Sum normal forces per foot -> (4,)
        return jp.sum(is_foot * normal_forces[None, :], axis=1)

    def _compute_rewards(
        self,
        data,
        action,
        last_act,
        local_linvel,
        local_angvel,
        height_above_ground,
        gravity_proj,
        cmd,
    ) -> dict:
        """
        Compute all reward components.

        Args:
            height_above_ground: Scalar, body z minus local terrain z.
        """
        # Velocity command tracking (local frame)
        lin_error_sq = jp.sum(jp.square(local_linvel[:2] - cmd[:2]))
        rew_lin = jp.exp(-lin_error_sq / 0.25)
        rew_yaw = jp.exp(-jp.square(local_angvel[2] - cmd[2]) / 0.25)

        rew_height = jp.exp(-10.0 * jp.square(height_above_ground - 0.27))
        rew_vz = -0.5 * jp.square(local_linvel[2])

        rew_upright = 0.5 * (-gravity_proj[2])

        # Joint regularization (relative to reference)
        correction = data.qpos[7:] - self.default_joints
        correction_dev = jp.sum(jp.square(correction))
        pen_joint = correction_dev * -0.3

        # Penalties
        pen_action_rate = -0.02 * jp.sum(jp.square(action - last_act))
        pen_action = -0.05 * jp.sum(jp.square(action))
        pen_angvel_xy = -0.05 * jp.sum(jp.square(local_angvel[:2]))

        return {
            "lin": rew_lin,
            "yaw": rew_yaw,
            "height": rew_height,
            "vz": rew_vz,
            "upright": rew_upright,
            "pen_rate": pen_action_rate,
            "pen_act": pen_action,
            "pen_angvel": pen_angvel_xy,
            "pen_joint": pen_joint,
        }

    def compute_reward_from_critic_obs(
        self, critic_obs: jax.Array, next_critic_obs: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Compute reward from critic observations for JAVE gradient targets."""
        action = jp.clip(action, -1.0, 1.0)

        next_local_linvel = next_critic_obs[..., 0:3]
        next_local_angvel = next_critic_obs[..., 3:6]
        next_gravity_proj = next_critic_obs[..., 6:9]
        cmd = critic_obs[..., 9:12]
        next_height = next_critic_obs[..., 12]
        next_joint_offsets = next_critic_obs[..., 13:25]
        last_act = critic_obs[..., 37:49]

        qpos_prefix = jp.zeros(next_joint_offsets.shape[:-1] + (7,))
        qpos = jp.concatenate(
            [qpos_prefix, self.default_joints + next_joint_offsets], axis=-1
        )
        rewards = self._compute_rewards(
            _RewardData(qpos),
            action,
            last_act,
            next_local_linvel,
            next_local_angvel,
            next_height,
            next_gravity_proj,
            cmd,
        )
        return sum(rewards.values())

    def _compute_done(
        self, height_above_ground, gravity_proj, qpos, step, episode_max_len
    ):
        """
        Compute termination conditions.

        Returns:
            done:     1.0 if the episode should reset (fall, NaN, or time-limit)
            terminal: 1.0 only for true terminations (fall / NaN).
                      0.0 for time-limit truncations.
                      Used by the training loop to distinguish truncation from
                      termination for the SHAC bootstrap correction.

        Grace period (step < 20) suppresses fall/tilt checks only.
        NaN in qpos always terminates immediately.
        Time limit (episode_max_len) forces periodic episode cycling.
        """
        fall_or_tilt = jp.array(0.0)
        fall_or_tilt = jp.where(
            height_above_ground < self.termination_height, 1.0, fall_or_tilt
        )
        fall_or_tilt = jp.where(gravity_proj[2] > -0.3, 1.0, fall_or_tilt)
        fall_or_tilt = jp.where(step < 20, 0.0, fall_or_tilt)  # Grace period

        # NaN always bypasses grace period
        nan_done = jp.where(jp.any(~jp.isfinite(qpos)), 1.0, jp.array(0.0))

        # True termination (fall or NaN) = no bootstrap
        terminal = jp.maximum(fall_or_tilt, nan_done)

        # Time-limit truncation = bootstrap
        # # Per-episode randomized limit
        time_limit = jp.where(step >= episode_max_len, 1.0, jp.array(0.0))

        done = jp.maximum(terminal, time_limit)
        return done, terminal

    def _apply_obs_noise(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        """Apply PPO-style observation noise for actor inputs only."""
        frame_noise = jp.concatenate(
            [
                jp.ones(3) * 0.2,  # angular velocity
                jp.ones(3) * 0.05,  # projected gravity
                jp.zeros(3),  # commands
                jp.ones(12) * 0.01,  # joint positions
                jp.ones(12) * 0.01,  # joint velocities
                jp.zeros(12),  # actions
            ],
            axis=0,
        )
        noise_buff = jp.tile(frame_noise, self.actor_history_len)
        return obs + (2.0 * jax.random.uniform(rng, obs.shape) - 1.0) * noise_buff

    def normalize_actor_obs(self, normalizer, norm_state, obs: jax.Array) -> jax.Array:
        """Normalize stacked actor frames with shared per-feature statistics."""
        frames = obs.reshape(*obs.shape[:-1], self.actor_history_len, -1)
        normalized = normalizer.normalize(norm_state, frames)
        return normalized.reshape(*obs.shape[:-1], self.actor_obs_dim)

    def _get_actor_obs(self, data, info) -> jax.Array:
        """
        Construct the actor observation vector.

        """
        quat = data.qpos[3:7]
        inv_q = quat_inv(quat)

        return jp.concatenate(
            [
                quat_rotate(data.qvel[3:6], inv_q),
                quat_rotate(jp.array([0.0, 0.0, -1.0]), inv_q),
                info["cmd"],
                data.qpos[7:] - self.default_joints,
                data.qvel[6:],
                info["last_act"],
            ]
        )

    def _get_critic_obs(self, data, info) -> jax.Array:
        """
        Construct the critic observation vector.

        """
        quat = data.qpos[3:7]
        inv_q = quat_inv(quat)
        height_above_ground = data.qpos[2] - info["ground_z"]

        return jp.concatenate(
            [
                quat_rotate(data.qvel[:3], inv_q),
                quat_rotate(data.qvel[3:6], inv_q),
                quat_rotate(jp.array([0.0, 0.0, -1.0]), inv_q),
                info["cmd"],
                jp.array([height_above_ground]),
                data.qpos[7:] - self.default_joints,
                data.qvel[6:],
                info["last_act"],
            ]
        )

    def _get_obs(self, data, info) -> jax.Array:
        """
        Construct observation vector.

        Observation includes:
        - Local linear velocity (3)
        - Local angular velocity (3)
        - Gravity in body frame (3)
        - Command [vx, vy, yaw_rate] (3)
        - Height above local ground (1)
        - Joint positions relative to default (12)
        - Joint velocities (12)
        - Previous action (12)

        Total: 49 dimensions
        """
        quat = data.qpos[3:7]
        inv_q = quat_inv(quat)

        # Height above local ground
        height_above_ground = data.qpos[2] - info["ground_z"]

        return jp.concatenate(
            [
                quat_rotate(data.qvel[:3], inv_q),  # Local linear velocity (3)
                quat_rotate(data.qvel[3:6], inv_q),  # Local angular velocity (3)
                quat_rotate(
                    jp.array([0.0, 0.0, -1.0]), inv_q
                ),  # Gravity in body frame (3)
                info["cmd"],  # Command (3)
                jp.array([height_above_ground]),  # Height above ground (1)
                data.qpos[7:] - self.default_joints,  # Joint positions (12)
                data.qvel[6:],  # Joint velocities (12)
                info["last_act"],  # Previous action (12)
            ]
        )  # Total: 49
