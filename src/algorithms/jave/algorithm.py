"""JAVE training for Go2 locomotion.

See ./docs/algorithms.md for further details
"""

import os
import time
import pickle
import json
from datetime import datetime

# Set to True to enable per-foot normal force logging
DEBUG_FOOT_CONTACTS = False

import jax
import jax.numpy as jp
import optax
import numpy as np

from src.core.data_structures import (
    Normalizer,
    TrainState,
    init_replay_buffer,
    add_to_replay_buffer,
    sample_replay_buffer,
    slim_checkpoint_state,
)
from src.core.networks import Actor, Critic, LearnedDynamicsModel
from src.envs.go2.environment import Go2Env
from src.envs.go2.terrain import differentiated_ou_foot_forces
from src.core.utils import compute_grad_norm


def load_checkpoint(path: str):
    """
    Load a training checkpoint.

    Args:
        path: Path to a .pkl file, or a training folder containing one.
              When given a folder, searches in order:
              checkpoint_latest.pkl, policy_best.pkl, policy_final.pkl

    Returns:
        Tuple of (state, hparams, step) where:
            - state: TrainState object
            - hparams: dict of hyperparameters (or None)
            - step: training step count
    """
    if os.path.isdir(path):
        for name in ["checkpoint_latest.pkl", "policy_best.pkl", "policy_final.pkl"]:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No checkpoint found in {path}")

    print(f"Loading checkpoint from {path}")
    with open(path, "rb") as f:
        state = pickle.load(f)

    # Try to load hyperparameters from same directory
    hparams = None
    hparams_path = os.path.join(os.path.dirname(path), "hparams.json")
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparams = json.load(f)

    return state, hparams, int(state.step)


def train(
    # General
    total_steps: int = 100_000,
    unroll_length: int = 12,
    num_envs: int = 256,
    actor_lr: float = 5e-3,
    critic_lr: float = 5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    target_update_rate: float = 0.01,
    critic_iterations: int = 16,
    use_lr_decay: bool = False,
    xml_path: str = "src/envs/go2/models/scene_mjx.xml",
    action_scale: float = 0.5,
    # Commands
    cmd_vel_x_range: tuple = (-2.0, 2.0),
    cmd_vel_y_range: tuple = (-1.0, 1.0),
    cmd_yaw_rate_range: tuple = (-1.5, 1.5),
    cmd_zero_prob: tuple = (0.1, 0.7, 0.5),
    cmd_ctrl_interval_range: tuple = (60, 140),
    # Randomization
    action_noise_std_start: float = 0.5,
    action_noise_std_end: float = 0.32,
    friction_range: tuple = (0.5, 2.0),
    mass_range: tuple = (0.85, 1.15),
    kp_range: tuple = (25.0, 45.0),
    kd_range: tuple = (0.3, 0.7),
    com_offset_range: tuple = (0.05, 0.05, 0.04),
    push_velocity_range: tuple = (-1.0, 1.0),
    push_interval_s: float = 4.0,
    terrain_flat_prob: float = 0.2,
    terrain_slope_max: float = 5.0,
    terrain_bump_std: float = 0.4,
    terrain_bump_decay: float = 0.4,
    terrain: bool = False,
    # Annealing
    zero_difficulty_frac: float = 0.0,
    curriculum_grace: int = None,
    curriculum_steps: int = None,
    # Misc.
    diagnose: bool = False,
    seed: int = 0,
    resume_from: str = None,
    checkpoint_interval: int = 10_000,
    max_episode_length: int = 5000,
    actor_history_len: int = 10,
    env_variant: str = "blind_nolinvel_nokinref",
    # JAVE: Jacobian-Augmented Value Estimation
    vg_weight: float = 0.1,
    vg_warmup_steps: int = 5_000,
    ldm_hidden: tuple = (256, 256),
    ldm_lr: float = 3e-4,
    ldm_iterations: int = 4,
    ldm_batch_size: int = 256,
    ldm_buffer_capacity: int = 100_000,
):
    """
    Train a quadruped locomotion policy using jave.

    Args:
        total_steps: Total environment steps to train
        unroll_length: Number of steps per trajectory rollout (short horizon h)
        num_envs: Number of parallel environments (N)
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        gamma: Discount factor
        target_update_rate: Soft update rate for target critic (1-alpha)
        critic_iterations: Number of critic gradient steps per actor update
        use_lr_decay: Linear LR decay to 62% over training
        action_scale: Scale factor for actions
        cmd_vel_x_range: (min, max) for forward velocity command (m/s)
        cmd_vel_y_range: (min, max) for lateral velocity command (m/s)
        cmd_yaw_rate_range: (min, max) for yaw rate command (rad/s)
        cmd_zero_prob: Per-component probability of zeroing (vx, vy, yaw)
        cmd_ctrl_interval_range: (min, max) steps between random command samples
        action_noise_std_start: Std dev of Gaussian action noise at step 0
        action_noise_std_end: Std dev of Gaussian action noise at total_steps
        friction_range: (lo, hi) multiplicative factor for geom_friction per episode
        mass_range: (lo, hi) multiplicative factor for body_mass per episode
        kp_range: (lo, hi) absolute range for actuator position gain per episode
        kd_range: (lo, hi) absolute range for actuator velocity gain per episode
        push_velocity_range: Interval root x/y velocity disturbance range.
        push_interval_s: Seconds between velocity pushes.
        terrain_flat_prob: Fraction of terrain episodes that use nominal gravity.
        terrain: Enable implicit tilted-gravity terrain randomization.
        zero_difficulty_frac: Fraction of envs that are held at difficulty=0 each
                              unroll, regardless of curriculum progress. These envs
                              see nominal gravity, nominal gains, and no COM offset.
        curriculum_grace: Steps at difficulty=0 before ramping starts.
        curriculum_steps: Steps over which difficulty ramps 0->1 (after grace).
        diagnose: Enable detailed diagnostic logging
        seed: Random seed
        resume_from: Path to checkpoint .pkl file or training folder to resume from
        checkpoint_interval: Save checkpoint every N steps
        vg_weight: Weight beta for gradient Bellman loss in critic training.
        vg_warmup_steps: Steps before activating gradient Bellman loss
        ldm_hidden: Learned dynamics model hidden layer sizes
        ldm_lr: Learned dynamics model learning rate
        ldm_iterations: LDM gradient steps per train_step
        ldm_batch_size: Mini-batch size for LDM training
        ldm_buffer_capacity: Replay buffer capacity

    Returns:
        Tuple of (final_state, save_directory)
    """
    # Handle checkpoint resumption
    resumed_state = None
    resumed_step = 0
    resumed_hparams = None

    if resume_from:
        resumed_state, resumed_hparams, resumed_step = load_checkpoint(resume_from)
        if resumed_hparams:
            print(f"Resuming from step {resumed_step}")
            print(
                f"  Loaded hparams: action_scale={resumed_hparams.get('action_scale')}"
            )
            action_scale = resumed_hparams.get("action_scale", action_scale)
            action_noise_std_start = resumed_hparams.get(
                "action_noise_std_start", action_noise_std_start
            )
            action_noise_std_end = resumed_hparams.get(
                "action_noise_std_end", action_noise_std_end
            )
            xml_path = resumed_hparams.get("xml_path", xml_path)
            env_variant = resumed_hparams.get("env_variant", env_variant)
            if "kp_range" in resumed_hparams:
                kp_range = tuple(resumed_hparams["kp_range"])
            if "kd_range" in resumed_hparams:
                kd_range = tuple(resumed_hparams["kd_range"])
            if "com_offset_range" in resumed_hparams:
                com_offset_range = tuple(resumed_hparams["com_offset_range"])
            if "push_velocity_range" in resumed_hparams:
                push_velocity_range = tuple(resumed_hparams["push_velocity_range"])
            if "push_interval_s" in resumed_hparams:
                push_interval_s = resumed_hparams["push_interval_s"]
            if "terrain_bump_std" in resumed_hparams:
                terrain_bump_std = resumed_hparams["terrain_bump_std"]
            if "terrain_bump_decay" in resumed_hparams:
                terrain_bump_decay = resumed_hparams["terrain_bump_decay"]
            if "cmd_ctrl_interval_range" in resumed_hparams:
                cmd_ctrl_interval_range = tuple(
                    resumed_hparams["cmd_ctrl_interval_range"]
                )
            if "zero_difficulty_frac" in resumed_hparams:
                zero_difficulty_frac = resumed_hparams["zero_difficulty_frac"]
            if "curriculum_grace" in resumed_hparams:
                curriculum_grace = resumed_hparams["curriculum_grace"]
            if "curriculum_steps" in resumed_hparams:
                curriculum_steps = resumed_hparams["curriculum_steps"]
            if "max_episode_length" in resumed_hparams:
                max_episode_length = resumed_hparams["max_episode_length"]
            if "actor_history_len" in resumed_hparams:
                actor_history_len = resumed_hparams["actor_history_len"]

    # Compute curriculum defaults relative to total_steps
    if curriculum_grace is None:
        curriculum_grace = total_steps // 10  # 10% grace at difficulty=0
    if curriculum_steps is None:
        curriculum_steps = int(total_steps * 0.8)  # ramp over 80%

    _curriculum_steps = max(curriculum_steps, 1)  # avoid division by zero

    _curriculum_grace_jax = jp.array(curriculum_grace, dtype=jp.int32)
    _curriculum_steps_jax = jp.array(_curriculum_steps, dtype=jp.float32)

    env = Go2Env(
        variant=env_variant,
        xml_path=xml_path,
        action_scale=action_scale,
        cmd_vel_x_range=cmd_vel_x_range,
        cmd_vel_y_range=cmd_vel_y_range,
        cmd_yaw_rate_range=cmd_yaw_rate_range,
        cmd_zero_prob=cmd_zero_prob,
        cmd_ctrl_interval_range=cmd_ctrl_interval_range,
        friction_range=friction_range,
        mass_range=mass_range,
        kp_range=kp_range,
        kd_range=kd_range,
        com_offset_range=com_offset_range,
        terrain_flat_prob=terrain_flat_prob,
        terrain_slope_max=terrain_slope_max if terrain else 0.0,
        max_episode_length=max_episode_length,
        actor_history_len=actor_history_len,
    )
    actor_norm = Normalizer(env.actor_frame_obs_dim)
    critic_norm = Normalizer(env.critic_obs_dim)

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_runs/jave_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Algorithm: JAVE, Save dir: {save_dir}")
    print(
        f"Domain randomization: action_noise={action_noise_std_start}->{action_noise_std_end}, "
        f"friction={friction_range}, mass={mass_range}, "
        f"kp={kp_range}, kd={kd_range}, "
        f"com_offset=+/-{com_offset_range}, "
        f"velocity_push={push_velocity_range} every {push_interval_s}s"
    )
    print(
        f"Curriculum: grace={curriculum_grace}, curriculum={curriculum_steps} steps, "
        f"terrain={'ON' if terrain else 'OFF'}, "
        f"terrain_flat_prob={terrain_flat_prob}, "
        f"terrain_slope_max={terrain_slope_max} deg, "
        f"terrain_bump_std={terrain_bump_std if terrain else 0.0}"
    )

    best_reward = (
        resumed_hparams.get("best_reward", -np.inf) if resumed_hparams else -np.inf
    )

    # Save hyperparameters up front so interrupted runs can be resumed. 
    hparams = {
        "algorithm": "jave",
        "total_steps": total_steps,
        "unroll_length": unroll_length,
        "num_envs": num_envs,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "target_update_rate": target_update_rate,
        "critic_iterations": critic_iterations,
        "use_lr_decay": use_lr_decay,
        "xml_path": xml_path,
        "action_scale": action_scale,
        "cmd_vel_x_range": list(cmd_vel_x_range),
        "cmd_vel_y_range": list(cmd_vel_y_range),
        "cmd_yaw_rate_range": list(cmd_yaw_rate_range),
        "cmd_zero_prob": list(cmd_zero_prob),
        "cmd_ctrl_interval_range": list(cmd_ctrl_interval_range),
        "action_noise_std_start": action_noise_std_start,
        "action_noise_std_end": action_noise_std_end,
        "friction_range": list(friction_range),
        "mass_range": list(mass_range),
        "kp_range": list(kp_range),
        "kd_range": list(kd_range),
        "com_offset_range": list(com_offset_range),
        "push_velocity_range": list(push_velocity_range),
        "push_interval_s": push_interval_s,
        "terrain_flat_prob": terrain_flat_prob,
        "terrain_slope_max": terrain_slope_max,
        "terrain_bump_std": terrain_bump_std,
        "terrain_bump_decay": terrain_bump_decay,
        "terrain": terrain,
        "zero_difficulty_frac": zero_difficulty_frac,
        "curriculum_grace": curriculum_grace,
        "curriculum_steps": curriculum_steps,
        "seed": seed,
        "best_reward": best_reward,
        "max_episode_length": max_episode_length,
        "actor_history_len": actor_history_len,
        "env_variant": env_variant,
        "vg_weight": vg_weight,
        "vg_warmup_steps": vg_warmup_steps,
        "ldm_hidden": list(ldm_hidden),
        "ldm_lr": ldm_lr,
        "ldm_iterations": ldm_iterations,
        "ldm_batch_size": ldm_batch_size,
        "ldm_buffer_capacity": ldm_buffer_capacity,
    }
    with open(f"{save_dir}/hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    # Initialize random keys
    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    # Initialize networks
    actor = Actor(env.action_dim)
    critic = Critic()

    actor_dummy = jp.zeros((1, env.actor_obs_dim), dtype=jp.float32)
    critic_dummy = jp.zeros((1, env.critic_obs_dim), dtype=jp.float32)
    actor_params = actor.init(k1, actor_dummy)
    critic_params = critic.init(k2, critic_dummy)
    target_critic_params = critic_params

    actor_normalizer = actor_norm.init()
    critic_normalizer = critic_norm.init()

    # Initialize learned dynamics model and replay buffer
    learned_dynamics_model = LearnedDynamicsModel(
        obs_dim=env.critic_obs_dim, hidden=ldm_hidden
    )
    dummy_action = jp.zeros((1, env.action_dim), dtype=jp.float32)
    ldm_params = learned_dynamics_model.init(k4, critic_dummy, dummy_action)
    replay_buffer = init_replay_buffer(
        ldm_buffer_capacity, env.critic_obs_dim, env.action_dim
    )

    if vg_weight > 0:
        print(
            f"JAVE: gradient Bellman critic training enabled, "
            f"vg_weight={vg_weight}, warmup={vg_warmup_steps}"
        )
        print(
            f"  Learned dynamics model: arch={ldm_hidden}, lr={ldm_lr}, "
            f"iters={ldm_iterations}, batch={ldm_batch_size}, "
            f"buffer={ldm_buffer_capacity}"
        )
    else:
        print("JAVE: gradient Bellman critic training disabled (vg_weight=0)")

    # Linear LR decay
    if use_lr_decay:
        total_iters = total_steps // (num_envs * unroll_length)
        lr_floor = 0.62
        actor_schedule = optax.linear_schedule(
            init_value=actor_lr,
            end_value=actor_lr * lr_floor,
            transition_steps=total_iters,
        )

        critic_schedule = optax.linear_schedule(
            init_value=critic_lr,
            end_value=critic_lr * lr_floor,
            transition_steps=total_iters * critic_iterations,
        )
        print(
            f"LR decay: linear over {total_iters} iters, "
            f"actor {actor_lr:.1e} --> {actor_lr * lr_floor:.1e} ({total_iters} steps), "
            f"critic {critic_lr:.1e} --> {critic_lr * lr_floor:.1e} ({total_iters * critic_iterations} steps)"
        )
    else:
        actor_schedule = actor_lr
        critic_schedule = critic_lr

    # Initialize optimizers
    actor_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(actor_schedule))
    critic_opt = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(critic_schedule)
    )
    ldm_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(ldm_lr))

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    ldm_opt_state = ldm_opt.init(ldm_params)

    # Initialize environments at difficulty=0 (flat ground)
    env_keys = jax.random.split(k3, num_envs)
    env_state = jax.vmap(env.reset)(env_keys, jp.zeros(num_envs))

    _push_interval_steps = max(int(round(push_interval_s / env.dt)), 1)
    _push_velocity_lo = jp.array(push_velocity_range[0], dtype=jp.float64)
    _push_velocity_hi = jp.array(push_velocity_range[1], dtype=jp.float64)
    _foot_body_ids = env._foot_body_ids
    _nominal_weight = env.nominal_total_mass * env.base_gravity_mag
    _terrain_bump_std = terrain_bump_std if terrain else 0.0

    def _actor_obs_from_state(env_state):
        return env_state.obs

    def _critic_obs_from_state(env_state):
        return env._get_critic_obs(env_state.data, env_state.info)

    def actor_loss(
        actor_params,
        target_critic_params,
        actor_norm_state,
        critic_norm_state,
        env_state,
        randomization,
        current_noise_std,
    ):
        """Short-horizon actor objective with sampled perturbations."""
        action_noise, velocity_pushes, terrain_bump_innovations = randomization

        def rollout_step(carry, inputs):
            state, foot_bump_ou = carry
            noise_t, velocity_push_t, terrain_bump_innov_t = inputs

            push_due = (state.info["step"] > 0) & (
                (state.info["step"] % _push_interval_steps) == 0
            )
            pushed_qvel = state.data.qvel.at[:2].set(velocity_push_t)
            state = state.replace(
                data=state.data.replace(
                    qvel=jp.where(push_due, pushed_qvel, state.data.qvel)
                )
            )

            foot_bump_ou, terrain_bump_forces = differentiated_ou_foot_forces(
                foot_bump_ou,
                terrain_bump_innov_t,
                jax.lax.stop_gradient(state.info["foot_normal_forces"]),
                difficulty=state.info["difficulty"],
                std=_terrain_bump_std,
                decay=terrain_bump_decay,
                robot_weight=_nominal_weight,
            )
            xfrc = state.data.xfrc_applied
            for i in range(4):
                xfrc = xfrc.at[_foot_body_ids[i], :3].add(terrain_bump_forces[i])
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

            # Actor sees noisy observations; critic/training targets keep raw obs.
            obs_rng, env_rng = jax.random.split(state.info["rng"])
            state = state.replace(info={**state.info, "rng": env_rng})
            actor_obs = _actor_obs_from_state(state)
            noisy_actor_obs = env._apply_obs_noise(actor_obs, obs_rng)

            # Compute action with reparameterized noise
            actor_obs_norm = env.normalize_actor_obs(
                actor_norm, actor_norm_state, noisy_actor_obs
            ).astype(jp.float32)
            action = actor.apply(actor_params, actor_obs_norm).astype(jp.float64)

            # Reparameterized action noise
            noisy_action = action + current_noise_std * noise_t.astype(jp.float64)
            noisy_action = jp.clip(noisy_action, -1.0, 1.0)

            next_state = env.step(state, noisy_action)
            foot_bump_ou = jp.where(next_state.done, jp.zeros((4, 3)), foot_bump_ou)

            return (next_state, foot_bump_ou), {
                "reward": next_state.reward,
                "done": next_state.done,
                "terminal": next_state.info["terminal"],
                "actor_obs": actor_obs,
                "critic_obs": _critic_obs_from_state(state),
                "action": noisy_action,
                "bootstrap_critic_obs": next_state.info["bootstrap_critic_obs"],
                "vel_x": next_state.metrics["vel_x"],
                "vel_y": next_state.metrics["vel_y"],
                "yaw_rate": next_state.metrics["yaw_rate"],
                "cmd_x": next_state.metrics["cmd_x"],
                "cmd_y": next_state.metrics["cmd_y"],
                "cmd_yaw": next_state.metrics["cmd_yaw"],
                "height": next_state.metrics["height"],
                "tilt": next_state.metrics["tilt"],
                "foot_normal_FL": next_state.metrics["foot_normal_FL"],
                "foot_normal_FR": next_state.metrics["foot_normal_FR"],
                "foot_normal_RL": next_state.metrics["foot_normal_RL"],
                "foot_normal_RR": next_state.metrics["foot_normal_RR"],
            }

        env_state = jax.lax.stop_gradient(env_state)

        (final_state, final_foot_bump_ou), traj = jax.lax.scan(
            rollout_step,
            (env_state, env_state.info["foot_bump_ou"]),
            (action_noise, velocity_pushes, terrain_bump_innovations),
            length=unroll_length,
        )
        final_state = final_state.replace(
            info={**final_state.info, "foot_bump_ou": final_foot_bump_ou}
        )

        bootstrap_critic_obs = critic_norm.normalize(
            critic_norm_state, traj["bootstrap_critic_obs"]
        ).astype(jp.float32)
        bootstrap_v = critic.apply(target_critic_params, bootstrap_critic_obs).squeeze()

        # Accumulate discounted returns, handling episode boundaries. Time-limit
        # truncations bootstrap from the pre-reset observation stored by env.step.
        def accum_return(carry, x):
            total, running, discount = carry
            r, done, terminal, v_next = x
            next_discount = discount * gamma
            running = running + discount * r
            trunc_bootstrap = (1.0 - terminal) * next_discount * v_next
            total = total + jp.where(done, running + trunc_bootstrap, 0.0)
            running = jp.where(done, 0.0, running)
            discount = jp.where(done, 1.0, next_discount)
            return (total, running, discount), None

        (total_ret, running, final_discount), _ = jax.lax.scan(
            accum_return,
            (0.0, 0.0, 1.0),
            (traj["reward"], traj["done"], traj["terminal"], bootstrap_v),
        )

        final_critic_obs = critic_norm.normalize(
            critic_norm_state, _critic_obs_from_state(final_state)
        ).astype(jp.float32)
        final_v = critic.apply(target_critic_params, final_critic_obs).squeeze()
        final_bootstrap = jp.where(traj["done"][-1], 0.0, final_discount * final_v)

        total_ret = total_ret + running + final_bootstrap

        return -total_ret / unroll_length, (traj, final_state)

    def critic_loss_from_data(
        critic_params,
        target_critic_params,
        critic_norm_state,
        traj_obs,
        traj_rewards,
        traj_dones,
        traj_terminals,
        traj_bootstrap_obs,
        final_obs,
    ):
        """
        Critic TD(lambda) loss using trajectory data collected by the actor.

        Implements Eq. 7 from the SHAC paper (Xu et al., ICLR 2022).
        All in float32 precision.
        """

        flat_obs = traj_obs.reshape(-1, env.critic_obs_dim)
        flat_bootstrap_obs = traj_bootstrap_obs.reshape(-1, env.critic_obs_dim)
        flat_obs_norm = critic_norm.normalize(critic_norm_state, flat_obs).astype(
            jp.float32
        )
        flat_bootstrap_obs_norm = critic_norm.normalize(
            critic_norm_state, flat_bootstrap_obs
        ).astype(jp.float32)
        final_obs_norm = critic_norm.normalize(critic_norm_state, final_obs).astype(
            jp.float32
        )

        # Predicted values V(s_t)
        values = critic.apply(critic_params, flat_obs_norm).squeeze()  # (H,)

        next_v = critic.apply(target_critic_params, flat_bootstrap_obs_norm).squeeze()
        final_v = critic.apply(target_critic_params, final_obs_norm).squeeze()  # scalar

        rewards = traj_rewards.reshape(-1).astype(jp.float32)  # (H,)
        dones = traj_dones.reshape(-1).astype(jp.float32)  # (H,)
        terminals = traj_terminals.reshape(-1).astype(jp.float32)  # (H,)

        def scan_fn(g_next, inputs):
            """
            TD(lambda) backward scan.
            Processes steps t = H-1, H-2, ..., 0.
            """
            r, done, terminal, v_next = inputs
            g_normal = r + gamma * (
                (1.0 - gae_lambda) * v_next + gae_lambda * g_next
            )  # Normal step
            g_trunc = r + gamma * v_next  # Time-limit trunc.
            g_term = r  # true term.
            g = jp.where(terminal, g_term, jp.where(done, g_trunc, g_normal))
            return g, g

        _, targets_reversed = jax.lax.scan(
            scan_fn,
            final_v,  # float32 scalar (determines the carry dtype)
            (rewards[::-1], dones[::-1], terminals[::-1], next_v[::-1]),
        )
        targets = targets_reversed[::-1]

        return jp.mean(jp.square(values - jax.lax.stop_gradient(targets)))

    def learned_dynamics_model_loss(ldm_params, batch):
        """
        Learned dynamics model training loss: MSE on observation residual.

        The LDM predicts delta_obs_norm from (obs_norm, action). Reward is
        computed analytically in the gradient Bellman target.
        Only non-terminal transitions contribute (at done=1, next_obs is
        from a reset, not from dynamics).
        """
        obs, action, next_obs, done = batch
        delta_obs_pred = learned_dynamics_model.apply(ldm_params, obs, action)
        delta_obs_true = next_obs - obs

        mask = 1.0 - done
        n_valid = jp.maximum(jp.sum(mask), 1.0)

        return (
            jp.sum(jp.sum(jp.square(delta_obs_pred - delta_obs_true), axis=-1) * mask)
            / n_valid
        )

    def denormalize_critic_obs(obs_norm, norm_state):
        return obs_norm * jp.sqrt(norm_state.var + critic_norm.eps) + norm_state.mean

    def analytical_reward_from_critic_obs(
        obs_norm, next_obs_norm, action, norm_state
    ):
        obs = denormalize_critic_obs(obs_norm, norm_state)
        next_obs = denormalize_critic_obs(next_obs_norm, norm_state)
        return env.compute_reward_from_critic_obs(obs, next_obs, action)

    def compute_gradient_bellman_targets(
        obs_norm_batch,
        actions_batch,
        ldm_params,
        target_critic_params,
        critic_norm_state,
    ):
        """Compute action-frozen one-step gradient Bellman targets for critic obs."""
        actions_sg = jax.lax.stop_gradient(actions_batch)
        wp = jax.lax.stop_gradient(ldm_params)
        tc = jax.lax.stop_gradient(target_critic_params)
        ns = jax.tree_util.tree_map(jax.lax.stop_gradient, critic_norm_state)

        def one_step_return(o, a):
            delta_o = learned_dynamics_model.apply(wp, o, a)
            next_o = o + delta_o
            r_hat = analytical_reward_from_critic_obs(o, next_o, a, ns)
            next_v = critic.apply(tc, next_o).squeeze()
            return r_hat + gamma * next_v

        return jax.vmap(jax.grad(one_step_return, argnums=0))(
            obs_norm_batch, actions_sg
        )

    def gradient_bellman_loss(critic_params, obs_norm_batch, vg_targets):
        """Match critic input gradients to Bellman gradient targets."""

        def single_critic_grad(o):
            return jax.grad(lambda x: critic.apply(critic_params, x).squeeze())(o)

        pred_grads = jax.vmap(single_critic_grad)(obs_norm_batch)
        targets_sg = jax.lax.stop_gradient(vg_targets)
        return jp.mean(jp.sum(jp.square(pred_grads - targets_sg), axis=-1))

    @jax.jit
    def train_step(state: TrainState):
        key, noise_key, push_key, bump_key, diff_mask_key, ldm_train_key = (
            jax.random.split(state.key, 6)
        )

        # Curriculum: difficulty=0 during grace, then ramp to 1
        difficulty = jp.clip(
            (state.step - _curriculum_grace_jax).astype(jp.float32)
            / _curriculum_steps_jax,
            0.0,
            1.0,
        )

        # Per-env difficulty: a fixed fraction of envs are held at difficulty=0
        # The mask is resampled every unroll
        zero_diff_mask = (
            jax.random.uniform(diff_mask_key, (num_envs,)) < zero_difficulty_frac
        )
        per_env_difficulty = jp.where(
            zero_diff_mask, jp.zeros(num_envs), jp.full((num_envs,), difficulty)
        )

        # Inject per-env difficulty into all non-zeroed-out env states
        updated_env_state = state.env_state.replace(
            info={**state.env_state.info, "difficulty": per_env_difficulty}
        )

        # Pre-sample all stochastic inputs (reparameterization)
        all_action_noise = jax.random.normal(
            noise_key, (num_envs, unroll_length, env.action_dim)
        )
        all_velocity_pushes = jax.random.uniform(
            push_key,
            (num_envs, unroll_length, 2),
            minval=_push_velocity_lo,
            maxval=_push_velocity_hi,
        )
        all_terrain_bump_innovations = jax.random.normal(
            bump_key, (num_envs, unroll_length, 4, 3)
        )
        all_randomization = (
            all_action_noise,
            all_velocity_pushes,
            all_terrain_bump_innovations,
        )

        # Linear noise schedule: start -> end over [0, total_steps]
        progress = jp.clip(state.step / total_steps, 0.0, 1.0)
        current_noise_std = action_noise_std_start + progress * (
            action_noise_std_end - action_noise_std_start
        )

        # Actor update
        actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        (losses, (trajs, final_states)), grads = jax.vmap(
            actor_grad_fn, in_axes=(None, None, None, None, 0, 0, None)
        )(
            state.actor_params,
            state.target_critic_params,
            state.normalizer,
            state.critic_normalizer,
            updated_env_state,
            all_randomization,
            current_noise_std,
        )

        grads = jax.tree_util.tree_map(lambda g: jp.nanmean(g, axis=0), grads)
        grads = jax.tree_util.tree_map(
            lambda g: jp.where(jp.isfinite(g), g, 0.0), grads
        )

        actor_grad_norm = compute_grad_norm(grads)

        updates, new_actor_opt = actor_opt.update(grads, state.actor_opt)
        new_actor_params = optax.apply_updates(state.actor_params, updates)

        # Construct normalized transitions for learned dynamics training.
        final_critic_obs = jax.vmap(_critic_obs_from_state)(final_states)
        traj_critic_obs_flat = trajs["critic_obs"].reshape(-1, env.critic_obs_dim)
        traj_next_critic_obs = jp.concatenate(
            [trajs["critic_obs"][:, 1:, :], final_critic_obs[:, None, :]], axis=1
        )
        traj_next_critic_obs_flat = traj_next_critic_obs.reshape(-1, env.critic_obs_dim)
        traj_actions_flat = trajs["action"].reshape(-1, env.action_dim)
        traj_dones_flat = trajs["done"].reshape(-1)

        # Normalize for LDM (operates in normalized observation space)
        obs_norm_flat = critic_norm.normalize(
            state.critic_normalizer, traj_critic_obs_flat
        ).astype(jp.float32)
        next_obs_norm_flat = critic_norm.normalize(
            state.critic_normalizer, traj_next_critic_obs_flat
        ).astype(jp.float32)

        new_replay_buffer = add_to_replay_buffer(
            state.replay_buffer,
            obs_norm_flat,
            traj_actions_flat,
            next_obs_norm_flat,
            traj_dones_flat,
            ldm_buffer_capacity,
        )

        # Train learned dynamics model
        def ldm_update_step(carry, key_i):
            w_params, w_opt_state = carry
            batch = sample_replay_buffer(new_replay_buffer, key_i, ldm_batch_size)
            loss_val, w_grads = jax.value_and_grad(learned_dynamics_model_loss)(
                w_params, batch
            )
            w_grads = jax.tree_util.tree_map(
                lambda g: jp.where(jp.isfinite(g), g, 0.0), w_grads
            )
            w_updates, new_w_opt = ldm_opt.update(w_grads, w_opt_state)
            new_w_params = optax.apply_updates(w_params, w_updates)
            return (new_w_params, new_w_opt), loss_val

        ldm_train_keys = jax.random.split(ldm_train_key, ldm_iterations)
        (new_ldm_params, new_ldm_opt), ldm_losses = jax.lax.scan(
            ldm_update_step,
            (state.ldm_params, state.ldm_opt),
            ldm_train_keys,
            length=ldm_iterations,
        )

        # Keep old LDM params if buffer too small
        buffer_ready = new_replay_buffer.size >= ldm_batch_size
        new_ldm_params = jax.tree_util.tree_map(
            lambda new, old: jp.where(buffer_ready, new, old),
            new_ldm_params,
            state.ldm_params,
        )
        new_ldm_opt = jax.tree_util.tree_map(
            lambda new, old: jp.where(buffer_ready, new, old),
            new_ldm_opt,
            state.ldm_opt,
        )

        # Critic updates (TD + gradient Bellman)
        all_obs = trajs["critic_obs"]
        all_rewards = trajs["reward"]
        all_dones = trajs["done"]
        all_terminals = trajs["terminal"]
        all_bootstrap_obs = trajs["bootstrap_critic_obs"]
        all_final_obs = final_critic_obs

        # Pre-compute gradient Bellman targets if vg.
        if vg_weight > 0:
            vg_active = jp.where(state.step >= vg_warmup_steps, 1.0, 0.0)

            vg_targets = compute_gradient_bellman_targets(
                obs_norm_flat,
                traj_actions_flat,
                new_ldm_params,
                state.target_critic_params,
                state.critic_normalizer,
            )
            # Mask out non-finite and zero during warmup
            vg_targets = jp.where(jp.isfinite(vg_targets), vg_targets, 0.0)
            vg_targets = vg_targets * vg_active

            _effective_vg = vg_weight * vg_active

        def single_env_critic_loss(
            critic_params,
            target_critic_params,
            norm_state,
            obs,
            rewards,
            dones,
            terminals,
            bootstrap_obs,
            final_obs,
        ):
            return critic_loss_from_data(
                critic_params,
                target_critic_params,
                norm_state,
                obs,
                rewards,
                dones,
                terminals,
                bootstrap_obs,
                final_obs,
            )

        def critic_update_step(carry, _):
            c_params, c_opt_state = carry

            c_losses, c_grads = jax.vmap(
                jax.value_and_grad(single_env_critic_loss, argnums=0),
                in_axes=(None, None, None, 0, 0, 0, 0, 0, 0),
            )(
                c_params,
                state.target_critic_params,
                state.critic_normalizer,
                all_obs,
                all_rewards,
                all_dones,
                all_terminals,
                all_bootstrap_obs,
                all_final_obs,
            )

            c_grads = jax.tree_util.tree_map(lambda g: jp.nanmean(g, axis=0), c_grads)

            if vg_weight > 0:
                # Gradient Bellman loss
                vg_loss_val, vg_grads = jax.value_and_grad(gradient_bellman_loss)(
                    c_params,
                    obs_norm_flat,
                    vg_targets,
                )

                c_grads = jax.tree_util.tree_map(
                    lambda g_td, g_vg: g_td + _effective_vg * g_vg,
                    c_grads,
                    vg_grads,
                )
            else:
                vg_loss_val = jp.float32(0.0)

            c_grads = jax.tree_util.tree_map(
                lambda g: jp.where(jp.isfinite(g), g, 0.0), c_grads
            )

            c_updates, new_c_opt = critic_opt.update(c_grads, c_opt_state)
            new_c_params = optax.apply_updates(c_params, c_updates)

            return (new_c_params, new_c_opt), (
                jp.mean(c_losses),
                vg_loss_val,
            )

        (new_critic_params, new_critic_opt), (critic_losses, vg_losses) = (
            jax.lax.scan(
                critic_update_step,
                (state.critic_params, state.critic_opt),
                None,
                length=critic_iterations,
            )
        )

        # Soft target update
        new_target = optax.incremental_update(
            new_critic_params, state.target_critic_params, target_update_rate
        )

        # Update actor and critic normalizers from their own observation streams.
        flat_actor_obs = trajs["actor_obs"].reshape(-1, env.actor_frame_obs_dim)
        safe_actor_obs = jp.where(
            jp.isfinite(flat_actor_obs), flat_actor_obs, state.normalizer.mean
        )
        new_actor_norm = actor_norm.update(state.normalizer, safe_actor_obs)

        flat_critic_obs = trajs["critic_obs"].reshape(-1, env.critic_obs_dim)
        safe_critic_obs = jp.where(
            jp.isfinite(flat_critic_obs),
            flat_critic_obs,
            state.critic_normalizer.mean,
        )
        new_critic_norm = critic_norm.update(state.critic_normalizer, safe_critic_obs)

        new_state = state.replace(
            key=key,
            env_state=final_states,
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            target_critic_params=new_target,
            normalizer=new_actor_norm,
            critic_normalizer=new_critic_norm,
            actor_opt=new_actor_opt,
            critic_opt=new_critic_opt,
            step=state.step + num_envs * unroll_length,
            ldm_params=new_ldm_params,
            ldm_opt=new_ldm_opt,
            replay_buffer=new_replay_buffer,
        )

        # Collect metrics
        metrics = {
            "reward": jp.mean(trajs["reward"]),
            "vel_x": jp.mean(trajs["vel_x"]),
            "vel_y": jp.mean(trajs["vel_y"]),
            "yaw_rate": jp.mean(trajs["yaw_rate"]),
            "cmd_x": jp.mean(trajs["cmd_x"]),
            "cmd_y": jp.mean(trajs["cmd_y"]),
            "cmd_yaw": jp.mean(trajs["cmd_yaw"]),
            "contact": jp.mean(final_states.metrics["contact_force"]),
            "actor_grad": actor_grad_norm,
            "critic_loss": critic_losses[-1],
            "actor_loss": jp.mean(losses),
            "action_noise_current": current_noise_std,
            "ldm_loss": ldm_losses[-1],
            "vg_loss": vg_losses[-1],
            "vg_active": vg_active if vg_weight > 0 else jp.float32(0.0),
            "track_vx": jp.mean(jp.abs(trajs["vel_x"] - trajs["cmd_x"])),
            "track_vy": jp.mean(jp.abs(trajs["vel_y"] - trajs["cmd_y"])),
            "track_yaw": jp.mean(jp.abs(trajs["yaw_rate"] - trajs["cmd_yaw"])),
            "track_vx_sq": jp.mean((trajs["vel_x"] - trajs["cmd_x"]) ** 2),
            "track_vy_sq": jp.mean((trajs["vel_y"] - trajs["cmd_y"]) ** 2),
            "track_yaw_sq": jp.mean((trajs["yaw_rate"] - trajs["cmd_yaw"]) ** 2),
            "rew_vel_x": jp.mean(final_states.metrics["rew_vel_x"]),
            "rew_vel_y": jp.mean(final_states.metrics["rew_vel_y"]),
            "rew_yaw": jp.mean(final_states.metrics["rew_yaw"]),
            "rew_vz": jp.mean(final_states.metrics["rew_vz"]),
            "pen_rate": jp.mean(final_states.metrics["pen_rate"]),
            "height": jp.mean(trajs["height"]),
            "tilt": jp.mean(trajs["tilt"]),
            "difficulty": difficulty,
            "foot_normal_FL": jp.mean(trajs["foot_normal_FL"]),
            "foot_normal_FR": jp.mean(trajs["foot_normal_FR"]),
            "foot_normal_RL": jp.mean(trajs["foot_normal_RL"]),
            "foot_normal_RR": jp.mean(trajs["foot_normal_RR"]),
        }

        return new_state, metrics

    if resumed_state is not None:
        # Restore learned params and optimizer states
        print(
            f"Restoring learned parameters and optimizer states from step {resumed_step}"
        )
        state = TrainState(
            key=key,
            env_state=env_state,
            actor_params=resumed_state.actor_params,
            critic_params=resumed_state.critic_params,
            target_critic_params=resumed_state.target_critic_params,
            normalizer=resumed_state.normalizer,
            actor_opt=resumed_state.actor_opt,
            critic_opt=resumed_state.critic_opt,
            step=resumed_step,
            critic_normalizer=resumed_state.critic_normalizer,
            ldm_params=resumed_state.ldm_params,
            ldm_opt=resumed_state.ldm_opt,
            replay_buffer=(
                resumed_state.replay_buffer
                if resumed_state.replay_buffer is not None
                else replay_buffer
            ),
        )
    else:
        state = TrainState(
            key=key,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            normalizer=actor_normalizer,
            actor_opt=actor_opt_state,
            critic_opt=critic_opt_state,
            step=0,
            critic_normalizer=critic_normalizer,
            ldm_params=ldm_params,
            ldm_opt=ldm_opt_state,
            replay_buffer=replay_buffer,
        )

    print("Compiling...")
    start_comp_time = time.perf_counter()
    warmup_state, _ = train_step(state)
    compile_time = time.perf_counter() - start_comp_time
    print(f"Compilation took {compile_time:.1f}s")

    # Warm up normalizer from the compilation step.
    state = state.replace(
        normalizer=warmup_state.normalizer,
        critic_normalizer=warmup_state.critic_normalizer,
        replay_buffer=warmup_state.replay_buffer,
    )

    print("Training...")

    if diagnose:
        header = (
            f"{'Step':>7} | {'Rew':>7} | {'TrkVx':>7} | {'TrkVy':>7} | "
            f"{'TrkYaw':>7} | {'RewVx':>7} | {'RewVy':>7} | {'RewYaw':>7} | "
            f"{'PenRate':>7} | {'Height':>7} | "
            f"{'Tilt':>7} | {'Diff':>5} | {'AGrad':>7} | {'Status'}"
        )
    else:
        header = (
            f"{'Step':>7} | {'Rew':>7} | {'TrkVx':>7} | {'TrkVy':>7} | "
            f"{'TrkYaw':>7} | {'AGrad':>7} | {'CLoss':>7} | "
            f"{'LDMLoss':>7} | {'VGLoss':>7} | {'Diff':>5} | {'Status':>8}"
        )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    start = time.time()
    log = []
    diag_log = []
    last_checkpoint_step = state.step

    steps_per_iter = num_envs * unroll_length
    start_iter = resumed_step // steps_per_iter
    total_iters = total_steps // steps_per_iter

    for i in range(start_iter, total_iters):
        state, metrics = train_step(state)

        if i % 10 == 0:
            jax.block_until_ready(state.step)

            vel_x = float(metrics["vel_x"])
            vel_y = float(metrics["vel_y"])
            yaw_rate = float(metrics["yaw_rate"])
            reward = float(metrics["reward"])

            # Per-env tracking errors
            cmd_x = float(metrics["cmd_x"])
            cmd_y = float(metrics["cmd_y"])
            cmd_yaw = float(metrics["cmd_yaw"])
            trk_vx = float(metrics["track_vx"])
            trk_vy = float(metrics["track_vy"])
            trk_yaw = float(metrics["track_yaw"])
            diff = float(metrics["difficulty"])
            max_err = max(trk_vx, trk_vy, trk_yaw)

            if max_err < 0.1:
                status = "TRACK :D"
            elif max_err < 0.25:
                status = "CLOSE :)"
            elif max_err < 0.4:
                status = "TRYING"
            else:
                status = "LEARN"

            if diagnose:
                print(
                    f"{state.step:7d} | {reward:7.2f} | {trk_vx:7.3f} | {trk_vy:7.3f} | "
                    f"{trk_yaw:7.3f} | "
                    f"{metrics['rew_vel_x']:7.2f} | "
                    f"{metrics['rew_vel_y']:7.2f} | "
                    f"{metrics['rew_yaw']:7.2f} | "
                    f"{metrics['pen_rate']:7.3f} | {metrics['height']:7.3f} | "
                    f"{metrics['tilt']:7.2f} | {diff:5.2f} | {metrics['actor_grad']:7.1f} | {status}"
                )

                diag_log.append(
                    {
                        "step": int(state.step),
                        "reward": reward,
                        "difficulty": diff,
                        "vel_x": vel_x,
                        "vel_y": vel_y,
                        "yaw_rate": yaw_rate,
                        "cmd_x": cmd_x,
                        "cmd_y": cmd_y,
                        "cmd_yaw": cmd_yaw,
                        "track_vx": trk_vx,
                        "track_vy": trk_vy,
                        "track_yaw": trk_yaw,
                        "rew_vel_x": float(metrics["rew_vel_x"]),
                        "rew_vel_y": float(metrics["rew_vel_y"]),
                        "rew_yaw": float(metrics["rew_yaw"]),
                        "pen_rate": float(metrics["pen_rate"]),
                        "height": float(metrics["height"]),
                        "tilt": float(metrics["tilt"]),
                        "actor_grad": float(metrics["actor_grad"]),
                        "critic_loss": float(metrics["critic_loss"]),
                        "ldm_loss": float(metrics["ldm_loss"]),
                        "vg_loss": float(metrics["vg_loss"]),
                        "vg_active": float(metrics["vg_active"]),
                    }
                )
            else:
                ldm_str = (
                    f"{float(metrics['ldm_loss']):7.4f}" if vg_weight > 0 else "    n/a"
                )
                vg_str = (
                    f"{float(metrics['vg_loss']):7.4f}" if vg_weight > 0 else "    n/a"
                )
                print(
                    f"{state.step:7d} | {reward:7.3f} | {trk_vx:7.3f} | {trk_vy:7.3f} | "
                    f"{trk_yaw:7.3f} | "
                    f"{metrics['actor_grad']:7.2f} | {metrics['critic_loss']:7.4f} | "
                    f"{ldm_str} | {vg_str} | {diff:5.2f} | {status}"
                )

            if DEBUG_FOOT_CONTACTS:
                print(
                    f"         foot GRF (N):  "
                    f"FL={float(metrics['foot_normal_FL']):7.2f}  "
                    f"FR={float(metrics['foot_normal_FR']):7.2f}  "
                    f"RL={float(metrics['foot_normal_RL']):7.2f}  "
                    f"RR={float(metrics['foot_normal_RR']):7.2f}"
                )

            log.append(
                [
                    state.step,
                    reward,
                    vel_x,
                    vel_y,
                    yaw_rate,
                    cmd_x,
                    cmd_y,
                    cmd_yaw,
                    float(metrics["actor_loss"]),
                    float(metrics["contact"]),
                    float(metrics["actor_grad"]),
                    float(metrics["critic_loss"]),
                    float(metrics["track_vx_sq"]),
                    float(metrics["track_vy_sq"]),
                    float(metrics["track_yaw_sq"]),
                    diff,
                    float(metrics["ldm_loss"]),
                    float(metrics["vg_loss"]),
                    float(metrics["vg_active"]),
                ]
            )

            # Save best policy
            if reward > best_reward and state.step > 5000:
                best_reward = reward
                with open(f"{save_dir}/policy_best.pkl", "wb") as f:
                    pickle.dump(slim_checkpoint_state(state), f)
                print(f"  >> New best! Reward: {best_reward:.3f}")

            # Periodic checkpoint
            if state.step - last_checkpoint_step >= checkpoint_interval:
                ckpt_path = os.path.join(save_dir, "checkpoint_latest.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump(slim_checkpoint_state(state), f)
                last_checkpoint_step = state.step
                print(f"  >> Checkpoint saved at step {state.step}")

    # Save final state and logs
    with open(f"{save_dir}/policy_final.pkl", "wb") as f:
        pickle.dump(slim_checkpoint_state(state), f)
    np.save(f"{save_dir}/log.npy", np.array(log))

    if diagnose and diag_log:
        with open(f"{save_dir}/diag_log.json", "w") as f:
            json.dump(diag_log, f, indent=2)
        print(f"Diagnostic log saved to {save_dir}/diag_log.json")

    elapsed = time.time() - start
    cmd_str = (
        f"vx=[{cmd_vel_x_range[0]:.2f},{cmd_vel_x_range[1]:.2f}], "
        f"vy=[{cmd_vel_y_range[0]:.2f},{cmd_vel_y_range[1]:.2f}], "
        f"yaw=[{cmd_yaw_rate_range[0]:.2f},{cmd_yaw_rate_range[1]:.2f}] "
        f"| zero_prob={cmd_zero_prob} interval={cmd_ctrl_interval_range}"
    )
    print("=" * (160 if diagnose else 120))
    print(f"Training complete in {elapsed:.1f}s (compilation: {compile_time:.1f}s)")
    print(f"Command ranges: {cmd_str}")
    print("=" * 100)
    print(f"Training complete in {elapsed:.1f}s (compile: {compile_time:.1f}s)")
    print(
        f"Curriculum: grace={curriculum_grace}, curriculum_steps={curriculum_steps}, "
        f"terrain={'ON' if terrain else 'OFF'}, terrain_flat_prob={terrain_flat_prob}, "
        f"terrain_slope_max={terrain_slope_max} deg, "
        f"terrain_bump_std={terrain_bump_std if terrain else 0.0}"
    )
    print(f"Best reward: {best_reward:.3f}")

    # Update hyperparameters with the final result.
    hparams["best_reward"] = best_reward
    with open(f"{save_dir}/hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    return state, save_dir
