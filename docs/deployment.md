# Deployment

## Exporting a trained policy

The C++ deployment executables consume a `.npz` file, not the pickled
training checkpoints. Export a checkpoint with:

```bash
python -m src.deploy.export_policy training_runs/<run>/policy_best.pkl
```

This writes `training_runs/<run>/policy_best_deploy.npz` containing the actor
weights, observation-normalizer statistics, actuator gains, and environment
metadata. The environment variant is read from the run's `hparams.json`, so
variant-specific data (e.g. the kinematic gait reference for
`blind_linvel_kinref`) is included automatically.

> **Warning:** the C++ deployment clamps velocity commands to ranges
> hard-coded in each variant's `policy.hpp`. They are not read from the
> `.npz`. Keep them in sync with the training command ranges printed by the
> export script.

## Building the C++ deployment

C++ deployment code is co-located with each Go2 variant and can be built from the repository root.

Build without ROS2:

```bash
cmake -S . -B build -DOPEN_DIFFLOCO_ENABLE_ROS2=OFF
cmake --build build
```

Build with ROS2 command-message support:

```bash
cmake -S . -B build-ros2 -DOPEN_DIFFLOCO_ENABLE_ROS2=ON
cmake --build build-ros2
```

The `OPEN_DIFFLOCO_ENABLE_ROS2` option enables ROS2 velocity-command input.
Terminal and wireless command sources are available in the ROS2-free build.

Command sources:

- `terminal`: keyboard command input inside the C++ deployment executable.
- `wireless`: command input from `LowState.wireless_remote`.
- `ros2`: command input from the configured ROS2 velocity command topic,
  available only with `OPEN_DIFFLOCO_ENABLE_ROS2=ON`.

Velocity command publishers are available under `src/deploy/`:

```bash
python -m src.deploy.terminal_command --control diffloco
python -m src.deploy.wireless_command --net lo --control diffloco --topic /velocity_command
```

They publish command messages for the C++ deployment `ros2` command source.
