# Deployment

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
