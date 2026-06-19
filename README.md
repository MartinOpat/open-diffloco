# Open-DiffLoco

Open-DiffLoco is a framework for training deployable blind quadruped
locomotion policies with differentiable simulation in MuJoCo MJX.

It implements the Short-Horizon Actor-Critic [(SHAC)](https://short-horizon-actor-critic.github.io/) algorithm, as well as a new method, Jacobian Augmented Value Estimation or JAVE, which further improves SHAC. This same idea could also be adopted to other successors of SHAC, e.g. Adaptive-Horizon Actor-Critic [(AHAC)](https://adaptive-horizon-actor-critic.github.io/).

Documentation:

- [Go2 variants and configs](docs/go2_variants.md)
- [Algorithms](docs/algorithms.md)
- [Implicit terrain](docs/terrain.md)
- [Deployment](docs/deployment.md)

TODO: Document features which are implemented and have been tested but have been found not strictly necessary and thus are disabled by default, namely:
  - Training curriculum
  - Learning rate decay
  - Stochastic policy formulation (std as a learned parameter of the NN)

## Go2 Variants

Training uses `embodiment: go2` plus one of these `variant` values:

- `blind_nolinvel_nokinref`: no actor linear velocity and no kinematic reference.
- `blind_linvel_nokinref`: actor linear velocity and no kinematic reference.
- `blind_linvel_kinref`: actor linear velocity, gait phase observations, and a kinematic reference (based on [Luo et al.](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb)) in control and rewards.
- `highspeed_nokinref`: high-speed locomotion without a kinematic reference.

Example:

```bash
python -m src.main --config src/configs/jave_go2.yaml
python -m src.main go2 --algorithm shac --variant blind_linvel_nokinref
```

## Algorithms

- SHAC: `src/algorithms/shac/algorithm.py`
- JAVE: `src/algorithms/jave/algorithm.py`

See [docs/algorithms.md](docs/algorithms.md) for algorithm notes.

## C++ Deployment

Build without ROS2:

```bash
cmake -S . -B build -DOPEN_DIFFLOCO_ENABLE_ROS2=OFF
cmake --build build
```

The build still requires the Unitree C++ SDK, Eigen, and zlib. The ROS2 flag is
off by default.

Build with ROS2 command-message support:

```bash
cmake -S . -B build-ros2 -DOPEN_DIFFLOCO_ENABLE_ROS2=ON
cmake --build build-ros2
```

Each variant has its own executable:

- `deploy_blind_nolinvel_nokinref`
- `deploy_blind_linvel_nokinref`
- `deploy_blind_linvel_kinref`
- `deploy_highspeed_nokinref`

Terminal control is the default command source:

```bash
./build/src/envs/go2/variants/blind_nolinvel_nokinref/deploy_cpp/deploy_blind_nolinvel_nokinref \
  --policy path/to/policy_deploy.npz \
  --interface lo \
  --domain-id 1 \
  --command-source terminal
```

Wireless control reads the Unitree remote directly from `LowState.wireless_remote`:

```bash
./build/src/envs/go2/variants/blind_nolinvel_nokinref/deploy_cpp/deploy_blind_nolinvel_nokinref \
  --policy path/to/policy_deploy.npz \
  --interface enp3s0 \
  --command-source wireless
```

ROS2 command-message control is available when built with
`OPEN_DIFFLOCO_ENABLE_ROS2=ON`. The `ros2` command source reads velocity
commands from the configured command topic.

```bash
./build-ros2/src/envs/go2/variants/blind_nolinvel_nokinref/deploy_cpp/deploy_blind_nolinvel_nokinref \
  --policy path/to/policy_deploy.npz \
  --interface lo \
  --domain-id 1 \
  --command-source ros2 \
  --cmd-topic /velocity_command
```

Velocity command publishers are available under `src/deploy/`:

```bash
python -m src.deploy.terminal_command --control diffloco
python -m src.deploy.wireless_command --net lo --control diffloco --topic /velocity_command
```
