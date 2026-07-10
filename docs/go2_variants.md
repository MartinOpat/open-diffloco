# Go2 Variants

Select variants with the `variant` field in a config file or with
`python -m src.main go2 --variant ...`.

| Variant | Actor linear velocity | Kinematic reference | High-speed reward shaping |
| --- | --- | --- | --- |
| `blind_nolinvel_nokinref` | no | no | no |
| `blind_linvel_nokinref` | yes | no | no |
| `blind_linvel_kinref` | yes | yes, including gait phase observations | no |
| `highspeed_nokinref` | no | no | yes |

Each variant has its own environment implementation in
`src/envs/go2/variants/<variant>/environment.py` and its own C++ deployment
code in `src/envs/go2/variants/<variant>/deploy_cpp/`.

## Command ranges

Velocity command ranges default per variant.
Can be overridden with
`--cmd-vel-x-range`, `--cmd-vel-y-range`, and `--cmd-yaw-rate-range` (or the
matching config keys):

| Variant | vx (m/s) | vy (m/s) | yaw rate (rad/s) |
| --- | --- | --- | --- |
| `blind_*` | (-2.0, 2.0) | (-1.0, 1.0) | (-1.5, 1.5) |
| `highspeed_nokinref` | (-3.0, 3.0) | (0.0, 0.0) | (-1.0, 1.0) |
