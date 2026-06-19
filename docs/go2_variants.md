# Go2 Variants

Select variants with the `variant` field in a config file or with
`python -m src.main go2 --variant ...`.

| Variant | Actor linear velocity | Kinematic reference | High-speed commands |
| --- | --- | --- | --- |
| `blind_nolinvel_nokinref` | no | no | no |
| `blind_linvel_nokinref` | yes | no | no |
| `blind_linvel_kinref` | yes | yes, including gait phase observations | no |
| `highspeed_nokinref` | yes | no | yes |

Each variant has its own environment implementation in
`src/envs/go2/variants/<variant>/environment.py` and its own C++ deployment
code in `src/envs/go2/variants/<variant>/deploy_cpp/`.

