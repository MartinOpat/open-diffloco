# Implicit Terrain

Terrain randomization is disabled by default. Enable it with `--terrain` or the
corresponding config value.

The current terrain model combines a per-episode tilted gravity vector with
contact-scaled dOU foot bumps. This approximates slope and local ground
unevenness without relying on MuJoCo contact geometry features that are not yet
sufficiently differentiable for this training loop. The main knob is
`--terrain-slope`, the maximum slope angle at full curriculum difficulty.

Random push disturbances use velocity pushes, to further improve robustness.
