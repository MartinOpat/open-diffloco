# Algorithms

Open-DiffLoco includes SHAC and JAVE:

- SHAC: `src/algorithms/shac/algorithm.py`
- JAVE: `src/algorithms/jave/algorithm.py`

NOTE: JAVE can be further improved to allow more faithful calculation of the critic gradient
in an asymmetric critic-actor setup by designing differentiable transition
functions between the critic and actor observations.
