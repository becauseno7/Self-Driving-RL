# Neon Highway V2: smooth speed control

V2 replaces instant speed jumps with a simple cruise-control model.

## New action meaning

The five discrete actions remain the same, but the speed actions now modify an
intention rather than teleporting the vehicle to a new speed:

- `LANE LEFT`: request the adjacent left lane,
- `HOLD`: preserve the current target lane and target speed,
- `LANE RIGHT`: request the adjacent right lane,
- `SPEED +`: raise target speed by 1 m/s (3.6 km/h),
- `SPEED -`: lower target speed by 1 m/s (3.6 km/h).

The car then accelerates or brakes smoothly toward the target. Its controller is
limited to 2.8 m/s² acceleration and 5.2 m/s² braking.

Repeated `SPEED +` or `SPEED -` presses now mean that the agent is intentionally
moving its target, while repeated `HOLD` means it is maintaining that target.

## Observation

Target speed must be visible to the agent; otherwise identical observations
could produce different acceleration. V2 therefore uses 16 values:

```text
[actual speed, target speed, current lane, target lane,
 lane 1 ahead gap, relative speed, behind gap,
 lane 2 ahead gap, relative speed, behind gap,
 lane 3 ahead gap, relative speed, behind gap,
 lane 4 ahead gap, relative speed, behind gap]
```

Because V1 used 15 values, V1 models cannot run in V2. The CLI detects this and
prints a clear retraining command instead of failing inside the neural network.

## Dashboard feedback

The Driver State panel now displays:

- actual speed,
- target speed,
- longitudinal acceleration,
- throttle pressure,
- brake pressure,
- the selected action.

Brake lights are connected to actual brake pressure, not merely to the selected
action. Small comfort costs for acceleration magnitude and target-speed changes
encourage smoother control without making safety secondary.

Unsafe lane changes now receive an immediate safety penalty. This gives DQN a
dense learning signal at the decision that caused most V2 side impacts, instead
of making it infer the mistake only from a crash several simulation steps later.

For longer experiments, `--preset high` increases the Q-network from two
128-unit layers to two 256-unit layers, expands replay memory, and uses larger
batches with a conservative update ratio. Pair it with `--device cuda
--headless` to use the configured NVIDIA GPU efficiently. Long runs evaluate
periodically and retain the safest checkpoint instead of assuming the final
DQN update must be the best one.

## Reward-tuning result

Target-speed oscillation was measured from the held-out action counts rather
than judged only by watching the animation:

| Target-change cost | Hold target | Speed adjustments | Completion |
|---|---:|---:|---:|
| none | 30% | 65% | 70% |
| medium | 45% | 49% | 55% |
| selected | 58% | 35% | 55% |

## High-compute safety result

The longer CUDA experiment did not assume its final update was its best. It
evaluated every 25,000 steps and selected the 125,000-step checkpoint. Across
two separate sets of 100 deterministic traffic seeds, it reduced crash rate
from 43% to 33% and from 45% to 28% compared with the previous V2 model. Side
impacts fell from 30–33 per set to 12–14.

The selected policy intentionally trades some raw completion performance for a
much steadier control intention. Longer training did not help this configuration
in the first seed, which is a reminder that DQN training is not guaranteed to
improve monotonically.

## Selected V2 benchmark

The selected model was trained for 30,000 steps with seed 7 and evaluated on 20
fixed episodes beginning at seed 10,000.

| Metric | Random driver | Smooth V2 DQN |
|---|---:|---:|
| Completion rate | 0% | 55% |
| Crash rate | 100% | 45% |
| Mean episode length | 66.40 | 310.00 |
| Mean return | -7.35 | +15.19 |
