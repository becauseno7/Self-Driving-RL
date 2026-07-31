# Neon Highway V6: preference-guided driving

V5 learned the route and became safe, but it still spent too much time behind
traffic and often reversed lane changes. V6 asks a more human question than
"did the episode finish?": which of two complete drives was the better overall
drive?

## What RLAIF means here

RLAIF is reinforcement learning from AI feedback. We collected matched
trajectories on identical hard-mode traffic seeds, then ranked them with this
ordered rubric:

1. Avoid crashes and near misses.
2. Finish the route and resolve traffic challenges.
3. Take safe, useful passing opportunities instead of waiting behind traffic.
4. Maintain progress without being repeatedly passed.
5. Avoid rapid lane changes, reversals, and control jerk.

The preference dataset contains 96 trajectories, 160 matched comparisons, and
128 individually reviewed AI labels. Labels include a short pair-specific
reason so the supervision is auditable. The resulting constrained
Bradley-Terry reward model reached 87.5% validation accuracy and 83.3% test
accuracy. Its largest positive signals were cruise quality, distance,
completion, and overtakes; its largest penalties were crashes, near misses,
being passed, missed passing chances, and reversals.

The checked-in preference labels live at
`data/rlaif/codex_preferences_v1.json`. Generated trajectories, checkpoints,
and model artifacts live under `runs/rlaif/` and are intentionally ignored by
Git.

## Why V6 is a guarded residual

Two direct experiments were useful but were not promoted:

- Reward-only QR-DQN fine-tuning became calmer and overtook more, but its
  independent crash rate rose to 6% and it missed more easy passes.
- Directly distilling corrections into the QR-DQN value head changed unrelated
  actions because many V5 action values are separated by tiny margins.

The promoted architecture therefore freezes V5 and trains a small residual
network with three choices: leave V5 unchanged, make a calm correction, or
take a pass. It sees the 33-value road observation, V5's proposed action, time
since the last lane change, and the last lane-change direction.

Two protections limit distribution shift:

- Confidence gating preserves ordinary V5 decisions unless the residual is
  sufficiently sure.
- A deterministic safety shield reconstructs the road checks from observable
  state. A pass must have adequate front clearance, rear clearance, rear TTC,
  and useful progress. A calm correction may only cancel a recent low-threat
  reversal and must preserve V5's pedal command.

The neural model decides which valid opportunities are worth acting on; the
shield only prevents invalid proposals from reaching the car.

## Reproduce training

Collect matched trajectories and fit the preference reward:

```powershell
uv run sdr-rlaif collect `
  --episodes 16 --seed 70000 `
  --output runs/rlaif/v1/preferences.json

uv run sdr-rlaif train-reward `
  --dataset runs/rlaif/v1/preferences.json `
  --labels data/rlaif/codex_preferences_v1.json `
  --target-reward-std 2 `
  --output runs/rlaif/v1/reward_model_v2.json
```

Train the residual on the RTX 4070. Completed teacher trajectories are used so
crashed demonstrations cannot become positive supervision:

```powershell
uv run sdr-rlaif override-train `
  --base-model runs/game/v5-good-driver-2p5m-restart/model.zip `
  --reward-model runs/rlaif/v1/reward_model_v2.json `
  --run-name override-v6-full-1 `
  --train-episodes 120 --validation-episodes 30 `
  --epochs 25 --device cuda --eval-episodes 100
```

The final run contained 54,000 driving states: 47,840 unchanged V5 choices,
5,896 calm corrections, and 264 pass corrections. Gate calibration used the
development seeds beginning at 80,000. The untouched final evaluation used
seeds beginning at 130,000.

Package the selected gates:

```powershell
uv run sdr-rlaif override-calibrate `
  --input runs/rlaif/override-v6-full-1/override_model.pt `
  --output runs/rlaif/v6-good-driver/override_model.pt `
  --calm-threshold 0.80 --passing-threshold 0.50 `
  --development-seed 80000 --heldout-seed 130000
```

## Held-out result

Both policies saw the same 100 hard-mode routes, seeds 130,000 through 130,099.

| Metric | V5 frozen base | V6 RLAIF | Change |
|---|---:|---:|---:|
| Completion | 98% | 98% | equal |
| Crash | 2% | 2% | equal |
| Mean speed | 73.1 km/h | 72.4 km/h | -0.7 km/h |
| Net passes / route | +5.60 | +5.35 | -0.25 |
| Passing response | 83.85% | 86.59% | +2.74 pp |
| Missed pass opportunities | 0.71 | 0.41 | -42% |
| Blocked steps | 10.55% | 5.08% | -52% |
| Lane changes | 16.93 | 8.94 | -47% |
| Rapid lane changes | 12.99 | 3.82 | -71% |
| Lane reversals | 11.10 | 3.63 | -67% |
| Minimum TTC | 1.57 s | 2.23 s | +0.66 s |

V6 does not maximize raw overtakes. It gives up a small amount of speed and
net passes in exchange for taking a larger share of easy opportunities,
spending far less time blocked, and eliminating most wasteful weaving without
losing held-out safety. That trade is why it is the recommended "good driver"
rather than an assertive racing policy.

## Watch or evaluate

```powershell
uv run sdr-rlaif watch `
  --base-model runs/game/v5-good-driver-2p5m-restart/model.zip `
  --override-model runs/rlaif/v6-good-driver/override_model.pt `
  --difficulty hard --episodes 10

uv run sdr-rlaif override-evaluate `
  --base-model runs/game/v5-good-driver-2p5m-restart/model.zip `
  --override-model runs/rlaif/v6-good-driver/override_model.pt `
  --difficulty hard --episodes 100 --seed 130000
```

For continuous driving, add `--endless`. The viewer resets the residual's
short lane-change memory after every crash before starting the next run.
