# Neon Highway V5: learning to drive through traffic

V4 learned to survive the staged waves, but its safest strategy between them
was often to sit behind traffic. V5 gives "good driver" a measurable meaning:
make progress through slower traffic, pass only through a safe gap, cruise at a
sensible highway speed, and do not weave merely to collect reward.

## Metrics before reward

The frozen V4 checkpoint and a hand-written proactive driver were evaluated on
the same 100 development episodes beginning at seed 60,000. Final model
selection will use a different held-out seed range.

| Development metric | Frozen V4 policy | Proactive heuristic |
|---|---:|---:|
| Completion | 70% | 99% |
| Crash | 30% | 1% |
| Mean speed | 67 km/h | 81 km/h |
| Net overtakes per route | +2.23 | +10.05 |
| Lane changes per route | 8.51 | 5.17 |
| Safe passing opportunities answered | 43% | 100% |
| Time blocked despite a safe passing lane | 18.2% | 0% |

This proves the scenario is navigable without reckless speed or constant lane
changes. The heuristic is a ceiling check, not the final solution: the project
still trains a neural policy from experience.

## What counts as a pass

An overtake is counted only when the ego car genuinely crosses from behind a
traffic car to ahead of it during normal physics. Counting happens before
off-screen recycling, so a teleported car cannot create a fake pass. A crash
step never earns an overtake.

A safe passing option exists only when all of these are true:

- the current leader is within 32 m and at least 1 m/s slower;
- an adjacent lane has the existing safe front gap, rear gap and rear TTC;
- that lane provides at least 8 m more useful clearance and a better traffic
  flow than the current lane.

The agent is not rewarded merely for starting a lane change. It still pays the
comfort cost, so weaving without completing passes is strictly worse.

## V5 reward

The ordinary progress reward now reaches its maximum at 27 m/s (97 km/h)
instead of paying the agent to chase the absolute 122 km/h limit:

```text
progress = 0.01 + 0.09 * clip((speed - 8) / (27 - 8), 0, 1)
```

The first 300k pilot learned to pass, but it traded safety for speed: its late
policy reached +4.1 net passes while crashing in 52% of validation routes.
That failed the safety-first gate. The revised terms make a crash dominate the
value of several passes, and charge only lane changes that actually begin:

```text
+0.60   completed overtake
-0.35   another vehicle passes the ego
-0.005  per step spent behind a slower leader while a safe pass is available
-0.05   lane change that actually starts
-0.12   choosing the wrong lane when a safe passing lane is available
-0.75   lane change whose initial front/rear gap is unsafe
-30     crash
+20     route completion
```

The potential-based safety teacher now watches both lanes throughout a merge,
including minimum front/rear clearance as well as TTC. It remains
potential-based, so it supplies earlier learning feedback without making slow
driving intrinsically valuable.

## Stable QR-DQN training

V4 used hard target-network copies and validation swung from 20% to 72% between
nearby checkpoints. The V5 GPU preset uses `tau=0.02` Polyak updates every 500
steps, halves the learning rate to `1.5e-4`, expands replay to 750,000 samples,
and explores for 30% of the run.

The intended long run is:

```powershell
uv run sdr-game learn `
  --run-name v5-good-driver-2p5m `
  --preset gpu --algo qrdqn --device cuda `
  --headless --envs 8 --timesteps 2500000 `
  --validation-freq 125000 --validation-episodes 100 `
  --eval-episodes 100
```

Success is safety first: at least the frozen policy's 70% development
completion, then higher speed and net overtakes with fewer lane changes and
less blocked time. Final reporting uses untouched seeds beginning at 120,000.

## Pilot result

The revised 300k-step safety pilot passed its promotion gate. On the same
development seeds used above it completed 78% of routes, crashed on 22%, drove
at 67 km/h, made +1.31 net passes, and used 6.76 lane changes per route. On the
separate default evaluation seeds it reached 79% completion and +2.20 net
passes. This is the configuration promoted to the long run; the pilot model is
`runs/game/v5-safe-driver-pilot-300k/best_model.zip`.

## Long-run result

The promoted configuration trained for 2.5 million steps on the RTX 4070. Two
models were retained and evaluated once on the untouched 100-episode seed
range beginning at 120,000:

| Held-out metric | Safety-selected model | Final live model |
|---|---:|---:|
| Completion | **96%** | 91% |
| Crash | **4%** | 9% |
| Mean speed | 74 km/h | **90 km/h** |
| Net overtakes per route | +5.98 | **+15.79** |
| Lane changes per route | **16.58** | 24.79 |
| Side impacts | **1** | 6 |

The safety-selected model is the default: it more than doubles the frozen V4
policy's development net progress and reaches 96% completion on held-out
evaluation. The final live model is useful as an experimental
assertive policy, but its extra speed and overtakes do not justify twice as
many crashes and substantially more lane changes.

```text
runs/game/v5-good-driver-2p5m-restart/model.zip       # recommended
runs/game/v5-good-driver-2p5m-restart/last_model.zip  # assertive experiment
```
