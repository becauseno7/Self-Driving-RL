# Neon Highway V4: honest physics and a horizon worth planning over

V4 is a correctness release. V3 reported 56% completion on hard mode, but five
bugs meant the agent was learning against a simulation that did not behave the
way its sensors described, and the discount factor made the goal invisible.

## Physics bugs fixed

**Traffic drove through traffic.** The follower model only treated cars strictly
ahead as obstacles and capped deceleration too late to matter. A 34 m/s car
placed 30 m behind an 8 m/s car passed through it and ended 108 m in front.
Traffic now uses the Intelligent Driver Model, and `_resolve_traffic_overlap`
projects out any remaining overlap front-to-back. The ego is deliberately
excluded from that projection, because being rear-ended has to stay possible.

**The middle of every lane change was a blind spot.** The ego moves 0.24 lanes
per step and `LANE_COLLISION_WIDTH` was 0.42, so at lane position 1.48 the
nearest lane centre was 0.48 away and nothing could hit the car. A blocker
parked exactly on top of the ego went undetected for that frame. Collisions now
test the lateral distance swept between the previous and current position, and
the width is 0.55 — it must exceed 0.5 or the hole reopens. Traffic yields to a
merging ego from 0.8 lanes away, so it no longer ignores a car cutting in.

**Collisions blamed the wrong car.** `_detect_collision` returned the first
match in list order. With a car at +4.0 m and another at −0.5 m closing 33 m/s,
it reported the far one. Every impact-type breakdown in the V3 notes was built
on this. It now reports the nearest car.

**Sensor gaps were centre-to-centre.** A reported gap of 4.5 m was already a
crash. Gaps are now bumper-to-bumper, so distance and time-to-collision mean
what the names say.

**Recycled cars ping-ponged.** A car passing +190 m was re-inserted behind the
lane's rearmost car, which could put it past the −55 m recycle line so it
teleported straight back. Re-insertion is now clamped relative to the ego.

There is also an absolute step limit. `completed` required every wave cleared,
so an episode where a wave never staged had no way to end at all.

## Actions: steering and pedal are chosen together

Through V3 one discrete slot held both. A full brake needed roughly 3 s of
committed input, and no lane change could be issued during it — the hard waves
are built to require exactly that combination. The action space now factors as
`steer * 3 + pedal` over `{LEFT, KEEP, RIGHT} x {BRAKE, COAST, GAS}`, giving 9
flat actions that DQN can still consume.

## Discount horizon

V3 trained at `gamma = 0.98`. Episodes are 450 steps, so the effective horizon
was 50 steps — five seconds. The route-completion bonus was worth 0.00056 at the
start of an episode. The agent was not failing to plan; it could not see the
goal. V4 uses 0.995 and scales the terminal rewards to match.

| Reward, seen from step 0 | V3 (gamma 0.98) | V4 (gamma 0.995) |
|---|---:|---:|
| Route completion | +0.0006 | +1.57 |
| Wave cleared, 70 steps out | +0.24 | +2.11 |

Time-to-collision shaping is now potential-based, so it speeds learning up
without changing which policy is optimal. The old raw per-step threat penalty
paid the agent to crawl.

## Observation: 20 -> 33 values

```text
[ 0] speed              [ 1] target speed
[ 2] lane position      [ 3] target lane
[ 4] lane-change progress
[ 5] longitudinal acceleration
[ 6] time remaining     [ 7] wave active   [ 8] waves cleared
then per lane (x4): front gap, front relative speed, front urgency,
                    rear gap,  rear relative speed,  rear urgency
```

The clock and wave state are new and necessary: the reward has terminal and
wave-completion terms that were previously invisible to the agent, which made
the task non-Markov in exactly the place it mattered. "Urgency" is normalized
inverse time-to-collision — the quantity every decision turns on, which the
network previously had to reconstruct by dividing two badly-scaled inputs.
Relative speeds are now scaled by their true ±26 m/s span rather than by
`MAX_SPEED`, which left a third of the channel unused.

## Algorithm and symmetry

The V3 policy had collapsed onto one escape direction:

| First wave escapes | V3 completion |
|---|---:|
| LEFT | 62% (n=101) |
| RIGHT | 48% (n=99) |

with LANE LEFT chosen in 6% of steps against LANE RIGHT in 2%. The task is
symmetric by construction, so that gap is the signature of Q-value
overestimation, which SB3's DQN has no Double-Q correction for.

V4 adds `--algo qrdqn` (distributional, from sb3-contrib) and a `MirrorSymmetry`
wrapper that mirrors a random half of training episodes. Mirroring is an exact
symmetry of this environment — `tests/test_symmetry.py` asserts a mirrored
episode earns an identical return — so it is free data that also forces both
directions to share one set of weights.

## Throughput

`lane_sensors()` ran five times per step and was the environment's bottleneck;
it is now memoized within a step and dropped before `step` and `reset` return,
so external callers still always see the truth. Training also runs on a
`SubprocVecEnv`:

```powershell
uv run sdr-game learn --preset gpu --algo qrdqn --headless --envs 8 --timesteps 2000000
```

V3 did 0.25 gradient steps per env step from one environment. The `gpu` preset
doubles the replay ratio and collects eight times faster.

## Configurable route length

`EPISODE_SECONDS` was a fixed 45 s with waves hard-coded at steps 0, 150 and
300. It is now the `episode_seconds` constructor argument and a `--seconds`
flag on every subcommand, and the wave schedule derives from it: one wave per
`CHALLENGE_INTERVAL_STEPS` (15 s), with the count trimmed so the last wave
still has time to be cleared.

```text
 45 s ->  450 steps ->  3 waves at 0, 150, 300        (unchanged default)
 90 s ->  900 steps ->  6 waves
180 s -> 1800 steps -> 12 waves
```

Holding wave density constant keeps the task coherent at any length. The cost
is that longer routes are strictly harder, since survival compounds: the
45 s-trained model drops from 75% to 22% completion on a 120 s route while
clearing twice as many waves (2.70 -> 5.45). Train at the length you intend to
measure.

## Endless mode

`--endless` turns the route into a lap. When a route would have finished, the
lap is banked and the car keeps driving; only a crash ends the episode.

The design decision that matters is what a lap does to the observation. The
episode clock and wave counter **recycle**: `lap_step` returns to zero, the
wave counter resets, and feature 6 (time remaining) climbs back to 1.0. Each
lap therefore looks to the agent exactly like a fresh route, which is why a
policy trained on fixed 45 s routes runs in endless mode without retraining. A
naive implementation that let the clock run to zero and stay there would have
pushed every policy off-distribution the moment the first route ended.

A banked lap pays `COMPLETION_BONUS`, the same as finishing a fixed route,
because it is the same achievement. The potential-based shaping is untouched:
its potential is zeroed only on a genuine terminal state, and a lap is not one.

`ENDLESS_STEP_LIMIT` (100,000 steps, ~2.8 simulated hours) still bounds the
episode so a flawless policy cannot hang a training run.

Completion rate carries no information in endless mode — every episode ends in
a crash sooner or later — so `EvaluationSummary` gained `mean_laps`, and
`evaluate --endless` reports laps and steps survived. The 2M-step model
averages 1.80 laps across two 30-episode seed sets (1.87 and 1.73), roughly
1,000 steps or 100 simulated seconds per life.

## Results

`runs/game/v4-qrdqn-2m`: QR-DQN, 2,000,000 steps, 8 parallel environments,
mirrored training episodes, hard mode, ~45 minutes on the RTX 4070.

The two V3 defects this release targeted are gone:

| Measure | V3 | V4 |
|---|---:|---:|
| Completion when the wave escapes LEFT | 62% | 62% |
| Completion when the wave escapes RIGHT | 48% | 62% |
| Directional gap | 14.0 points | 0.8 points |
| Steering mix (LEFT / RIGHT) | 6% / 2% | 4.6% / 5.4% |
| Lane changes combined with a pedal input | impossible | 60% |

Measured over 300 hard episodes from seed 10,000.

Completion rate across five independent 100-episode seed sets:

| Seeds | 10,000 | 20,000 | 30,000 | 50,000 | 70,000 |
|---|---:|---:|---:|---:|---:|
| Completion | 55% | 68% | 69% | 66% | 68% |

Seeds 10,000-10,099 are the project's standard evaluation set and are
noticeably harder for this policy than any other sample; the honest headline is
a range of 55-69% with a mean near 65%, not any single number. Reporting one
100-episode figure — as the V1-V3 notes did — hides more variance than it shows.
Waves cleared rose from 2.17 to 2.47 and mean episode length from 327 to 365
steps on the same seed set.

These are not directly comparable to V3's 56%, because V4 removed the lane
change blind spot, widened the collision box past half a lane, and switched to
bumper-to-bumper gaps. Part of what V3 scored was a simulation that could not
hit it.

## What is still wrong

Validation completion oscillates violently between checkpoints:

```text
 500k 55%    900k 28%   1.3M 37%   1.7M 22%
 600k 52%   1.0M 20%    1.4M 50%   1.8M 62%
 700k 45%   1.1M 68%    1.5M 27%   1.9M 52%
 800k 60%   1.2M 65%    1.6M 43%   2.0M 72%
```

A 50-point swing between neighbouring checkpoints is policy churn, not sampling
noise, and it is now the largest single obstacle to a high success rate. The
`best_model` mechanism papers over it by keeping whichever checkpoint got lucky.
Worth trying next, roughly in order of expected value:

1. Polyak-averaged target updates (`tau` well below 1) instead of hard copies
   every 5,000 steps, which is the usual cause of this oscillation.
2. A lower learning rate late in training, or a cosine schedule.
3. PPO or another on-policy method, which trades sample efficiency for
   stability and would not need the best-checkpoint crutch at all.
4. Validating on 200+ episodes so checkpoint selection stops rewarding luck.

The remaining crashes on the hard seed set are 20 side, 17 front, 8 rear. Side
impacts dominating means merges are still being started into gaps that close.

## Compatibility

V4 changes both the observation shape (20 -> 33) and the action count (5 -> 9),
so V3 checkpoints are rejected on both counts. V3 remains available in Git.
