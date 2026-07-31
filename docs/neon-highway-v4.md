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

`--endless` removes the finish line: one continuous drive until a crash. There
is no route boundary, no lap, and nothing that resets mid-drive. Waves keep
arriving every `CHALLENGE_INTERVAL_STEPS` and the wave counter accumulates.

The design question is what the episode clock means without a deadline. It
reports a constant 1.0 -- "plenty of road left" -- and the waves-cleared
fraction reports 0.0 because a fraction has no meaning when waves never stop.
A fixed-route checkpoint is shape-compatible and can be watched endlessly, but
it was not optimized for that observation distribution or survival objective.
Train with `--endless` when endless survival is the target metric.

An earlier attempt banked each finished route as a lap and recycled the clock.
It worked, but the recycling was an artifact with no counterpart in the task,
so continuous driving replaced it. Later smooth-wave and collision changes
made the old three-to-four-minute benchmark stale; a fresh endless checkpoint
is needed before publishing a new headline number.

`ENDLESS_STEP_LIMIT` (100,000 steps, ~2.8 simulated hours) still bounds the
episode so a flawless policy cannot hang a training run.

Completion rate carries no information when every episode ends in a crash, so
`EvaluationSummary` and checkpoint selection use survival time in endless mode.
`evaluate --endless` reports mean and longest survival directly.

## Presentation: real-time, and nothing pops

Two things made the game read as buggy rather than as a simulation.

**The world ran at six times real time.** One simulation step covers `DT` =
0.1 s and was drawn as exactly one frame, so at 60 fps the world advanced six
seconds per real second in 14-pixel jumps. Rendering was never the bottleneck
-- a frame costs 5.1 ms, a 198 fps ceiling. The renderer now splits each step
into `round(fps * DT / speed)` interpolated frames, drawing positions as
`previous + alpha * (current - previous)` from the previous-position fields the
physics already kept. At 60 fps that is six frames per step, real time, and
0.5 pixels of motion per frame.

`--speed` sets world seconds per real second. `watch` and `random` default to
2.0, which gives three interpolated frames per simulation step at 60 fps;
`learn` defaults to 12.0 so rendered training remains fast.

**Cars teleported in view.** The environment now protects a continuity window
from 34 m behind the car to 102 m ahead, deliberately wider than the camera.
Wave staging placed vehicles at -9, -21, +17, +30 and +49 m, so up to six cars
could previously appear out of nothing every wave -- every 15 s in endless mode.
Recycling could also drop a car into view with a fresh colour and body shape,
so one car visibly became another.

Staging now respects the camera. A car already on screen is *persuaded* rather
than moved: its desired speed changes and the IDM closes the gap over the next
few seconds, which is how a squeeze forms on a real road. A car off screen may
be repositioned, but never nearer than the edge of the view, so it drives into
frame. Recycling is clamped the same way. `VISIBLE_BEHIND` and `VISIBLE_AHEAD`
encode the camera in the environment, and
`test_nothing_ever_teleports_or_restyles_on_screen` asserts that across 2,000
steps no visible car ever moves further than `MAX_SPEED * DT` or changes
appearance.

The opening wave is exempt: nothing has been drawn at step zero, so there is no
continuity to break and the scenario is set up exactly as before.

**Cars and collisions now share one believable footprint.** The first physical
sprite fix made a 4.6 m x 1.9 m car honest under the old camera, but that camera
projected it as a 92 x 29 px horizontal bar. V4.1 narrows the road and moves the
longitudinal scale closer: cars now occupy about 67 x 78 px and use distinct
coupe, sedan and SUV silhouettes with real windows, mirrors, wheels, lighting
and body contours.

The renderer still derives those dimensions from `CAR_LENGTH`, `CAR_WIDTH`,
`LANE_WIDTH` and the camera scale. No decorative bodywork extends beyond the
physical footprint. `LANE_COLLISION_WIDTH` remains
`CAR_WIDTH / LANE_WIDTH` = 0.514, so the merge midpoint is collidable and two
sprites touch when the physics reports contact.

Collision detection is now a continuous two-dimensional sweep. It computes the
time interval of longitudinal overlap and the time interval of lateral overlap,
then registers a crash only when those intervals intersect. This catches a fast
diagonal merge or a full pass between 0.1-second steps without combining
longitudinal and lateral proximity that happened at different times. Exact
body-boundary contact counts as a crash.

Smooth wave formation changed the task. Waves now build over seconds instead of appearing
complete, and they no longer reshuffle traffic into known-safe geometry, so
whatever congestion exists persists. Fixed-route completion is unaffected
(64% mean over two 100-episode sets, against 65% before), but endless survival
roughly halved:

| | Mean survival | Longest run |
|---|---:|---:|
| Teleporting waves | 3m 14s - 4m 28s | 19m 02s |
| Smooth waves | 1m 53s - 2m 37s | 8m 32s |

Traffic speed does not decay over long runs (mean desired speed holds at
23-25 m/s across 4,000 steps), so this is a genuinely harder task rather than a
leak. Retraining against it should recover the difference.

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

After the V4.1 simultaneous-axis collision sweep, the same frozen checkpoint
completed 72%, 65% and 70% across seeds 10,000, 30,000 and 90,000. This is a
regression check for the physics patch, not a newly trained model result.

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
