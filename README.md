# Self-Driving RL

A learning-focused reinforcement learning project in which an agent learns
highway driving in simulation.

This is deliberately **not** an attempt to control a real vehicle. The first
environment is small enough to understand and train on a normal computer while
still containing useful driving decisions: lane changes, speed control,
collision avoidance, and delayed consequences.

## First milestone

We use `highway-fast-v0` with:

- a kinematics observation (positions and velocities of nearby cars),
- five discrete actions (left, idle, right, faster, slower),
- DQN, an algorithm that learns a value for each possible action,
- fixed evaluation seeds and metrics so improvements are measurable.

The first experiment asks:

> Does a trained DQN driver achieve a higher return and lower crash rate than
> a random driver on the same 20 evaluation episodes?

Read [docs/rl-primer.md](docs/rl-primer.md) before changing the reward or model.

## Setup

Install [uv](https://docs.astral.sh/uv/) and run:

```powershell
uv sync --extra dev
```

Inspect the environment and measure the random baseline:

```powershell
uv run sdr-inspect --seed 7
uv run sdr-random --episodes 20
```

Train the reference DQN (20,000 steps is the first meaningful run):

```powershell
uv run sdr-train --timesteps 20000 --seed 7
```

The command prints the model path when it finishes. Evaluate it with:

```powershell
uv run sdr-evaluate runs\dqn\<run-name>\model.zip --episodes 20
```

Open training curves:

```powershell
uv run tensorboard --logdir runs
```

Run code quality checks:

```powershell
uv run pytest
uv run ruff check .
```

## What gets saved

Each training run creates a folder under `runs/dqn/` containing:

- `config.json`: seed, environment, hyperparameters, and package versions,
- `model.zip`: the learned Q-network,
- `monitor.csv`: episode rewards and lengths during training,
- `evaluation.json`: held-out evaluation metrics,
- `tensorboard/`: learning curves.

Generated runs are intentionally excluded from Git.

## Learning roadmap

1. **RL loop and baseline** — run a random policy and understand one transition.
2. **Reference DQN** — train and evaluate a known implementation.
3. **DQN from scratch** — implement epsilon-greedy exploration, replay memory,
   Bellman targets, and a target network in PyTorch.
4. **Experiment properly** — compare several seeds and inspect crashes rather
   than trusting one average reward.
5. **Harder driving** — tune observations/rewards, add continuous control, then
   progress to image observations or a simulator such as CARLA.

The project is configured for the CUDA 12.6 PyTorch build, which runs on the
RTX 4070 with the installed NVIDIA driver. The small standard MLP may still be
environment-bound; the high-compute preset uses a larger network and extra
gradient updates so the GPU has more useful work to do.

## Neon Highway game

The project now includes a custom top-down 2D driving game with procedural car
graphics, animated traffic, visible sensor rays, collision feedback, and a live
learning HUD.

Watch a driver with no training:

```powershell
uv run sdr-game random
```

Watch DQN learn from its crashes and save the result:

```powershell
uv run sdr-game learn --timesteps 30000
```

Run the larger GPU-oriented configuration headlessly:

```powershell
uv run sdr-game learn --preset high --device cuda --headless --timesteps 200000
```

Long runs validate periodically on a separate fixed seed set. The final
`model.zip` is the safest checkpoint seen during training; `last_model.zip`
is also kept so later-policy regression remains inspectable.

Each 0.1 s simulation step is drawn as several interpolated frames rather
than one, so motion is smooth instead of jumping 14 pixels per frame.

`--speed` sets world seconds per real second. Pace and smoothness trade off
against each other at a fixed refresh rate, because the simulation itself only
ticks at 10 Hz:

| `--speed` | Frames per step at 60 fps | Feel |
|---|---:|---|
| 1 | 6 | real time, very smooth, slow |
| 2 (default) | 3 | brisk and smooth |
| 3 | 2 | faster with less interpolation |
| 6 | 1 | original pace, original choppiness |

Raise `--fps` as well if your monitor runs above 60 Hz: frames per step is
`round(fps * 0.1 / speed)`, so `--fps 120 --speed 3` gives four. `learn`
defaults to `--speed 12`, which keeps rendered training exactly as fast as it
was. Use `--headless` for maximum training speed. Press `Space` to pause, `H`
to hide sensor rays, or `Esc` to stop safely and save the current model.

Nothing is ever teleported where the player can see it. Traffic that needs to
form a challenge changes its speed and closes the gap over a few seconds;
vehicles are only repositioned off screen, and drive into frame from there.

Watch the newest trained model without exploration:

```powershell
uv run sdr-game watch
```

Game runs are written to `runs/game/<run-name>/` with the random baseline,
trained evaluation, model, configuration, monitor log, and TensorBoard data.

### V1 systems

The V1 game adds adaptive traffic following, escalating traffic pressure,
time-to-collision safety shaping, decomposed rewards, route-completion bonuses,
swept collision detection, impact classification, crash particles, live
learning curves, and action Q-value displays.

See [docs/neon-highway-v1.md](docs/neon-highway-v1.md) for a readable tour of
the game logic and dashboard.

### V2 speed control

V2 adds a visible target speed and smooth cruise-control dynamics. `SPEED +`
and `SPEED -` adjust the target by 3.6 km/h, while throttle and brake pressure
move the actual car speed toward it. The dashboard shows both speeds,
acceleration, and pedal pressure.

This changes the observation from 15 to 16 values, so V2 requires a newly
trained model. See [docs/neon-highway-v2.md](docs/neon-highway-v2.md).

Watch the selected local V2 policy with:

```powershell
uv run sdr-game watch --model runs/game/v2-high-stable-200k/model.zip
```

The selected high-compute checkpoint completed 67% and 72% of two separate
100-episode evaluation sets. The previous V2 model completed 57% and 55% on
the same traffic seeds. Periodic validation selected the 125k-step checkpoint;
training continued to 200k without overwriting that stronger policy.

V2 is preserved in Git at commit `030120a`. V3 changes the observation shape,
so V2 checkpoints are intentionally rejected by the current environment.

### V3 360-degree hard mode

V3 adds rear relative speed for every lane, raising the observation from 16 to
20 values. The game draws rear sensor rays and reports rear time-to-collision.
Hard mode presents three randomized, solvable traffic waves per episode: a slow
leader, a fast-closing rear vehicle, a safe escape lane, and sometimes a nearby
unsafe lane. A route only completes after all three waves are cleared, so a
passive HOLD policy now completes 0% of hard episodes.

The V3 model completed 56% and 55% of two separate 100-episode hard evaluation
sets. See [docs/neon-highway-v3.md](docs/neon-highway-v3.md) for the observation
layout and scenario design. Note that its impact-type breakdown was computed
with the collision-reporting bug fixed in V4, so those counts are unreliable.

### V4 corrected physics and a usable planning horizon

V4 is mostly a correctness release. Five physics bugs meant V3 learned against a
simulation that did not match its own sensors, and `gamma = 0.98` over 450-step
episodes gave a five-second horizon in which route completion was worth 0.0006.

- Traffic used to drive straight through traffic; it now follows the
  Intelligent Driver Model with an explicit non-overlap constraint.
- The middle frame of every lane change could not be hit by anything.
- Collisions were attributed to the first car in a list, not the nearest.
- Sensor gaps were centre-to-centre, so a reported 4.5 m gap was already a crash.
- Recycled traffic could teleport back and forth between frames.
- Collision detection now solves continuous overlap on both axes, so a
  diagonal merge only crashes when the two physical bodies overlap at the
  same instant—and cannot pass through between simulation frames.

Steering and pedal are now chosen together (9 actions instead of 5), so the
agent can brake and merge in one step — the manoeuvre the hard waves are built
to require, and the one V3's action space forbade. The observation grows from 20
to 33 values, adding the episode clock, wave state, and per-lane inverse
time-to-collision.

Train the V4 configuration:

```powershell
uv run sdr-game learn --preset gpu --algo qrdqn --headless --envs 8 --timesteps 2000000
```

`--algo qrdqn` is distributional and avoids the Q-value overestimation that left
the V3 policy completing 62% of left-escape waves against 48% of right-escape
ones. Training episodes are randomly mirrored, which is an exact symmetry of the
environment, so both directions share one set of weights.

Watch the trained V4 policy:

```powershell
uv run sdr-game watch --model runs/game/v4-qrdqn-2m/model.zip --difficulty hard
```

Score it headlessly on several independent seed sets, which is the only honest
way to read the completion rate:

```powershell
uv run sdr-game evaluate --episodes 100 --seed 10000 20000 30000 50000 70000
```

### Route length

`--seconds` sets the route length for `random`, `learn`, `evaluate`, and
`watch`. The default is 45 s. A hard-mode wave stages every 15 s, so a longer
route means more waves rather than the same three followed by empty road:

```powershell
uv run sdr-game watch --difficulty hard --seconds 120
```

Longer routes are strictly harder, because every extra wave is another chance
to crash. The 2M-step model, which was trained on 45 s routes, scores this on
seeds 30,000:

| Route | Waves | Completion | Waves cleared |
|---|---:|---:|---:|
| 45 s (default) | 3 | 75% | 2.70 |
| 120 s | 8 | 22% | 5.45 |

Train on the length you intend to measure — pass the same `--seconds` to
`learn` and the run will validate and evaluate at that length too.

### Endless mode

`--endless` removes the finish line entirely. It is one continuous drive, and
the only thing that restarts it is a crash:

```powershell
uv run sdr-game watch --difficulty hard --endless
```

There is no route boundary and nothing resets mid-drive: the clock only counts
up and waves keep arriving every 15 s, accumulating. A fixed-route checkpoint
can be watched in this mode, but it was not optimized for continuous survival.
Train with `--endless` when endless driving is the metric you want to improve;
checkpoint selection then ranks policies by survival time.

The header shows time survived, distance, and the longest run of the session.
`CHALLENGES` shows waves cleared so far.

Completion rate is meaningless here, since every endless run ends in a crash
eventually, so `evaluate --endless` reports survival time:

```powershell
uv run sdr-game evaluate --endless --episodes 30 --seed 30000 50000
```

The bundled 2M-step checkpoint predates smooth wave formation and the V4.1
continuous collision solver. Treat its endless score as a baseline for the
next retraining run, not as the expected ceiling.

### V5 good-driver objective

V5 teaches the agent to move through traffic rather than merely wait for a
forced escape. Real overtakes, being passed, safe passing opportunities,
blocked time and lane-change efficiency are now first-class evaluation
metrics. The reward values completed passes but gives no bonus for starting a
lane change, so unnecessary weaving still loses comfort reward.

The frozen V4 policy completes 70% of the 100-episode development set but
averages 67 km/h, answers 43% of safe passing opportunities and makes only 2.2
net passes per route. A safe hand-written driver reaches 99%, 81 km/h and 10.1
net passes with fewer lane changes, showing that the desired behaviour is
achievable. See [docs/neon-highway-v5.md](docs/neon-highway-v5.md) for the exact
metrics, reward and reproducible training command.

### V6 RLAIF good driver

V6 learns driving preferences from AI-ranked, matched trajectories while
keeping the proven V5 policy frozen underneath. A confidence-gated residual
can take a clearly useful pass or suppress a pointless reversal; an independent
safety shield rejects any proposal that fails the lane-gap, rear-TTC, threat,
or pedal-consistency checks.

Watch the calibrated driver:

```powershell
uv run sdr-rlaif watch `
  --base-model runs/game/v5-good-driver-2p5m-restart/model.zip `
  --override-model runs/rlaif/v6-good-driver/override_model.pt `
  --difficulty hard --episodes 10
```

Add `--endless` to keep restarting only after crashes. On the untouched
100-route seed range beginning at 130,000, V6 matched V5 at 98% completion and
2% crashes while cutting lane changes from 16.93 to 8.94, reversals from 11.10
to 3.63, missed passing opportunities from 0.71 to 0.41, and blocked time from
10.55% to 5.08%. See [docs/neon-highway-v6-rlaif.md](docs/neon-highway-v6-rlaif.md)
for the preference rubric, failed experiments, training commands, and full
evaluation.

## Collaboration style

The code is split into small files with one job each. Before each major change,
we will identify the concept it teaches, the metric it should improve, and one
piece for you to predict or modify. That keeps the project understandable while
still letting the implementation move quickly.
