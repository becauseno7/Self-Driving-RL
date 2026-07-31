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

Training is intentionally capped at 120 rendered frames per second. Use
`--headless` for maximum speed. Press `Space` to pause, `H` to hide sensor rays,
or `Esc` to stop safely and save the current model.

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

`--endless` removes the finish line. A finished route is banked as a lap and
the car drives straight into the next one, so the only thing that restarts the
episode is a crash:

```powershell
uv run sdr-game watch --difficulty hard --endless
```

Laps deliberately recycle the episode clock and the wave counter, so each lap
looks to the agent exactly like a fresh route. A policy trained on fixed routes
therefore stays in distribution and needs no retraining. A banked lap pays the
same bonus as finishing a fixed route.

The header shows the current lap and distance driven; `CHALLENGES` shows waves
cleared this lap. Completion rate is meaningless here — every endless episode
ends in a crash eventually — so `evaluate --endless` reports laps and steps
survived instead:

```powershell
uv run sdr-game evaluate --endless --episodes 30 --seed 30000 50000
```

The 2M-step model averages 1.8 laps, about 1,000 steps or 100 simulated
seconds, before it crashes.

The 2M-step run takes about 45 minutes on the RTX 4070. It closed the
directional gap from 14.0 points to 0.8, and 60% of its lane changes now carry a
pedal input at the same time. Completion runs 55-69% depending on which
100-episode seed set you measure — the standard seed-10,000 set is the hardest
sample of the five tried, so no single number is representative.

Validation completion still swings by up to 50 points between neighbouring
checkpoints, which is now the main obstacle to a high success rate.
[docs/neon-highway-v4.md](docs/neon-highway-v4.md) has the full accounting and
what to try next.

## Collaboration style

The code is split into small files with one job each. Before each major change,
we will identify the concept it teaches, the metric it should improve, and one
piece for you to predict or modify. That keeps the project understandable while
still letting the implementation move quickly.
