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

## Collaboration style

The code is split into small files with one job each. Before each major change,
we will identify the concept it teaches, the metric it should improve, and one
piece for you to predict or modify. That keeps the project understandable while
still letting the implementation move quickly.
