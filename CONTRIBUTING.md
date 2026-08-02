# Contributing

Thanks for helping make Self-Driving RL clearer, safer, and easier to learn
from. Small, evidence-backed changes are especially welcome.

## Set up a development environment

Self-Driving RL supports Python 3.11 and 3.12. On the original Windows/Linux
CUDA workstation:

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
```

For a platform-native CPU environment, use standard packaging tools so the
repository's CUDA-specific `uv` source override does not apply:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

## Before opening a pull request

- Keep the simulator deterministic for a fixed seed.
- Add or update tests for behavior changes, then run the full test suite and
  Ruff.
- Explain the concept the change teaches and the metric it should affect.
- For policy, reward, physics, observation, or action changes, report a
  matched-seed baseline and candidate comparison. Include completion, crash,
  and driving-quality metrics—not return alone.
- Update the focused guide under `docs/` when an interface, command, metric, or
  experiment contract changes.
- Do not commit generated `runs/`, TensorBoard logs, caches, local paths,
  credentials, or unreviewed model files.
- Keep real-vehicle or safety-critical claims out of code and documentation.

Observation/action compatibility is part of the model format. The v1.0 driver
expects 33 observations and 9 joint steering/pedal actions. A shape change
requires new models, explicit migration notes, and rejection of incompatible
checkpoints.

## Model promotion

A visually convincing route is not enough to promote a driver. Use a held-out
seed range that was not used for training, tuning, or checkpoint selection.
Preserve the safety rate first, then show a meaningful gain in the metric the
experiment targets. Human DAgger candidates must pass the promotion gate in
`self_driving_rl.dagger`; rejected candidates must not silently become defaults.

## Pull-request description

Please include:

1. the problem and intended learning value;
2. the implementation scope;
3. commands run and their results;
4. matched-seed evaluation evidence for behavior changes;
5. any compatibility, artifact, or documentation impact.

The project license is pending. Until it is selected, ask the maintainer before
submitting a substantial external contribution; no contribution terms should
be inferred from the current absence of a `LICENSE` file.
