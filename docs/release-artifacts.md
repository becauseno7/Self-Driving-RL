# Release artifact provenance

Self-Driving RL v1.0 uses two learned artifacts plus deterministic controller
code. The code, documentation, and learned artifacts are approved under the MIT
License. The artifacts are staged under `release/huggingface/` for inspection,
but are not public downloads until publication is explicitly approved.

## Frozen stack

1. `model.zip` is the safety/validation-selected checkpoint from the V5
   `v5-good-driver-2p5m-restart` QR-DQN run. Training reached 2.5 million steps;
   the selected checkpoint is intentionally used instead of `last_model.zip`.
2. `override_model.pt` is the calibrated V6 confidence-gated RLAIF residual.
   It keeps V5 frozen and may make only shielded calm/pass corrections.
3. The repository's V7/V8 deterministic `LongitudinalIntentPolicy` supplies
   persistent passing intent and proportional comfort/emergency braking. It is
   code, not a third learned artifact.
4. DAgger model candidates are excluded. Their held-out promotion gates returned
   `HOLD`, including a net-passing regression in the final candidate.

The simulator observation has 33 values and its flat action space has 9 joint
steering/pedal actions. The V6 residual stores an `observation_size` of 35
because it appends two lane-change memory values to the 33 simulator values,
then conditions on V5's proposed action.

## Verify the staged artifacts

From the repository root on Windows:

```powershell
Get-FileHash release\huggingface\model.zip -Algorithm SHA256
Get-FileHash release\huggingface\override_model.pt -Algorithm SHA256
```

Expected values:

| File | Bytes | SHA-256 |
|---|---:|---|
| `model.zip` | 3,594,202 | `5780BBEE5CE2009459F3AA796AA4982FBF33222DCC182883D31AFAA16C597039` |
| `override_model.pt` | 35,439 | `06C3A0CEE04AAF6B8822781FE78F867F513A3019EBBF7FD8D91E10F117146BEC` |

Do not load a checksum mismatch. Stable-Baselines3 archives use cloudpickle and
must be treated as trusted executable-adjacent data.

## Final evaluation evidence

The release result is stored in
`release/huggingface/evaluation/v8-braking-final-unseen-100.json`. It was
produced from 100 contiguous hard/dynamic routes, seeds 690000 through 690099,
at the default 45-second route length:

```powershell
uv run sdr-rlaif override-evaluate `
  --base-model release/huggingface/model.zip `
  --override-model release/huggingface/override_model.pt `
  --longitudinal-intent --dynamic-traffic `
  --difficulty hard --device cpu `
  --episodes 100 --seed 690000
```

It recorded 100% completion, zero crashes, 8.43 mean net passes, 90.2% passing
response, and 0.04% clear-road deep-slowdown steps. These numbers describe this
simulator and seed set only.

## Publication targets

- Hugging Face: `slicedonions/self-driving-rl-v1`
- GitHub source tag: `v1.0.0`
- GitHub release: `https://github.com/becauseno7/Self-Driving-RL/releases/tag/v1.0.0`
- Browser replay: `https://becauseno7.github.io/Self-Driving-RL/`

These targets remain unpublished until the user approves the external release
actions. The authoritative machine-readable inventory is
`release/huggingface/artifact-manifest.json`.
