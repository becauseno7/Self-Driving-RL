---
title: Self-Driving RL Live
emoji: 🚗
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
license: mit
models:
  - slicedonions/self-driving-rl-v1
short_description: Watch the frozen v1.0 RL driver make live decisions in dynamic traffic.
---

# Self-Driving RL Live

This Space runs the genuine frozen Self-Driving RL v1.0 stack in a fresh,
server-side simulator session for every connected viewer:

- V5 validation-selected QR-DQN checkpoint;
- V6 confidence-gated RLAIF residual;
- V7/V8 persistent-intent, braking, and safety controller;
- `NeonHighwayEnv-v5` in hard mode with dynamic traffic by default.

The browser receives simulator state at 10 Hz and renders it smoothly. It does
not replay a recording and it does not replace policy decisions with scripted
browser behavior. Every **New traffic** request creates a new deterministic
seed, and a crashed endless run automatically restarts on the next seed.

## Run locally

From the source repository root:

```powershell
$env:SDR_BASE_MODEL = "release/huggingface/model.zip"
$env:SDR_OVERRIDE_MODEL = "release/huggingface/override_model.pt"
uv run --with fastapi==0.141.1 --with "uvicorn[standard]==0.52.1" `
  --with huggingface_hub==1.26.0 uvicorn space.app:app --host 127.0.0.1 --port 7860
```

Then open `http://127.0.0.1:7860`.

## Scope

This is a simulator-only education and research demonstration. It is not a
real-vehicle controller, perception stack, or safety claim. Source and model
artifacts are MIT licensed.

