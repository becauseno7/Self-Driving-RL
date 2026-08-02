---
title: Self-Driving RL Live
emoji: 🚗
colorFrom: green
colorTo: gray
sdk: static
app_file: static/index.html
license: mit
models:
  - slicedonions/self-driving-rl-v1
short_description: Watch a learned RL driver navigate dynamic traffic live.
---

# Self-Driving RL Live

This free Static Space runs the genuine frozen Self-Driving RL v1.0 stack in a
fresh, in-browser simulator session for every viewer:

- V5 validation-selected QR-DQN checkpoint;
- V6 confidence-gated RLAIF residual;
- V7/V8 persistent-intent, braking, and safety controller;
- `NeonHighwayEnv-v5` in hard mode with dynamic traffic by default.

Pyodide executes the original `NeonHighwayEnv-v5` and V7/V8 controller source;
ONNX Runtime Web executes parity-checked exports of the V5 and V6 learned
networks. The app steps the policy at 10 Hz and renders it smoothly. It does not
replay a recording or replace policy decisions with scripted browser behavior.
Every **New traffic** request creates a new deterministic seed, and a crashed
endless run automatically restarts on the next seed.

## Run locally

Export and validate the browser networks from the source repository root:

```powershell
uv run --with onnx --with onnxruntime python tools/export_browser_policy.py `
  --base-model release/huggingface/model.zip `
  --override-model release/huggingface/override_model.pt
```

Serve the repository root with a local HTTP server, then open
`space/static/index.html`. The
checked-in `app.py` and `Dockerfile` preserve an optional server-backed version
for paid Docker hosting, but the public Space needs no hosted compute.

## Scope

This is a simulator-only education and research demonstration. It is not a
real-vehicle controller, perception stack, or safety claim. Source and model
artifacts are MIT licensed.
