---
language:
  - en
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - qrdqn
  - rlaif
  - gymnasium
  - simulated-driving
  - stable-baselines3
license: other
license_name: "PENDING: no public code or model license has been approved"
model-index:
  - name: Self-Driving RL v1.0 layered driver
    results:
      - task:
          type: reinforcement-learning
          name: Hard/dynamic Neon Highway route completion
        dataset:
          type: NeonHighwayEnv-v5
          name: 100 deterministic hard/dynamic seeds from 690000
        metrics:
          - type: completion_rate
            value: 1.0
            name: Completion rate
          - type: crash_rate
            value: 0.0
            name: Crash rate
          - type: mean_net_overtakes
            value: 8.43
            name: Mean net passes per route
          - type: passing_response_rate
            value: 0.9022403258655805
            name: Passing response rate
          - type: clear_road_deep_slowdown_rate
            value: 0.0004
            name: Clear-road deep-slowdown rate
---

# Self-Driving RL v1.0

This model package contains the two learned layers of the recommended
Self-Driving RL driver:

- `model.zip`: the validation-selected checkpoint from a 2.5M-step V5 QR-DQN
  run;
- `override_model.pt`: the calibrated V6 confidence-gated preference residual.

The complete driver also requires the v1.0 repository code. Its V7/V8
`LongitudinalIntentPolicy` adds deterministic persistent speed intent, smooth
braking, and safety logic. No DAgger weights are included: the evaluated DAgger
candidates were rejected by the matched-seed promotion gate and are not part of
the recommended driver.

> **License and publication status:** pending. `license: other` above is a
> schema-compatible placeholder, not a grant of permission. The user must
> approve a public code/model license before this folder is uploaded. The final
> Hugging Face repository ID and download URLs are also pending.

## Model details

| Layer | Role | Format |
|---|---|---|
| V5 | Frozen QR-DQN base policy | Stable-Baselines3 / `sb3-contrib` ZIP |
| V6 | Sparse RLAIF calm/pass corrections | PyTorch weights-only checkpoint |
| V7/V8 | Persistent intent, braking, and safety controller | Repository Python code |

- **Simulator observation:** 33 normalized float values (nine ego/route values
  plus six values per lane across four lanes).
- **Simulator action:** 9 discrete joint actions: left/keep/right ×
  brake/coast/gas.
- **Base network:** `[256, 256]` with 64 quantiles.
- **Base run:** QR-DQN, 2,500,000 steps, training seed 7, eight parallel
  hard-mode environments, random mirror augmentation, 45-second routes.
- **Checkpoint selection:** periodic validation selected `model.zip`; the final
  live checkpoint was more assertive and less safe, so it was not packaged.
- **Preference layer:** 35 road/context values plus the one-hot V5 proposal. The
  35 values are the 33-value observation and two lane-change memory features.
- **V6 gates:** calm threshold 0.80, passing threshold 0.50; calibration used
  development seed 80000 and held-out seed 130000.

The full V5 configuration is in `config/v5-training-config.json` and V6
provenance is in `config/v6-override-provenance.json`.

## Intended uses

- teaching reinforcement-learning evaluation, reward design, checkpoint
  selection, and distribution shift;
- reproducing the frozen policy in the bundled Neon Highway simulator;
- studying layered learned/deterministic control and confidence-gated
  preference residuals;
- comparing new policies on fixed, held-out simulator seeds.

## Unsupported uses

- any real vehicle, robot, actuator, driver-assistance system, or
  safety-critical controller;
- claims about autonomous-driving safety, regulatory compliance, or road
  readiness;
- inputs from cameras, lidar, maps, GPS, real traffic, weather, pedestrians, or
  another simulator without retraining and independent validation;
- loading the artifacts from an untrusted or checksum-mismatched source.

## Install

The package has not been uploaded. After the user approves publication, replace
`<PENDING_HF_REPOSITORY_ID>`; do not run this placeholder unchanged:

```bash
git clone https://github.com/becauseno7/Self-Driving-RL.git
cd Self-Driving-RL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . huggingface_hub
hf download <PENDING_HF_REPOSITORY_ID> model.zip override_model.pt --local-dir models
```

Verify both downloads before use:

```text
model.zip
  bytes: 3594202
  sha256: 5780BBEE5CE2009459F3AA796AA4982FBF33222DCC182883D31AFAA16C597039

override_model.pt
  bytes: 35439
  sha256: 06C3A0CEE04AAF6B8822781FE78F867F513A3019EBBF7FD8D91E10F117146BEC
```

## Loading and watching

The supported high-level loading example is the project CLI:

```bash
sdr-rlaif watch \
  --base-model models/model.zip \
  --override-model models/override_model.pt \
  --longitudinal-intent --dynamic-traffic \
  --difficulty hard --device cpu --episodes 10
```

The equivalent Python composition is:

```python
from pathlib import Path

from self_driving_rl.longitudinal import LongitudinalIntentPolicy
from self_driving_rl.rlaif import load_override_policy

preference_policy = load_override_policy(
    Path("models/model.zip"),
    Path("models/override_model.pt"),
    device="cpu",
)
driver = LongitudinalIntentPolicy(preference_policy)
# action = int(driver(observation))  # observation.shape == (33,)
```

Stable-Baselines3 model ZIPs can deserialize cloudpickle. That is a trusted
code-loading boundary: use only approved artifacts, verify the SHA-256 digest,
and prefer a minimally privileged environment. The override loader uses
`torch.load(..., weights_only=True)`.

## Evaluation

The final release evaluation ran the complete stack on 100 contiguous,
previously unused seeds (`690000`–`690099`) in hard mode with dynamic traffic.
Every route was 45 simulated seconds.

| Metric | Result |
|---|---:|
| Completion | 100% |
| Crash rate | 0% |
| Mean net passes / route | 8.43 |
| Passing response | 90.2% |
| Clear-road deep slowdown | 0.04% of steps |

The raw result is in
`evaluation/v8-braking-final-unseen-100.json`. Reproduce it with:

```bash
sdr-rlaif override-evaluate \
  --base-model model.zip \
  --override-model override_model.pt \
  --longitudinal-intent --dynamic-traffic \
  --difficulty hard --device cpu \
  --episodes 100 --seed 690000
```

## Training and preference provenance

V5 was trained in the custom `NeonHighwayEnv-v5` task with QR-DQN from
`sb3-contrib==2.9.0`, PyTorch `2.13.0+cu126`, Gymnasium `1.3.0`, a replay size of
750,000, batch size 512, gamma 0.995, learning rate 0.00015, Polyak factor 0.02,
four gradient steps per environment step, and exploration from 1.0 to 0.02 over
30% of training. Periodic 100-route validation selected the packaged model.

V6 kept that checkpoint frozen. Its preference source comprised 96 matched
trajectories, 160 comparisons, and 128 individually reviewed AI labels using an
ordered safety/completion/passing/calmness rubric. A residual learned leave,
calm-correction, or pass-correction choices from 54,000 completed-trajectory
states. Confidence calibration and an independent front/rear gap and TTC shield
limit deployment-time interventions.

## Limitations and provenance caveats

- The 100-route result is a deterministic simulator evaluation, not a confidence
  bound for real driving or every simulator seed.
- The policy sees privileged structured kinematics rather than raw perception.
- Traffic behavior and challenge generation are authored and much narrower than
  real traffic.
- The selected checkpoint, thresholds, reward terms, and controller rules were
  informed by earlier development ranges. The 690000-690099 range was reserved
  for the reported final run.
- RLAIF labels are small-scale, rubric-dependent, and AI-generated/reviewed;
  they do not represent a broad human preference population.
- V7/V8 behavior lives in repository code and therefore must be paired with the
  v1.0 source release; the two weight files alone do not reproduce the result.
- DAgger experiments are documented, but every candidate remained `HOLD`; none
  is silently bundled here.
- Exact reproducibility still depends on the pinned Python stack, platform
  numerics, and repository source. The required source tag is `v1.0.0`; it will
  remain unavailable until publication is approved.

See `artifact-manifest.json` for the machine-readable file inventory and the
project README for the architecture and learning guides.
