# Security policy

## Supported versions

Security fixes are prepared for the current `main` branch and the latest v1.0.x
release. Older experiment snapshots and locally generated checkpoints are not
maintained as separate products.

## Report a vulnerability privately

Please do not open a public issue for a vulnerability. Use GitHub's private
[security advisory form](https://github.com/becauseno7/Self-Driving-RL/security/advisories/new)
and include:

- the affected version or commit;
- a minimal reproduction or proof of concept;
- the impact and any known prerequisites;
- whether the issue involves a model artifact, dependency, browser replay, or
  Python code.

Avoid including secrets or personal data that are not necessary to reproduce
the issue. The maintainer will acknowledge a complete report when it is seen,
coordinate remediation and disclosure, and credit reporters who want credit.

## Model-file safety

The base policy uses the Stable-Baselines3 ZIP format, which may deserialize
cloudpickle content. Loading an untrusted or tampered archive can execute code
with the user's permissions. Download models only from an approved release,
verify the documented SHA-256 digest, and load them in a minimally privileged
environment. Do not upload unknown checkpoints to this project for inspection.

## Safety boundary

This project is a simulator and is not a real-vehicle controller. Reports that
the simulated policy handles a scenario poorly are valuable model-quality
issues, but the software must never be connected to a car, robot, or
safety-critical control system. A model's simulator completion rate is not a
security or safety certification.
