# Reddit launch draft

## Suggested title

I taught an RL agent to drive through dynamic highway traffic — then rejected
the “more advanced” checkpoints when they drove worse

## Post body

I built a small 2D highway simulator and trained a self-driving agent to manage
speed, change lanes, and pass traffic. The fun part was watching it fail in very
human-looking ways: sitting behind slow cars, tapping the throttle repeatedly,
braking too hard, and accepting only the easiest passing gaps.

The final driver is deliberately layered:

- a QR-DQN policy proposes one of nine joint steering/pedal actions;
- a small preference model makes sparse confidence-gated corrections;
- deterministic intent and safety logic smooths braking, remembers a pass, and
  rejects dangerous merges.

I also collected human DAgger demonstrations. They improved some moments, but
the learned candidates failed the matched-seed passing-quality gate, so I did
not ship them just because they were newer.

On 100 previously unused 45-second hard/dynamic simulator routes, the frozen
stack completed 100/100 with zero crashes, averaged 8.43 net passes, answered
90.2% of labelled safe passing opportunities, and spent 0.04% of clear-road
steps in a deep slowdown. These are simulator results, not evidence that this
can control a real car.

The repository includes the simulator, training and evaluation CLIs, model
provenance, rejected-experiment notes, tests, and a browser replay generated
from three real frozen-policy trajectories. The browser page is deterministic
playback rather than pretending that PyTorch inference is running in your tab.

Browser replay: https://becauseno7.github.io/Self-Driving-RL/

Code: https://github.com/becauseno7/Self-Driving-RL/releases/tag/v1.0.0

Models and evaluation record: https://huggingface.co/slicedonions/self-driving-rl-v1

I used AI heavily while building it, but treated it as a learning project: I
made the driving decisions, labelled demonstrations, compared matched-seed
evaluations, and kept notes about what each layer does. Feedback on the reward
design, evaluation protocol, or next simulator challenge is welcome.

## Media sequence

1. A 15–25 second clip showing a clean pass in dynamic traffic.
2. The calm native Analysis view with intent and safety telemetry visible.
3. The browser Policy Roadbook at desktop width.
4. A compact results card with the simulator-only caveat on the same frame.

## Before posting

- Confirm every release URL works in a signed-out browser.
- Keep the simulator-only qualification next to the evaluation numbers.
- Upload a native video rather than a screen recording containing terminal or
  desktop clutter.
- Do not call the project autonomous driving, FSD, or real-world safe.
- Prefer one clear post over cross-posting identical text everywhere at once.
