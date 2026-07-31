# Neon Highway V3: 360-degree hard mode

V3 addresses two ways the V2 score could be misleading: the agent could see a
rear gap but not the rear vehicle's closing speed, and some randomly generated
episodes required little intervention.

## 360-degree observation

The observation now contains 20 normalized values:

```text
[actual speed, target speed, current lane, target lane,
 lane 1 front gap, front relative speed, rear gap, rear relative speed,
 lane 2 front gap, front relative speed, rear gap, rear relative speed,
 lane 3 front gap, front relative speed, rear gap, rear relative speed,
 lane 4 front gap, front relative speed, rear gap, rear relative speed]
```

A positive rear relative speed means the vehicle behind is closing. Lane-change
safety uses both rear distance and rear time-to-collision. The renderer draws
front rays in cyan and rear rays in pink, changing to amber or red as TTC falls.

Because V2 used 16 observations, its checkpoints cannot run in V3. V2 remains
available at Git commit `030120a`.

## Hard challenge waves

Hard episodes contain challenge waves at 0, 15, and 30 simulated seconds. Each
wave places a slower leader and a faster rear vehicle around the agent while
leaving one randomized adjacent lane safely usable. When two adjacent lanes
exist, the other is deliberately unsafe. A wave is credited only after the
agent survives it for seven seconds, and route completion requires clearing all
three waves.

This design tests a real decision instead of requiring an arbitrary number of
button presses. The agent may brake, merge, or combine both, but passive HOLD
cannot receive a completion merely because traffic happened to be easy.

## Baselines and selected model

All figures below use deterministic hard-mode traffic seeds.

| Policy | Episodes | Completion | Crash | Waves cleared |
|---|---:|---:|---:|---:|
| Passive HOLD | 100 | 0% | 100% | 0.00 |
| Observation-only heuristic | 100 | 36% | 64% | 1.78 |
| V3 DQN, seeds 10,000+ | 100 | 56% | 44% | 2.17 |
| V3 DQN, seeds 50,000+ | 100 | 55% | 45% | 2.20 |

The selected DQN was trained for 300,000 steps on the RTX 4070. Validation ran
every 25,000 steps over 30 fixed episodes, and the best checkpoint was promoted
automatically. On the first 100-episode test it recorded 13 front, 14 rear, and
17 side impacts. On the second it recorded 7 front, 15 rear, and 23 side
impacts. These are intentionally harder than the V2 test distribution and
should not be compared as if the traffic scenarios were identical.

The remaining rear and side crashes are useful V3 targets. Evaluation now logs
minimum rear TTC explicitly so future reward changes and algorithms can be
compared against this baseline rather than judged only from animation.
