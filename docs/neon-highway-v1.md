# Neon Highway V1

V1 is the second major pass over the custom driving game. It improves the task
logic, crash diagnosis, learning signal, and the information shown while the
agent drives.

## What stayed compatible

The observation is still a 15-number vector:

```text
[speed, current lane, target lane,
 lane 1 ahead gap, relative speed, behind gap,
 lane 2 ahead gap, relative speed, behind gap,
 lane 3 ahead gap, relative speed, behind gap,
 lane 4 ahead gap, relative speed, behind gap]
```

Existing models can therefore load in V1, although they were trained under the
old traffic and reward rules and will not behave optimally.

## Better traffic logic

Traffic vehicles now have a desired cruising speed and continuously calculate a
safe following gap. When approaching another vehicle or the RL car, they brake
progressively and later return to their desired speed. This replaces the old
last-second rear-yield rule.

Traffic spacing also becomes gently tighter as an episode progresses. This
creates increasing pressure without introducing unavoidable random hazards.

## Reward components

The displayed reward is the sum of five inspectable parts:

- `progress`: rewards useful forward speed,
- `safety`: penalizes low time-to-collision and recognizes safe evasions,
- `comfort`: adds a small cost when starting a lane change,
- `rules`: penalizes impossible actions such as leaving the road,
- `terminal`: gives `-10` for a crash or `+5` for completing the route.

This decomposition makes reward debugging much easier. If the agent finds an
unwanted strategy, we can identify which incentive is responsible.

## Crash handling

Collision detection checks the motion between consecutive frames, so a fast car
cannot pass through another car between updates. Each impact records:

- front, side, or rear contact,
- low, medium, or high severity,
- relative impact speed,
- lane and traffic-car speed,
- reward components, time-to-collision, and near-miss count.

The crash presentation shows these details with an impact animation. The final
transition is then stored in replay memory before the environment resets.

## Reading the new dashboard

The left dashboard describes the training session: recent returns, completion
and crash counts, collision types, near misses, and safe evasions.

The right dashboard explains the current decision:

- speed, lane, selected action, and immediate reward,
- safety gauge, front gap, time-to-collision, and difficulty,
- the five reward components,
- the Q-value assigned to each action by DQN.

The action with the highest Q-value is what the deterministic agent believes
will produce the greatest discounted future return.

## V1 reference benchmark

The first V1 DQN was trained for 30,000 steps with seed 7 and evaluated on 20
episodes beginning at seed 10,000.

| Metric | Random driver | V1 DQN |
|---|---:|---:|
| Completion rate | 0% | 60% |
| Crash rate | 100% | 40% |
| Mean episode length | 60.55 | 332.55 |
| Mean return | -7.44 | +15.80 |

This is a baseline, not a final claim. A proper algorithm comparison will use
multiple training seeds.
