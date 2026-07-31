# RL primer for this project

## The loop

At time step `t`:

1. The environment gives the agent an observation `s_t`.
2. The agent selects an action `a_t`.
3. The simulator advances and returns a reward `r_t` and new observation
   `s_(t+1)`.
4. The episode ends naturally (`terminated`) or because its time limit is
   reached (`truncated`).

The agent is not given the correct action. It must discover behavior that
maximizes the discounted sum of future rewards:

```text
return = r_t + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
```

`gamma` controls how much future rewards matter. Our initial value of `0.8`
comes from the HighwayEnv reference example and is an experiment setting, not a
law of nature.

## What the agent sees

The default kinematics observation is a small matrix. Each row represents the
ego vehicle or a nearby vehicle; columns describe features such as whether the
vehicle is present, relative position, and velocity. The neural network receives
numbers—not rendered road pixels.

This is an intentional simplification. It separates learning to make driving
decisions from learning visual perception.

## What DQN learns

DQN approximates a function `Q(observation, action)`: the expected future
return after choosing an action in an observation and behaving well afterward.
For each observation, the network outputs one Q-value per discrete action. A
greedy driver chooses the action with the largest Q-value.

During training, the policy sometimes chooses a random action. This
epsilon-greedy exploration lets it discover outcomes that its current network
would never try.

The central learning target is:

```text
target = reward + gamma * max(next_action_values)
```

There are important details around terminal states and a slowly updated target
network. We will implement those explicitly in the from-scratch milestone.

## Why start with a random baseline?

A reward number has no meaning by itself. The random policy gives us a lower
reference point. Later, a simple rule-based driver can provide a stronger
reference. Every agent is evaluated on the same sequence of seeds so that the
traffic scenarios are comparable.

Our first metrics are:

- mean episode return,
- variability of return,
- crash rate,
- mean episode length,
- mean speed,
- action frequencies.

High reward without a reasonable crash rate is not a convincing driving policy.

## One detail worth noticing in the code

Gymnasium returns both `terminated` and `truncated`. They both end the current
evaluation episode, but they have different mathematical meanings during
learning: a time limit does not imply that the future value of a state is zero.
Stable-Baselines3 handles this distinction for the reference agent.
