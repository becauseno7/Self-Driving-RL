# V8: human DAgger

V8 adds a human teaching loop on top of the frozen V7 driver. It is deliberately
a small residual rather than a replacement policy: the trained QR-DQN, V6 RLAIF
preferences, V7 persistent speed planner, and projected-gap protection remain in
the stack.

## What you teach

You label high-level lane judgment:

- `A` or `LEFT`: the driver should move left.
- `K`: the driver should keep its lane.
- `D` or `RIGHT`: the driver should move right.
- `W` or `UP`: the road permits faster progress.
- `S` or `DOWN`: slow down or create more space.
- `ENTER`: explicitly approve the complete lane and speed decision.
- `U` or `BACKSPACE`: remove your most recent label from this session.
- `ESC`: save everything collected so far and exit.

Silence is not a label. This matters: assuming that every moment without an
intervention is human approval would create thousands of low-quality negative
examples and drown out the useful corrections.

Throttle and braking are not taught as tenth-of-a-second key presses. Faster
and slower are three-second high-level intentions with a fixed target chosen
when you label them. The V7 planner translates that intention through its speed
hysteresis and emergency braking, so DAgger can learn pace without reintroducing
pedal chatter. Faster guidance is rejected whenever front risk is elevated.

## One DAgger round

Start a collection session:

```powershell
uv run sdr-dagger collect
```

Let the driver act most of the time. Correct clear mistakes, especially:

- staying behind a slow car despite an easy safe gap;
- choosing the worse of two passing lanes;
- beginning a pointless reversal;
- leaving a good passing lane too early;
- changing lanes when keeping the lane is calmer and equally fast.

Also press `ENTER` on representative good decisions. A useful first dataset has
at least 30 labels, including eight lane corrections, eight speed corrections,
four examples in each speed direction, and eight complete approvals.
Roughly 80-150 thoughtful labels is a stronger first round than thousands of
rapid, ambiguous key presses.

Train the gated residual:

```powershell
uv run sdr-dagger train
```

The trainer holds out 20% of each label class and learns four outputs: whether
to alter lane choice, the desired steering decision, whether to alter speed,
and the desired speed guidance. Lane and speed receive separate confidence
thresholds. A threshold is accepted only if it preserves at least 98% of its
held-out approvals and at least 80% of its overrides are correct.

Compare V7 and DAgger on the same unseen traffic seeds:

```powershell
uv run sdr-dagger evaluate --episodes 100
```

The evaluator reports `PROMOTE` only after at least 50 matched routes, all
safety and non-regression checks pass, and there is a measurable gain in net
passing, avoidable following, or lane reversals. Offline imitation accuracy by
itself is never a promotion decision.

Watch the result:

```powershell
uv run sdr-dagger watch --endless
```

## The aggregation step

DAgger is iterative. For round two, collect states visited by the newly learned
driver and append them to the same dataset:

```powershell
uv run sdr-dagger collect `
  --dagger-model runs/dagger/human-v1/dagger_model.pt

uv run sdr-dagger train --overwrite
```

This corrects distribution shift: the student changes which situations it
visits, so the expert labels those new situations and the dataset grows across
rounds. Every sample stores its session, traffic seed, and episode step.

## Safety boundary

The learned residual can preserve the current lane or propose a left/right
decision while retaining V7's pedal command. Before any learned lane change is
executed, an independent shield checks:

- lane boundaries and whether a merge is already in progress;
- current and projected front gap;
- current and projected rear gap;
- rear time-to-collision;
- whether the ego car can clear its current lane safely.

Unsafe human key presses are not recorded and are shown as rejected in the HUD.
Unsafe learned proposals fall back to V7 and increment the safety-veto counter.
Learned left/right corrections also have a 3.5-second maneuver cooldown, which
prevents one label from cascading into repeated lane hops across nearby states.

The final decision to promote a DAgger model is still on-road evaluation, not
offline imitation accuracy. It must match the base safety rate and improve the
driving metrics we care about: net overtakes, passing response, avoidable
following, lane reversals, unjustified braking, and clear-road stalls.
