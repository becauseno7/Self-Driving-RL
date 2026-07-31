"""Print one observation and transition in a human-readable form."""

from __future__ import annotations

import argparse

import numpy as np

from self_driving_rl.environment import ENV_CONFIG, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def format_observation(observation: np.ndarray) -> str:
    features = ENV_CONFIG["observation"]["features"]
    header = "vehicle  " + "  ".join(f"{feature:>8}" for feature in features)
    rows = [header]
    for index, vehicle in enumerate(observation):
        label = "ego" if index == 0 else f"nearby-{index}"
        values = "  ".join(f"{float(value):8.3f}" for value in vehicle)
        rows.append(f"{label:<8} {values}")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    env = make_env()
    try:
        observation, _ = env.reset(seed=args.seed)
        actions = env.unwrapped.action_type.actions_indexes
        idle_action = int(actions["IDLE"])

        print("Discrete actions:")
        for name, index in actions.items():
            print(f"  {index}: {name}")

        print("\nInitial normalized observation:")
        print(format_observation(observation))

        next_observation, reward, terminated, truncated, info = env.step(idle_action)
        print("\nAfter taking IDLE for one decision step:")
        print(format_observation(next_observation))
        print(
            f"\nreward={float(reward):.3f}, speed={float(info['speed']):.2f} m/s, "
            f"terminated={terminated}, truncated={truncated}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
