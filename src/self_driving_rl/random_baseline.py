"""Measure a policy that has no learning or driving knowledge."""

from __future__ import annotations

import argparse
import json

from self_driving_rl.environment import make_env
from self_driving_rl.metrics import evaluate_in_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")

    env = make_env()
    try:
        summary = evaluate_in_env(
            env,
            lambda _observation: int(env.action_space.sample()),
            episodes=args.episodes,
            seed=args.seed,
        )
    finally:
        env.close()

    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
