"""Evaluate a saved DQN on held-out, reproducible traffic scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN

from self_driving_rl.environment import make_env
from self_driving_rl.metrics import evaluate_in_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")

    model = DQN.load(args.model, device="cpu")
    env = make_env()
    try:
        summary = evaluate_in_env(
            env,
            lambda observation: int(model.predict(observation, deterministic=True)[0]),
            episodes=args.episodes,
            seed=args.seed,
        )
    finally:
        env.close()

    result = summary.to_dict()
    rendered = json.dumps(result, indent=2)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
