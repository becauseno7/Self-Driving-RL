"""Train the first reference DQN and save everything needed to reproduce it."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from self_driving_rl.environment import ENV_CONFIG, ENV_ID, make_env
from self_driving_rl.metrics import evaluate_in_env

DQN_CONFIG: dict[str, Any] = {
    "policy_kwargs": {"net_arch": [256, 256]},
    "learning_rate": 5e-4,
    "buffer_size": 15_000,
    "learning_starts": 200,
    "batch_size": 32,
    "gamma": 0.8,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update_interval": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def package_versions() -> dict[str, str]:
    names = ["gymnasium", "highway-env", "numpy", "stable-baselines3", "torch"]
    return {name: importlib.metadata.version(name) for name in names}


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    args = parse_args()
    if args.timesteps < 1:
        raise SystemExit("--timesteps must be at least 1")
    if args.eval_episodes < 1:
        raise SystemExit("--eval-episodes must be at least 1")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / "dqn" / run_name
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    experiment_config = {
        "algorithm": "DQN",
        "environment_id": ENV_ID,
        "environment_config": ENV_CONFIG,
        "dqn_config": DQN_CONFIG,
        "timesteps": args.timesteps,
        "training_seed": args.seed,
        "evaluation_seed": 10_000,
        "evaluation_episodes": args.eval_episodes,
        "package_versions": package_versions(),
        "git_revision": git_revision(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(experiment_config, indent=2) + "\n",
        encoding="utf-8",
    )

    training_env = Monitor(make_env(), str(run_dir / "monitor.csv"))
    model = DQN(
        "MlpPolicy",
        training_env,
        **DQN_CONFIG,
        seed=args.seed,
        device="cpu",
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
    )

    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model_path = run_dir / "model"
        model.save(model_path)
    finally:
        training_env.close()

    evaluation_env = make_env()
    try:
        summary = evaluate_in_env(
            evaluation_env,
            lambda observation: int(model.predict(observation, deterministic=True)[0]),
            episodes=args.eval_episodes,
            seed=10_000,
        )
    finally:
        evaluation_env.close()

    (run_dir / "evaluation.json").write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved model and metrics to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
