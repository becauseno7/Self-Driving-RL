"""Run, train, or watch the Neon Highway reinforcement-learning game."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from self_driving_rl.game_env import NeonHighwayEnv
from self_driving_rl.metrics import evaluate_in_env

GAME_DQN_CONFIG: dict[str, Any] = {
    "policy_kwargs": {"net_arch": [128, 128]},
    "learning_rate": 7e-4,
    "buffer_size": 50_000,
    "learning_starts": 750,
    "batch_size": 64,
    "gamma": 0.98,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 500,
    "exploration_fraction": 0.4,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
}


class GameHudCallback(BaseCallback):
    def __init__(self, game_env: NeonHighwayEnv, total_steps: int) -> None:
        super().__init__()
        self.game_env = game_env
        self.total_steps = total_steps
        self.recent_returns: deque[float] = deque(maxlen=20)
        self.best_return = float("-inf")
        self.collisions = 0
        self.completions = 0
        self.collision_types: Counter[str] = Counter()

    def _q_values(self) -> list[float]:
        observations = self.locals.get("new_obs")
        if observations is None:
            return [0.0] * 5
        observation_tensor, _ = self.model.policy.obs_to_tensor(observations)
        with th.no_grad():
            values = self.model.q_net(observation_tensor).detach().cpu().numpy()[0]
        return [float(value) for value in values]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("crashed", False):
                self.collisions += 1
                collision = info.get("collision")
                if isinstance(collision, dict) and collision.get("kind"):
                    self.collision_types[str(collision["kind"])] += 1
            if info.get("completed", False):
                self.completions += 1
            if "episode" in info:
                episode_return = float(info["episode"]["r"])
                self.recent_returns.append(episode_return)
                self.best_return = max(self.best_return, episode_return)

        epsilon = float(getattr(self.model, "exploration_rate", 1.0))
        self.game_env.hud_data.update(
            {
                "mode": "EXPLORING" if epsilon > 0.2 else "REFINING",
                "epsilon": epsilon,
                "training_step": self.num_timesteps,
                "training_total": self.total_steps,
                "mean_return": float(np.mean(self.recent_returns)) if self.recent_returns else 0.0,
                "best_return": self.best_return if self.best_return != float("-inf") else 0.0,
                "collisions": self.collisions,
                "completions": self.completions,
                "recent_returns": list(self.recent_returns),
                "q_values": self._q_values(),
                "collision_types": dict(self.collision_types),
            }
        )
        return not any(bool(info.get("user_quit", False)) for info in infos)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    random_parser = subparsers.add_parser("random", help="Watch an untrained random driver fail")
    random_parser.add_argument("--episodes", type=int, default=10)
    random_parser.add_argument("--seed", type=int, default=7)
    random_parser.add_argument("--fps", type=int, default=60)

    learn_parser = subparsers.add_parser("learn", help="Watch DQN learn while it drives")
    learn_parser.add_argument("--timesteps", type=int, default=30_000)
    learn_parser.add_argument("--seed", type=int, default=7)
    learn_parser.add_argument("--fps", type=int, default=120)
    learn_parser.add_argument("--run-name", type=str, default=None)
    learn_parser.add_argument("--eval-episodes", type=int, default=20)
    learn_parser.add_argument("--headless", action="store_true", help="Train as fast as possible")

    watch_parser = subparsers.add_parser("watch", help="Watch the newest or selected trained model")
    watch_parser.add_argument("--model", type=Path, default=None)
    watch_parser.add_argument("--episodes", type=int, default=10)
    watch_parser.add_argument("--seed", type=int, default=10_000)
    watch_parser.add_argument("--fps", type=int, default=60)
    return parser


def run_policy(
    env: NeonHighwayEnv,
    choose_action: Any,
    *,
    episodes: int,
    seed: int,
    mode: str,
    epsilon: float,
    q_values_provider: Any | None = None,
) -> None:
    completed = 0
    crashes = 0
    completions = 0
    best_return = float("-inf")
    recent_returns: deque[float] = deque(maxlen=20)
    collision_types: Counter[str] = Counter()

    while completed < episodes and not env.quit_requested:
        observation, _ = env.reset(seed=seed + completed)
        terminated = truncated = False
        while not (terminated or truncated) and not env.quit_requested:
            q_values = (
                q_values_provider(observation) if q_values_provider is not None else [0.0] * 5
            )
            action = int(choose_action(observation))
            env.hud_data.update(
                {
                    "mode": mode,
                    "epsilon": epsilon,
                    "best_return": best_return if best_return != float("-inf") else 0.0,
                    "collisions": crashes,
                    "completions": completions,
                    "recent_returns": list(recent_returns),
                    "q_values": q_values,
                    "collision_types": dict(collision_types),
                }
            )
            observation, _, terminated, truncated, info = env.step(action)

        if not info.get("user_quit", False):
            completed += 1
            crashes += int(bool(info.get("crashed", False)))
            completions += int(bool(info.get("completed", False)))
            collision = info.get("collision")
            if isinstance(collision, dict) and collision.get("kind"):
                collision_types[str(collision["kind"])] += 1
            best_return = max(best_return, env.episode_return)
            recent_returns.append(env.episode_return)


def random_mode(args: argparse.Namespace) -> None:
    env = NeonHighwayEnv(render_mode="human", render_fps=args.fps)
    env.action_space.seed(args.seed)
    try:
        run_policy(
            env,
            lambda _observation: env.action_space.sample(),
            episodes=args.episodes,
            seed=args.seed,
            mode="RANDOM DRIVER",
            epsilon=1.0,
        )
    finally:
        env.close()


def _evaluation(model: DQN, *, episodes: int, seed: int) -> dict[str, Any]:
    env = NeonHighwayEnv()
    try:
        return evaluate_in_env(
            env,
            lambda observation: int(model.predict(observation, deterministic=True)[0]),
            episodes=episodes,
            seed=seed,
        ).to_dict()
    finally:
        env.close()


def _model_q_values(model: DQN, observation: np.ndarray) -> list[float]:
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    with th.no_grad():
        values = model.q_net(observation_tensor).detach().cpu().numpy()[0]
    return [float(value) for value in values]


def _random_evaluation(*, episodes: int, seed: int) -> dict[str, Any]:
    env = NeonHighwayEnv()
    try:
        return evaluate_in_env(
            env,
            lambda _observation: int(env.action_space.sample()),
            episodes=episodes,
            seed=seed,
        ).to_dict()
    finally:
        env.close()


def learn_mode(args: argparse.Namespace) -> None:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / "game" / run_name
    if run_dir.exists():
        raise SystemExit(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    config = {
        "environment": NeonHighwayEnv.VERSION,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "evaluation_seed": 10_000,
        "evaluation_episodes": args.eval_episodes,
        "rendered_training": not args.headless,
        "render_fps": args.fps,
        "dqn": GAME_DQN_CONFIG,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["gymnasium", "numpy", "pygame-ce", "stable-baselines3", "torch"]
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    baseline = _random_evaluation(episodes=args.eval_episodes, seed=10_000)
    (run_dir / "random_baseline.json").write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )

    game_env = NeonHighwayEnv(
        render_mode=None if args.headless else "human",
        render_fps=args.fps,
    )
    monitored_env = Monitor(game_env, str(run_dir / "monitor.csv"))
    model = DQN(
        "MlpPolicy",
        monitored_env,
        **GAME_DQN_CONFIG,
        seed=args.seed,
        device="cpu",
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    callback = GameHudCallback(game_env, args.timesteps)

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    except KeyboardInterrupt:
        print("\nTraining interrupted; saving current model.")
    finally:
        model.save(run_dir / "model")
        monitored_env.close()

    evaluation = _evaluation(model, episodes=args.eval_episodes, seed=10_000)
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved game agent to {run_dir.resolve()}")
    print(
        f"Random crash rate: {baseline['crash_rate']:.0%}  ->  "
        f"Agent crash rate: {evaluation['crash_rate']:.0%}"
    )
    print(
        f"Random completion rate: {baseline['completion_rate']:.0%}  ->  "
        f"Agent completion rate: {evaluation['completion_rate']:.0%}"
    )


def _latest_model() -> Path:
    candidates = sorted(
        Path("runs/game").glob("*/model.zip"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise SystemExit("No trained game model found. Run: uv run sdr-game learn")
    return candidates[-1]


def watch_mode(args: argparse.Namespace) -> None:
    model_path = args.model or _latest_model()
    model = DQN.load(model_path, device="cpu")
    env = NeonHighwayEnv(render_mode="human", render_fps=args.fps)
    try:
        run_policy(
            env,
            lambda observation: model.predict(observation, deterministic=True)[0],
            episodes=args.episodes,
            seed=args.seed,
            mode="TRAINED AGENT",
            epsilon=0.0,
            q_values_provider=lambda observation: _model_q_values(model, observation),
        )
    finally:
        env.close()


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "episodes", 1) < 1:
        raise SystemExit("--episodes must be at least 1")

    if args.mode == "random":
        random_mode(args)
    elif args.mode == "learn":
        if args.timesteps < 1 or args.eval_episodes < 1:
            raise SystemExit("--timesteps and --eval-episodes must be at least 1")
        learn_mode(args)
    elif args.mode == "watch":
        watch_mode(args)


if __name__ == "__main__":
    main()
