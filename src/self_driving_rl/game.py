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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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

HIGH_COMPUTE_DQN_CONFIG: dict[str, Any] = {
    "policy_kwargs": {"net_arch": [256, 256]},
    "learning_rate": 2e-4,
    "buffer_size": 250_000,
    "learning_starts": 5_000,
    "batch_size": 128,
    "gamma": 0.98,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 2_000,
    "exploration_fraction": 0.25,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.02,
}

DQN_PRESETS = {
    "standard": GAME_DQN_CONFIG,
    "high": HIGH_COMPUTE_DQN_CONFIG,
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


class SafetyEvalCallback(BaseCallback):
    """Evaluate periodically and retain the safest checkpoint from a long run."""

    def __init__(
        self,
        run_dir: Path,
        *,
        eval_freq: int,
        episodes: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.seed = seed
        self.next_evaluation = eval_freq
        self.best_score = (float("-inf"), float("-inf"))
        self.history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_evaluation:
            return True

        result = _evaluation(self.model, episodes=self.episodes, seed=self.seed)
        record = {"timesteps": self.num_timesteps, **result}
        self.history.append(record)
        (self.run_dir / "validation_history.json").write_text(
            json.dumps(self.history, indent=2) + "\n",
            encoding="utf-8",
        )

        score = (result["completion_rate"], result["mean_return"])
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.run_dir / "best_model")
            (self.run_dir / "best_validation.json").write_text(
                json.dumps(record, indent=2) + "\n",
                encoding="utf-8",
            )

        print(
            f"Validation at {self.num_timesteps:,} steps: "
            f"completion {result['completion_rate']:.0%}, "
            f"crash {result['crash_rate']:.0%}"
        )
        self.next_evaluation += self.eval_freq
        return True


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
    learn_parser.add_argument(
        "--preset",
        choices=sorted(DQN_PRESETS),
        default="standard",
        help="DQN capacity/training preset",
    )
    learn_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device (auto uses CUDA when available)",
    )
    learn_parser.add_argument("--validation-freq", type=int, default=25_000)
    learn_parser.add_argument("--validation-episodes", type=int, default=20)
    learn_parser.add_argument("--validation-seed", type=int, default=20_000)

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


def _evaluation(
    model: DQN,
    *,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    env = NeonHighwayEnv()
    try:
        _ensure_model_compatible(model, env)
        return evaluate_in_env(
            env,
            lambda observation: int(model.predict(observation, deterministic=True)[0]),
            episodes=episodes,
            seed=seed,
        ).to_dict()
    finally:
        env.close()


def _ensure_model_compatible(model: DQN, env: NeonHighwayEnv) -> None:
    model_shape = getattr(model.observation_space, "shape", None)
    environment_shape = env.observation_space.shape
    if model_shape != environment_shape:
        raise SystemExit(
            "Model/environment mismatch: "
            f"the model expects observations shaped {model_shape}, but "
            f"{env.VERSION} provides {environment_shape}. "
            "Run `uv run sdr-game watch` without --model to automatically select a "
            "compatible checkpoint, or train a fresh V2 model with: "
            "uv run sdr-game learn --timesteps 30000"
        )


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
    dqn_config = DQN_PRESETS[args.preset]

    config = {
        "environment": NeonHighwayEnv.VERSION,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "preset": args.preset,
        "evaluation_seed": 10_000,
        "evaluation_episodes": args.eval_episodes,
        "validation_frequency": args.validation_freq,
        "validation_episodes": args.validation_episodes,
        "validation_seed": args.validation_seed,
        "rendered_training": not args.headless,
        "render_fps": args.fps,
        "device": args.device,
        "dqn": dqn_config,
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
        **dqn_config,
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    callback = CallbackList(
        [
            GameHudCallback(game_env, args.timesteps),
            SafetyEvalCallback(
                run_dir,
                eval_freq=args.validation_freq,
                episodes=args.validation_episodes,
                seed=args.validation_seed,
            ),
        ]
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    except KeyboardInterrupt:
        print("\nTraining interrupted; saving current model.")
    finally:
        model.save(run_dir / "last_model")
        monitored_env.close()

    best_model_path = run_dir / "best_model.zip"
    if best_model_path.exists():
        model = DQN.load(best_model_path, device=args.device)
    model.save(run_dir / "model")

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


def _latest_model(
    model_directory: Path = Path("runs/game"),
    required_shape: tuple[int, ...] | None = None,
) -> Path:
    if required_shape is None:
        required_shape = NeonHighwayEnv().observation_space.shape

    candidates = sorted(
        model_directory.glob("*/model.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No trained game model found. Run: uv run sdr-game learn")

    for path in candidates:
        try:
            model = DQN.load(path, device="cpu")
        except (OSError, ValueError, KeyError):
            continue
        if getattr(model.observation_space, "shape", None) == required_shape:
            return path

    raise SystemExit(
        f"No model compatible with {NeonHighwayEnv.VERSION} observations shaped "
        f"{required_shape} was found. Train one with: "
        "uv run sdr-game learn --timesteps 30000"
    )


def watch_mode(args: argparse.Namespace) -> None:
    model_path = args.model or _latest_model()
    model = DQN.load(model_path, device="cpu")
    env = NeonHighwayEnv(render_mode="human", render_fps=args.fps)
    try:
        _ensure_model_compatible(model, env)
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
        if min(
            args.timesteps,
            args.eval_episodes,
            args.validation_freq,
            args.validation_episodes,
        ) < 1:
            raise SystemExit("Training, evaluation, and validation values must be at least 1")
        learn_mode(args)
    elif args.mode == "watch":
        watch_mode(args)


if __name__ == "__main__":
    main()
