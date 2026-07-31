"""Run, train, or watch the Neon Highway reinforcement-learning game."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from self_driving_rl.game_env import ACTION_COUNT, NeonHighwayEnv
from self_driving_rl.metrics import evaluate_in_env, format_duration
from self_driving_rl.symmetry import MirrorSymmetry

# gamma must stay in step with NeonHighwayEnv.SHAPING_GAMMA: episodes are 450
# steps, so the 0.98 used through V3 gave a 50-step horizon and made the
# route-completion bonus worth 0.0006 at the start of an episode.
GAME_DQN_CONFIG: dict[str, Any] = {
    "policy_kwargs": {"net_arch": [128, 128]},
    "learning_rate": 7e-4,
    "buffer_size": 50_000,
    "learning_starts": 750,
    "batch_size": 64,
    "gamma": 0.995,
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
    "gamma": 0.995,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 2_000,
    "exploration_fraction": 0.25,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.02,
}

# Sized for a many-env headless run. V3 did 0.25 gradient steps per env step
# from a single environment, which left the GPU mostly idle; with --envs 8 this
# doubles the replay ratio while collecting experience eight times faster.
GPU_DQN_CONFIG: dict[str, Any] = {
    "policy_kwargs": {"net_arch": [256, 256]},
    "learning_rate": 1.5e-4,
    "buffer_size": 750_000,
    "learning_starts": 25_000,
    "batch_size": 512,
    "gamma": 0.995,
    # Small, frequent target updates avoid the 20%-72% policy swings seen in
    # V4's hard-copy checkpoints while still letting QR-DQN learn quickly.
    "tau": 0.02,
    "train_freq": (1, "step"),
    "gradient_steps": 4,
    "target_update_interval": 500,
    "exploration_fraction": 0.3,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.02,
}

DQN_PRESETS = {
    "standard": GAME_DQN_CONFIG,
    "high": HIGH_COMPUTE_DQN_CONFIG,
    "gpu": GPU_DQN_CONFIG,
}

ALGORITHMS: dict[str, type[OffPolicyAlgorithm]] = {"dqn": DQN, "qrdqn": QRDQN}

# QR-DQN learns a distribution over returns per action instead of a single mean.
# SB3's DQN has no Double-Q correction, and its overestimation showed up in V3
# as a policy that had all but abandoned one of the two escape directions.
QRDQN_POLICY_KWARGS: dict[str, Any] = {"n_quantiles": 64}


def build_algorithm_kwargs(algorithm: str, dqn_config: dict[str, Any]) -> dict[str, Any]:
    config = {key: value for key, value in dqn_config.items()}
    if algorithm == "qrdqn":
        policy_kwargs = dict(config.get("policy_kwargs", {}))
        policy_kwargs.update(QRDQN_POLICY_KWARGS)
        config["policy_kwargs"] = policy_kwargs
    return config


def load_model(
    path: Path,
    device: str = "cpu",
    env: Any | None = None,
) -> OffPolicyAlgorithm:
    """Load a checkpoint without needing to be told which algorithm wrote it."""
    errors: list[str] = []
    for name, algorithm_class in ALGORITHMS.items():
        try:
            return algorithm_class.load(path, device=device, env=env)
        except (OSError, ValueError, KeyError, RuntimeError, AttributeError) as error:
            errors.append(f"{name}: {error}")
    raise SystemExit(f"Could not load {path} as any known algorithm.\n" + "\n".join(errors))


def action_values(model: OffPolicyAlgorithm, observation_tensor: th.Tensor) -> np.ndarray:
    """Per-action values for the HUD, for both a Q-net and a quantile net."""
    with th.no_grad():
        if isinstance(model, QRDQN):
            # (batch, quantiles, actions) -> mean over quantiles is the Q value.
            values = model.quantile_net(observation_tensor).mean(dim=1)
        else:
            values = model.q_net(observation_tensor)
    return values.detach().cpu().numpy()[0]


def make_training_env(
    difficulty_mode: str,
    seed: int,
    index: int,
    mirror: bool,
    episode_seconds: float | None = None,
    endless: bool = False,
) -> Any:
    """Factory for SubprocVecEnv workers; each env gets its own seed stream."""

    def _init() -> Monitor:
        env: gym.Env = NeonHighwayEnv(
            difficulty_mode=difficulty_mode,
            episode_seconds=episode_seconds,
            endless=endless,
        )
        if mirror:
            env = MirrorSymmetry(env)
        env.reset(seed=seed + 1_000 * index)
        return Monitor(env)

    return _init


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
            return [0.0] * ACTION_COUNT
        observation_tensor, _ = self.model.policy.obs_to_tensor(observations)
        values = action_values(self.model, observation_tensor)
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
        difficulty_mode: str,
        episode_seconds: float | None = None,
        endless: bool = False,
    ) -> None:
        super().__init__()
        self.episode_seconds = episode_seconds
        self.endless = endless
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.seed = seed
        self.difficulty_mode = difficulty_mode
        self.next_evaluation = eval_freq
        self.best_score = (float("-inf"), float("-inf"))
        self.history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_evaluation:
            return True

        result = _evaluation(
            self.model,
            episodes=self.episodes,
            seed=self.seed,
            difficulty_mode=self.difficulty_mode,
            episode_seconds=self.episode_seconds,
            endless=self.endless,
        )
        record = {"timesteps": self.num_timesteps, **result}
        self.history.append(record)
        (self.run_dir / "validation_history.json").write_text(
            json.dumps(self.history, indent=2) + "\n",
            encoding="utf-8",
        )

        score = validation_score(result, endless=self.endless)
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
    random_parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        help=(
            "World seconds per real second. Pace and smoothness trade off at a "
            "fixed refresh rate: at 60 fps you get round(6 / speed) frames per "
            "simulation step, so 1=six frames, 3=two, 6=one. Raise --fps too if "
            "your monitor runs above 60 Hz."
        ),
    )
    random_parser.add_argument(
        "--difficulty",
        choices=sorted(NeonHighwayEnv.DIFFICULTY_MODES),
        default="hard",
    )
    random_parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Route length in simulated seconds (default 45; a wave is added every 15 s)",
    )
    random_parser.add_argument(
        "--endless",
        action="store_true",
        help="Never finish: one continuous drive that only a crash restarts",
    )

    learn_parser = subparsers.add_parser("learn", help="Watch DQN learn while it drives")
    learn_parser.add_argument("--timesteps", type=int, default=30_000)
    learn_parser.add_argument("--seed", type=int, default=7)
    learn_parser.add_argument("--fps", type=int, default=120)
    learn_parser.add_argument(
        "--speed",
        type=float,
        default=12.0,
        help="World seconds per real second (12.0 keeps rendered training at full speed)",
    )
    learn_parser.add_argument("--run-name", type=str, default=None)
    learn_parser.add_argument("--eval-episodes", type=int, default=20)
    learn_parser.add_argument("--headless", action="store_true", help="Train as fast as possible")
    learn_parser.add_argument(
        "--envs",
        type=int,
        default=1,
        help="Parallel environments (headless only; 8-16 keeps a modern GPU busy)",
    )
    learn_parser.add_argument(
        "--preset",
        choices=sorted(DQN_PRESETS),
        default="standard",
        help="DQN capacity/training preset",
    )
    learn_parser.add_argument(
        "--algo",
        choices=sorted(ALGORITHMS),
        default="dqn",
        help="qrdqn is distributional and does not overestimate like plain DQN",
    )
    learn_parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="Disable random left/right mirroring of training episodes",
    )
    learn_parser.set_defaults(mirror=True)
    learn_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device (auto uses CUDA when available)",
    )
    learn_parser.add_argument("--validation-freq", type=int, default=25_000)
    learn_parser.add_argument("--validation-episodes", type=int, default=20)
    learn_parser.add_argument("--validation-seed", type=int, default=20_000)
    learn_parser.add_argument(
        "--difficulty",
        choices=sorted(NeonHighwayEnv.DIFFICULTY_MODES),
        default="hard",
    )
    learn_parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Route length in simulated seconds (default 45; a wave is added every 15 s)",
    )
    learn_parser.add_argument(
        "--endless",
        action="store_true",
        help="Never finish: one continuous drive that only a crash restarts",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Score a trained model headlessly on fixed traffic seeds",
    )
    evaluate_parser.add_argument("--model", type=Path, default=None)
    evaluate_parser.add_argument("--episodes", type=int, default=100)
    evaluate_parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[10_000],
        help="One or more starting seeds; each is scored as a separate set",
    )
    evaluate_parser.add_argument("--output", type=Path, default=None)
    evaluate_parser.add_argument(
        "--difficulty",
        choices=sorted(NeonHighwayEnv.DIFFICULTY_MODES),
        default="hard",
    )
    evaluate_parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Route length in simulated seconds (default 45; a wave is added every 15 s)",
    )
    evaluate_parser.add_argument(
        "--endless",
        action="store_true",
        help="Never finish: one continuous drive that only a crash restarts",
    )

    watch_parser = subparsers.add_parser("watch", help="Watch the newest or selected trained model")
    watch_parser.add_argument("--model", type=Path, default=None)
    watch_parser.add_argument("--episodes", type=int, default=10)
    watch_parser.add_argument("--seed", type=int, default=10_000)
    watch_parser.add_argument("--fps", type=int, default=60)
    watch_parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        help=(
            "World seconds per real second. Pace and smoothness trade off at a "
            "fixed refresh rate: at 60 fps you get round(6 / speed) frames per "
            "simulation step, so 1=six frames, 3=two, 6=one. Raise --fps too if "
            "your monitor runs above 60 Hz."
        ),
    )
    watch_parser.add_argument(
        "--difficulty",
        choices=sorted(NeonHighwayEnv.DIFFICULTY_MODES),
        default="hard",
    )
    watch_parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Route length in simulated seconds (default 45; a wave is added every 15 s)",
    )
    watch_parser.add_argument(
        "--endless",
        action="store_true",
        help="Never finish: one continuous drive that only a crash restarts",
    )
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
    longest_survival = 0.0
    best_return = float("-inf")
    recent_returns: deque[float] = deque(maxlen=20)
    collision_types: Counter[str] = Counter()

    while completed < episodes and not env.quit_requested:
        observation, _ = env.reset(seed=seed + completed)
        reset_policy = getattr(choose_action, "reset", None)
        if callable(reset_policy):
            reset_policy()
        terminated = truncated = False
        while not (terminated or truncated) and not env.quit_requested:
            q_values = (
                q_values_provider(observation)
                if q_values_provider is not None
                else [0.0] * ACTION_COUNT
            )
            action = int(choose_action(observation))
            policy_hud = getattr(choose_action, "hud_data", None)
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
                    "longest_survival": longest_survival,
                }
            )
            if isinstance(policy_hud, dict):
                env.hud_data.update(policy_hud)
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
            if env.elapsed_seconds > longest_survival:
                longest_survival = env.elapsed_seconds
                if env.endless:
                    print(f"  New longest run: {format_duration(longest_survival)}")


def random_mode(args: argparse.Namespace) -> None:
    env = NeonHighwayEnv(
        render_mode="human",
        render_fps=args.fps,
        render_speed=args.speed,
        difficulty_mode=args.difficulty,
        episode_seconds=args.seconds,
        endless=args.endless,
    )
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
    model: BaseAlgorithm,
    *,
    episodes: int,
    seed: int,
    difficulty_mode: str,
    episode_seconds: float | None = None,
    endless: bool = False,
) -> dict[str, Any]:
    env = NeonHighwayEnv(
        difficulty_mode=difficulty_mode, episode_seconds=episode_seconds, endless=endless
    )
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


HELP_TRAIN_FRESH = (
    "Run `uv run sdr-game watch` without --model to automatically select a "
    "compatible checkpoint, or train a fresh model with: "
    "uv run sdr-game learn --preset gpu --algo qrdqn --headless --envs 8 "
    "--timesteps 2500000 --validation-freq 125000 --validation-episodes 100"
)


def _model_is_compatible(model: BaseAlgorithm, env: NeonHighwayEnv) -> bool:
    return (
        getattr(model.observation_space, "shape", None) == env.observation_space.shape
        and getattr(model.action_space, "n", None) == env.action_space.n
    )


def _ensure_model_compatible(model: BaseAlgorithm, env: NeonHighwayEnv) -> None:
    if _model_is_compatible(model, env):
        return
    model_shape = getattr(model.observation_space, "shape", None)
    model_actions = getattr(model.action_space, "n", None)
    raise SystemExit(
        "Model/environment mismatch: the model expects observations shaped "
        f"{model_shape} with {model_actions} actions, but {env.VERSION} provides "
        f"{env.observation_space.shape} with {env.action_space.n}. " + HELP_TRAIN_FRESH
    )


def _model_q_values(model: BaseAlgorithm, observation: np.ndarray) -> list[float]:
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    return [float(value) for value in action_values(model, observation_tensor)]


def _random_evaluation(
    *,
    episodes: int,
    seed: int,
    difficulty_mode: str,
    episode_seconds: float | None = None,
    endless: bool = False,
) -> dict[str, Any]:
    env = NeonHighwayEnv(
        difficulty_mode=difficulty_mode,
        episode_seconds=episode_seconds,
        endless=endless,
    )
    try:
        return evaluate_in_env(
            env,
            lambda _observation: int(env.action_space.sample()),
            episodes=episodes,
            seed=seed,
        ).to_dict()
    finally:
        env.close()


def validation_score(result: dict[str, Any], *, endless: bool) -> tuple[float, float]:
    """Rank checkpoints by the outcome the selected mode is meant to optimize."""
    primary = (
        float(result["mean_survival_seconds"])
        if endless
        else float(result["completion_rate"])
    )
    return primary, float(result["mean_return"])


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
        "algorithm": args.algo,
        "parallel_envs": args.envs,
        "mirror_augmentation": args.mirror,
        "difficulty": args.difficulty,
        "driving_objective": {
            "cruise_speed_mps": NeonHighwayEnv.CRUISE_SPEED,
            "overtake_bonus": NeonHighwayEnv.OVERTAKE_BONUS,
            "passed_by_traffic_cost": NeonHighwayEnv.PASSED_BY_TRAFFIC_COST,
            "blocked_with_safe_pass_cost": NeonHighwayEnv.BLOCKED_WITH_SAFE_PASS_COST,
            "lane_change_cost": NeonHighwayEnv.LANE_CHANGE_COMFORT_COST,
            "unsafe_lane_change_cost": NeonHighwayEnv.UNSAFE_LANE_CHANGE_COST,
            "crash_penalty": NeonHighwayEnv.CRASH_PENALTY,
        },
        "episode_seconds": args.seconds or NeonHighwayEnv.EPISODE_SECONDS,
        "endless": args.endless,
        "evaluation_seed": 10_000,
        "evaluation_episodes": args.eval_episodes,
        "validation_frequency": args.validation_freq,
        "validation_episodes": args.validation_episodes,
        "validation_seed": args.validation_seed,
        "rendered_training": not args.headless,
        "render_fps": args.fps,
        "render_speed": args.speed,
        "device": args.device,
        "dqn": build_algorithm_kwargs(args.algo, dqn_config),
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "gymnasium",
                "numpy",
                "pygame-ce",
                "sb3-contrib",
                "stable-baselines3",
                "torch",
            ]
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    baseline = _random_evaluation(
        episodes=args.eval_episodes,
        seed=10_000,
        difficulty_mode=args.difficulty,
        episode_seconds=args.seconds,
        endless=args.endless,
    )
    (run_dir / "random_baseline.json").write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )

    game_env: NeonHighwayEnv | None = None
    if args.envs > 1:
        # The live HUD reads one env's state directly, so parallel collection
        # and rendered training are mutually exclusive.
        training_env: Any = SubprocVecEnv(
            [
                make_training_env(
                    args.difficulty,
                    args.seed,
                    index,
                    args.mirror,
                    args.seconds,
                    args.endless,
                )
                for index in range(args.envs)
            ]
        )
        training_env = VecMonitor(training_env, str(run_dir / "monitor.csv"))
    else:
        game_env = NeonHighwayEnv(
            render_mode=None if args.headless else "human",
            render_fps=args.fps,
            render_speed=args.speed,
            difficulty_mode=args.difficulty,
            episode_seconds=args.seconds,
            endless=args.endless,
        )
        wrapped: gym.Env = MirrorSymmetry(game_env) if args.mirror else game_env
        training_env = Monitor(wrapped, str(run_dir / "monitor.csv"))

    model = ALGORITHMS[args.algo](
        "MlpPolicy",
        training_env,
        **build_algorithm_kwargs(args.algo, dqn_config),
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    callbacks: list[BaseCallback] = [
        SafetyEvalCallback(
            run_dir,
            eval_freq=args.validation_freq,
            episodes=args.validation_episodes,
            seed=args.validation_seed,
            difficulty_mode=args.difficulty,
            episode_seconds=args.seconds,
            endless=args.endless,
        )
    ]
    if game_env is not None:
        callbacks.insert(0, GameHudCallback(game_env, args.timesteps))
    callback = CallbackList(callbacks)

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    except KeyboardInterrupt:
        print("\nTraining interrupted; saving current model.")
    finally:
        model.save(run_dir / "last_model")
        training_env.close()

    best_model_path = run_dir / "best_model.zip"
    if best_model_path.exists():
        model = load_model(best_model_path, device=args.device)
    model.save(run_dir / "model")

    evaluation = _evaluation(
        model,
        episodes=args.eval_episodes,
        seed=10_000,
        difficulty_mode=args.difficulty,
        episode_seconds=args.seconds,
        endless=args.endless,
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved game agent to {run_dir.resolve()}")
    if args.endless:
        print(
            f"Random mean survival: {format_duration(baseline['mean_survival_seconds'])}  ->  "
            f"Agent mean survival: {format_duration(evaluation['mean_survival_seconds'])}"
        )
    else:
        print(
            f"Random crash rate: {baseline['crash_rate']:.0%}  ->  "
            f"Agent crash rate: {evaluation['crash_rate']:.0%}"
        )
        print(
            f"Random completion rate: {baseline['completion_rate']:.0%}  ->  "
            f"Agent completion rate: {evaluation['completion_rate']:.0%}"
        )


def _latest_model(model_directory: Path = Path("runs/game")) -> Path:
    reference_env = NeonHighwayEnv()
    candidates = sorted(
        model_directory.glob("*/model.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No trained game model found. Run: uv run sdr-game learn")

    for path in candidates:
        try:
            model = load_model(path, device="cpu")
        except (OSError, ValueError, KeyError, SystemExit):
            continue
        if _model_is_compatible(model, reference_env):
            return path

    raise SystemExit(
        f"No model compatible with {NeonHighwayEnv.VERSION} was found "
        f"(needs observations shaped {reference_env.observation_space.shape} and "
        f"{reference_env.action_space.n} actions). " + HELP_TRAIN_FRESH
    )


def evaluate_mode(args: argparse.Namespace) -> None:
    """Score one model across one or more independent seed sets.

    A single 100-episode figure hides more variance than it shows: the same V4
    checkpoint scores 55% on seeds 10,000 and 69% on seeds 30,000.
    """
    model_path = args.model or _latest_model()
    model = load_model(model_path, device="cpu")
    print(f"Model: {model_path}  ({type(model).__name__}, {args.difficulty} mode)\n")

    results: dict[str, Any] = {}
    for seed in args.seed:
        result = _evaluation(
            model,
            episodes=args.episodes,
            seed=seed,
            difficulty_mode=args.difficulty,
            episode_seconds=args.seconds,
            endless=args.endless,
        )
        results[str(seed)] = result
        span = f"  seeds {seed:>7,}-{seed + args.episodes - 1:<7,} "
        if args.endless:
            # Nothing ever "completes", so survival is the whole story.
            print(
                span
                + f"mean {format_duration(result['mean_survival_seconds']):>8}  "
                f"longest {format_duration(result['longest_survival_seconds']):>8}  "
                f"waves {result['mean_challenges_resolved']:>5.2f}  "
                f"net passes {result['mean_net_overtakes']:+5.1f}  "
                f"return {result['mean_return']:+.1f}"
            )
        else:
            print(
                span + f"completion {result['completion_rate']:>4.0%}  "
                f"crash {result['crash_rate']:>4.0%}  "
                f"speed {result['mean_speed'] * 3.6:>3.0f} km/h  "
                f"net passes {result['mean_net_overtakes']:+4.1f}  "
                f"return {result['mean_return']:+.1f}"
            )

    if len(results) > 1:
        if args.endless:
            means = [result["mean_survival_seconds"] for result in results.values()]
            best = max(result["longest_survival_seconds"] for result in results.values())
            print(
                f"\n  across {len(means)} sets: mean survival "
                f"{format_duration(min(means))}-{format_duration(max(means))}, "
                f"longest single run {format_duration(best)}"
            )
        else:
            values = [result["completion_rate"] for result in results.values()]
            print(
                f"\n  across {len(values)} sets: completion "
                f"{min(values):.0%}-{max(values):.0%}, mean {float(np.mean(values)):.0%}"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


def watch_mode(args: argparse.Namespace) -> None:
    model_path = args.model or _latest_model()
    model = load_model(model_path, device="cpu")
    env = NeonHighwayEnv(
        render_mode="human",
        render_fps=args.fps,
        render_speed=args.speed,
        difficulty_mode=args.difficulty,
        episode_seconds=args.seconds,
        endless=args.endless,
    )
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
            args.envs,
        ) < 1:
            raise SystemExit("Training, evaluation, and validation values must be at least 1")
        if args.envs > 1 and not args.headless:
            raise SystemExit(
                "--envs > 1 requires --headless: the live HUD follows a single environment."
            )
        learn_mode(args)
    elif args.mode == "evaluate":
        evaluate_mode(args)
    elif args.mode == "watch":
        watch_mode(args)


if __name__ == "__main__":
    main()
