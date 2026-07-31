"""Shared evaluation logic so every policy is measured the same way."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

Policy = Callable[[NDArray[np.floating[Any]]], int]


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    seed: int
    mean_return: float
    std_return: float
    crash_rate: float
    completion_rate: float
    mean_episode_length: float
    mean_speed: float
    mean_min_ttc: float
    action_counts: dict[str, int]
    collision_types: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_in_env(
    env: gym.Env,
    policy: Policy,
    *,
    episodes: int,
    seed: int,
) -> EvaluationSummary:
    """Run complete episodes and aggregate driving-oriented metrics."""
    returns: list[float] = []
    lengths: list[int] = []
    speeds: list[float] = []
    crashes = 0
    completions = 0
    action_counts: Counter[str] = Counter()
    collision_types: Counter[str] = Counter()
    episode_minimum_ttcs: list[float] = []

    for episode in range(episodes):
        episode_seed = seed + episode
        observation, _ = env.reset(seed=episode_seed)
        env.action_space.seed(episode_seed)
        terminated = truncated = False
        episode_return = 0.0
        episode_length = 0
        final_info: dict[str, Any] = {}
        minimum_ttc = float("inf")

        while not (terminated or truncated):
            action = int(policy(observation))
            observation, reward, terminated, truncated, final_info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            action_counts[str(action)] += 1

            if "speed" in final_info:
                speeds.append(float(final_info["speed"]))
            if "ttc" in final_info:
                ttc = float(final_info["ttc"])
                if np.isfinite(ttc):
                    minimum_ttc = min(minimum_ttc, ttc)

        returns.append(episode_return)
        lengths.append(episode_length)
        crashes += int(bool(final_info.get("crashed", False)))
        completions += int(bool(final_info.get("completed", truncated and not terminated)))
        collision = final_info.get("collision")
        if isinstance(collision, dict) and collision.get("kind"):
            collision_types[str(collision["kind"])] += 1
        if np.isfinite(minimum_ttc):
            episode_minimum_ttcs.append(minimum_ttc)

    return EvaluationSummary(
        episodes=episodes,
        seed=seed,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        crash_rate=crashes / episodes,
        completion_rate=completions / episodes,
        mean_episode_length=float(np.mean(lengths)),
        mean_speed=float(np.mean(speeds)) if speeds else float("nan"),
        mean_min_ttc=(
            float(np.mean(episode_minimum_ttcs)) if episode_minimum_ttcs else float("nan")
        ),
        action_counts=dict(sorted(action_counts.items())),
        collision_types=dict(sorted(collision_types.items())),
    )
