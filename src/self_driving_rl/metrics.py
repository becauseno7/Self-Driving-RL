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
    # Episodes that hit the absolute step limit without crashing or completing.
    # crash_rate + completion_rate + timeout_rate should always be 1.0.
    timeout_rate: float
    # Endless mode never completes, so laps survived is the metric that moves.
    mean_laps: float
    mean_episode_length: float
    mean_speed: float
    mean_min_ttc: float
    mean_min_rear_ttc: float
    mean_challenges_presented: float
    mean_challenges_resolved: float
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
    timeouts = 0
    laps: list[int] = []
    action_counts: Counter[str] = Counter()
    collision_types: Counter[str] = Counter()
    episode_minimum_ttcs: list[float] = []
    episode_minimum_rear_ttcs: list[float] = []
    challenges_presented: list[int] = []
    challenges_resolved: list[int] = []

    for episode in range(episodes):
        episode_seed = seed + episode
        observation, _ = env.reset(seed=episode_seed)
        env.action_space.seed(episode_seed)
        terminated = truncated = False
        episode_return = 0.0
        episode_length = 0
        final_info: dict[str, Any] = {}
        minimum_ttc = float("inf")
        minimum_rear_ttc = float("inf")

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
            if "rear_ttc" in final_info:
                rear_ttc = float(final_info["rear_ttc"])
                if np.isfinite(rear_ttc):
                    minimum_rear_ttc = min(minimum_rear_ttc, rear_ttc)

        returns.append(episode_return)
        lengths.append(episode_length)
        crashes += int(bool(final_info.get("crashed", False)))
        completions += int(bool(final_info.get("completed", truncated and not terminated)))
        timeouts += int(bool(final_info.get("timed_out", False)))
        laps.append(int(final_info.get("laps_completed", 0)))
        collision = final_info.get("collision")
        if isinstance(collision, dict) and collision.get("kind"):
            collision_types[str(collision["kind"])] += 1
        if np.isfinite(minimum_ttc):
            episode_minimum_ttcs.append(minimum_ttc)
        if np.isfinite(minimum_rear_ttc):
            episode_minimum_rear_ttcs.append(minimum_rear_ttc)
        challenges_presented.append(int(final_info.get("challenges_presented", 0)))
        challenges_resolved.append(int(final_info.get("challenges_resolved", 0)))

    return EvaluationSummary(
        episodes=episodes,
        seed=seed,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        crash_rate=crashes / episodes,
        completion_rate=completions / episodes,
        timeout_rate=timeouts / episodes,
        mean_laps=float(np.mean(laps)),
        mean_episode_length=float(np.mean(lengths)),
        mean_speed=float(np.mean(speeds)) if speeds else float("nan"),
        mean_min_ttc=(
            float(np.mean(episode_minimum_ttcs)) if episode_minimum_ttcs else float("nan")
        ),
        mean_min_rear_ttc=(
            float(np.mean(episode_minimum_rear_ttcs))
            if episode_minimum_rear_ttcs
            else float("nan")
        ),
        mean_challenges_presented=float(np.mean(challenges_presented)),
        mean_challenges_resolved=float(np.mean(challenges_resolved)),
        action_counts=dict(sorted(action_counts.items())),
        collision_types=dict(sorted(collision_types.items())),
    )
