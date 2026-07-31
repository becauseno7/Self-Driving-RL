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


def format_duration(seconds: float) -> str:
    """Render simulated seconds as 12.4s, 3m 07s, or 1h 04m."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        minutes, remainder = divmod(int(round(seconds)), 60)
        return f"{minutes}m {remainder:02d}s"
    hours, remainder = divmod(int(round(seconds)), 3600)
    return f"{hours}h {remainder // 60:02d}m"


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
    # Endless mode never completes, so survival time is the metric that moves.
    mean_survival_seconds: float
    longest_survival_seconds: float
    mean_episode_length: float
    mean_speed: float
    mean_distance_km: float
    mean_overtakes: float
    mean_passed_by_traffic: float
    mean_net_overtakes: float
    mean_lane_changes: float
    mean_unsafe_lane_changes: float
    mean_unproductive_lane_changes: float
    mean_rapid_lane_changes: float
    mean_lane_reversals: float
    mean_missed_passing_opportunities: float
    overtakes_per_100km: float
    lane_changes_per_100km: float
    lane_changes_per_overtake: float
    passing_response_rate: float
    passing_miss_rate: float
    blocked_step_rate: float
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
    survival_seconds: list[float] = []
    action_counts: Counter[str] = Counter()
    collision_types: Counter[str] = Counter()
    episode_minimum_ttcs: list[float] = []
    episode_minimum_rear_ttcs: list[float] = []
    challenges_presented: list[int] = []
    challenges_resolved: list[int] = []
    distances: list[float] = []
    overtakes: list[int] = []
    passed_by_traffic: list[int] = []
    lane_changes: list[int] = []
    unsafe_lane_changes: list[int] = []
    unproductive_lane_changes: list[int] = []
    rapid_lane_changes: list[int] = []
    lane_reversals: list[int] = []
    missed_passing_opportunities: list[int] = []
    passing_opportunities = 0
    passing_actions = 0
    blocked_steps = 0

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
        survival_seconds.append(float(final_info.get("elapsed_seconds", 0.0)))
        collision = final_info.get("collision")
        if isinstance(collision, dict) and collision.get("kind"):
            collision_types[str(collision["kind"])] += 1
        if np.isfinite(minimum_ttc):
            episode_minimum_ttcs.append(minimum_ttc)
        if np.isfinite(minimum_rear_ttc):
            episode_minimum_rear_ttcs.append(minimum_rear_ttc)
        challenges_presented.append(int(final_info.get("challenges_presented", 0)))
        challenges_resolved.append(int(final_info.get("challenges_resolved", 0)))
        distances.append(float(final_info.get("distance_m", 0.0)))
        overtakes.append(int(final_info.get("overtakes", 0)))
        passed_by_traffic.append(int(final_info.get("passed_by_traffic", 0)))
        lane_changes.append(int(final_info.get("lane_changes", 0)))
        unsafe_lane_changes.append(int(final_info.get("unsafe_lane_changes", 0)))
        unproductive_lane_changes.append(int(final_info.get("unproductive_lane_changes", 0)))
        rapid_lane_changes.append(int(final_info.get("rapid_lane_changes", 0)))
        lane_reversals.append(int(final_info.get("lane_reversals", 0)))
        missed_passing_opportunities.append(
            int(final_info.get("missed_passing_opportunities", 0))
        )
        passing_opportunities += int(final_info.get("passing_opportunities", 0))
        passing_actions += int(final_info.get("passing_actions", 0))
        blocked_steps += int(final_info.get("blocked_steps", 0))

    distance_km = float(np.sum(distances)) / 1000.0
    total_overtakes = int(np.sum(overtakes))
    total_lane_changes = int(np.sum(lane_changes))

    return EvaluationSummary(
        episodes=episodes,
        seed=seed,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        crash_rate=crashes / episodes,
        completion_rate=completions / episodes,
        timeout_rate=timeouts / episodes,
        mean_survival_seconds=float(np.mean(survival_seconds)),
        longest_survival_seconds=float(np.max(survival_seconds)),
        mean_episode_length=float(np.mean(lengths)),
        mean_speed=float(np.mean(speeds)) if speeds else float("nan"),
        mean_distance_km=float(np.mean(distances)) / 1000.0,
        mean_overtakes=float(np.mean(overtakes)),
        mean_passed_by_traffic=float(np.mean(passed_by_traffic)),
        mean_net_overtakes=float(np.mean(np.subtract(overtakes, passed_by_traffic))),
        mean_lane_changes=float(np.mean(lane_changes)),
        mean_unsafe_lane_changes=float(np.mean(unsafe_lane_changes)),
        mean_unproductive_lane_changes=float(np.mean(unproductive_lane_changes)),
        mean_rapid_lane_changes=float(np.mean(rapid_lane_changes)),
        mean_lane_reversals=float(np.mean(lane_reversals)),
        mean_missed_passing_opportunities=float(np.mean(missed_passing_opportunities)),
        overtakes_per_100km=(
            100.0 * total_overtakes / distance_km if distance_km > 0.0 else 0.0
        ),
        lane_changes_per_100km=(
            100.0 * total_lane_changes / distance_km if distance_km > 0.0 else 0.0
        ),
        lane_changes_per_overtake=(
            total_lane_changes / total_overtakes if total_overtakes else float("inf")
        ),
        passing_response_rate=(
            passing_actions / passing_opportunities if passing_opportunities else 0.0
        ),
        passing_miss_rate=(
            int(np.sum(missed_passing_opportunities)) / passing_opportunities
            if passing_opportunities
            else 0.0
        ),
        blocked_step_rate=blocked_steps / max(int(np.sum(lengths)), 1),
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
