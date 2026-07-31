from __future__ import annotations

import numpy as np

from self_driving_rl.environment import make_env
from self_driving_rl.game_env import IDLE, NeonHighwayEnv
from self_driving_rl.metrics import evaluate_in_env, format_duration


def test_evaluation_summary_has_expected_shape() -> None:
    env = make_env()
    try:
        summary = evaluate_in_env(
            env,
            lambda _observation: 1,  # IDLE
            episodes=2,
            seed=321,
        )
    finally:
        env.close()

    assert summary.episodes == 2
    assert summary.seed == 321
    assert 0.0 <= summary.crash_rate <= 1.0
    assert summary.mean_episode_length > 0
    assert summary.action_counts == {"1": int(summary.mean_episode_length * 2)}
    assert summary.mean_overtakes == 0.0
    assert summary.mean_net_overtakes == 0.0


def test_format_duration_reads_naturally_at_every_scale() -> None:
    assert format_duration(0.0) == "0.0s"
    assert format_duration(12.35) == "12.3s"
    assert format_duration(59.9) == "59.9s"
    assert format_duration(60.0) == "1m 00s"
    assert format_duration(146.5) == "2m 26s"
    assert format_duration(1142.0) == "19m 02s"
    assert format_duration(3600.0) == "1h 00m"
    assert format_duration(3860.0) == "1h 04m"


def test_endless_summary_reports_survival_not_completion() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard", endless=True)
    try:
        summary = evaluate_in_env(env, lambda _observation: IDLE, episodes=4, seed=900)
    finally:
        env.close()

    assert summary.completion_rate == 0.0, "endless mode cannot complete"
    assert summary.crash_rate == 1.0
    assert summary.longest_survival_seconds >= summary.mean_survival_seconds > 0.0


def test_outcome_rates_account_for_every_episode() -> None:
    """Crash, completion and timeout are the only three ways an episode ends."""
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        summary = evaluate_in_env(env, lambda _observation: IDLE, episodes=8, seed=4_000)
    finally:
        env.close()

    total = summary.crash_rate + summary.completion_rate + summary.timeout_rate
    assert np.isclose(total, 1.0)
