from __future__ import annotations

import numpy as np

from self_driving_rl.environment import make_env
from self_driving_rl.game_env import IDLE, NeonHighwayEnv
from self_driving_rl.metrics import evaluate_in_env


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


def test_outcome_rates_account_for_every_episode() -> None:
    """Crash, completion and timeout are the only three ways an episode ends."""
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        summary = evaluate_in_env(env, lambda _observation: IDLE, episodes=8, seed=4_000)
    finally:
        env.close()

    total = summary.crash_rate + summary.completion_rate + summary.timeout_rate
    assert np.isclose(total, 1.0)
