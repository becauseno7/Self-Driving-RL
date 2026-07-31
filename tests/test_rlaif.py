from __future__ import annotations

import numpy as np
import torch as th

from self_driving_rl.game_env import IDLE, NeonHighwayEnv
from self_driving_rl.rlaif import (
    FEATURE_NAMES,
    OverrideThresholds,
    PreferenceFeatureAccumulator,
    PreferenceOverrideNet,
    PreferenceOverridePolicy,
    PreferenceRewardModel,
    PreferenceRewardWrapper,
    _apply_override_gate,
    overall_driving_score,
)


def test_preference_features_are_additive_counter_deltas() -> None:
    features = PreferenceFeatureAccumulator()
    first = features.observe(
        {
            "speed": 27.0,
            "distance_m": 2.7,
            "lane_changes": 1,
            "rapid_lane_changes": 1,
            "blocked_steps": 1,
            "acceleration": 1.0,
        }
    )
    second = features.observe(
        {
            "speed": 27.0,
            "distance_m": 5.4,
            "lane_changes": 1,
            "rapid_lane_changes": 1,
            "blocked_steps": 1,
            "acceleration": 1.0,
            "completed": True,
        }
    )

    lane_change_index = FEATURE_NAMES.index("lane_changes")
    completion_index = FEATURE_NAMES.index("completion")
    assert first[lane_change_index] == 1.0
    assert second[lane_change_index] == 0.0
    assert second[completion_index] == 1.0
    np.testing.assert_allclose(features.values, first + second)


def test_preference_wrapper_preserves_base_reward_and_reports_bonus() -> None:
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    weights[FEATURE_NAMES.index("distance_km")] = 1.0
    model = PreferenceRewardModel(
        feature_names=FEATURE_NAMES,
        feature_scales=tuple([1.0] * len(FEATURE_NAMES)),
        weights=tuple(float(value) for value in weights),
        episode_reward_scale=1.0,
        clip_per_step=1.0,
    )
    env = PreferenceRewardWrapper(NeonHighwayEnv(difficulty_mode="standard"), model)
    try:
        env.reset(seed=91)
        _, reward, _, _, info = env.step(IDLE)
    finally:
        env.close()

    assert info["preference_reward"] > 0.0
    assert np.isclose(reward, info["base_reward"] + info["preference_reward"])


def test_overall_score_requires_safety_and_values_clean_progress() -> None:
    base = {
        "completion_rate": 0.95,
        "mean_return": 50.0,
        "mean_net_overtakes": 6.0,
        "mean_lane_changes": 10.0,
        "mean_lane_reversals": 1.0,
        "mean_missed_passing_opportunities": 1.0,
        "blocked_step_rate": 0.05,
    }
    unsafe = {**base, "completion_rate": 0.89, "mean_net_overtakes": 20.0}
    wasteful = {**base, "mean_lane_changes": 20.0, "mean_lane_reversals": 5.0}

    assert overall_driving_score(unsafe) == float("-inf")
    assert overall_driving_score(base) > overall_driving_score(wasteful)


def test_override_network_predicts_kind_and_action_for_each_state() -> None:
    network = PreferenceOverrideNet(observation_size=33, action_count=9)
    kind_logits, action_logits = network(
        th.zeros((4, 33)), th.tensor([0, 1, 2, 3])
    )

    assert kind_logits.shape == (4, 3)
    assert action_logits.shape == (4, 9)


def test_override_gate_only_changes_confident_decisions() -> None:
    base_actions = np.asarray([0, 1, 2, 3], dtype=np.int64)
    proposed_actions = np.asarray([8, 7, 6, 5], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
            [0.10, 0.79, 0.11],
        ],
        dtype=np.float32,
    )

    actions, kinds = _apply_override_gate(
        base_actions,
        probabilities,
        proposed_actions,
        OverrideThresholds(calm=0.80, passing=0.80),
    )

    np.testing.assert_array_equal(actions, [0, 7, 6, 3])
    np.testing.assert_array_equal(kinds, [0, 1, 2, 0])


def test_override_safety_shield_reconstructs_environment_pass_options() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard")
    rng = np.random.default_rng(928)
    try:
        observation, _ = env.reset(seed=928)
        for _ in range(250):
            assert PreferenceOverridePolicy._passing_options(
                observation
            ) == env.passing_lane_options()
            observation, _, terminated, truncated, _ = env.step(
                int(rng.integers(0, env.action_space.n))
            )
            if terminated or truncated:
                observation, _ = env.reset(seed=int(rng.integers(1, 100_000)))
    finally:
        env.close()
