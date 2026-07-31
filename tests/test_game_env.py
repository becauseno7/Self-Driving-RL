from __future__ import annotations

import numpy as np

from self_driving_rl.game_env import IDLE, NeonHighwayEnv


def test_game_environment_contract() -> None:
    env = NeonHighwayEnv()
    try:
        observation, info = env.reset(seed=42)
        next_observation, reward, terminated, truncated, next_info = env.step(1)
    finally:
        env.close()

    assert observation.shape == (15,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert env.action_space.n == 5
    assert isinstance(info, dict)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_info["action"] == "HOLD"
    assert next_info["collision"] is None
    assert set(next_info["reward_components"]) == {
        "progress",
        "safety",
        "comfort",
        "rules",
        "terminal",
    }
    assert np.isclose(sum(next_info["reward_components"].values()), reward)
    assert next_info["ttc"] > 0


def test_game_reset_is_reproducible() -> None:
    first = NeonHighwayEnv()
    second = NeonHighwayEnv()
    try:
        first_observation, _ = first.reset(seed=123)
        second_observation, _ = second.reset(seed=123)
    finally:
        first.close()
        second.close()

    np.testing.assert_array_equal(first_observation, second_observation)


def test_swept_collision_records_impact_details() -> None:
    env = NeonHighwayEnv()
    try:
        env.reset(seed=9)
        car = env.traffic[0]
        car.lane = env.target_lane
        env.lane_position = float(env.target_lane)
        env.previous_ego_position = 0.0
        env.ego_position = 3.0
        car.previous_position = 5.0
        car.position = 2.0

        collision = env._detect_collision()
    finally:
        env.close()

    assert collision is not None
    assert collision.kind in {"FRONT IMPACT", "REAR IMPACT"}
    assert collision.severity in {"LOW", "MEDIUM", "HIGH"}
    assert collision.impact_speed >= 0


def test_traffic_brakes_for_agent_ahead() -> None:
    env = NeonHighwayEnv()
    try:
        env.reset(seed=12)
        car = env.traffic[0]
        car.lane = env.target_lane
        car.position = env.ego_position - 10.0
        car.speed = 29.0
        car.desired_speed = 29.0
        env.ego_speed = 18.0

        target_speed = env._traffic_target_speed(car)
    finally:
        env.close()

    assert target_speed < car.desired_speed


def test_route_completion_adds_terminal_bonus() -> None:
    env = NeonHighwayEnv()
    try:
        env.reset(seed=21)
        env.step_count = env.max_episode_steps - 1
        for car in env.traffic:
            car.position = env.ego_position + 100.0 + car.lane * 10.0
            car.previous_position = car.position

        _, reward, terminated, truncated, info = env.step(IDLE)
    finally:
        env.close()

    assert not terminated
    assert truncated
    assert info["completed"] is True
    assert info["reward_components"]["terminal"] == 5.0
    assert reward > 5.0
