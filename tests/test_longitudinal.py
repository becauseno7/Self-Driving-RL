from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from self_driving_rl.game_env import (
    FASTER,
    LANE_LEFT,
    LANE_RIGHT,
    PEDAL_BRAKE,
    PEDAL_GAS,
    SLOWER,
    STEER_KEEP,
    STEER_RIGHT,
    NeonHighwayEnv,
    decode_action,
)
from self_driving_rl.longitudinal import (
    DrivingIntent,
    LongitudinalIntentPolicy,
    observed_passing_options,
)


class ConstantPolicy:
    def __init__(self, action: int) -> None:
        self.action = action
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def __call__(self, _observation: NDArray[np.floating[Any]]) -> int:
        return self.action


def _clear_traffic(env: NeonHighwayEnv) -> None:
    for index, car in enumerate(env.traffic):
        car.position = env.ego_position + 200.0 + 10.0 * index
        car.previous_position = car.position
        car.speed = env.ego_speed
    env._invalidate_sensors()


def test_open_road_controller_ignores_raw_braking_and_recovers_cruise() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    raw_policy = ConstantPolicy(SLOWER)
    policy = LongitudinalIntentPolicy(raw_policy)
    try:
        observation, _ = env.reset(seed=711)
        _clear_traffic(env)
        observation = env._observation()
        actions = []
        for _ in range(9):
            action = policy(observation)
            actions.append(action)
            observation, _, terminated, truncated, _ = env.step(action)
            assert not (terminated or truncated)
    finally:
        env.close()

    pedals = [decode_action(action)[1] for action in actions]
    assert PEDAL_BRAKE not in pedals
    assert pedals.count(PEDAL_GAS) <= 3
    assert policy.intent == DrivingIntent.CRUISE
    assert env.target_speed > 22.0


def test_controller_commits_to_a_safe_pass_after_the_lane_change_starts() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(LANE_LEFT))
    try:
        observation, _ = env.reset(seed=712)
        _clear_traffic(env)
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.position = env.ego_position + env.CAR_LENGTH + 20.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed - 5.0
        env._invalidate_sensors()
        observation = env._observation()
        assert 0 in observed_passing_options(observation)

        first_action = policy(observation)
        observation, _, _, _, _ = env.step(first_action)
        second_action = policy(observation)
    finally:
        env.close()

    assert decode_action(first_action)[0] == decode_action(LANE_LEFT)[0]
    assert decode_action(second_action)[0] == STEER_KEEP
    assert policy.intent == DrivingIntent.PASS_LEFT
    assert policy.desired_speed > policy.config.cruise_speed


def test_controller_takes_persistent_easy_pass_when_lane_policy_waits() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    try:
        observation, _ = env.reset(seed=718)
        _clear_traffic(env)
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.position = env.ego_position + env.CAR_LENGTH + 20.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed - 5.0
        env._invalidate_sensors()
        observation = env._observation()

        actions = []
        for _ in range(policy.config.pass_opportunity_dwell_steps):
            action = policy(observation)
            actions.append(action)
            observation, _, terminated, truncated, _ = env.step(action)
            assert not (terminated or truncated)
    finally:
        env.close()

    assert all(decode_action(action)[0] == STEER_KEEP for action in actions[:-1])
    assert decode_action(actions[-1])[0] == decode_action(LANE_LEFT)[0]
    assert policy.lane_veto_reason == "taking persistent safe pass"


def test_controller_can_override_raw_gas_with_emergency_braking() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    try:
        observation, _ = env.reset(seed=713)
        _clear_traffic(env)
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.position = env.ego_position + env.CAR_LENGTH + 5.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed - 10.0
        env._invalidate_sensors()

        action = policy(env._observation())
    finally:
        env.close()

    assert decode_action(action)[1] == PEDAL_BRAKE
    assert policy.intent == DrivingIntent.EMERGENCY
    assert policy.intervened


def test_closing_traffic_cancels_pass_commitment() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    try:
        _, _ = env.reset(seed=717)
        _clear_traffic(env)
        policy._pass_steps_remaining = 20
        policy._pass_direction = 1
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.position = env.ego_position + env.CAR_LENGTH + 15.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed - 8.0
        env._invalidate_sensors()

        action = policy(env._observation())
    finally:
        env.close()

    assert policy.intent in {DrivingIntent.FOLLOW, DrivingIntent.EMERGENCY}
    assert policy._pass_steps_remaining == 0
    assert decode_action(action)[1] == PEDAL_BRAKE


def test_following_gap_uses_time_headway_without_double_counting_minimum() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    try:
        _, _ = env.reset(seed=716)
        _clear_traffic(env)
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.position = env.ego_position + 45.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed - 0.3
        env._invalidate_sensors()

        policy(env._observation())
    finally:
        env.close()

    assert policy.intent == DrivingIntent.CRUISE


def test_controller_vetoes_lane_change_with_fast_closing_rear_car() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(LANE_LEFT))
    try:
        _, _ = env.reset(seed=714)
        _clear_traffic(env)
        rear_car = next(car for car in env.traffic if car.lane == env.target_lane - 1)
        rear_car.position = env.ego_position - 16.0
        rear_car.previous_position = rear_car.position
        rear_car.speed = env.ego_speed + 8.0
        env._invalidate_sensors()

        action = policy(env._observation())
    finally:
        env.close()

    assert decode_action(action)[0] == STEER_KEEP
    assert policy.lane_intervened
    assert policy.lane_veto_reason == "unsafe projected merge gap"


def test_controller_blocks_direction_reversal_during_committed_pass() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    raw_policy = ConstantPolicy(LANE_LEFT)
    policy = LongitudinalIntentPolicy(raw_policy)
    try:
        _, _ = env.reset(seed=715)
        _clear_traffic(env)
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.position = env.ego_position + env.CAR_LENGTH + 20.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed - 5.0
        env._invalidate_sensors()
        first_action = policy(env._observation())
        observation, _, _, _, _ = env.step(first_action)
        while float(observation[4]) >= 0.05:
            observation, _, terminated, truncated, _ = env.step(policy(observation))
            assert not (terminated or truncated)

        raw_policy.action = LANE_RIGHT
        reversal = policy(observation)
    finally:
        env.close()

    assert decode_action(reversal)[0] == STEER_KEEP
    assert policy.lane_veto_reason == "blocked pass-direction reversal"
    assert STEER_RIGHT == decode_action(LANE_RIGHT)[0]
