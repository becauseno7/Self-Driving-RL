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
    SpeedGuidance,
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


def test_human_speed_guidance_changes_the_plan_without_raw_pedal_control() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    faster = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    slower = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    try:
        env.reset(seed=720)
        _clear_traffic(env)
        env.ego_speed = env.CRUISE_SPEED
        env.target_speed = env.CRUISE_SPEED
        observation = env._observation()

        faster.set_speed_guidance(
            SpeedGuidance.FASTER, current_speed=env.ego_speed
        )
        faster(observation)
        slower.set_speed_guidance(
            SpeedGuidance.SLOWER, current_speed=env.ego_speed
        )
        slower_action = slower(observation)
    finally:
        env.close()

    assert faster.desired_speed > faster.config.cruise_speed
    assert faster.reason == "human-taught faster progress"
    assert slower.desired_speed < slower.config.cruise_speed
    assert slower.reason == "human-taught slower progress"
    assert decode_action(slower_action)[1] == PEDAL_BRAKE


def test_faster_guidance_never_overrides_emergency_braking() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    try:
        env.reset(seed=721)
        _clear_traffic(env)
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.position = env.ego_position + env.CAR_LENGTH + 5.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed - 10.0
        env._invalidate_sensors()
        policy.set_speed_guidance(
            SpeedGuidance.FASTER, current_speed=env.ego_speed
        )

        action = policy(env._observation())
    finally:
        env.close()

    assert policy.intent == DrivingIntent.EMERGENCY
    assert policy.reason == "critical closing TTC"
    assert decode_action(action)[1] == PEDAL_BRAKE


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


def test_speed_matched_slow_leader_keeps_pass_desire_alive() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    try:
        observation, _ = env.reset(seed=719)
        _clear_traffic(env)
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.position = env.ego_position + 20.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed
        env._invalidate_sensors()
        observation = env._observation()

        assert not observed_passing_options(
            observation,
            include_speed_matched_slow_leaders=False,
        )
        assert observed_passing_options(observation)
        actions = [
            policy(observation)
            for _ in range(policy.config.pass_opportunity_dwell_steps)
        ]
    finally:
        env.close()

    assert all(decode_action(action)[0] == STEER_KEEP for action in actions[:-1])
    assert decode_action(actions[-1])[0] != STEER_KEEP
    assert policy.intent in {DrivingIntent.PASS_LEFT, DrivingIntent.PASS_RIGHT}


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


def test_comfort_braking_is_pulsed_but_emergency_braking_is_continuous() -> None:
    def pedal_sequence(*, gap: float, speed_deficit: float) -> tuple[list[int], list[str]]:
        env = NeonHighwayEnv(difficulty_mode="standard")
        policy = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
        try:
            observation, _ = env.reset(seed=722)
            _clear_traffic(env)
            leader = next(car for car in env.traffic if car.lane == env.target_lane)
            leader.position = env.ego_position + env.CAR_LENGTH + gap
            leader.previous_position = leader.position
            leader.speed = env.ego_speed - speed_deficit
            env._invalidate_sensors()
            observation = env._observation()
            pedals: list[int] = []
            modes: list[str] = []
            for _ in range(4):
                action = policy(observation)
                pedals.append(decode_action(action)[1])
                modes.append(policy.braking_mode)
                observation, _, terminated, truncated, _ = env.step(action)
                assert not (terminated or truncated)
            return pedals, modes
        finally:
            env.close()

    comfort_pedals, comfort_modes = pedal_sequence(gap=12.0, speed_deficit=4.0)
    emergency_pedals, emergency_modes = pedal_sequence(
        gap=8.0, speed_deficit=8.0
    )

    assert comfort_pedals[0] == PEDAL_BRAKE
    assert comfort_pedals.count(PEDAL_BRAKE) < len(comfort_pedals)
    assert not any(
        first == second == PEDAL_BRAKE
        for first, second in zip(comfort_pedals, comfort_pedals[1:], strict=False)
    )
    assert "COMFORT BRAKE" in comfort_modes
    assert emergency_pedals == [PEDAL_BRAKE] * 4
    assert emergency_modes == ["EMERGENCY"] * 4


def test_comfort_braking_hysteresis_ignores_small_threshold_jitter() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    try:
        env.reset(seed=723)
        _clear_traffic(env)
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.speed = env.ego_speed - 4.0

        leader.position = env.ego_position + env.CAR_LENGTH + 12.0
        leader.previous_position = leader.position
        env._invalidate_sensors()
        policy(env._observation())
        assert policy._comfort_braking_active

        # Urgency is now between the 0.30 release and 0.42 engage thresholds.
        leader.position = env.ego_position + env.CAR_LENGTH + 15.5
        leader.previous_position = leader.position
        env._invalidate_sensors()
        policy(env._observation())
        assert policy._comfort_braking_active

        leader.position = env.ego_position + env.CAR_LENGTH + 30.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed
        env._invalidate_sensors()
        policy(env._observation())
    finally:
        env.close()

    assert not policy._comfort_braking_active


def test_ordinary_follow_target_stays_close_to_the_leader_speed() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    policy = LongitudinalIntentPolicy(ConstantPolicy(FASTER))
    try:
        env.reset(seed=724)
        _clear_traffic(env)
        leader = next(car for car in env.traffic if car.lane == env.target_lane)
        leader.position = env.ego_position + env.CAR_LENGTH + 12.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed - 4.0
        env._invalidate_sensors()
        observation = env._observation()
        road = policy._road_state(observation)
        desired_gap = max(
            policy.config.minimum_follow_gap,
            policy.config.follow_time_headway * env.ego_speed,
        )
        follow_target = policy._following_speed(road, env.ego_speed, desired_gap)
    finally:
        env.close()

    assert (
        follow_target
        >= leader.speed - policy.config.maximum_comfort_speed_deficit - 1e-5
    )


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
