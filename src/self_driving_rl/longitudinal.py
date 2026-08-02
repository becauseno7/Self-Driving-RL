"""Persistent driving intent and smooth longitudinal control for Neon Highway.

The learned policy proposes lane choices. This module adds persistent passing
intent, projected-gap protection, and a slower, explainable desired-speed
planner in place of tenth-of-a-second pedal jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from self_driving_rl.game_env import (
    ACTION_NAMES,
    PEDAL_BRAKE,
    PEDAL_COAST,
    PEDAL_GAS,
    STEER_KEEP,
    STEER_LEFT,
    STEER_RIGHT,
    NeonHighwayEnv,
    decode_action,
    encode_action,
)


class DrivingIntent(StrEnum):
    CRUISE = "CRUISE"
    FOLLOW = "FOLLOW"
    SEARCH_PASS = "SEARCH PASS"
    PASS_LEFT = "PASS LEFT"
    PASS_RIGHT = "PASS RIGHT"
    RETURN = "RETURN"
    EMERGENCY = "EMERGENCY"


class SpeedGuidance(IntEnum):
    """Persistent high-level human guidance; never a raw pedal command."""

    BASE = 0
    FASTER = 1
    SLOWER = 2


@dataclass(frozen=True)
class LongitudinalConfig:
    cruise_speed: float = NeonHighwayEnv.CRUISE_SPEED
    passing_speed: float = 30.0
    minimum_follow_gap: float = 14.0
    follow_time_headway: float = 1.15
    follow_trigger_margin: float = 7.0
    critical_front_urgency: float = 0.72
    braking_front_urgency: float = 0.42
    braking_release_urgency: float = 0.30
    rear_pressure_urgency: float = 0.55
    speed_deadband: float = 0.75
    command_interval_steps: int = 4
    recovery_command_interval_steps: int = 3
    comfort_brake_min_interval_steps: int = 2
    comfort_brake_max_interval_steps: int = 4
    emergency_command_interval_steps: int = 1
    pass_commitment_steps: int = 35
    pass_opportunity_dwell_steps: int = 5
    desired_speed_smoothing: float = 0.22
    comfort_deceleration_smoothing: float = 0.35
    maximum_comfort_speed_deficit: float = 1.0
    merge_projection_seconds: float = 0.6
    merge_origin_exposure_seconds: float = 0.3
    merge_front_buffer: float = 3.0
    merge_rear_buffer: float = 3.0
    recognize_speed_matched_slow_leaders: bool = True
    guidance_duration_steps: int = 30
    guidance_speed_delta: float = 4.0


def observed_lane_reading(
    observation: NDArray[np.floating[Any]], lane: int
) -> tuple[float, float, float, float, float, float]:
    """Decode one lane's normalized sensor block."""
    offset = 9 + 6 * lane
    speed_span = NeonHighwayEnv.MAX_SPEED - NeonHighwayEnv.MIN_SPEED
    return (
        float(observation[offset]) * NeonHighwayEnv.SENSOR_DISTANCE,
        float(observation[offset + 1]) * speed_span,
        float(observation[offset + 2]),
        float(observation[offset + 3]) * NeonHighwayEnv.SENSOR_DISTANCE,
        float(observation[offset + 4]) * speed_span,
        float(observation[offset + 5]),
    )


def observed_passing_options(
    observation: NDArray[np.floating[Any]],
    *,
    include_speed_matched_slow_leaders: bool = True,
) -> tuple[int, ...]:
    """Reconstruct safe and useful adjacent passing lanes from observation."""
    if float(observation[4]) >= 0.05:
        return ()
    target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
    front_gap, front_relative, _, _, _, _ = observed_lane_reading(
        observation, target_lane
    )
    speed_span = NeonHighwayEnv.MAX_SPEED - NeonHighwayEnv.MIN_SPEED
    ego_speed = NeonHighwayEnv.MIN_SPEED + float(observation[0]) * speed_span
    leader_speed = ego_speed + front_relative
    slow_leader = (
        include_speed_matched_slow_leaders
        and leader_speed
        < NeonHighwayEnv.CRUISE_SPEED
        - NeonHighwayEnv.PASSING_SLOW_LEADER_MARGIN
    )
    if (
        front_gap >= NeonHighwayEnv.PASSING_TRIGGER_GAP
        or (
            front_relative >= -NeonHighwayEnv.PASSING_MIN_CLOSING_SPEED
            and not slow_leader
        )
    ):
        return ()

    options: list[int] = []
    for candidate in (target_lane - 1, target_lane + 1):
        if not 0 <= candidate < NeonHighwayEnv.LANES:
            continue
        ahead_gap, ahead_relative, _, behind_gap, behind_relative, _ = (
            observed_lane_reading(observation, candidate)
        )
        rear_closing_speed = max(behind_relative, 0.0)
        rear_ttc = (
            behind_gap / rear_closing_speed
            if rear_closing_speed > 0.1
            else float("inf")
        )
        safe = (
            ahead_gap > NeonHighwayEnv.SAFE_LANE_CHANGE_FRONT_GAP
            and behind_gap > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_GAP
            and rear_ttc > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_TTC
        )
        materially_better = (
            ahead_gap > front_gap + NeonHighwayEnv.PASSING_CLEARANCE_GAIN
            and (
                ahead_relative
                > front_relative + NeonHighwayEnv.PASSING_MIN_CLOSING_SPEED
                or ahead_gap
                > front_gap + 2.0 * NeonHighwayEnv.PASSING_CLEARANCE_GAIN
            )
        )
        if safe and materially_better:
            options.append(candidate)
    return tuple(options)


class LongitudinalIntentPolicy:
    """Wrap a lane policy with persistent intent and smooth target-speed control."""

    def __init__(self, lane_policy: Any, config: LongitudinalConfig | None = None) -> None:
        self.lane_policy = lane_policy
        self.config = config or LongitudinalConfig()
        self.reset()

    def reset(self) -> None:
        reset_policy = getattr(self.lane_policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        self.intent = DrivingIntent.CRUISE
        self.reason = "open-road cruise"
        self.desired_speed = self.config.cruise_speed
        self.raw_action = 0
        self.last_action = 0
        self.intervened = False
        self.lane_intervened = False
        self.lane_veto_reason = ""
        self._step = 0
        self._last_speed_command_step = -1_000_000
        self._pass_steps_remaining = 0
        self._pass_direction = 0
        self._pass_opportunity_steps = 0
        self._slow_leader_steps = 0
        self._comfort_braking_active = False
        self.braking_mode = "COAST"
        self.braking_reason = "open-road speed hold"
        self.speed_guidance = SpeedGuidance.BASE
        self.speed_guidance_target = self.config.cruise_speed
        self._speed_guidance_steps_remaining = 0

    def set_speed_guidance(
        self,
        guidance: SpeedGuidance | int,
        *,
        current_speed: float,
        duration_steps: int | None = None,
    ) -> None:
        """Latch a human speed preference for the smooth planner to execute."""
        selected = SpeedGuidance(int(guidance))
        if selected == SpeedGuidance.BASE:
            self.speed_guidance = selected
            self.speed_guidance_target = self.config.cruise_speed
            self._speed_guidance_steps_remaining = 0
            return
        delta = self.config.guidance_speed_delta
        if selected == SpeedGuidance.FASTER:
            target = min(self.config.passing_speed, current_speed + delta)
        else:
            target = max(NeonHighwayEnv.MIN_SPEED, current_speed - delta)
        self.speed_guidance = selected
        self.speed_guidance_target = float(target)
        self._speed_guidance_steps_remaining = (
            self.config.guidance_duration_steps
            if duration_steps is None
            else max(1, int(duration_steps))
        )

    @property
    def hud_data(self) -> dict[str, Any]:
        data = {
            "driving_intent": str(self.intent),
            "desired_speed": self.desired_speed,
            "speed_reason": self.reason,
            "raw_action": ACTION_NAMES[self.raw_action],
            "speed_intervened": self.intervened,
            "lane_intervened": self.lane_intervened,
            "lane_veto_reason": self.lane_veto_reason,
            "speed_guidance": self.speed_guidance.name,
            "speed_guidance_target": self.speed_guidance_target,
            "braking_mode": self.braking_mode,
            "braking_reason": self.braking_reason,
        }
        preference_decision = getattr(self.lane_policy, "last_decision", None)
        if preference_decision is not None:
            data["preference_decision"] = str(preference_decision)
        return data

    @staticmethod
    def _speed(observation: NDArray[np.floating[Any]], index: int) -> float:
        speed_span = NeonHighwayEnv.MAX_SPEED - NeonHighwayEnv.MIN_SPEED
        return NeonHighwayEnv.MIN_SPEED + float(observation[index]) * speed_span

    @staticmethod
    def _relevant_lanes(
        observation: NDArray[np.floating[Any]],
    ) -> tuple[int, ...]:
        lane_position = float(observation[2]) * (NeonHighwayEnv.LANES - 1)
        target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
        lanes = {
            lane
            for lane in range(NeonHighwayEnv.LANES)
            if abs(lane_position - lane) <= NeonHighwayEnv.LANE_COLLISION_WIDTH
        }
        lanes.add(target_lane)
        return tuple(sorted(lanes))

    def _following_speed(
        self,
        road: dict[str, float],
        speed: float,
        desired_gap: float,
    ) -> float:
        """A comfortable follow target that never dives far below its leader."""
        leader_speed = speed + road["front_relative"]
        gap_adjustment = np.clip(
            (road["front_gap"] - desired_gap) / 8.0,
            -self.config.maximum_comfort_speed_deficit,
            2.0,
        )
        return float(
            np.clip(
                leader_speed + gap_adjustment,
                NeonHighwayEnv.MIN_SPEED,
                self.config.cruise_speed,
            )
        )

    def _road_state(
        self, observation: NDArray[np.floating[Any]]
    ) -> dict[str, float]:
        target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
        readings = [
            observed_lane_reading(observation, lane)
            for lane in self._relevant_lanes(observation)
        ]
        target_reading = observed_lane_reading(observation, target_lane)
        return {
            "front_gap": target_reading[0],
            "front_relative": target_reading[1],
            "front_urgency": max(reading[2] for reading in readings),
            "rear_urgency": max(reading[5] for reading in readings),
        }

    def _select_intent(
        self,
        observation: NDArray[np.floating[Any]],
        raw_action: int,
        road: dict[str, float],
        speed: float,
    ) -> tuple[DrivingIntent, float, str]:
        steer, _ = decode_action(raw_action)
        target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
        candidate_lane = target_lane + (-1 if steer == STEER_LEFT else 1)
        passing_options = observed_passing_options(
            observation,
            include_speed_matched_slow_leaders=(
                self.config.recognize_speed_matched_slow_leaders
            ),
        )
        starts_pass = steer != STEER_KEEP and candidate_lane in passing_options
        if starts_pass:
            self._pass_steps_remaining = self.config.pass_commitment_steps
            self._pass_direction = -1 if steer == STEER_LEFT else 1
        elif self._pass_steps_remaining > 0:
            self._pass_steps_remaining -= 1

        if road["front_urgency"] >= self.config.critical_front_urgency:
            self._pass_steps_remaining = 0
            self._pass_direction = 0
            leader_speed = speed + road["front_relative"]
            desired = max(
                NeonHighwayEnv.MIN_SPEED,
                min(leader_speed, speed - 2.0),
            )
            return DrivingIntent.EMERGENCY, desired, "critical closing TTC"

        desired_gap = max(
            self.config.minimum_follow_gap,
            self.config.follow_time_headway * speed,
        )
        if self._comfort_braking_active:
            self._pass_steps_remaining = 0
            self._pass_direction = 0
            desired = self._following_speed(road, speed, desired_gap)
            return DrivingIntent.FOLLOW, desired, "pass paused for closing traffic"

        if self._pass_steps_remaining > 0:
            intent = (
                DrivingIntent.PASS_LEFT
                if self._pass_direction < 0
                else DrivingIntent.PASS_RIGHT
            )
            return intent, self.config.passing_speed, "committed safe pass"

        if self._slow_leader_steps > 0:
            desired = self._following_speed(road, speed, desired_gap)
            waited_seconds = self._slow_leader_steps * NeonHighwayEnv.DT
            return (
                DrivingIntent.SEARCH_PASS,
                desired,
                f"waiting {waited_seconds:.1f}s for safe pass gap",
            )

        following = (
            road["front_gap"] < desired_gap + self.config.follow_trigger_margin
            and (
                road["front_relative"] < -0.2
                or road["front_gap"] < desired_gap
            )
        )
        if following:
            desired = self._following_speed(road, speed, desired_gap)
            return DrivingIntent.FOLLOW, desired, "stable time headway"

        if steer != STEER_KEEP:
            return DrivingIntent.RETURN, self.config.cruise_speed, "lane positioning"
        return DrivingIntent.CRUISE, self.config.cruise_speed, "open-road cruise"

    def _protected_steer(
        self,
        observation: NDArray[np.floating[Any]],
        raw_steer: int,
    ) -> tuple[int, str]:
        """Veto merges that cannot remain safe throughout the lateral motion."""
        if raw_steer == STEER_KEEP:
            return raw_steer, ""
        if float(observation[4]) >= 0.05:
            return STEER_KEEP, "finishing committed lane change"

        direction = -1 if raw_steer == STEER_LEFT else 1
        if self._pass_steps_remaining > 0 and direction == -self._pass_direction:
            return STEER_KEEP, "blocked pass-direction reversal"

        target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
        candidate_lane = target_lane + direction
        if not 0 <= candidate_lane < NeonHighwayEnv.LANES:
            return STEER_KEEP, "road boundary"

        ahead_gap, ahead_relative, _, behind_gap, behind_relative, _ = (
            observed_lane_reading(observation, candidate_lane)
        )
        projection = self.config.merge_projection_seconds
        projected_ahead_gap = ahead_gap + min(ahead_relative, 0.0) * projection
        projected_behind_gap = behind_gap - max(behind_relative, 0.0) * projection
        rear_closing_speed = max(behind_relative, 0.0)
        rear_ttc = (
            behind_gap / rear_closing_speed
            if rear_closing_speed > 0.1
            else float("inf")
        )
        safe_candidate = (
            ahead_gap
            > NeonHighwayEnv.SAFE_LANE_CHANGE_FRONT_GAP
            + self.config.merge_front_buffer
            and projected_ahead_gap > NeonHighwayEnv.SAFE_LANE_CHANGE_FRONT_GAP
            and behind_gap
            > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_GAP
            + self.config.merge_rear_buffer
            and projected_behind_gap > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_GAP
            and rear_ttc > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_TTC
        )
        if not safe_candidate:
            return STEER_KEEP, "unsafe projected merge gap"

        origin_gap, origin_relative, _, _, _, _ = observed_lane_reading(
            observation, target_lane
        )
        projected_origin_gap = origin_gap + min(origin_relative, 0.0) * (
            self.config.merge_origin_exposure_seconds
        )
        if projected_origin_gap <= NeonHighwayEnv.CAR_LENGTH + 1.5:
            return STEER_KEEP, "too late to clear current lane"
        return raw_steer, ""

    def _planned_steer(
        self,
        observation: NDArray[np.floating[Any]],
        raw_steer: int,
    ) -> tuple[int, str]:
        """Respect the learned lane choice, then take persistent easy openings."""
        protected_steer, reason = self._protected_steer(observation, raw_steer)
        passing_options = observed_passing_options(
            observation,
            include_speed_matched_slow_leaders=(
                self.config.recognize_speed_matched_slow_leaders
            ),
        )
        lane_change_finished = float(observation[4]) < 0.05
        can_consider_pass = (
            lane_change_finished
            and self._pass_steps_remaining <= 0
            and bool(passing_options)
        )
        if can_consider_pass:
            self._pass_opportunity_steps += 1
        else:
            self._pass_opportunity_steps = 0

        if (
            protected_steer != STEER_KEEP
            or self._pass_opportunity_steps < self.config.pass_opportunity_dwell_steps
        ):
            return protected_steer, reason

        target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
        ranked_options = sorted(
            passing_options,
            key=lambda lane: observed_lane_reading(observation, lane)[0],
            reverse=True,
        )
        for candidate_lane in ranked_options:
            candidate_steer = (
                STEER_LEFT if candidate_lane < target_lane else STEER_RIGHT
            )
            planned_steer, veto_reason = self._protected_steer(
                observation, candidate_steer
            )
            if planned_steer != STEER_KEEP:
                self._pass_opportunity_steps = 0
                return planned_steer, "taking persistent safe pass"
            reason = veto_reason
        return protected_steer, reason

    def _speed_command(
        self,
        *,
        target_speed: float,
        speed: float,
        road: dict[str, float],
    ) -> int:
        emergency = self.intent == DrivingIntent.EMERGENCY
        slowing = target_speed > self.desired_speed + 0.25
        recovering = target_speed < self.desired_speed - self.config.speed_deadband
        if emergency:
            interval = self.config.emergency_command_interval_steps
        elif slowing and self._comfort_braking_active:
            urgency_span = max(
                self.config.critical_front_urgency
                - self.config.braking_release_urgency,
                1e-6,
            )
            severity = float(
                np.clip(
                    (road["front_urgency"] - self.config.braking_release_urgency)
                    / urgency_span,
                    0.0,
                    1.0,
                )
            )
            interval = int(
                round(
                    self.config.comfort_brake_max_interval_steps
                    - severity
                    * (
                        self.config.comfort_brake_max_interval_steps
                        - self.config.comfort_brake_min_interval_steps
                    )
                )
            )
        elif recovering and not self._comfort_braking_active:
            interval = self.config.recovery_command_interval_steps
        else:
            interval = self.config.command_interval_steps
        ready = self._step - self._last_speed_command_step >= interval
        if not ready:
            if slowing:
                self.braking_mode = "COAST"
                self.braking_reason = "metering target-speed reduction"
            elif recovering:
                self.braking_mode = "RECOVER"
                self.braking_reason = "smooth return to speed plan"
            else:
                self.braking_mode = "COAST"
                self.braking_reason = (
                    "matching slower traffic"
                    if self.intent
                    in {DrivingIntent.FOLLOW, DrivingIntent.SEARCH_PASS}
                    else "target speed settled"
                )
            return PEDAL_COAST

        if emergency and (
            target_speed > self.desired_speed
            or speed > self.desired_speed + self.config.speed_deadband
        ):
            self._last_speed_command_step = self._step
            self.braking_mode = "EMERGENCY"
            self.braking_reason = "critical closing TTC"
            return PEDAL_BRAKE

        if self._comfort_braking_active and slowing:
            self._last_speed_command_step = self._step
            self.braking_mode = "COMFORT BRAKE"
            self.braking_reason = "proportional following response"
            return PEDAL_BRAKE

        if recovering:
            self._last_speed_command_step = self._step
            self.braking_mode = "RECOVER"
            self.braking_reason = "smooth return to speed plan"
            return PEDAL_GAS

        if target_speed > self.desired_speed + self.config.speed_deadband:
            if (
                road["rear_urgency"] >= self.config.rear_pressure_urgency
                and road["front_urgency"] < self.config.braking_front_urgency
            ):
                self.braking_mode = "COAST"
                self.braking_reason = "rear pressure blocks extra braking"
                return PEDAL_COAST
            self._last_speed_command_step = self._step
            self.braking_mode = "COMFORT BRAKE"
            self.braking_reason = "settling stable following speed"
            return PEDAL_BRAKE
        self.braking_mode = "COAST"
        self.braking_reason = (
            "matching slower traffic"
            if self.intent in {DrivingIntent.FOLLOW, DrivingIntent.SEARCH_PASS}
            else "target speed settled"
        )
        return PEDAL_COAST

    def __call__(self, observation: NDArray[np.floating[Any]]) -> int:
        self._step += 1
        raw_action = int(self.lane_policy(observation))
        raw_steer, raw_pedal = decode_action(raw_action)
        speed = self._speed(observation, 0)
        target_lane = int(
            round(float(observation[3]) * (NeonHighwayEnv.LANES - 1))
        )
        front_gap, front_relative, _, _, _, _ = observed_lane_reading(
            observation, target_lane
        )
        slow_leader = (
            float(observation[4]) < 0.05
            and front_gap < NeonHighwayEnv.PASSING_TRIGGER_GAP
            and speed + front_relative
            < self.config.cruise_speed
            - NeonHighwayEnv.PASSING_SLOW_LEADER_MARGIN
        )
        self._slow_leader_steps = (
            self._slow_leader_steps + 1 if slow_leader else 0
        )
        protected_steer, lane_veto_reason = self._planned_steer(
            observation, raw_steer
        )
        protected_action = encode_action(protected_steer, raw_pedal)
        target_speed = self._speed(observation, 1)
        road = self._road_state(observation)
        if road["front_urgency"] >= self.config.critical_front_urgency:
            self._comfort_braking_active = False
        elif road["front_urgency"] >= self.config.braking_front_urgency:
            self._comfort_braking_active = True
        elif road["front_urgency"] <= self.config.braking_release_urgency:
            self._comfort_braking_active = False
        intent, raw_desired_speed, reason = self._select_intent(
            observation, protected_action, road, speed
        )
        active_guidance = (
            self.speed_guidance
            if self._speed_guidance_steps_remaining > 0
            else SpeedGuidance.BASE
        )
        if (
            active_guidance == SpeedGuidance.FASTER
            and intent
            in {
                DrivingIntent.CRUISE,
                DrivingIntent.PASS_LEFT,
                DrivingIntent.PASS_RIGHT,
                DrivingIntent.RETURN,
            }
            and road["front_urgency"] < self.config.braking_front_urgency
        ):
            raw_desired_speed = max(raw_desired_speed, self.speed_guidance_target)
            reason = "human-taught faster progress"
        elif active_guidance == SpeedGuidance.SLOWER:
            raw_desired_speed = min(raw_desired_speed, self.speed_guidance_target)
            reason = "human-taught slower progress"
        if self._speed_guidance_steps_remaining > 0:
            self._speed_guidance_steps_remaining -= 1
            if self._speed_guidance_steps_remaining == 0:
                self.speed_guidance = SpeedGuidance.BASE
        self.intent = intent
        self.reason = reason
        # Emergency targets take effect immediately. Ordinary following moves
        # toward its target proportionally, avoiding a one-frame plunge from
        # cruise speed to a slow leader's speed.
        if intent == DrivingIntent.EMERGENCY:
            self.desired_speed = raw_desired_speed
        else:
            smoothing = (
                self.config.comfort_deceleration_smoothing
                + (
                    1.0 - self.config.comfort_deceleration_smoothing
                )
                * float(
                    np.clip(
                        (
                            road["front_urgency"]
                            - self.config.braking_release_urgency
                        )
                        / max(
                            self.config.critical_front_urgency
                            - self.config.braking_release_urgency,
                            1e-6,
                        ),
                        0.0,
                        1.0,
                    )
                )
                if raw_desired_speed < self.desired_speed
                else self.config.desired_speed_smoothing
            )
            self.desired_speed += smoothing * (raw_desired_speed - self.desired_speed)
        self.desired_speed = float(
            np.clip(
                self.desired_speed,
                NeonHighwayEnv.MIN_SPEED,
                NeonHighwayEnv.MAX_SPEED,
            )
        )
        pedal = self._speed_command(
            target_speed=target_speed,
            speed=speed,
            road=road,
        )
        action = encode_action(protected_steer, pedal)
        self.raw_action = raw_action
        self.last_action = action
        self.lane_intervened = protected_steer != raw_steer
        self.lane_veto_reason = lane_veto_reason
        self.intervened = pedal != raw_pedal or self.lane_intervened
        return action


__all__ = [
    "DrivingIntent",
    "LongitudinalConfig",
    "LongitudinalIntentPolicy",
    "SpeedGuidance",
    "observed_lane_reading",
    "observed_passing_options",
]
