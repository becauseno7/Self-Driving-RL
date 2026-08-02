"""Pyodide runtime for the exact v1.0 simulator and deterministic controller."""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import numpy as np


def _install_gymnasium_stub() -> None:
    """Provide only the Gymnasium surface used by NeonHighwayEnv."""

    class Env:
        @classmethod
        def __class_getitem__(cls, _item: Any) -> type[Env]:
            return cls

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> None:
            del options
            self.np_random = np.random.default_rng(seed)

        def close(self) -> None:
            return None

    class Discrete:
        def __init__(self, n: int) -> None:
            self.n = int(n)

        def contains(self, value: Any) -> bool:
            try:
                number = int(value)
            except (TypeError, ValueError):
                return False
            return 0 <= number < self.n

    class Box:
        def __init__(self, *, low: Any, high: Any, dtype: Any) -> None:
            self.low = low
            self.high = high
            self.dtype = dtype
            self.shape = np.asarray(low).shape

    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Env = Env
    gymnasium.spaces = types.SimpleNamespace(Discrete=Discrete, Box=Box)
    sys.modules["gymnasium"] = gymnasium


def _load_project_modules() -> None:
    game_source = str(globals()["game_env_source"])
    longitudinal_source = str(globals()["longitudinal_source"])
    package = types.ModuleType("self_driving_rl")
    package.__path__ = []
    sys.modules["self_driving_rl"] = package

    game_module = types.ModuleType("self_driving_rl.game_env")
    game_module.__package__ = "self_driving_rl"
    sys.modules[game_module.__name__] = game_module
    exec(compile(game_source, "game_env.py", "exec"), game_module.__dict__)

    longitudinal_module = types.ModuleType("self_driving_rl.longitudinal")
    longitudinal_module.__package__ = "self_driving_rl"
    sys.modules[longitudinal_module.__name__] = longitudinal_module
    exec(
        compile(longitudinal_source, "longitudinal.py", "exec"),
        longitudinal_module.__dict__,
    )


_install_gymnasium_stub()
_load_project_modules()

from self_driving_rl.game_env import (  # noqa: E402
    ACTION_NAMES,
    STEER_KEEP,
    STEER_LEFT,
    NeonHighwayEnv,
    decode_action,
)
from self_driving_rl.longitudinal import LongitudinalIntentPolicy  # noqa: E402


def _round(value: Any, digits: int = 3) -> float:
    return round(float(value), digits)


def _finite_ttc(value: Any) -> float:
    return min(float(value), 999.0)


class BrowserPreferencePolicy:
    """Browser equivalent of the frozen V6 confidence gate and safety shield."""

    CALM_THRESHOLD = 0.8
    PASSING_THRESHOLD = 0.5

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_decision = "V5 BASE"
        self.pending_action = 4
        self._previous_target_lane: int | None = None
        self._previous_route_remaining: float | None = None
        self._steps_since_lane_change = 1_000_000
        self._last_lane_change_direction = 0
        self._prepared_context = np.asarray([1.0, 0.0], dtype=np.float32)

    def __call__(self, _observation: Any) -> int:
        return int(self.pending_action)

    def prepare(self, observation: np.ndarray[Any, np.dtype[np.float32]]) -> list[float]:
        target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
        route_remaining = float(observation[6])
        reset = (
            self._previous_route_remaining is not None
            and route_remaining > self._previous_route_remaining + 0.25
        )
        if self._previous_target_lane is None or reset:
            self._steps_since_lane_change = 1_000_000
            self._last_lane_change_direction = 0
        elif target_lane != self._previous_target_lane:
            self._steps_since_lane_change = 1
            self._last_lane_change_direction = int(
                np.sign(target_lane - self._previous_target_lane)
            )
        else:
            self._steps_since_lane_change += 1
        self._previous_target_lane = target_lane
        self._previous_route_remaining = route_remaining
        self._prepared_context = np.asarray(
            [
                min(
                    self._steps_since_lane_change
                    / NeonHighwayEnv.RAPID_LANE_CHANGE_STEPS,
                    1.0,
                ),
                float(self._last_lane_change_direction),
            ],
            dtype=np.float32,
        )
        return self._prepared_context.tolist()

    @staticmethod
    def _passing_options(
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[int, ...]:
        if float(observation[4]) >= 0.05:
            return ()
        target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
        speed_span = NeonHighwayEnv.MAX_SPEED - NeonHighwayEnv.MIN_SPEED

        def lane_reading(lane: int) -> tuple[float, float, float, float]:
            offset = 9 + 6 * lane
            return (
                float(observation[offset]) * NeonHighwayEnv.SENSOR_DISTANCE,
                float(observation[offset + 1]) * speed_span,
                float(observation[offset + 3]) * NeonHighwayEnv.SENSOR_DISTANCE,
                float(observation[offset + 4]) * speed_span,
            )

        front_gap, front_relative, _, _ = lane_reading(target_lane)
        if (
            front_gap >= NeonHighwayEnv.PASSING_TRIGGER_GAP
            or front_relative >= -NeonHighwayEnv.PASSING_MIN_CLOSING_SPEED
        ):
            return ()
        options: list[int] = []
        for candidate in (target_lane - 1, target_lane + 1):
            if not 0 <= candidate < NeonHighwayEnv.LANES:
                continue
            ahead_gap, ahead_relative, behind_gap, behind_relative = lane_reading(
                candidate
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

    @staticmethod
    def _current_threat_level(
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> float:
        current_lane = int(round(float(observation[2]) * (NeonHighwayEnv.LANES - 1)))
        offset = 9 + 6 * current_lane
        return max(float(observation[offset + 2]), float(observation[offset + 5]))

    def _shield(
        self,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        base_action: int,
        proposed_action: int,
        kind: int,
    ) -> tuple[int, int]:
        base_steer, base_pedal = decode_action(base_action)
        proposed_steer, proposed_pedal = decode_action(proposed_action)
        same_pedal = proposed_pedal == base_pedal
        context = self._prepared_context
        if kind == 1:
            base_direction = -1 if base_steer == STEER_LEFT else 1
            valid = (
                base_steer != STEER_KEEP
                and proposed_steer == STEER_KEEP
                and same_pedal
                and float(context[0]) < 1.0
                and base_direction == -int(context[1])
                and float(observation[4]) < 0.05
                and self._current_threat_level(observation) < 0.6
            )
        elif kind == 2:
            target_lane = int(
                round(float(observation[3]) * (NeonHighwayEnv.LANES - 1))
            )
            proposed_lane = target_lane + (-1 if proposed_steer == STEER_LEFT else 1)
            valid = (
                proposed_steer != STEER_KEEP
                and same_pedal
                and proposed_lane in self._passing_options(observation)
            )
        else:
            valid = False
        return (proposed_action, kind) if valid else (base_action, 0)

    def choose(
        self,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        base_action: int,
        probabilities: list[float],
        proposed_action: int,
    ) -> int:
        calm_strength = probabilities[1] / self.CALM_THRESHOLD
        passing_strength = probabilities[2] / self.PASSING_THRESHOLD
        if probabilities[1] >= self.CALM_THRESHOLD and calm_strength >= passing_strength:
            kind = 1
        elif (
            probabilities[2] >= self.PASSING_THRESHOLD
            and passing_strength > calm_strength
        ):
            kind = 2
        else:
            kind = 0
        selected_action = proposed_action if kind else base_action
        if selected_action == base_action:
            kind = 0
        selected_action, kind = self._shield(
            observation, base_action, selected_action, kind
        )
        self.pending_action = int(selected_action)
        self.last_decision = {0: "V5 BASE", 1: "CALM", 2: "PASS"}[kind]
        return self.pending_action


class BrowserDriverRuntime:
    def __init__(self) -> None:
        self.preference = BrowserPreferencePolicy()
        self.controller = LongitudinalIntentPolicy(self.preference)
        self.env: NeonHighwayEnv | None = None
        self.observation: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self.info: dict[str, Any] = {}
        self.episode = 0
        self.seed = 0
        self._car_ids: dict[int, int] = {}
        self._next_car_id = 0

    def reset(self, seed: int, difficulty: str, dynamic_traffic: bool) -> str:
        self.seed = int(seed)
        if self.env is not None:
            self.env.close()
        self.env = NeonHighwayEnv(
            difficulty_mode=difficulty,
            dynamic_traffic=bool(dynamic_traffic),
            endless=True,
        )
        self.observation, self.info = self.env.reset(seed=self.seed)
        self.controller.reset()
        self._car_ids.clear()
        self._next_car_id = 0
        self.episode += 1
        return self._state_json(inference_ms=0.0, done=False)

    def prepare_json(self) -> str:
        assert self.observation is not None
        context = self.preference.prepare(self.observation)
        return json.dumps(
            {
                "observation": self.observation.tolist(),
                "override_observation": self.observation.tolist() + context,
            },
            separators=(",", ":"),
        )

    def step(
        self,
        base_action: int,
        probabilities: list[float],
        proposed_action: int,
        inference_ms: float,
    ) -> str:
        assert self.env is not None and self.observation is not None
        self.preference.choose(
            self.observation,
            int(base_action),
            [float(value) for value in probabilities],
            int(proposed_action),
        )
        action = int(self.controller(self.observation))
        self.observation, _, terminated, truncated, self.info = self.env.step(action)
        return self._state_json(
            inference_ms=float(inference_ms), done=bool(terminated or truncated)
        )

    def _car_id(self, car: Any) -> int:
        identity = id(car)
        if identity not in self._car_ids:
            self._car_ids[identity] = self._next_car_id
            self._next_car_id += 1
        return self._car_ids[identity]

    def _state_json(self, *, inference_ms: float, done: bool) -> str:
        assert self.env is not None
        env = self.env
        hud = self.controller.hud_data
        sensors = [
            [_round(ahead), _round(ahead_rel), _round(behind), _round(behind_rel)]
            for ahead, ahead_rel, behind, behind_rel in env.lane_sensors()
        ]
        cars = [
            [
                self._car_id(car),
                _round(car.position - env.ego_position),
                _round(env.traffic_lateral_position(car)),
                _round(car.speed),
                int(car.braking),
                int(car.color_index),
                int(car.style),
            ]
            for car in env.traffic
        ]
        cars.sort(key=lambda item: item[0])
        info = self.info
        payload = {
            "frame": {
                "e": [
                    _round(env.ego_position),
                    _round(env.lane_position),
                    int(env.target_lane),
                    _round(env.ego_speed),
                    _round(env.target_speed),
                    _round(env.longitudinal_acceleration),
                    _round(env.throttle),
                    _round(env.brake),
                ],
                "c": cars,
                "s": sensors,
                "t": {
                    "action_index": int(env.last_action),
                    "action": ACTION_NAMES[int(env.last_action)],
                    "episode_return": _round(env.episode_return),
                    "intent": str(hud.get("driving_intent", "CRUISE")),
                    "desired_speed": _round(
                        hud.get("desired_speed", env.target_speed)
                    ),
                    "reason": str(hud.get("speed_reason", "open-road cruise")),
                    "braking_mode": str(hud.get("braking_mode", "COAST")),
                    "preference_decision": str(
                        hud.get("preference_decision", "V5 BASE")
                    ),
                    "challenge": str(info.get("challenge", env.challenge_name)),
                    "overtakes": int(info.get("overtakes", env.overtakes)),
                    "passed_by_traffic": int(
                        info.get("passed_by_traffic", env.passed_by_traffic)
                    ),
                    "lane_changes": int(
                        info.get("lane_changes", env.lane_changes)
                    ),
                    "near_misses": int(info.get("near_misses", env.near_misses)),
                    "traffic_lane_changes": int(
                        info.get("traffic_lane_changes", env.traffic_lane_changes)
                    ),
                    "ttc": _round(_finite_ttc(info.get("ttc", 999.0))),
                    "rear_ttc": _round(_finite_ttc(info.get("rear_ttc", 999.0))),
                    "elapsed_seconds": _round(env.elapsed_seconds, 1),
                    "inference_ms": _round(inference_ms),
                },
            },
            "seed": self.seed,
            "episode": self.episode,
            "done": done,
            "outcome": (
                "crashed"
                if info.get("crashed")
                else "timed_out"
                if done
                else None
            ),
            "collision": info.get("collision"),
        }
        return json.dumps(payload, separators=(",", ":"), allow_nan=False)


runtime = BrowserDriverRuntime()
