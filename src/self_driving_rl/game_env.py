"""A fair, learnable top-down highway environment with rich driving telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4

ACTION_NAMES = {
    LANE_LEFT: "LANE LEFT",
    IDLE: "HOLD",
    LANE_RIGHT: "LANE RIGHT",
    FASTER: "FASTER",
    SLOWER: "SLOWER",
}


@dataclass
class TrafficCar:
    lane: int
    position: float
    previous_position: float
    speed: float
    desired_speed: float
    color_index: int
    style: int
    braking: bool = False


@dataclass(frozen=True)
class CollisionEvent:
    kind: str
    severity: str
    impact_speed: float
    relative_position: float
    traffic_speed: float
    lane: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NeonHighwayEnv(gym.Env[NDArray[np.float32], int]):
    """Endless highway task designed for readable mistakes and useful learning."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    VERSION = "NeonHighwayEnv-v1"
    LANES = 4
    DT = 0.1
    MIN_SPEED = 8.0
    MAX_SPEED = 34.0
    SENSOR_DISTANCE = 90.0
    EPISODE_SECONDS = 45.0
    CAR_LENGTH = 4.6
    LANE_COLLISION_WIDTH = 0.42

    def __init__(
        self,
        render_mode: str | None = None,
        render_fps: int = 60,
        traffic_per_lane: int = 4,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if traffic_per_lane < 2:
            raise ValueError("traffic_per_lane must be at least 2")

        self.render_mode = render_mode
        self.render_fps = render_fps
        self.traffic_per_lane = traffic_per_lane
        self.action_space = gym.spaces.Discrete(5)

        # speed, lane position, target lane, then for each lane:
        # distance ahead, relative speed ahead, distance behind.
        low = np.array([0.0, 0.0, 0.0] + [0.0, -1.0, 0.0] * self.LANES, dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0] + [1.0, 1.0, 1.0] * self.LANES, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.ego_position = 0.0
        self.previous_ego_position = 0.0
        self.ego_speed = 22.0
        self.lane_position = 1.0
        self.target_lane = 1
        self.traffic: list[TrafficCar] = []
        self.step_count = 0
        self.episode_index = 0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_reward_components = self._empty_reward_components()
        self.last_action = IDLE
        self.crashed = False
        self.completed = False
        self.last_collision: CollisionEvent | None = None
        self.near_misses = 0
        self.safe_lane_changes = 0
        self.invalid_actions = 0
        self._near_miss_active = False
        self.quit_requested = False
        self.renderer: Any | None = None

        # The training callback updates these without coupling rendering to SB3.
        self.hud_data: dict[str, Any] = {
            "mode": "READY",
            "epsilon": 1.0,
            "training_step": 0,
            "training_total": 0,
            "mean_return": 0.0,
            "best_return": 0.0,
            "collisions": 0,
            "completions": 0,
            "recent_returns": [],
            "q_values": [0.0] * 5,
            "collision_types": {},
        }

    @property
    def max_episode_steps(self) -> int:
        return int(self.EPISODE_SECONDS / self.DT)

    @property
    def difficulty(self) -> float:
        """Traffic pressure rises gently during an episode."""
        return float(np.clip(self.step_count / self.max_episode_steps, 0.0, 1.0))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self.ego_position = 0.0
        self.previous_ego_position = 0.0
        self.ego_speed = 22.0
        self.target_lane = int(self.np_random.integers(1, 3))
        self.lane_position = float(self.target_lane)
        self.step_count = 0
        self.episode_index += 1
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_reward_components = self._empty_reward_components()
        self.last_action = IDLE
        self.crashed = False
        self.completed = False
        self.last_collision = None
        self.near_misses = 0
        self.safe_lane_changes = 0
        self.invalid_actions = 0
        self._near_miss_active = False
        self._spawn_traffic()

        observation = self._observation()
        info = self._info()
        if self.render_mode == "human" and not self.quit_requested:
            self.render()
        return observation, info

    def _spawn_traffic(self) -> None:
        self.traffic = []
        for lane in range(self.LANES):
            position = float(self.np_random.uniform(38.0, 56.0))
            for _ in range(self.traffic_per_lane - 1):
                self.traffic.append(self._new_car(lane, position))
                position += float(self.np_random.uniform(42.0, 64.0))

            behind = -float(self.np_random.uniform(28.0, 62.0))
            self.traffic.append(self._new_car(lane, behind))

    def _new_car(self, lane: int, position: float) -> TrafficCar:
        desired_speed = float(self.np_random.uniform(16.0, 29.0))
        return TrafficCar(
            lane=lane,
            position=position,
            previous_position=position,
            speed=desired_speed,
            desired_speed=desired_speed,
            color_index=int(self.np_random.integers(0, 8)),
            style=int(self.np_random.integers(0, 3)),
        )

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.quit_requested:
            return self._observation(), 0.0, False, True, {**self._info(), "user_quit": True}

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.last_action = action
        action_result = self._apply_action(action)
        self._update_motion()
        self._recycle_traffic()
        self.last_collision = self._detect_collision()
        self.crashed = self.last_collision is not None
        self.step_count += 1
        self.completed = self.step_count >= self.max_episode_steps and not self.crashed
        self._update_near_misses()

        reward = self._reward(action_result)
        self.last_reward = reward
        self.episode_return += reward

        terminated = self.crashed
        truncated = self.completed
        info = self._info()
        observation = self._observation()

        if self.render_mode == "human":
            self.render()
            if self.quit_requested:
                truncated = True
                info["user_quit"] = True

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: int) -> dict[str, bool]:
        invalid = False
        lane_change_started = False
        safe_lane_change = False
        lane_change_finished = abs(self.lane_position - self.target_lane) < 0.05

        if action in {LANE_LEFT, LANE_RIGHT} and lane_change_finished:
            direction = -1 if action == LANE_LEFT else 1
            candidate_lane = self.target_lane + direction
            if 0 <= candidate_lane < self.LANES:
                current_gap, current_relative, _ = self.lane_sensors()[self.target_lane]
                ahead_gap, _, behind_gap = self.lane_sensors()[candidate_lane]
                danger_here = current_gap < 22.0 and current_relative < 0.0
                safe_lane_change = ahead_gap > 18.0 and behind_gap > 13.0
                self.target_lane = candidate_lane
                lane_change_started = True
                if danger_here and safe_lane_change:
                    self.safe_lane_changes += 1
            else:
                invalid = True
                self.invalid_actions += 1
        elif action == FASTER:
            if self.ego_speed >= self.MAX_SPEED:
                invalid = True
                self.invalid_actions += 1
            self.ego_speed = min(self.MAX_SPEED, self.ego_speed + 1.5)
        elif action == SLOWER:
            if self.ego_speed <= self.MIN_SPEED:
                invalid = True
                self.invalid_actions += 1
            self.ego_speed = max(self.MIN_SPEED, self.ego_speed - 1.5)

        return {
            "invalid": invalid,
            "lane_change_started": lane_change_started,
            "safe_lane_change": safe_lane_change,
        }

    def _update_motion(self) -> None:
        self.previous_ego_position = self.ego_position
        lane_delta = np.clip(self.target_lane - self.lane_position, -0.24, 0.24)
        self.lane_position += float(lane_delta)
        if abs(self.lane_position - self.target_lane) < 0.02:
            self.lane_position = float(self.target_lane)
        self.ego_position += self.ego_speed * self.DT

        for car in self.traffic:
            car.previous_position = car.position
            target_speed = self._traffic_target_speed(car)
            speed_delta = float(np.clip(target_speed - car.speed, -5.5 * self.DT, 2.0 * self.DT))
            car.braking = speed_delta < -0.08
            car.speed = float(np.clip(car.speed + speed_delta, self.MIN_SPEED, self.MAX_SPEED))
            car.position += car.speed * self.DT

    def _traffic_target_speed(self, car: TrafficCar) -> float:
        leaders: list[tuple[float, float]] = [
            (other.position, other.speed)
            for other in self.traffic
            if other is not car and other.lane == car.lane and other.position > car.position
        ]
        if abs(float(car.lane) - self.lane_position) < self.LANE_COLLISION_WIDTH:
            if self.ego_position > car.position:
                leaders.append((self.ego_position, self.ego_speed))

        if not leaders:
            return car.desired_speed

        lead_position, lead_speed = min(leaders, key=lambda leader: leader[0])
        clear_gap = lead_position - car.position - self.CAR_LENGTH
        safe_gap = 7.0 + car.speed * 0.7
        if clear_gap >= safe_gap * 1.6:
            return car.desired_speed

        gap_ratio = float(np.clip(clear_gap / safe_gap, 0.0, 1.0))
        cautious_speed = lead_speed - (1.0 - gap_ratio) * 2.5
        return min(car.desired_speed, max(self.MIN_SPEED, cautious_speed))

    def _recycle_traffic(self) -> None:
        pressure = self.difficulty
        gap_low = 48.0 - 10.0 * pressure
        gap_high = 70.0 - 12.0 * pressure

        for lane in range(self.LANES):
            lane_cars = [car for car in self.traffic if car.lane == lane]
            for car in lane_cars:
                relative = car.position - self.ego_position
                if relative < -55.0:
                    farthest = max(other.position for other in lane_cars)
                    car.position = farthest + float(self.np_random.uniform(gap_low, gap_high))
                    self._reroll_car(car)
                elif relative > 190.0:
                    nearest = min(other.position for other in lane_cars)
                    car.position = nearest - float(self.np_random.uniform(42.0, 62.0))
                    self._reroll_car(car)

    def _reroll_car(self, car: TrafficCar) -> None:
        desired_speed = float(self.np_random.uniform(16.0, 29.0))
        car.previous_position = car.position
        car.speed = desired_speed
        car.desired_speed = desired_speed
        car.color_index = int(self.np_random.integers(0, 8))
        car.style = int(self.np_random.integers(0, 3))
        car.braking = False

    def _detect_collision(self) -> CollisionEvent | None:
        for car in self.traffic:
            previous_gap = car.previous_position - self.previous_ego_position
            current_gap = car.position - self.ego_position
            crossed_between_frames = previous_gap * current_gap <= 0.0
            longitudinal_overlap = abs(current_gap) < self.CAR_LENGTH or crossed_between_frames
            lateral_gap = abs(float(car.lane) - self.lane_position)

            if longitudinal_overlap and lateral_gap < self.LANE_COLLISION_WIDTH:
                changing_lanes = abs(self.lane_position - self.target_lane) > 0.05
                if changing_lanes or lateral_gap > 0.2:
                    kind = "SIDE IMPACT"
                elif current_gap >= 0.0:
                    kind = "FRONT IMPACT"
                else:
                    kind = "REAR IMPACT"

                lateral_impact = 4.0 if changing_lanes else 0.0
                impact_speed = abs(self.ego_speed - car.speed) + lateral_impact
                if impact_speed < 4.0:
                    severity = "LOW"
                elif impact_speed < 9.0:
                    severity = "MEDIUM"
                else:
                    severity = "HIGH"
                return CollisionEvent(
                    kind=kind,
                    severity=severity,
                    impact_speed=impact_speed,
                    relative_position=current_gap,
                    traffic_speed=car.speed,
                    lane=car.lane,
                )
        return None

    def current_threat(self) -> dict[str, float]:
        current_lane = int(np.clip(round(self.lane_position), 0, self.LANES - 1))
        ahead_gap, relative_speed, _ = self.lane_sensors()[current_lane]
        closing_speed = max(-relative_speed, 0.0)
        ttc = ahead_gap / closing_speed if closing_speed > 0.1 else float("inf")
        threat = float(np.clip(1.0 - ttc / 5.0, 0.0, 1.0)) if np.isfinite(ttc) else 0.0
        return {
            "gap": ahead_gap,
            "relative_speed": relative_speed,
            "ttc": ttc,
            "level": threat,
        }

    def _update_near_misses(self) -> None:
        threat = self.current_threat()
        near_miss_now = threat["ttc"] < 1.25 and not self.crashed
        if near_miss_now and not self._near_miss_active:
            self.near_misses += 1
        self._near_miss_active = near_miss_now

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        return {
            "progress": 0.0,
            "safety": 0.0,
            "comfort": 0.0,
            "rules": 0.0,
            "terminal": 0.0,
        }

    def _reward(self, action_result: dict[str, bool]) -> float:
        speed_fraction = (self.ego_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        components = self._empty_reward_components()
        components["progress"] = 0.025 + 0.075 * float(np.clip(speed_fraction, 0.0, 1.0))

        threat = self.current_threat()
        if threat["level"] > 0.0:
            components["safety"] -= 0.22 * threat["level"]
        if action_result["safe_lane_change"] and threat["level"] > 0.15:
            components["safety"] += 0.05
        if action_result["lane_change_started"]:
            components["comfort"] -= 0.012
        if action_result["invalid"]:
            components["rules"] -= 0.08
        if self.crashed:
            components["terminal"] = -10.0
        elif self.completed:
            components["terminal"] = 5.0

        self.last_reward_components = components
        return float(sum(components.values()))

    def lane_sensors(self) -> list[tuple[float, float, float]]:
        """Return (ahead gap, ahead relative speed, behind gap) for each lane."""
        readings: list[tuple[float, float, float]] = []
        for lane in range(self.LANES):
            ahead = [
                car
                for car in self.traffic
                if car.lane == lane and car.position >= self.ego_position
            ]
            behind = [
                car for car in self.traffic if car.lane == lane and car.position < self.ego_position
            ]

            if ahead:
                nearest_ahead = min(ahead, key=lambda car: car.position)
                ahead_gap = min(nearest_ahead.position - self.ego_position, self.SENSOR_DISTANCE)
                relative_speed = nearest_ahead.speed - self.ego_speed
            else:
                ahead_gap = self.SENSOR_DISTANCE
                relative_speed = 0.0

            behind_gap = (
                min(self.ego_position - car.position for car in behind)
                if behind
                else self.SENSOR_DISTANCE
            )
            readings.append((ahead_gap, relative_speed, min(behind_gap, self.SENSOR_DISTANCE)))
        return readings

    def _observation(self) -> NDArray[np.float32]:
        speed = (self.ego_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        lane = self.lane_position / (self.LANES - 1)
        target_lane = self.target_lane / (self.LANES - 1)
        values = [float(np.clip(speed, 0.0, 1.0)), lane, target_lane]

        for ahead_gap, relative_speed, behind_gap in self.lane_sensors():
            values.extend(
                [
                    ahead_gap / self.SENSOR_DISTANCE,
                    float(np.clip(relative_speed / self.MAX_SPEED, -1.0, 1.0)),
                    behind_gap / self.SENSOR_DISTANCE,
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        threat = self.current_threat()
        return {
            "speed": self.ego_speed,
            "crashed": self.crashed,
            "completed": self.completed,
            "lane": self.lane_position,
            "target_lane": self.target_lane,
            "action": ACTION_NAMES[self.last_action],
            "episode_return": self.episode_return,
            "episode_step": self.step_count,
            "episode_progress": self.step_count / self.max_episode_steps,
            "difficulty": self.difficulty,
            "near_misses": self.near_misses,
            "safe_lane_changes": self.safe_lane_changes,
            "invalid_actions": self.invalid_actions,
            "ttc": threat["ttc"],
            "threat_level": threat["level"],
            "closest_gap": threat["gap"],
            "reward_components": dict(self.last_reward_components),
            "collision": self.last_collision.to_dict() if self.last_collision else None,
        }

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode is None:
            return None
        if self.renderer is None:
            from self_driving_rl.game_renderer import NeonRenderer

            self.renderer = NeonRenderer(human=self.render_mode == "human", fps=self.render_fps)

        frame, keep_running = self.renderer.draw(self)
        if not keep_running:
            self.quit_requested = True
        return frame if self.render_mode == "rgb_array" else None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
