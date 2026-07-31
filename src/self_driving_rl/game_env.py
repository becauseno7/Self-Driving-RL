"""A fair, learnable top-down highway environment with rich driving telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

# Steering and speed are chosen together. Through V3 they shared one discrete
# slot, so an agent that wanted to brake could not also merge -- exactly the
# combination the hard waves are built to require. The action index factors as
# `steer * 3 + pedal`, which keeps a flat Discrete space that DQN can use.
STEER_LEFT, STEER_KEEP, STEER_RIGHT = 0, 1, 2
PEDAL_BRAKE, PEDAL_COAST, PEDAL_GAS = 0, 1, 2
ACTION_COUNT = 9


def encode_action(steer: int, pedal: int) -> int:
    return steer * 3 + pedal


def decode_action(action: int) -> tuple[int, int]:
    return divmod(int(action), 3)


# Names kept for the single-purpose actions so existing call sites and tests
# read the same way they did when these were the only five choices.
LANE_LEFT = encode_action(STEER_LEFT, PEDAL_COAST)
IDLE = encode_action(STEER_KEEP, PEDAL_COAST)
LANE_RIGHT = encode_action(STEER_RIGHT, PEDAL_COAST)
FASTER = encode_action(STEER_KEEP, PEDAL_GAS)
SLOWER = encode_action(STEER_KEEP, PEDAL_BRAKE)

_STEER_NAMES = {STEER_LEFT: "LEFT", STEER_KEEP: "", STEER_RIGHT: "RIGHT"}
_PEDAL_NAMES = {PEDAL_BRAKE: "BRAKE", PEDAL_COAST: "", PEDAL_GAS: "GAS"}
ACTION_NAMES = {
    encode_action(steer, pedal): (
        "+".join(part for part in (_STEER_NAMES[steer], _PEDAL_NAMES[pedal]) if part) or "HOLD"
    )
    for steer in (STEER_LEFT, STEER_KEEP, STEER_RIGHT)
    for pedal in (PEDAL_BRAKE, PEDAL_COAST, PEDAL_GAS)
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

    VERSION = "NeonHighwayEnv-v4"
    DIFFICULTY_MODES = {"standard", "hard"}
    LANES = 4
    DT = 0.1
    MIN_SPEED = 8.0
    MAX_SPEED = 34.0
    SENSOR_DISTANCE = 90.0
    EPISODE_SECONDS = 45.0
    # The stretch of road the player can actually see, mirroring the renderer's
    # camera. Nothing may be teleported inside this window: a car that changes
    # position or colour on screen reads as a bug, not as traffic.
    VISIBLE_BEHIND = 34.0
    VISIBLE_AHEAD = 102.0
    CAR_LENGTH = 4.6
    CAR_WIDTH = 1.9
    LANE_WIDTH = 3.7
    # Two cars touch when their centres are one car width apart. Expressed in
    # lane units this is the exact footprint the renderer draws, so anything
    # that looks like a collision is one. It also has to exceed 0.5, or the
    # midpoint of a lane change would sit further than this from every lane
    # centre and nothing could hit the ego there.
    LANE_COLLISION_WIDTH = CAR_WIDTH / LANE_WIDTH
    TRAFFIC_YIELD_WIDTH = 0.8
    TARGET_SPEED_STEP = 1.0
    MAX_ACCELERATION = 2.8
    MAX_BRAKING = 5.2
    SPEED_CONTROLLER_GAIN = 1.8
    TARGET_CHANGE_COMFORT_COST = 0.006
    UNSAFE_LANE_CHANGE_COST = 0.28
    SAFE_LANE_CHANGE_FRONT_GAP = 15.0
    SAFE_LANE_CHANGE_REAR_GAP = 11.0
    SAFE_LANE_CHANGE_REAR_TTC = 3.5
    # A wave is staged every CHALLENGE_INTERVAL_STEPS, so a longer route is a
    # longer endurance test rather than the same three waves with dead time
    # bolted on the end. The default 45 s route gives the familiar three.
    CHALLENGE_INTERVAL_STEPS = 150
    CHALLENGE_CLEAR_STEPS = 70
    STEP_LIMIT_MARGIN = 150
    # Endless mode still needs a ceiling so a flawless policy cannot hang a
    # training run. 100,000 steps is ~2.8 simulated hours.
    ENDLESS_STEP_LIMIT = 100_000

    # Terminal and event rewards. These are sized for SHAPING_GAMMA: at 0.995 a
    # bonus 450 steps away still retains ~10% of its value, so completing the
    # route is worth chasing. They are far too large for a 0.98 agent, which
    # cannot see past ~50 steps.
    CRASH_PENALTY = -15.0
    COMPLETION_BONUS = 15.0
    CHALLENGE_BONUS = 3.0
    TIMEOUT_PENALTY = -2.0

    # Time-to-collision shaping is potential-based (Ng et al., 1999) so it
    # speeds learning up without changing which policy is optimal. A raw per
    # step threat penalty instead pays the agent to crawl.
    SHAPING_GAMMA = 0.995
    SAFETY_POTENTIAL_WEIGHT = 2.0
    THREAT_HORIZON = 6.0

    # Intelligent Driver Model constants for traffic cars. Together with the
    # non-overlap constraint in _resolve_traffic_overlap they stop traffic from
    # driving through itself.
    IDM_MIN_GAP = 6.0
    IDM_TIME_HEADWAY = 1.2
    IDM_MAX_ACCELERATION = 2.4
    IDM_COMFORT_BRAKING = 4.5
    IDM_EMERGENCY_BRAKING = 9.0
    IDM_EXPONENT = 4.0

    def __init__(
        self,
        render_mode: str | None = None,
        render_fps: int = 60,
        render_speed: float = 1.0,
        traffic_per_lane: int = 4,
        difficulty_mode: str = "hard",
        episode_seconds: float | None = None,
        endless: bool = False,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if traffic_per_lane < 2:
            raise ValueError("traffic_per_lane must be at least 2")
        if difficulty_mode not in self.DIFFICULTY_MODES:
            raise ValueError(f"Unsupported difficulty mode: {difficulty_mode}")
        if episode_seconds is not None and episode_seconds < self.DT * 50:
            raise ValueError(f"episode_seconds must be at least {self.DT * 50}")

        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_speed = render_speed
        self.traffic_per_lane = traffic_per_lane
        self.difficulty_mode = difficulty_mode
        self.episode_seconds = (
            self.EPISODE_SECONDS if episode_seconds is None else float(episode_seconds)
        )
        # Endless mode drops the finish line entirely: one continuous drive
        # until a crash. Waves keep arriving on the same schedule, and the
        # episode clock reports "plenty of road left" for as long as it lasts.
        self.endless = bool(endless)
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)

        # Ego block, then six readings per lane. See _observation for the
        # layout; the ego block carries the episode clock and wave state
        # because the reward has terminal and wave-completion terms that are
        # otherwise invisible to the agent.
        ego_low = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
        ego_high = [1.0] * 9
        lane_low = [0.0, -1.0, 0.0, 0.0, -1.0, 0.0] * self.LANES
        lane_high = [1.0] * 6 * self.LANES
        low = np.array(ego_low + lane_low, dtype=np.float32)
        high = np.array(ego_high + lane_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.ego_position = 0.0
        self.previous_ego_position = 0.0
        self.ego_speed = 22.0
        self.target_speed = 22.0
        self.longitudinal_acceleration = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.lane_position = 1.0
        self.previous_lane_position = 1.0
        self.target_lane = 1
        self.traffic: list[TrafficCar] = []
        self._sensor_cache: list[tuple[float, float, float, float]] | None = None
        self.step_count = 0
        self.episode_index = 0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_reward_components = self._empty_reward_components()
        self.last_action = IDLE
        self.crashed = False
        self.completed = False
        self.timed_out = False
        self.last_collision: CollisionEvent | None = None
        self.near_misses = 0
        self.safe_lane_changes = 0
        self.invalid_actions = 0
        self.challenges_presented = 0
        self.challenges_resolved = 0
        self.challenge_active = False
        self.challenge_started_at = 0
        self.challenge_name = "OPEN ROAD"
        self._next_challenge_index = 0
        self._near_miss_active = False
        self._previous_potential = 0.0
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
            "q_values": [0.0] * ACTION_COUNT,
            "collision_types": {},
        }

    @property
    def max_episode_steps(self) -> int:
        return int(self.episode_seconds / self.DT)

    @property
    def challenge_steps(self) -> tuple[int, ...]:
        """Step at which each hard-mode wave stages, one per interval.

        The final wave must still have room to be cleared, so a route only
        gains a wave once there is time to survive it.
        """
        latest_useful = self.max_episode_steps - self.CHALLENGE_CLEAR_STEPS
        count = max(1, latest_useful // self.CHALLENGE_INTERVAL_STEPS + 1)
        return tuple(index * self.CHALLENGE_INTERVAL_STEPS for index in range(count))

    @property
    def absolute_step_limit(self) -> int:
        """Hard cap so an episode can never run forever if a wave never stages."""
        if self.endless:
            return self.ENDLESS_STEP_LIMIT
        return self.max_episode_steps + self.STEP_LIMIT_MARGIN

    @property
    def elapsed_seconds(self) -> float:
        """Simulated seconds survived so far this episode."""
        return self.step_count * self.DT

    def _invalidate_sensors(self) -> None:
        self._sensor_cache = None

    def _sensors(self) -> list[tuple[float, float, float, float]]:
        """Sensor readings memoized within one step.

        `lane_sensors` recomputed five times per step was the environment's
        throughput bottleneck. The cache is dropped before `step` and `reset`
        return, so anything mutating traffic from outside still sees the truth.
        """
        if self._sensor_cache is None:
            self._sensor_cache = self.lane_sensors()
        return self._sensor_cache

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
        self.target_speed = 22.0
        self.longitudinal_acceleration = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.target_lane = int(self.np_random.integers(1, 3))
        self.lane_position = float(self.target_lane)
        self.previous_lane_position = self.lane_position
        self.step_count = 0
        self.episode_index += 1
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_reward_components = self._empty_reward_components()
        self.last_action = IDLE
        self.crashed = False
        self.completed = False
        self.timed_out = False
        self.last_collision = None
        self.near_misses = 0
        self.safe_lane_changes = 0
        self.invalid_actions = 0
        self.challenges_presented = 0
        self.challenges_resolved = 0
        self.challenge_active = False
        self.challenge_started_at = 0
        self.challenge_name = "OPEN ROAD"
        self._next_challenge_index = 0
        self._near_miss_active = False
        self._spawn_traffic()
        self._maybe_stage_hard_challenge()
        self._invalidate_sensors()
        self._previous_potential = self._safety_potential()

        observation = self._observation()
        info = self._info()
        if self.render_mode == "human" and not self.quit_requested:
            self.render()
        self._invalidate_sensors()
        return observation, info

    def _spawn_traffic(self) -> None:
        self.traffic = []
        if self.difficulty_mode == "hard":
            first_low, first_high = 32.0, 47.0
            gap_low, gap_high = 36.0, 54.0
        else:
            first_low, first_high = 38.0, 56.0
            gap_low, gap_high = 42.0, 64.0
        for lane in range(self.LANES):
            position = float(self.np_random.uniform(first_low, first_high))
            for _ in range(self.traffic_per_lane - 1):
                self.traffic.append(self._new_car(lane, position))
                position += float(self.np_random.uniform(gap_low, gap_high))

            behind = -float(self.np_random.uniform(28.0, 62.0))
            self.traffic.append(self._new_car(lane, behind))

    def _new_car(self, lane: int, position: float) -> TrafficCar:
        speed_range = (13.0, 31.0) if self.difficulty_mode == "hard" else (16.0, 29.0)
        desired_speed = float(self.np_random.uniform(*speed_range))
        return TrafficCar(
            lane=lane,
            position=position,
            previous_position=position,
            speed=desired_speed,
            desired_speed=desired_speed,
            color_index=int(self.np_random.integers(0, 8)),
            style=int(self.np_random.integers(0, 3)),
        )

    def _set_car_state(self, car: TrafficCar, offset: float, speed: float) -> None:
        car.position = self.ego_position + offset
        car.previous_position = car.position
        car.speed = speed
        car.desired_speed = speed
        car.braking = False

    def _is_visible(self, car: TrafficCar) -> bool:
        relative = car.position - self.ego_position
        return -self.VISIBLE_BEHIND <= relative <= self.VISIBLE_AHEAD

    def _nearest_to(self, lane: int, offset: float, used: set[int]) -> TrafficCar:
        """The car in `lane` already closest to where a wave wants one."""
        target = self.ego_position + offset
        candidates = [car for car in self.traffic if car.lane == lane and id(car) not in used]
        return min(candidates, key=lambda car: abs(car.position - target))

    def _stage_car(self, car: TrafficCar, offset: float, speed: float) -> None:
        """Set up one wave vehicle without the player seeing anything jump.

        A car already on screen is persuaded rather than moved: changing what
        it does instead of where it is produces the same squeeze a few seconds
        later, which is how the situation forms on a real road. A car off
        screen may be repositioned, but never closer than the edge of the
        view, so it drives into frame instead of appearing inside it.

        The opening wave is exempt. Nothing has been drawn yet at step zero, so
        there is no continuity to break and the scenario can be set up exactly.
        """
        if self.step_count == 0:
            self._set_car_state(car, offset, speed)
            return
        if self._is_visible(car):
            car.desired_speed = speed
            car.braking = speed < car.speed
            return
        if offset >= 0.0:
            offset = max(offset, self.VISIBLE_AHEAD + 6.0)
        else:
            offset = min(offset, -(self.VISIBLE_BEHIND + 6.0))
        self._set_car_state(car, offset, speed)

    def _arrange_challenge_lane(
        self,
        lane: int,
        *,
        rear_offset: float,
        rear_speed: float,
        front_offset: float,
        front_speed: float,
    ) -> None:
        # Pick whichever cars are already nearest the wanted spots, so the
        # smallest possible change produces the wave.
        used: set[int] = set()
        rear = self._nearest_to(lane, rear_offset, used)
        used.add(id(rear))
        front = self._nearest_to(lane, front_offset, used)
        used.add(id(front))
        self._stage_car(rear, rear_offset, rear_speed)
        self._stage_car(front, front_offset, front_speed)

        remaining = [car for car in self.traffic if car.lane == lane and id(car) not in used]
        for index, car in enumerate(sorted(remaining, key=lambda car: car.position), start=1):
            cruise_speed = float(self.np_random.uniform(18.0, 29.0))
            self._stage_car(car, front_offset + 46.0 * index, cruise_speed)

    def _maybe_stage_hard_challenge(self) -> None:
        if self.difficulty_mode != "hard" or self.challenge_active:
            return
        if self.endless:
            # No route to finish, so waves simply keep arriving on schedule.
            next_wave_step = self._next_challenge_index * self.CHALLENGE_INTERVAL_STEPS
        else:
            challenge_steps = self.challenge_steps
            if self._next_challenge_index >= len(challenge_steps):
                return
            next_wave_step = challenge_steps[self._next_challenge_index]
        if self.step_count < next_wave_step:
            return
        if abs(self.lane_position - self.target_lane) >= 0.05:
            return

        current_lane = self.target_lane
        adjacent_lanes = [
            lane for lane in (current_lane - 1, current_lane + 1) if 0 <= lane < self.LANES
        ]
        escape_lane = int(self.np_random.choice(adjacent_lanes))
        trap_lanes = [lane for lane in adjacent_lanes if lane != escape_lane]

        self._arrange_challenge_lane(
            current_lane,
            rear_offset=-float(self.np_random.uniform(21.0, 26.0)),
            rear_speed=float(self.np_random.uniform(28.0, 32.0)),
            front_offset=float(self.np_random.uniform(29.0, 34.0)),
            front_speed=float(self.np_random.uniform(14.0, 17.5)),
        )
        self._arrange_challenge_lane(
            escape_lane,
            rear_offset=-float(self.np_random.uniform(43.0, 50.0)),
            rear_speed=float(self.np_random.uniform(18.0, 23.0)),
            front_offset=float(self.np_random.uniform(45.0, 53.0)),
            front_speed=float(self.np_random.uniform(23.0, 29.0)),
        )
        for trap_lane in trap_lanes:
            self._arrange_challenge_lane(
                trap_lane,
                rear_offset=-float(self.np_random.uniform(9.0, 13.0)),
                rear_speed=float(self.np_random.uniform(29.0, 32.0)),
                front_offset=float(self.np_random.uniform(15.0, 20.0)),
                front_speed=float(self.np_random.uniform(14.0, 18.0)),
            )

        self._resolve_traffic_overlap()
        self._invalidate_sensors()
        direction = "LEFT" if escape_lane < current_lane else "RIGHT"
        self.challenge_name = f"SQUEEZE / ESCAPE {direction}"
        self.challenge_active = True
        self.challenge_started_at = self.step_count
        self.challenges_presented += 1
        self._next_challenge_index += 1

    def _update_challenge_progress(self) -> bool:
        if not self.challenge_active or self.crashed:
            return False
        if self.step_count - self.challenge_started_at < self.CHALLENGE_CLEAR_STEPS:
            return False
        self.challenge_active = False
        self.challenges_resolved += 1
        self.challenge_name = "WAVE CLEARED"
        return True

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.quit_requested:
            # Quitting is not a real terminal state: flag it so SB3 bootstraps
            # the value of the final observation instead of treating it as 0.
            info = {**self._info(), "user_quit": True, "TimeLimit.truncated": True}
            return self._observation(), 0.0, False, True, info

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self._invalidate_sensors()
        self.last_action = action
        action_result = self._apply_action(action)
        self._update_motion()
        self._recycle_traffic()
        self.last_collision = self._detect_collision()
        self.crashed = self.last_collision is not None
        self.step_count += 1
        challenge_resolved = self._update_challenge_progress()
        action_result["challenge_resolved"] = challenge_resolved
        all_challenges_cleared = (
            self.difficulty_mode != "hard"
            or self.challenges_resolved == len(self.challenge_steps)
        )
        self.completed = (
            not self.endless
            and self.step_count >= self.max_episode_steps
            and not self.crashed
            and all_challenges_cleared
        )
        # A wave that never stages would otherwise leave `completed` False
        # forever, and nothing else ends the episode. Resolved before the
        # reward so the timeout penalty lands on the step that ends it.
        self.timed_out = not self.completed and self.step_count >= self.absolute_step_limit
        self._update_near_misses()

        reward = self._reward(action_result)
        self.last_reward = reward
        self.episode_return += reward

        if not self.crashed and not self.completed and not self.timed_out:
            self._maybe_stage_hard_challenge()

        terminated = self.crashed
        truncated = self.completed or self.timed_out

        self._invalidate_sensors()
        info = self._info()
        observation = self._observation()
        if self.timed_out:
            info["TimeLimit.truncated"] = True

        if self.render_mode == "human":
            self.render()
            if self.quit_requested:
                truncated = True
                info["user_quit"] = True
                info["TimeLimit.truncated"] = True

        self._invalidate_sensors()
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: int) -> dict[str, bool]:
        invalid = False
        lane_change_started = False
        safe_lane_change = False
        speed_target_changed = False
        steer, pedal = decode_action(action)
        lane_change_finished = abs(self.lane_position - self.target_lane) < 0.05

        if steer != STEER_KEEP and lane_change_finished:
            direction = -1 if steer == STEER_LEFT else 1
            candidate_lane = self.target_lane + direction
            if 0 <= candidate_lane < self.LANES:
                sensors = self._sensors()
                current_gap, current_relative, _, _ = sensors[self.target_lane]
                ahead_gap, _, behind_gap, behind_relative = sensors[candidate_lane]
                danger_here = current_gap < 22.0 and current_relative < 0.0
                rear_closing_speed = max(behind_relative, 0.0)
                rear_ttc = (
                    behind_gap / rear_closing_speed
                    if rear_closing_speed > 0.1
                    else float("inf")
                )
                safe_lane_change = (
                    ahead_gap > self.SAFE_LANE_CHANGE_FRONT_GAP
                    and behind_gap > self.SAFE_LANE_CHANGE_REAR_GAP
                    and rear_ttc > self.SAFE_LANE_CHANGE_REAR_TTC
                )
                self.target_lane = candidate_lane
                lane_change_started = True
                if danger_here and safe_lane_change:
                    self.safe_lane_changes += 1
            else:
                invalid = True
                self.invalid_actions += 1

        # The pedal is applied independently, so braking through a merge is a
        # single action rather than a choice between the two.
        if pedal == PEDAL_GAS:
            if self.target_speed >= self.MAX_SPEED:
                invalid = True
                self.invalid_actions += 1
            self.target_speed = min(
                self.MAX_SPEED,
                self.target_speed + self.TARGET_SPEED_STEP,
            )
            speed_target_changed = not invalid
        elif pedal == PEDAL_BRAKE:
            if self.target_speed <= self.MIN_SPEED:
                invalid = True
                self.invalid_actions += 1
            self.target_speed = max(
                self.MIN_SPEED,
                self.target_speed - self.TARGET_SPEED_STEP,
            )
            speed_target_changed = not invalid

        return {
            "invalid": invalid,
            "lane_change_requested": steer != STEER_KEEP,
            "lane_change_started": lane_change_started,
            "safe_lane_change": safe_lane_change,
            "unsafe_lane_change": lane_change_started and not safe_lane_change,
            "speed_target_changed": speed_target_changed,
        }

    def _update_motion(self) -> None:
        self.previous_ego_position = self.ego_position
        self.previous_lane_position = self.lane_position
        self._update_ego_speed()
        lane_delta = np.clip(self.target_lane - self.lane_position, -0.24, 0.24)
        self.lane_position += float(lane_delta)
        if abs(self.lane_position - self.target_lane) < 0.02:
            self.lane_position = float(self.target_lane)
        self.ego_position += self.ego_speed * self.DT

        # One sorted pass per lane resolves every leader, instead of scanning
        # the whole traffic list once per car.
        for lane in range(self.LANES):
            lane_cars = sorted(
                (car for car in self.traffic if car.lane == lane),
                key=lambda car: car.position,
                reverse=True,
            )
            ego_in_lane = abs(float(lane) - self.lane_position) < self.TRAFFIC_YIELD_WIDTH
            leader: tuple[float, float] | None = None
            for car in lane_cars:
                if ego_in_lane and self.ego_position > car.position:
                    ego = (self.ego_position, self.ego_speed)
                    leader = ego if leader is None or ego[0] < leader[0] else leader
                car.previous_position = car.position
                acceleration = self._traffic_acceleration(car, leader)
                car.braking = acceleration < -0.8
                car.speed = float(
                    np.clip(car.speed + acceleration * self.DT, self.MIN_SPEED, self.MAX_SPEED)
                )
                leader = (car.position, car.speed)
                car.position += car.speed * self.DT

        self._resolve_traffic_overlap()
        self._invalidate_sensors()

    def _update_ego_speed(self) -> None:
        speed_error = self.target_speed - self.ego_speed
        requested_acceleration = speed_error * self.SPEED_CONTROLLER_GAIN
        acceleration = float(
            np.clip(
                requested_acceleration,
                -self.MAX_BRAKING,
                self.MAX_ACCELERATION,
            )
        )
        if abs(speed_error) < 0.02:
            self.ego_speed = self.target_speed
            acceleration = 0.0

        self.longitudinal_acceleration = acceleration
        self.throttle = max(acceleration / self.MAX_ACCELERATION, 0.0)
        self.brake = max(-acceleration / self.MAX_BRAKING, 0.0)
        self.ego_speed = float(
            np.clip(
                self.ego_speed + acceleration * self.DT,
                self.MIN_SPEED,
                self.MAX_SPEED,
            )
        )

    def _leader_of(self, car: TrafficCar) -> tuple[float, float] | None:
        """Nearest obstacle ahead of `car` in its lane, including the ego."""
        leaders: list[tuple[float, float]] = [
            (other.position, other.speed)
            for other in self.traffic
            if other is not car and other.lane == car.lane and other.position > car.position
        ]
        # A merging ego is a hazard well before it reaches the lane centre, so
        # traffic starts yielding at TRAFFIC_YIELD_WIDTH rather than at the
        # narrower width that counts as a collision.
        if abs(float(car.lane) - self.lane_position) < self.TRAFFIC_YIELD_WIDTH:
            if self.ego_position > car.position:
                leaders.append((self.ego_position, self.ego_speed))

        if not leaders:
            return None
        return min(leaders, key=lambda leader: leader[0])

    def _traffic_acceleration(
        self,
        car: TrafficCar,
        leader: tuple[float, float] | None = None,
    ) -> float:
        """Intelligent Driver Model acceleration, in m/s^2.

        `leader` is (position, speed) of the nearest obstacle ahead; it is
        resolved from the traffic list when the caller does not supply it.
        """
        free_road = self.IDM_MAX_ACCELERATION * (
            1.0 - (car.speed / max(car.desired_speed, 1e-3)) ** self.IDM_EXPONENT
        )

        if leader is None:
            leader = self._leader_of(car)
        if leader is None:
            return float(
                np.clip(free_road, -self.IDM_EMERGENCY_BRAKING, self.IDM_MAX_ACCELERATION)
            )

        lead_position, lead_speed = leader
        clear_gap = max(lead_position - car.position - self.CAR_LENGTH, 1e-3)
        approach_rate = car.speed - lead_speed
        desired_gap = self.IDM_MIN_GAP + max(
            0.0,
            car.speed * self.IDM_TIME_HEADWAY
            + (car.speed * approach_rate)
            / (2.0 * np.sqrt(self.IDM_MAX_ACCELERATION * self.IDM_COMFORT_BRAKING)),
        )
        interaction = self.IDM_MAX_ACCELERATION * (desired_gap / clear_gap) ** 2
        return float(
            np.clip(
                free_road - interaction,
                -self.IDM_EMERGENCY_BRAKING,
                self.IDM_MAX_ACCELERATION,
            )
        )

    def _resolve_traffic_overlap(self) -> None:
        """Never let two traffic cars occupy the same space.

        The IDM alone cannot guarantee this after teleports or an extreme speed
        difference, so overlaps are projected out front-to-back. The ego is
        deliberately excluded: being rear-ended is a real failure mode.
        """
        for lane in range(self.LANES):
            lane_cars = sorted(
                (car for car in self.traffic if car.lane == lane),
                key=lambda car: car.position,
                reverse=True,
            )
            for leader, follower in zip(lane_cars, lane_cars[1:], strict=False):
                limit = leader.position - self.CAR_LENGTH
                if follower.position > limit:
                    follower.position = limit
                    follower.speed = min(follower.speed, leader.speed)

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
                    # Recycling swaps the car's colour and shape, so it has to
                    # land out of sight or the player watches one car become
                    # another.
                    car.position = max(
                        car.position, self.ego_position + self.VISIBLE_AHEAD + 6.0
                    )
                    self._reroll_car(car)
                elif relative > 190.0:
                    # Re-insert relative to the ego, not to the lane's rearmost
                    # car. Anchoring on the lane could place the car past the
                    # -55 m recycle line, making it teleport again next step.
                    nearest = min(other.position for other in lane_cars)
                    car.position = min(
                        nearest - float(self.np_random.uniform(42.0, 62.0)),
                        self.ego_position + 190.0,
                    )
                    car.position = min(
                        max(car.position, self.ego_position - 50.0),
                        self.ego_position - self.VISIBLE_BEHIND - 6.0,
                    )
                    self._reroll_car(car)
        self._resolve_traffic_overlap()
        self._invalidate_sensors()

    def _reroll_car(self, car: TrafficCar) -> None:
        speed_range = (13.0, 31.0) if self.difficulty_mode == "hard" else (16.0, 29.0)
        desired_speed = float(self.np_random.uniform(*speed_range))
        car.previous_position = car.position
        car.speed = desired_speed
        car.desired_speed = desired_speed
        car.color_index = int(self.np_random.integers(0, 8))
        car.style = int(self.np_random.integers(0, 3))
        car.braking = False

    def _swept_lateral_gap(self, lane: int) -> float:
        """Smallest lateral distance to `lane` swept during this step.

        The ego moves up to 0.24 lanes per step, so testing only the end-of-step
        position leaves a hole in the middle of every merge where no lane centre
        is within LANE_COLLISION_WIDTH.
        """
        previous = float(lane) - self.previous_lane_position
        current = float(lane) - self.lane_position
        if previous * current <= 0.0:
            return 0.0
        return min(abs(previous), abs(current))

    def _detect_collision(self) -> CollisionEvent | None:
        hits: list[tuple[float, TrafficCar]] = []
        for car in self.traffic:
            previous_gap = car.previous_position - self.previous_ego_position
            current_gap = car.position - self.ego_position
            crossed_between_frames = previous_gap * current_gap <= 0.0
            longitudinal_overlap = abs(current_gap) < self.CAR_LENGTH or crossed_between_frames
            if longitudinal_overlap and self._swept_lateral_gap(car.lane) < (
                self.LANE_COLLISION_WIDTH
            ):
                hits.append((abs(current_gap), car))

        if not hits:
            return None

        # Report the car actually struck, not whichever one happens to come
        # first in the traffic list.
        _, car = min(hits, key=lambda hit: hit[0])
        current_gap = car.position - self.ego_position
        lateral_gap = abs(float(car.lane) - self.lane_position)
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

    def current_threat(self) -> dict[str, float]:
        current_lane = int(np.clip(round(self.lane_position), 0, self.LANES - 1))
        ahead_gap, relative_speed, _, _ = self._sensors()[current_lane]
        closing_speed = max(-relative_speed, 0.0)
        ttc = ahead_gap / closing_speed if closing_speed > 0.1 else float("inf")
        threat = (
            float(np.clip(1.0 - ttc / self.THREAT_HORIZON, 0.0, 1.0))
            if np.isfinite(ttc)
            else 0.0
        )
        return {
            "gap": ahead_gap,
            "relative_speed": relative_speed,
            "ttc": ttc,
            "level": threat,
        }

    def rear_threat(self) -> dict[str, float]:
        current_lane = int(np.clip(round(self.lane_position), 0, self.LANES - 1))
        _, _, behind_gap, relative_speed = self._sensors()[current_lane]
        closing_speed = max(relative_speed, 0.0)
        ttc = behind_gap / closing_speed if closing_speed > 0.1 else float("inf")
        threat = (
            float(np.clip(1.0 - ttc / self.THREAT_HORIZON, 0.0, 1.0))
            if np.isfinite(ttc)
            else 0.0
        )
        return {
            "gap": behind_gap,
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
            "shaping": 0.0,
            "comfort": 0.0,
            "rules": 0.0,
            "terminal": 0.0,
        }

    def _safety_potential(self) -> float:
        """Higher is safer. Only differences of this function are ever paid."""
        front = self.current_threat()["level"]
        rear = self.rear_threat()["level"]
        return -self.SAFETY_POTENTIAL_WEIGHT * (front + rear)

    def _reward(self, action_result: dict[str, bool]) -> float:
        speed_fraction = (self.ego_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        components = self._empty_reward_components()
        components["progress"] = 0.025 + 0.075 * float(np.clip(speed_fraction, 0.0, 1.0))

        # Potential-based shaping: gamma * Phi(s') - Phi(s). On a terminal step
        # the successor has no future, so its potential is 0 by definition.
        episode_over = self.crashed or self.completed or self.timed_out
        potential = 0.0 if episode_over else self._safety_potential()
        components["shaping"] = self.SHAPING_GAMMA * potential - self._previous_potential
        self._previous_potential = potential

        threat_level = self.current_threat()["level"]
        if action_result["safe_lane_change"] and threat_level > 0.15:
            components["safety"] += 0.05
        if action_result["unsafe_lane_change"]:
            components["safety"] -= self.UNSAFE_LANE_CHANGE_COST
        if action_result.get("challenge_resolved", False):
            components["safety"] += self.CHALLENGE_BONUS
        if action_result["lane_change_requested"]:
            components["comfort"] -= 0.012
        if action_result["speed_target_changed"]:
            components["comfort"] -= self.TARGET_CHANGE_COMFORT_COST
        components["comfort"] -= 0.004 * (abs(self.longitudinal_acceleration) / self.MAX_BRAKING)
        if action_result["invalid"]:
            components["rules"] -= 0.08
        if self.crashed:
            components["terminal"] = self.CRASH_PENALTY
        elif self.completed:
            components["terminal"] = self.COMPLETION_BONUS
        elif self.timed_out:
            components["terminal"] = self.TIMEOUT_PENALTY

        self.last_reward_components = components
        return float(sum(components.values()))

    def lane_sensors(self) -> list[tuple[float, float, float, float]]:
        """Return front/rear gaps and relative speeds for every lane."""
        readings: list[tuple[float, float, float, float]] = []
        for lane in range(self.LANES):
            ahead = [
                car
                for car in self.traffic
                if car.lane == lane and car.position >= self.ego_position
            ]
            behind = [
                car for car in self.traffic if car.lane == lane and car.position < self.ego_position
            ]

            # Gaps are bumper-to-bumper: a centre-to-centre distance below one
            # car length is already a collision, so reporting it as free space
            # made every distance a full car length too optimistic.
            if ahead:
                nearest_ahead = min(ahead, key=lambda car: car.position)
                ahead_gap = float(
                    np.clip(
                        nearest_ahead.position - self.ego_position - self.CAR_LENGTH,
                        0.0,
                        self.SENSOR_DISTANCE,
                    )
                )
                relative_speed = nearest_ahead.speed - self.ego_speed
            else:
                ahead_gap = self.SENSOR_DISTANCE
                relative_speed = 0.0

            if behind:
                nearest_behind = max(behind, key=lambda car: car.position)
                behind_gap = float(
                    np.clip(
                        self.ego_position - nearest_behind.position - self.CAR_LENGTH,
                        0.0,
                        self.SENSOR_DISTANCE,
                    )
                )
                behind_relative_speed = nearest_behind.speed - self.ego_speed
            else:
                behind_gap = self.SENSOR_DISTANCE
                behind_relative_speed = 0.0
            readings.append(
                (ahead_gap, relative_speed, behind_gap, behind_relative_speed)
            )
        return readings

    def _urgency(self, gap: float, closing_speed: float) -> float:
        """Normalized inverse time-to-collision: 0 when clear, 1 at impact.

        The network would otherwise have to learn a division between two
        badly-scaled inputs to recover the quantity every decision turns on.
        """
        if closing_speed <= 0.1:
            return 0.0
        ttc = gap / closing_speed
        return float(np.clip(1.0 - ttc / self.THREAT_HORIZON, 0.0, 1.0))

    def _observation(self) -> NDArray[np.float32]:
        speed_span = self.MAX_SPEED - self.MIN_SPEED
        speed = (self.ego_speed - self.MIN_SPEED) / speed_span
        target_speed = (self.target_speed - self.MIN_SPEED) / speed_span
        if self.longitudinal_acceleration >= 0.0:
            acceleration = self.longitudinal_acceleration / self.MAX_ACCELERATION
        else:
            acceleration = self.longitudinal_acceleration / self.MAX_BRAKING
        waves = len(self.challenge_steps)
        hard = self.difficulty_mode == "hard"

        values = [
            float(np.clip(speed, 0.0, 1.0)),
            float(np.clip(target_speed, 0.0, 1.0)),
            self.lane_position / (self.LANES - 1),
            self.target_lane / (self.LANES - 1),
            float(np.clip(abs(self.lane_position - self.target_lane), 0.0, 1.0)),
            float(np.clip(acceleration, -1.0, 1.0)),
            # Endless driving has no deadline, so the agent permanently sees
            # the "plenty of road left" state it spends most of training in.
            1.0
            if self.endless
            else float(np.clip(1.0 - self.step_count / self.max_episode_steps, 0.0, 1.0)),
            1.0 if (hard and self.challenge_active) else 0.0,
            # Endless waves never stop arriving, so "fraction of the route's
            # waves cleared" has no value to report.
            0.0
            if (hard and self.endless)
            else ((self.challenges_resolved / waves) if hard else 1.0),
        ]

        for ahead_gap, relative_speed, behind_gap, behind_relative_speed in self._sensors():
            values.extend(
                [
                    ahead_gap / self.SENSOR_DISTANCE,
                    # Scaled by the true relative-speed span, not MAX_SPEED,
                    # which left a third of the channel's range unused.
                    float(np.clip(relative_speed / speed_span, -1.0, 1.0)),
                    self._urgency(ahead_gap, max(-relative_speed, 0.0)),
                    behind_gap / self.SENSOR_DISTANCE,
                    float(np.clip(behind_relative_speed / speed_span, -1.0, 1.0)),
                    self._urgency(behind_gap, max(behind_relative_speed, 0.0)),
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        threat = self.current_threat()
        rear_threat = self.rear_threat()
        return {
            "speed": self.ego_speed,
            "target_speed": self.target_speed,
            "acceleration": self.longitudinal_acceleration,
            "throttle": self.throttle,
            "brake": self.brake,
            "crashed": self.crashed,
            "completed": self.completed,
            "timed_out": self.timed_out,
            "endless": self.endless,
            "elapsed_seconds": self.elapsed_seconds,
            "lane": self.lane_position,
            "target_lane": self.target_lane,
            "action": ACTION_NAMES[self.last_action],
            "episode_return": self.episode_return,
            "episode_step": self.step_count,
            "episode_progress": self.step_count / self.max_episode_steps,
            "difficulty": self.difficulty,
            "difficulty_mode": self.difficulty_mode,
            "challenge": self.challenge_name,
            "challenge_active": self.challenge_active,
            "challenges_presented": self.challenges_presented,
            "challenges_resolved": self.challenges_resolved,
            "near_misses": self.near_misses,
            "safe_lane_changes": self.safe_lane_changes,
            "invalid_actions": self.invalid_actions,
            "ttc": threat["ttc"],
            "threat_level": threat["level"],
            "closest_gap": threat["gap"],
            "rear_ttc": rear_threat["ttc"],
            "rear_threat_level": rear_threat["level"],
            "rear_gap": rear_threat["gap"],
            "reward_components": dict(self.last_reward_components),
            "collision": self.last_collision.to_dict() if self.last_collision else None,
        }

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode is None:
            return None
        if self.renderer is None:
            from self_driving_rl.game_renderer import NeonRenderer

            self.renderer = NeonRenderer(
                human=self.render_mode == "human",
                fps=self.render_fps,
                speed=self.render_speed,
            )

        frame, keep_running = self.renderer.draw(self)
        if not keep_running:
            self.quit_requested = True
        return frame if self.render_mode == "rgb_array" else None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
