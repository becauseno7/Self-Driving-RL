from __future__ import annotations

import numpy as np

from self_driving_rl.game_env import (
    ACTION_COUNT,
    ACTION_NAMES,
    FASTER,
    IDLE,
    LANE_LEFT,
    LANE_RIGHT,
    PEDAL_BRAKE,
    PEDAL_GAS,
    SLOWER,
    STEER_LEFT,
    STEER_RIGHT,
    NeonHighwayEnv,
    decode_action,
    encode_action,
)
from self_driving_rl.game_renderer import NeonRenderer


def test_game_environment_contract() -> None:
    env = NeonHighwayEnv()
    try:
        observation, info = env.reset(seed=42)
        next_observation, reward, terminated, truncated, next_info = env.step(IDLE)
    finally:
        env.close()

    assert observation.shape == (33,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert env.action_space.n == 9
    assert isinstance(info, dict)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_info["action"] == "HOLD"
    assert next_info["collision"] is None
    assert set(next_info["reward_components"]) == {
        "progress",
        "traffic",
        "safety",
        "shaping",
        "comfort",
        "rules",
        "terminal",
    }
    assert np.isclose(sum(next_info["reward_components"].values()), reward)
    assert next_info["ttc"] > 0


def test_target_speed_produces_smooth_acceleration() -> None:
    env = NeonHighwayEnv()
    try:
        env.reset(seed=8)
        starting_speed = env.ego_speed
        starting_target = env.target_speed

        _, _, _, _, faster_info = env.step(FASTER)
        speed_after_press = env.ego_speed

        for _ in range(20):
            env.step(IDLE)
    finally:
        env.close()

    assert env.target_speed == starting_target + env.TARGET_SPEED_STEP
    assert starting_speed < speed_after_press < env.target_speed
    assert env.ego_speed > speed_after_press
    assert abs(env.ego_speed - env.target_speed) < 0.05
    assert 0.0 <= env.throttle <= 1.0
    assert 0.0 <= env.brake <= 1.0
    assert faster_info["reward_components"]["comfort"] < 0.0


def test_unsafe_lane_change_gets_immediate_safety_penalty() -> None:
    env = NeonHighwayEnv()
    try:
        env.reset(seed=14)
        env.target_lane = 1
        env.lane_position = 1.0
        blocking_car = next(car for car in env.traffic if car.lane == 0)
        blocking_car.position = env.ego_position + 8.0

        action_result = env._apply_action(LANE_LEFT)
        env._reward(action_result)
    finally:
        env.close()

    assert action_result["unsafe_lane_change"] is True
    assert env.last_reward_components["safety"] <= -env.UNSAFE_LANE_CHANGE_COST


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


def test_hard_mode_starts_with_a_visible_360_degree_challenge() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        env.reset(seed=31)
        current_lane = env.target_lane
        ahead_gap, ahead_relative, behind_gap, behind_relative = env.lane_sensors()[
            current_lane
        ]
        adjacent = [
            reading
            for lane, reading in enumerate(env.lane_sensors())
            if abs(lane - current_lane) == 1
        ]
    finally:
        env.close()

    assert env.challenges_presented == 1
    assert env.challenge_active is True
    assert ahead_gap < 35.0
    assert ahead_relative < 0.0
    assert behind_gap < 27.0
    assert behind_relative > 0.0
    assert any(
        front_gap > 20.0
        and rear_gap > 16.0
        and (rear_relative <= 0.1 or rear_gap / rear_relative > 3.5)
        for front_gap, _, rear_gap, rear_relative in adjacent
    )


def test_passive_driver_does_not_complete_hard_opening() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        for seed in range(5):
            env.reset(seed=seed)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(IDLE)
            assert info["crashed"] is True
            assert info["completed"] is False
    finally:
        env.close()


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


def test_traffic_cars_never_drive_through_each_other() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=5)
        env.lane_position = 3.0
        env.target_lane = 3
        env.ego_position = 5_000.0
        lane_zero = [car for car in env.traffic if car.lane == 0]
        slow, fast = lane_zero[0], lane_zero[1]
        slow.position, slow.speed, slow.desired_speed = 100.0, env.MIN_SPEED, env.MIN_SPEED
        fast.position, fast.speed, fast.desired_speed = 70.0, env.MAX_SPEED, env.MAX_SPEED
        for car in (slow, fast):
            car.previous_position = car.position

        closest = float("inf")
        for _ in range(200):
            env._update_motion()
            closest = min(closest, slow.position - fast.position)
    finally:
        env.close()

    assert closest >= env.CAR_LENGTH - 1e-6, f"traffic overlapped: {closest:.2f} m"
    assert fast.position < slow.position, "the faster car teleported past the slower one"


def test_no_lateral_blind_spot_in_the_middle_of_a_lane_change() -> None:
    """The ego sweeps 0.24 lanes per step, so end-of-step tests miss the middle."""
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=1)
        env.target_lane = 1
        env.lane_position = 1.0
        for car in env.traffic:
            car.position = env.ego_position + 600.0
            car.previous_position = car.position
        blocker = env.traffic[0]
        blocker.lane = 2

        env.step(LANE_RIGHT)
        collided_at = None
        for _ in range(6):
            blocker.position = env.ego_position
            blocker.previous_position = env.ego_position
            blocker.speed = env.ego_speed
            if env._detect_collision() is not None:
                collided_at = env.lane_position
                break
            env.step(IDLE)
    finally:
        env.close()

    assert collided_at is not None, "a car sitting on the ego was never detected"
    assert collided_at < 2.0, (
        f"detection only happened at lane {collided_at}, after the merge finished"
    )


def test_no_lane_position_is_immune_to_every_lane() -> None:
    """The original bug: mid-merge, no lane centre was within the collision box."""
    env = NeonHighwayEnv()
    try:
        position = 1.0
        immune = []
        for _ in range(10):
            nearest = min(abs(lane - position) for lane in range(env.LANES))
            if nearest >= env.LANE_COLLISION_WIDTH:
                immune.append(round(position, 3))
            position += float(np.clip(2 - position, -0.24, 0.24))
            if abs(position - 2) < 0.02:
                position = 2.0
    finally:
        env.close()

    assert not immune, f"nothing can hit the ego at lane {immune}"


def test_diagonal_merge_contact_is_detected_at_the_same_instant() -> None:
    """Regression: visible bodywork merged together without ending the episode."""
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=101)
        for car in env.traffic:
            car.position = env.ego_position + 600.0
            car.previous_position = car.position

        blocker = env.traffic[0]
        blocker.lane = 2
        blocker.previous_position = env.previous_ego_position + env.CAR_LENGTH + 0.8
        blocker.position = env.ego_position + env.CAR_LENGTH - 0.8
        blocker.speed = env.ego_speed
        env.previous_lane_position = 1.30
        env.lane_position = 1.54
        env.target_lane = 2

        collision = env._detect_collision()
    finally:
        env.close()

    assert collision is not None
    assert collision.kind == "SIDE IMPACT"


def test_axis_overlaps_at_different_times_are_not_a_collision() -> None:
    """Swept tests must not combine a past longitudinal pass with a later merge."""
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=102)
        for car in env.traffic:
            car.position = env.ego_position + 600.0
            car.previous_position = car.position

        blocker = env.traffic[0]
        blocker.lane = 2
        blocker.previous_position = env.previous_ego_position - env.CAR_LENGTH - 0.2
        blocker.position = env.ego_position - env.CAR_LENGTH - 4.0
        env.previous_lane_position = 1.30
        env.lane_position = 1.54
        env.target_lane = 2

        collision = env._detect_collision()
    finally:
        env.close()

    assert collision is None


def test_touching_collision_boundaries_count_as_contact() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=103)
        for car in env.traffic:
            car.position = env.ego_position + 600.0
            car.previous_position = car.position

        blocker = env.traffic[0]
        blocker.lane = env.target_lane
        blocker.position = env.ego_position + env.CAR_LENGTH
        blocker.previous_position = blocker.position
        env.lane_position = float(env.target_lane)
        env.previous_lane_position = env.lane_position

        collision = env._detect_collision()
    finally:
        env.close()

    assert collision is not None


def test_collision_width_matches_a_real_car_in_a_real_lane() -> None:
    """What the renderer draws and what the physics hits must be one number."""
    env = NeonHighwayEnv()
    try:
        assert env.LANE_COLLISION_WIDTH == env.CAR_WIDTH / env.LANE_WIDTH
        # Must still clear 0.5, or the midpoint of a merge reopens the blind spot.
        assert env.LANE_COLLISION_WIDTH > 0.5
    finally:
        env.close()


def test_renderer_uses_the_exact_physics_footprint() -> None:
    env = NeonHighwayEnv()
    try:
        width, length = NeonRenderer.car_footprint(env)
        lane_pixels = NeonRenderer.ROAD_WIDTH / env.LANES

        assert np.isclose(width / lane_pixels, env.LANE_COLLISION_WIDTH)
        assert np.isclose(length / NeonRenderer.PIXELS_PER_METER, env.CAR_LENGTH)
        assert length > width, "the camera turned real cars into horizontal bars again"
    finally:
        env.close()


def test_collision_reports_the_nearest_car_not_the_first_in_the_list() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=4)
        for car in env.traffic:
            car.position = env.ego_position + 600.0
            car.previous_position = car.position
        lane = int(env.target_lane)
        env.lane_position = float(lane)
        env.previous_lane_position = float(lane)
        far, near = env.traffic[0], env.traffic[-1]
        for car, offset, speed in ((far, 4.0, 5.0), (near, -0.5, 33.0)):
            car.lane = lane
            car.position = env.ego_position + offset
            car.previous_position = car.position
            car.speed = speed

        collision = env._detect_collision()
    finally:
        env.close()

    assert collision is not None
    assert collision.relative_position == -0.5
    assert collision.traffic_speed == 33.0


def test_sensor_gaps_are_bumper_to_bumper() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=2)
        for car in env.traffic:
            car.position = env.ego_position + 600.0
            car.previous_position = car.position
        lane = int(env.target_lane)
        env.lane_position = float(lane)
        env.previous_lane_position = float(lane)
        lead = env.traffic[0]
        lead.lane = lane
        lead.position = env.ego_position + env.CAR_LENGTH + 10.0
        lead.previous_position = lead.position

        ahead_gap = env.lane_sensors()[lane][0]

        lead.position = env.ego_position + 4.5
        lead.previous_position = lead.position
        touching_gap = env.lane_sensors()[lane][0]
        touching_is_a_crash = env._detect_collision() is not None
    finally:
        env.close()

    assert ahead_gap == 10.0
    assert touching_gap == 0.0
    assert touching_is_a_crash


def test_episode_always_ends_even_if_a_wave_never_stages() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        env.reset(seed=3)
        env._next_challenge_index = len(env.challenge_steps)  # no wave can stage
        terminated = truncated = False
        steps = 0
        info: dict = {}
        while not (terminated or truncated) and steps <= env.absolute_step_limit + 10:
            for car in env.traffic:  # remove every crash opportunity
                car.position = env.ego_position + 500.0 + car.lane * 5.0
                car.previous_position = car.position
            _, _, terminated, truncated, info = env.step(IDLE)
            steps += 1
    finally:
        env.close()

    assert truncated
    assert steps == env.absolute_step_limit
    assert info["completed"] is False
    assert info["timed_out"] is True
    assert info["TimeLimit.truncated"] is True


def test_the_agent_can_brake_and_merge_in_the_same_step() -> None:
    """The hard waves need both at once; through V3 the action space forbade it."""
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=19)
        env.target_lane = 1
        env.lane_position = 1.0
        env.target_speed = env.ego_speed = 22.0
        starting_target = env.target_speed

        action_result = env._apply_action(encode_action(STEER_RIGHT, PEDAL_BRAKE))
    finally:
        env.close()

    assert env.target_lane == 2, "the merge did not start"
    assert env.target_speed == starting_target - env.TARGET_SPEED_STEP, "the brake was ignored"
    assert action_result["lane_change_started"] is True
    assert action_result["speed_target_changed"] is True


def test_action_names_cover_every_steer_and_pedal_pair() -> None:
    assert len(ACTION_NAMES) == ACTION_COUNT
    assert ACTION_NAMES[IDLE] == "HOLD"
    assert ACTION_NAMES[encode_action(STEER_LEFT, PEDAL_BRAKE)] == "LEFT+BRAKE"
    assert ACTION_NAMES[encode_action(STEER_RIGHT, PEDAL_GAS)] == "RIGHT+GAS"
    for action in range(ACTION_COUNT):
        assert encode_action(*decode_action(action)) == action


def test_route_length_is_configurable_and_scales_the_waves() -> None:
    """A longer route is more waves, not the same three with dead time after."""
    default = NeonHighwayEnv(difficulty_mode="hard")
    long_route = NeonHighwayEnv(difficulty_mode="hard", episode_seconds=180.0)
    try:
        assert default.max_episode_steps == 450
        assert default.challenge_steps == (0, 150, 300)

        assert long_route.max_episode_steps == 1800
        assert long_route.challenge_steps == tuple(range(0, 1651, 150))
        assert len(long_route.challenge_steps) == 12

        # every wave must still have room to be cleared before the route ends
        last = long_route.challenge_steps[-1]
        assert last + long_route.CHALLENGE_CLEAR_STEPS <= long_route.max_episode_steps
    finally:
        default.close()
        long_route.close()


def test_a_long_route_completes_only_after_every_wave() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard", episode_seconds=90.0)
    try:
        env.reset(seed=12)
        assert len(env.challenge_steps) == 6
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            for car in env.traffic:  # remove all crash risk, keep the waves
                if abs(car.position - env.ego_position) < 12.0:
                    car.position = env.ego_position + 300.0
                    car.previous_position = car.position
            _, _, terminated, truncated, info = env.step(IDLE)
    finally:
        env.close()

    assert info["episode_step"] == 900
    assert info["completed"] is True
    assert info["challenges_resolved"] == 6


def test_episode_seconds_rejects_a_route_shorter_than_the_physics_allows() -> None:
    try:
        NeonHighwayEnv(episode_seconds=1.0)
    except ValueError as error:
        assert "episode_seconds" in str(error)
    else:
        raise AssertionError("a 1 second route should have been rejected")


def test_nothing_ever_teleports_or_restyles_on_screen() -> None:
    """A car that jumps or changes colour in view reads as a bug, not traffic."""
    env = NeonHighwayEnv(difficulty_mode="hard", endless=True)
    offences: list[str] = []
    try:
        env.reset(seed=30_000)
        for _ in range(2_000):
            before = {
                id(car): (car.position, car.color_index, car.style) for car in env.traffic
            }
            ego_before = env.ego_position
            env.step(IDLE)
            if env.crashed:
                env.reset(seed=30_000)
                continue

            for car in env.traffic:
                previous_position, color, style = before[id(car)]
                was_visible = (
                    -env.VISIBLE_BEHIND <= previous_position - ego_before <= env.VISIBLE_AHEAD
                )
                now_visible = env._is_visible(car)
                if not (was_visible or now_visible):
                    continue
                # A car under its own power moves at most MAX_SPEED * DT.
                travelled = abs(car.position - previous_position)
                if travelled > env.MAX_SPEED * env.DT + 1e-6:
                    offences.append(f"jumped {travelled:.1f} m while visible")
                if (car.color_index, car.style) != (color, style):
                    offences.append("changed appearance while visible")
    finally:
        env.close()

    assert not offences, offences[:5]


def _clear_the_road(env: NeonHighwayEnv) -> None:
    for car in env.traffic:
        if abs(car.position - env.ego_position) < 12.0:
            car.position = env.ego_position + 300.0
            car.previous_position = car.position


def test_endless_mode_only_ends_on_a_crash() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard", endless=True)
    try:
        env.reset(seed=7)
        terminated = truncated = False
        steps = 0
        info: dict = {}
        while not (terminated or truncated) and steps < 3_000:
            _clear_the_road(env)
            _, _, terminated, truncated, info = env.step(IDLE)
            steps += 1
    finally:
        env.close()

    assert steps == 3_000, "endless mode ended without a crash"
    assert not terminated and not truncated
    assert info["completed"] is False, "endless mode should never complete"
    assert info["elapsed_seconds"] == 300.0


def test_endless_driving_is_one_continuous_run() -> None:
    """No route boundary: the clock only counts up and waves only accumulate."""
    env = NeonHighwayEnv(difficulty_mode="hard", endless=True)
    try:
        env.reset(seed=7)
        elapsed = []
        resolved = []
        for _ in range(1_200):
            _clear_the_road(env)
            _, _, _, _, info = env.step(IDLE)
            elapsed.append(info["elapsed_seconds"])
            resolved.append(info["challenges_resolved"])
    finally:
        env.close()

    assert elapsed == sorted(elapsed), "the clock reset at some point"
    assert resolved == sorted(resolved), "the wave counter reset at some point"
    assert resolved[-1] > len(NeonHighwayEnv(difficulty_mode="hard").challenge_steps), (
        "waves stopped arriving after one route's worth"
    )


def test_endless_observation_reports_no_deadline() -> None:
    """A route-trained policy must not see the clock run out mid-drive."""
    env = NeonHighwayEnv(difficulty_mode="hard", endless=True)
    try:
        env.reset(seed=7)
        time_remaining = []
        for _ in range(900):
            _clear_the_road(env)
            env.step(IDLE)
            observation = env._observation()
            time_remaining.append(float(observation[6]))
    finally:
        env.close()

    assert set(time_remaining) == {1.0}, "endless mode should report no deadline"


def test_endless_mode_never_pays_a_route_completion_bonus() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard", endless=True)
    try:
        env.reset(seed=7)
        terminal_components = []
        for _ in range(900):
            _clear_the_road(env)
            _, _, terminated, truncated, info = env.step(IDLE)
            terminal_components.append(info["reward_components"]["terminal"])
            assert not (terminated or truncated)
    finally:
        env.close()

    assert set(terminal_components) == {0.0}, "something ended the route"


def test_fixed_route_mode_is_unchanged_by_the_endless_machinery() -> None:
    fixed = NeonHighwayEnv(difficulty_mode="hard")
    try:
        observation, _ = fixed.reset(seed=21)
        assert fixed.endless is False
        assert observation[6] == 1.0
        for _ in range(40):
            fixed.step(IDLE)
            if fixed.crashed:
                break
        assert fixed._observation()[6] < 1.0, "the fixed-route clock stopped counting down"
    finally:
        fixed.close()


def test_observation_exposes_the_clock_and_wave_state() -> None:
    """The reward has terminal and wave terms, so the agent must see the clock."""
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        first, _ = env.reset(seed=44)
        for _ in range(120):
            env.step(FASTER if env.ego_speed < env.target_speed else IDLE)
            if env.crashed:
                break
        later = env._observation()
    finally:
        env.close()

    assert first[6] == 1.0, "time remaining should start full"
    assert later[6] < first[6], "time remaining must fall as the episode runs"
    assert first[7] == 1.0, "hard mode opens with a wave active"
    assert env.observation_space.contains(later)


def test_safety_shaping_is_potential_based() -> None:
    """A potential function telescopes: only the endpoints survive the sum.

    That is what makes the shaping unable to change which policy is optimal,
    so nothing rewards the agent for simply crawling along in clear traffic.
    """
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        env.reset(seed=77)
        starting_potential = env._previous_potential
        discounted_shaping = 0.0
        step = 0
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(IDLE)
            discounted_shaping += (
                env.SHAPING_GAMMA**step * info["reward_components"]["shaping"]
            )
            step += 1
    finally:
        env.close()

    # sum_t gamma^t (gamma * Phi_{t+1} - Phi_t) telescopes to -Phi_0, because
    # the potential of a terminal state is defined as 0. The shaping therefore
    # contributes a constant to every policy's value and cannot reorder them.
    assert np.isclose(discounted_shaping, -starting_potential, atol=1e-6)
    assert starting_potential < 0.0, "seed 77 should open with a real threat"


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

        acceleration = env._traffic_acceleration(car)
    finally:
        env.close()

    assert acceleration < 0.0


def test_route_completion_adds_terminal_bonus() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
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
    assert info["reward_components"]["terminal"] == env.COMPLETION_BONUS
    assert reward > env.COMPLETION_BONUS


def test_real_traffic_crossings_count_overtakes_and_being_passed() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=201)
        for index, car in enumerate(env.traffic):
            car.previous_position = 100.0 + index * 10.0
            car.position = car.previous_position

        env.previous_ego_position = 0.0
        env.ego_position = 1.0
        overtaken = env.traffic[0]
        overtaken.previous_position = 0.5
        overtaken.position = 0.5
        passing = env.traffic[1]
        passing.previous_position = -0.5
        passing.position = 1.5

        overtakes, passed_by_traffic = env._traffic_progress_events()
    finally:
        env.close()

    assert overtakes == 1
    assert passed_by_traffic == 1


def test_passing_option_requires_a_slower_leader_and_safe_clearer_lane() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=202)
        env.target_lane = 1
        env.lane_position = 1.0
        env.previous_lane_position = 1.0
        for index, car in enumerate(env.traffic):
            car.position = env.ego_position + 500.0 + index * 10.0
            car.previous_position = car.position
            car.speed = env.ego_speed

        leader, clear_front, clear_rear, trap = env.traffic[:4]
        leader.lane = 1
        leader.position = env.ego_position + env.CAR_LENGTH + 20.0
        leader.speed = env.ego_speed - 5.0
        clear_front.lane = 2
        clear_front.position = env.ego_position + env.CAR_LENGTH + 48.0
        clear_rear.lane = 2
        clear_rear.position = env.ego_position - env.CAR_LENGTH - 25.0
        trap.lane = 0
        trap.position = env.ego_position + env.CAR_LENGTH + 5.0
        for car in (leader, clear_front, clear_rear, trap):
            car.previous_position = car.position
        env._invalidate_sensors()

        options = env.passing_lane_options()
        action_result = env._apply_action(LANE_RIGHT, passing_options=options)
    finally:
        env.close()

    assert options == (2,)
    assert action_result["passing_maneuver"] is True


def test_traffic_reward_values_completed_passes_not_idle_weaving() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=203)
        waiting = env._apply_action(IDLE, passing_options=(2,))
        env._reward(waiting)
        waiting_traffic_reward = env.last_reward_components["traffic"]

        overtake = env._apply_action(IDLE, passing_options=())
        overtake["overtakes"] = 1
        env._reward(overtake)
        overtake_traffic_reward = env.last_reward_components["traffic"]

        weave = env._apply_action(LANE_RIGHT, passing_options=())
        env._reward(weave)
        weave_traffic_reward = env.last_reward_components["traffic"]
    finally:
        env.close()

    assert waiting_traffic_reward == -env.BLOCKED_WITH_SAFE_PASS_COST
    assert overtake_traffic_reward == env.OVERTAKE_BONUS
    assert weave_traffic_reward == 0.0


def test_preference_telemetry_counts_missed_passes_and_quick_reversals() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=206)
        env._passing_opportunity_active = True
        env.missed_passing_opportunities = 0
        for index, car in enumerate(env.traffic):
            car.position = env.ego_position + 300.0 + 10.0 * index
            car.previous_position = car.position
        env._invalidate_sensors()
        env.step(IDLE)

        env.target_lane = 1
        env.lane_position = 1.0
        env._apply_action(1)
        env.target_lane = 0
        env.lane_position = 0.0
        env.step_count += 10
        env._apply_action(7)
    finally:
        env.close()

    assert env.missed_passing_opportunities == 1
    assert env.rapid_lane_changes == 1
    assert env.lane_reversals == 1


def test_longitudinal_telemetry_finds_clear_braking_and_pedal_reversals() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=612)
        for index, car in enumerate(env.traffic):
            car.position = env.ego_position + 300.0 + 10.0 * index
            car.previous_position = car.position
        env._invalidate_sensors()

        env.step(SLOWER)
        _, _, _, _, info = env.step(FASTER)
    finally:
        env.close()

    assert info["unjustified_brakes"] == 1
    assert info["pedal_reversals"] == 1
    assert info["speed_target_changes"] == 2


def test_longitudinal_telemetry_counts_clear_road_stalling() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=613)
        env.ego_speed = env.CRUISE_SPEED - env.STALL_SPEED_MARGIN - 1.0
        env.target_speed = env.ego_speed
        for index, car in enumerate(env.traffic):
            car.position = env.ego_position + 300.0 + 10.0 * index
            car.previous_position = car.position
        env._invalidate_sensors()

        _, _, _, _, info = env.step(IDLE)
    finally:
        env.close()

    assert info["clear_road_stall_steps"] == 1


def test_speed_matched_slow_leader_remains_an_avoidable_pass() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=614)
        for index, car in enumerate(env.traffic):
            car.position = env.ego_position + 200.0 + 10.0 * index
            car.previous_position = car.position
            car.speed = env.ego_speed
        leader = env.traffic[0]
        leader.lane = env.target_lane
        leader.position = env.ego_position + 20.0
        leader.previous_position = leader.position
        leader.speed = env.ego_speed
        env._invalidate_sensors()

        options = env.passing_lane_options()
        _, _, _, _, info = env.step(IDLE)
    finally:
        env.close()

    assert options
    assert info["slow_leader_follow_steps"] == 1
    assert info["avoidable_following_steps"] == 1


def test_renderer_accepts_explanatory_driving_intent_hud() -> None:
    env = NeonHighwayEnv(render_mode="rgb_array", difficulty_mode="standard")
    try:
        env.reset(seed=720)
        env.hud_data.update(
            {
                "driving_intent": "PASS LEFT",
                "desired_speed": 28.0,
                "speed_reason": "taking persistent safe pass",
                "braking_mode": "COMFORT BRAKE",
                "braking_reason": "proportional following response",
            }
        )

        frame = env.render()
    finally:
        env.close()

    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3


def test_traffic_lane_change_moves_smoothly_and_occupies_both_lanes() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard", dynamic_traffic=True)
    try:
        env.reset(seed=721)
        car = next(car for car in env.traffic if car.lane == 1)

        env._start_traffic_lane_change(car, 2)
        occupied_at_start = env._traffic_occupied_lanes(car)
        env._update_traffic_lane_changes()
        lateral_after_one_step = env.traffic_lateral_position(car)
    finally:
        env.close()

    assert occupied_at_start == {1, 2}
    assert np.isclose(lateral_after_one_step, 1.0 + env.TRAFFIC_LANE_CHANGE_STEP)
    assert car.lane == 1
    assert car.target_lane == 2


def test_traffic_refuses_to_merge_across_ego_blind_spot() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard", dynamic_traffic=True)
    try:
        env.reset(seed=722)
        car = next(car for car in env.traffic if car.lane == 0)
        car.position = env.ego_position + env.CAR_LENGTH + 2.0
        car.previous_position = car.position
        car.speed = env.ego_speed
        env.lane_position = 1.0
        env.target_lane = 1
        env._invalidate_sensors()

        safe = env._traffic_lane_change_is_safe(car, 1)
    finally:
        env.close()

    assert not safe


def test_motivated_traffic_starts_safe_overtake_without_observation_change() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard", dynamic_traffic=True)
    try:
        observation, _ = env.reset(seed=723)
        for index, car in enumerate(env.traffic):
            car.position = env.ego_position + 200.0 + 20.0 * index
            car.previous_position = car.position
            car.speed = 20.0
            car.desired_speed = 20.0
        follower = env.traffic[0]
        leader = env.traffic[1]
        follower.lane = 1
        follower.position = env.ego_position + 50.0
        follower.previous_position = follower.position
        follower.speed = 25.0
        follower.desired_speed = 29.0
        leader.lane = 1
        leader.position = follower.position + 20.0
        leader.previous_position = leader.position
        leader.speed = 16.0
        leader.desired_speed = 16.0
        env._invalidate_sensors()

        env._maybe_start_traffic_lane_changes()
    finally:
        env.close()

    assert observation.shape == (33,)
    assert follower.target_lane in {0, 2}
    assert env.traffic_lane_changes >= 1


def test_dynamic_traffic_preserves_two_wave_actors_per_lane() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard", dynamic_traffic=True)
    try:
        env.reset(seed=724)
        lane_zero = [car for car in env.traffic if car.lane == 0]
        for car in lane_zero[2:]:
            car.lane = 2
        follower, leader = lane_zero[:2]
        follower.position = env.ego_position + 50.0
        follower.previous_position = follower.position
        follower.speed = 25.0
        follower.desired_speed = 30.0
        leader.position = follower.position + 20.0
        leader.previous_position = leader.position
        leader.speed = 15.0
        leader.desired_speed = 15.0
        env._invalidate_sensors()

        env._maybe_start_traffic_lane_changes()
    finally:
        env.close()

    assert follower.target_lane is None
    assert sum(car.lane == 0 for car in env.traffic) == 2


def test_close_ahead_merge_is_recorded_as_traffic_cut_in() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard", dynamic_traffic=True)
    try:
        env.reset(seed=725)
        car = next(car for car in env.traffic if car.lane == env.target_lane - 1)
        car.position = env.ego_position + 30.0
        car.previous_position = car.position

        env._start_traffic_lane_change(car, env.target_lane)
    finally:
        env.close()

    assert env.traffic_cut_ins == 1


def test_wrong_lane_does_not_evade_blocked_traffic_cost() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=204)
        wrong_way = env._apply_action(LANE_LEFT, passing_options=(2,))
        env._reward(wrong_way)
    finally:
        env.close()

    assert wrong_way["lane_change_started"] is True
    assert wrong_way["passing_maneuver"] is False
    assert env.last_reward_components["traffic"] == -(
        env.BLOCKED_WITH_SAFE_PASS_COST + env.UNPRODUCTIVE_LANE_CHANGE_COST
    )


def test_merge_safety_potential_sees_target_lane_immediately() -> None:
    env = NeonHighwayEnv(difficulty_mode="standard")
    try:
        env.reset(seed=205)
        env.target_lane = 1
        env.lane_position = 1.0
        for index, car in enumerate(env.traffic):
            car.position = env.ego_position + 300.0 + index * 10.0
            car.previous_position = car.position
        blocker = env.traffic[0]
        blocker.lane = 2
        blocker.position = env.ego_position + env.CAR_LENGTH + 2.0
        blocker.previous_position = blocker.position
        blocker.speed = env.ego_speed
        env._invalidate_sensors()

        before = env._safety_potential()
        env.target_lane = 2
        committed = env._safety_potential()
    finally:
        env.close()

    assert before == 0.0
    assert committed < -1.0
