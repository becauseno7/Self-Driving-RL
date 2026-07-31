from __future__ import annotations

import numpy as np

from self_driving_rl.game_env import (
    ACTION_COUNT,
    IDLE,
    LANE_LEFT,
    LANE_RIGHT,
    PEDAL_BRAKE,
    STEER_LEFT,
    STEER_RIGHT,
    NeonHighwayEnv,
    encode_action,
)
from self_driving_rl.symmetry import (
    MirrorSymmetry,
    mirror_action,
    mirror_observation,
)


def test_mirror_action_swaps_steering_and_keeps_the_pedal() -> None:
    assert mirror_action(LANE_LEFT) == LANE_RIGHT
    assert mirror_action(LANE_RIGHT) == LANE_LEFT
    assert mirror_action(IDLE) == IDLE
    assert mirror_action(encode_action(STEER_LEFT, PEDAL_BRAKE)) == encode_action(
        STEER_RIGHT, PEDAL_BRAKE
    )
    for action in range(ACTION_COUNT):
        assert mirror_action(mirror_action(action)) == action


def test_mirroring_an_observation_twice_is_the_identity() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        observation, _ = env.reset(seed=5)
        once = mirror_observation(observation, env.LANES)
        twice = mirror_observation(once, env.LANES)
    finally:
        env.close()

    np.testing.assert_allclose(twice, observation, atol=1e-7)
    assert not np.allclose(once, observation), "seed 5 is not laterally symmetric"


def test_mirrored_observation_reverses_the_lane_readings() -> None:
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        observation, _ = env.reset(seed=8)
        mirrored = mirror_observation(observation, env.LANES)
        lanes = observation[9:].reshape(env.LANES, 6)
        mirrored_lanes = mirrored[9:].reshape(env.LANES, 6)
    finally:
        env.close()

    np.testing.assert_allclose(mirrored_lanes, lanes[::-1], atol=1e-7)
    assert np.isclose(mirrored[2], 1.0 - observation[2])
    assert np.isclose(mirrored[3], 1.0 - observation[3])


def test_mirrored_episode_earns_the_same_return_as_the_original() -> None:
    """The exact test of a symmetry: the mirrored world must play identically."""
    plan = [LANE_LEFT, IDLE, IDLE, IDLE, IDLE, encode_action(STEER_LEFT, PEDAL_BRAKE)] * 40

    plain = NeonHighwayEnv(difficulty_mode="hard")
    mirrored = MirrorSymmetry(NeonHighwayEnv(difficulty_mode="hard"), probability=1.0)
    try:
        plain_observation, _ = plain.reset(seed=61)
        mirrored_observation, info = mirrored.reset(seed=61)
        assert info["mirrored"] is True
        np.testing.assert_allclose(
            mirrored_observation,
            mirror_observation(plain_observation, plain.LANES),
            atol=1e-7,
        )

        plain_return = mirrored_return = 0.0
        for action in plan:
            _, reward, terminated, truncated, _ = plain.step(action)
            plain_return += reward
            if terminated or truncated:
                break
        for action in plan:
            # The wrapper flips steering internally, so the mirrored agent must
            # issue the mirrored intent to trace the same path.
            _, reward, terminated, truncated, _ = mirrored.step(mirror_action(action))
            mirrored_return += reward
            if terminated or truncated:
                break
    finally:
        plain.close()
        mirrored.close()

    assert np.isclose(plain_return, mirrored_return, atol=1e-6)


def test_wrapper_leaves_spaces_untouched() -> None:
    env = MirrorSymmetry(NeonHighwayEnv(difficulty_mode="hard"))
    try:
        observation, _ = env.reset(seed=3)
        assert env.observation_space.contains(observation)
        assert env.action_space.n == ACTION_COUNT
        for _ in range(30):
            observation, _, terminated, truncated, _ = env.step(IDLE)
            assert env.observation_space.contains(observation)
            if terminated or truncated:
                break
    finally:
        env.close()
