"""Left/right mirroring, so the agent cannot learn a preferred escape side.

The highway task is symmetric by construction: a hard wave picks its escape
lane with a coin flip. The V3 DQN nevertheless completed 62% of episodes whose
first wave escaped left and only 48% of those escaping right, and it chose
LANE RIGHT in 2% of steps against LANE LEFT in 6%.

Mirroring an episode is an exact symmetry of this environment, so training on
randomly mirrored episodes is free extra data that also forces the two
directions to share one set of weights.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from self_driving_rl.game_env import (
    ACTION_COUNT,
    STEER_KEEP,
    STEER_LEFT,
    STEER_RIGHT,
    NeonHighwayEnv,
    decode_action,
    encode_action,
)

EGO_FEATURES = 9
LANE_FEATURES = 6


def mirror_action(action: int) -> int:
    """Swap the steering half of an action, leaving the pedal alone."""
    steer, pedal = decode_action(action)
    flipped = {STEER_LEFT: STEER_RIGHT, STEER_RIGHT: STEER_LEFT, STEER_KEEP: STEER_KEEP}[steer]
    return encode_action(flipped, pedal)


def mirror_observation(observation: NDArray[np.float32], lanes: int) -> NDArray[np.float32]:
    """Reverse lane order and flip the two lane-index features."""
    mirrored = observation.copy()
    # Features 2 and 3 are lane position and target lane, normalized to [0, 1].
    mirrored[2] = 1.0 - observation[2]
    mirrored[3] = 1.0 - observation[3]
    lane_block = observation[EGO_FEATURES:].reshape(lanes, LANE_FEATURES)
    mirrored[EGO_FEATURES:] = lane_block[::-1].reshape(-1)
    return mirrored


class MirrorSymmetry(gym.Wrapper):
    """Mirror the road for a random half of episodes.

    The flip is chosen once per episode rather than per step: a policy that saw
    the world flip mid-manoeuvre would be solving a different, harder task.
    """

    def __init__(self, env: NeonHighwayEnv, probability: float = 0.5) -> None:
        super().__init__(env)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self._mirrored = False
        self._rng = np.random.default_rng()

    @property
    def lanes(self) -> int:
        return int(self.env.unwrapped.LANES)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        observation, info = self.env.reset(seed=seed, options=options)
        self._mirrored = bool(self._rng.random() < self.probability)
        if self._mirrored:
            observation = mirror_observation(observation, self.lanes)
        info["mirrored"] = self._mirrored
        return observation, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._mirrored:
            action = mirror_action(int(action))
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._mirrored:
            observation = mirror_observation(observation, self.lanes)
        info["mirrored"] = self._mirrored
        return observation, reward, terminated, truncated, info


__all__ = [
    "ACTION_COUNT",
    "MirrorSymmetry",
    "mirror_action",
    "mirror_observation",
]
