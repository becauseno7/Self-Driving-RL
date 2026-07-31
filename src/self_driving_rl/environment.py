"""Environment construction kept in one place for reproducible experiments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401 - importing registers HighwayEnv with Gymnasium

ENV_ID = "highway-fast-v0"

# These values make the task definition visible and versionable. Unspecified
# settings use HighwayEnv 1.12.0 defaults.
ENV_CONFIG: dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "order": "sorted",
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 30,
    "policy_frequency": 5,
    "collision_reward": -1.0,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0.0,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
}



def make_env(*, render_mode: str | None = None) -> gym.Env:
    """Create a fresh copy of the project's driving environment."""
    return gym.make(ENV_ID, config=ENV_CONFIG, render_mode=render_mode)
