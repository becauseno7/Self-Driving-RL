from __future__ import annotations

import numpy as np

from self_driving_rl.environment import make_env


def test_environment_observation_and_step_contract() -> None:
    env = make_env()
    try:
        observation, info = env.reset(seed=123)

        assert observation.shape == (5, 5)
        assert observation.dtype == np.float32
        assert isinstance(info, dict)
        assert env.action_space.n == 5

        next_observation, reward, terminated, truncated, step_info = env.step(1)

        assert next_observation.shape == (5, 5)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    finally:
        env.close()
