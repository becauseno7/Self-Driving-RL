from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_driving_rl import game
from self_driving_rl.game_env import ACTION_COUNT, NeonHighwayEnv

CURRENT_SHAPE = NeonHighwayEnv().observation_space.shape


def _checkpoint(directory: Path, run_name: str, timestamp: int) -> Path:
    path = directory / run_name / "model.zip"
    path.parent.mkdir(parents=True)
    path.touch()
    os.utime(path, (timestamp, timestamp))
    return path


def _fake_model(shape: tuple[int, ...], actions: int) -> SimpleNamespace:
    return SimpleNamespace(
        observation_space=SimpleNamespace(shape=shape),
        action_space=SimpleNamespace(n=actions),
    )


def test_latest_model_skips_newer_incompatible_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = tmp_path / "runs" / "game"
    compatible = _checkpoint(model_directory, "v4", 100)
    stale_observation = _checkpoint(model_directory, "v3-newer", 200)

    models = {
        compatible: _fake_model(CURRENT_SHAPE, ACTION_COUNT),
        stale_observation: _fake_model((20,), ACTION_COUNT),
    }
    monkeypatch.setattr(game, "load_model", lambda path, device: models[Path(path)])

    assert game._latest_model(model_directory) == compatible


def test_latest_model_rejects_a_checkpoint_with_the_old_action_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V3 split steering and pedal across one slot, so it had five actions."""
    model_directory = tmp_path / "runs" / "game"
    compatible = _checkpoint(model_directory, "v4", 100)
    stale_actions = _checkpoint(model_directory, "five-action-newer", 200)

    models = {
        compatible: _fake_model(CURRENT_SHAPE, ACTION_COUNT),
        stale_actions: _fake_model(CURRENT_SHAPE, 5),
    }
    monkeypatch.setattr(game, "load_model", lambda path, device: models[Path(path)])

    assert game._latest_model(model_directory) == compatible


def test_latest_model_explains_when_retraining_is_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = tmp_path / "runs" / "game"
    _checkpoint(model_directory, "v3", 100)

    monkeypatch.setattr(
        game, "load_model", lambda path, device: _fake_model((20,), 5)
    )

    with pytest.raises(SystemExit, match="train a fresh model"):
        game._latest_model(model_directory)


def test_ensure_model_compatible_reports_both_mismatches() -> None:
    env = NeonHighwayEnv()
    try:
        game._ensure_model_compatible(_fake_model(CURRENT_SHAPE, ACTION_COUNT), env)
        with pytest.raises(SystemExit, match="Model/environment mismatch"):
            game._ensure_model_compatible(_fake_model((20,), ACTION_COUNT), env)
        with pytest.raises(SystemExit, match="Model/environment mismatch"):
            game._ensure_model_compatible(_fake_model(CURRENT_SHAPE, 5), env)
    finally:
        env.close()


def test_qrdqn_preset_adds_quantiles_without_touching_dqn() -> None:
    base = {"policy_kwargs": {"net_arch": [256, 256]}, "gamma": 0.995}

    dqn_kwargs = game.build_algorithm_kwargs("dqn", base)
    qrdqn_kwargs = game.build_algorithm_kwargs("qrdqn", base)

    assert "n_quantiles" not in dqn_kwargs["policy_kwargs"]
    assert qrdqn_kwargs["policy_kwargs"]["n_quantiles"] == 64
    assert qrdqn_kwargs["policy_kwargs"]["net_arch"] == [256, 256]
    assert base["policy_kwargs"] == {"net_arch": [256, 256]}, "the preset was mutated"
