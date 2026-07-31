from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_driving_rl import game


def _checkpoint(directory: Path, run_name: str, timestamp: int) -> Path:
    path = directory / run_name / "model.zip"
    path.parent.mkdir(parents=True)
    path.touch()
    os.utime(path, (timestamp, timestamp))
    return path


def test_latest_model_skips_newer_incompatible_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = tmp_path / "runs" / "game"
    compatible = _checkpoint(model_directory, "v2", 100)
    incompatible = _checkpoint(model_directory, "v1-newer", 200)
    shapes = {compatible: (16,), incompatible: (15,)}

    monkeypatch.setattr(
        game.DQN,
        "load",
        lambda path, device: SimpleNamespace(
            observation_space=SimpleNamespace(shape=shapes[Path(path)])
        ),
    )

    assert game._latest_model(model_directory, required_shape=(16,)) == compatible


def test_latest_model_explains_when_retraining_is_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = tmp_path / "runs" / "game"
    _checkpoint(model_directory, "v1", 100)

    monkeypatch.setattr(
        game.DQN,
        "load",
        lambda path, device: SimpleNamespace(
            observation_space=SimpleNamespace(shape=(15,))
        ),
    )

    with pytest.raises(SystemExit, match="Train one with"):
        game._latest_model(model_directory, required_shape=(16,))
