from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_driving_rl.game_env import IDLE
from tools.export_web_replays import (
    MANIFEST_SCHEMA,
    SCHEMA,
    export_replay,
    sha256_file,
    validate_replay,
    write_json,
)


class HoldPolicy:
    hud_data = {
        "driving_intent": "CRUISE",
        "desired_speed": 22.0,
        "speed_reason": "test policy",
    }

    def reset(self) -> None:
        pass

    def __call__(self, _observation: object) -> int:
        return IDLE


def _small_replay(seed: int = 91) -> dict:
    return export_replay(
        HoldPolicy(),
        seed=seed,
        commit="a" * 40,
        model_provenance={
            "base": {"path": "base.zip", "sha256": "b" * 64},
            "override": {"path": "override.pt", "sha256": "c" * 64},
        },
        episode_seconds=5.0,
    )


def test_export_is_byte_deterministic_and_schema_valid(tmp_path: Path) -> None:
    first = _small_replay()
    second = _small_replay()
    validate_replay(first)
    validate_replay(second)

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_json(first_path, first, compact=True)
    write_json(second_path, second, compact=True)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["schema"] == SCHEMA
    assert first["meta"]["dynamic_traffic"] is True
    assert first["meta"]["sample_hz"] == 10
    assert len(first["frames"]) == int(first["final"]["elapsed_seconds"] * 10) + 1
    assert first["fields"]["c"][0] == "stable_id"


def test_export_keeps_stable_traffic_ids_and_finite_json() -> None:
    replay = _small_replay(seed=92)
    expected = [car[0] for car in replay["frames"][0]["c"]]

    assert expected == list(range(16))
    assert all([car[0] for car in frame["c"]] == expected for frame in replay["frames"])
    assert "Infinity" not in json.dumps(replay)


def test_validator_rejects_changed_traffic_identity() -> None:
    replay = _small_replay()
    replay["frames"][-1]["c"][0][0] = 999

    with pytest.raises(ValueError, match="Traffic IDs changed"):
        validate_replay(replay)


def test_checked_in_replays_match_manifest_and_provenance() -> None:
    data_dir = Path(__file__).parents[1] / "web" / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["replay_schema"] == SCHEMA
    assert manifest["provenance"]["driver_stack"] == "Frozen v1.0 layered driver"
    assert [entry["seed"] for entry in manifest["replays"]] == [470_000, 470_001, 470_002]
    assert manifest["provenance"]["configuration"] == {
        "difficulty": "hard",
        "dynamic_traffic": True,
        "duration_seconds": 45.0,
        "sample_hz": 10,
    }

    for entry in manifest["replays"]:
        replay_path = data_dir / entry["file"]
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        validate_replay(replay)
        assert replay["meta"]["driver_stack"] == "Frozen v1.0 layered driver"
        assert replay_path.stat().st_size == entry["bytes"] < 1_000_000
        assert sha256_file(replay_path) == entry["sha256"]
        assert len(replay["frames"]) == entry["frames"] == 451
        assert replay["final"] == entry["final"]


def test_browser_copy_and_scrub_state_are_replay_truthful() -> None:
    root = Path(__file__).parents[1]
    html = (root / "web" / "index.html").read_text(encoding="utf-8")
    javascript = (root / "web" / "app.js").read_text(encoding="utf-8")
    scrub_handler = javascript.split('ui.scrub.addEventListener("input", () => {', 1)[1]
    scrub_handler = scrub_handler.split("});", 1)[0]

    assert "Live telemetry" not in html
    assert "Replay telemetry" in html
    assert "Frozen V7.1" not in html
    assert "Frozen v1.0 layered driver" in html
    assert "updatePlayState();" in scrub_handler
