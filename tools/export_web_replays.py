"""Export compact deterministic replays from the Frozen v1.0 layered driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from self_driving_rl.game_env import ACTION_NAMES, NeonHighwayEnv
from self_driving_rl.longitudinal import LongitudinalIntentPolicy
from self_driving_rl.rlaif import load_override_policy

SCHEMA = "self-driving-rl/replay@1"
MANIFEST_SCHEMA = "self-driving-rl/replay-manifest@1"
DEFAULT_SEEDS = (470_000, 470_001, 470_002)
DEFAULT_BASE_MODEL = Path("runs/game/v5-good-driver-2p5m-restart/model.zip")
DEFAULT_OVERRIDE_MODEL = Path("runs/rlaif/v6-good-driver/override_model.pt")
MODEL_HASHES = {
    "base": "5780BBEE5CE2009459F3AA796AA4982FBF33222DCC182883D31AFAA16C597039",
    "override": "06C3A0CEE04AAF6B8822781FE78F867F513A3019EBBF7FD8D91E10F117146BEC",
}


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_models(base_model: Path, override_model: Path) -> dict[str, dict[str, str]]:
    models = {
        "base": (base_model, MODEL_HASHES["base"]),
        "override": (override_model, MODEL_HASHES["override"]),
    }
    provenance: dict[str, dict[str, str]] = {}
    for name, (path, expected_hash) in models.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} model: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{name} model hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        provenance[name] = {"path": path.as_posix(), "sha256": actual_hash}
    return provenance


def _traffic_frame(env: NeonHighwayEnv, car_ids: dict[int, int]) -> list[list[Any]]:
    result: list[list[Any]] = []
    for car in env.traffic:
        result.append(
            [
                car_ids[id(car)],
                _round(car.position - env.ego_position),
                _round(env.traffic_lateral_position(car)),
                _round(car.speed),
                int(car.braking),
                int(car.color_index),
                int(car.style),
            ]
        )
    result.sort(key=lambda item: item[0])
    return result


def _finite_ttc(value: Any) -> float:
    number = float(value)
    return min(number, 999.0)


def capture_frame(
    env: NeonHighwayEnv,
    policy: Any,
    info: dict[str, Any],
    car_ids: dict[int, int],
) -> dict[str, Any]:
    hud = getattr(policy, "hud_data", {})
    sensors = [
        [_round(ahead), _round(ahead_rel), _round(behind), _round(behind_rel)]
        for ahead, ahead_rel, behind, behind_rel in env.lane_sensors()
    ]
    return {
        "e": [
            _round(env.ego_position),
            _round(env.lane_position),
            int(env.target_lane),
            _round(env.ego_speed),
            _round(env.target_speed),
            _round(env.longitudinal_acceleration),
            _round(env.throttle),
            _round(env.brake),
        ],
        "c": _traffic_frame(env, car_ids),
        "s": sensors,
        "x": [
            int(env.last_action),
            _round(env.episode_return),
            str(hud.get("driving_intent", "CRUISE")),
            _round(hud.get("desired_speed", env.target_speed)),
            str(hud.get("speed_reason", "open-road cruise")),
            str(info.get("challenge", env.challenge_name)),
            int(bool(info.get("challenge_active", env.challenge_active))),
            int(info.get("overtakes", env.overtakes)),
            int(info.get("passed_by_traffic", env.passed_by_traffic)),
            int(info.get("lane_changes", env.lane_changes)),
            int(info.get("near_misses", env.near_misses)),
            _round(_finite_ttc(info.get("ttc", 999.0))),
            _round(_finite_ttc(info.get("rear_ttc", 999.0))),
            str(info.get("threat_level", "clear")),
        ],
    }


def final_metrics(
    env: NeonHighwayEnv,
    info: dict[str, Any],
    speeds: list[float],
    minimum_ttc: float,
    minimum_rear_ttc: float,
    action_counts: Counter[int],
) -> dict[str, Any]:
    return {
        "outcome": (
            "completed"
            if info.get("completed")
            else "crashed"
            if info.get("crashed")
            else "timed_out"
        ),
        "elapsed_seconds": _round(info.get("elapsed_seconds", env.elapsed_seconds), 1),
        "episode_return": _round(info.get("episode_return", env.episode_return)),
        "distance_km": _round(float(info.get("distance_m", env.ego_position)) / 1000),
        "mean_speed_kmh": _round(sum(speeds) / max(len(speeds), 1) * 3.6, 1),
        "overtakes": int(info.get("overtakes", env.overtakes)),
        "passed_by_traffic": int(info.get("passed_by_traffic", env.passed_by_traffic)),
        "net_overtakes": int(info.get("net_overtakes", env.overtakes - env.passed_by_traffic)),
        "lane_changes": int(info.get("lane_changes", env.lane_changes)),
        "unsafe_lane_changes": int(
            info.get("unsafe_lane_changes", env.unsafe_lane_changes)
        ),
        "near_misses": int(info.get("near_misses", env.near_misses)),
        "traffic_lane_changes": int(
            info.get("traffic_lane_changes", env.traffic_lane_changes)
        ),
        "traffic_cut_ins": int(info.get("traffic_cut_ins", env.traffic_cut_ins)),
        "challenges_resolved": int(
            info.get("challenges_resolved", env.challenges_resolved)
        ),
        "challenges_presented": int(
            info.get("challenges_presented", env.challenges_presented)
        ),
        "minimum_ttc": _round(minimum_ttc),
        "minimum_rear_ttc": _round(minimum_rear_ttc),
        "action_counts": {
            ACTION_NAMES[action]: action_counts[action] for action in sorted(action_counts)
        },
        "collision": info.get("collision"),
    }


def export_replay(
    policy: Any,
    *,
    seed: int,
    commit: str,
    model_provenance: dict[str, dict[str, str]],
    episode_seconds: float = 45.0,
) -> dict[str, Any]:
    env = NeonHighwayEnv(
        difficulty_mode="hard",
        dynamic_traffic=True,
        episode_seconds=episode_seconds,
    )
    try:
        observation, info = env.reset(seed=seed)
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        car_ids = {id(car): index for index, car in enumerate(env.traffic)}
        frames = [capture_frame(env, policy, info, car_ids)]
        speeds: list[float] = []
        action_counts: Counter[int] = Counter()
        minimum_ttc = 999.0
        minimum_rear_ttc = 999.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(policy(observation))
            observation, _, terminated, truncated, info = env.step(action)
            action_counts[action] += 1
            speeds.append(float(info["speed"]))
            minimum_ttc = min(minimum_ttc, _finite_ttc(info["ttc"]))
            minimum_rear_ttc = min(minimum_rear_ttc, _finite_ttc(info["rear_ttc"]))
            frames.append(capture_frame(env, policy, info, car_ids))

        return {
            "schema": SCHEMA,
            "meta": {
                "seed": seed,
                "sample_hz": int(round(1.0 / env.DT)),
                "driver_stack": "Frozen v1.0 layered driver",
                "difficulty": "hard",
                "dynamic_traffic": True,
                "requested_duration_seconds": episode_seconds,
                "source_commit": commit,
                "models": model_provenance,
                "road": {
                    "lanes": env.LANES,
                    "lane_width_m": env.LANE_WIDTH,
                    "car_length_m": env.CAR_LENGTH,
                    "car_width_m": env.CAR_WIDTH,
                    "sensor_range_m": env.SENSOR_DISTANCE,
                },
            },
            "fields": {
                "e": [
                    "distance_m",
                    "lane",
                    "target_lane",
                    "speed_mps",
                    "target_speed_mps",
                    "acceleration_mps2",
                    "throttle",
                    "brake",
                ],
                "c": [
                    "stable_id",
                    "relative_distance_m",
                    "lane",
                    "speed_mps",
                    "braking",
                    "color_index",
                    "style",
                ],
                "s": [
                    "ahead_gap_m",
                    "ahead_relative_speed_mps",
                    "behind_gap_m",
                    "behind_relative_speed_mps",
                ],
                "x": [
                    "action_id",
                    "episode_return",
                    "driving_intent",
                    "desired_speed_mps",
                    "intent_reason",
                    "challenge",
                    "challenge_active",
                    "overtakes",
                    "passed_by_traffic",
                    "lane_changes",
                    "near_misses",
                    "ttc_seconds",
                    "rear_ttc_seconds",
                    "threat_level",
                ],
            },
            "action_names": {str(key): value for key, value in ACTION_NAMES.items()},
            "frames": frames,
            "final": final_metrics(
                env,
                info,
                speeds,
                minimum_ttc,
                minimum_rear_ttc,
                action_counts,
            ),
        }
    finally:
        env.close()


def validate_replay(replay: dict[str, Any]) -> None:
    if replay.get("schema") != SCHEMA:
        raise ValueError("Unsupported replay schema")
    meta = replay.get("meta", {})
    if meta.get("sample_hz") != 10 or meta.get("difficulty") != "hard":
        raise ValueError("Replay must be a 10 Hz hard-mode recording")
    if meta.get("dynamic_traffic") is not True:
        raise ValueError("Replay must use dynamic traffic")
    frames = replay.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("Replay has no usable frames")
    expected_cars = len(frames[0].get("c", []))
    stable_ids = [car[0] for car in frames[0]["c"]]
    if expected_cars == 0 or stable_ids != list(range(expected_cars)):
        raise ValueError("Traffic IDs are not stable contiguous slots")
    for frame in frames:
        if len(frame.get("e", [])) != 8 or len(frame.get("x", [])) != 14:
            raise ValueError("Malformed frame telemetry")
        if len(frame.get("s", [])) != int(meta["road"]["lanes"]):
            raise ValueError("Malformed lane sensors")
        if [car[0] for car in frame.get("c", [])] != stable_ids:
            raise ValueError("Traffic IDs changed during replay")


def write_json(path: Path, payload: dict[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if compact
        else json.dumps(payload, ensure_ascii=False, indent=2)
    )
    path.write_text(rendered + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--override-model", type=Path, default=DEFAULT_OVERRIDE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("web/data"))
    parser.add_argument("--seed", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--source-commit", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    commit = args.source_commit or source_commit()
    models = verify_models(args.base_model, args.override_model)
    policy = LongitudinalIntentPolicy(
        load_override_policy(args.base_model, args.override_model, device="cpu")
    )
    manifest_entries: list[dict[str, Any]] = []
    for seed in args.seed:
        replay = export_replay(
            policy,
            seed=seed,
            commit=commit,
            model_provenance=models,
        )
        validate_replay(replay)
        filename = f"replay-{seed}.json"
        path = args.output_dir / filename
        write_json(path, replay, compact=True)
        manifest_entries.append(
            {
                "seed": seed,
                "file": filename,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "frames": len(replay["frames"]),
                "final": replay["final"],
            }
        )
        print(f"{filename}: {path.stat().st_size:,} bytes")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "replay_schema": SCHEMA,
        "provenance": {
            "description": (
                "Generated offline by the real frozen Python policy; deterministic playback, "
                "not live browser inference."
            ),
            "driver_stack": "Frozen v1.0 layered driver",
            "generator": "tools/export_web_replays.py",
            "source_commit": commit,
            "models": models,
            "configuration": {
                "difficulty": "hard",
                "dynamic_traffic": True,
                "duration_seconds": 45.0,
                "sample_hz": 10,
            },
        },
        "replays": manifest_entries,
    }
    write_json(args.output_dir / "manifest.json", manifest, compact=False)


if __name__ == "__main__":
    main()
