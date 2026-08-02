"""Live Hugging Face Space for the frozen Self-Driving RL v1.0 driver."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

from self_driving_rl.game_env import ACTION_NAMES, NeonHighwayEnv
from self_driving_rl.longitudinal import LongitudinalIntentPolicy
from self_driving_rl.rlaif import load_override_policy

MODEL_REPOSITORY = os.getenv("SDR_MODEL_REPO", "slicedonions/self-driving-rl-v1")
BASE_SHA256 = "5780BBEE5CE2009459F3AA796AA4982FBF33222DCC182883D31AFAA16C597039"
OVERRIDE_SHA256 = "06C3A0CEE04AAF6B8822781FE78F867F513A3019EBBF7FD8D91E10F117146BEC"
MAX_SESSIONS = max(1, int(os.getenv("SDR_MAX_SESSIONS", "6")))
VALID_RATES = {0.5, 1.0, 2.0}
STATIC_DIR = Path(__file__).with_name("static")
_artifact_lock = threading.Lock()
_session_lock = asyncio.Lock()
_active_sessions = 0

app = FastAPI(
    title="Self-Driving RL Live",
    version="1.0.0",
    description="Live simulator-only inference for the frozen v1.0 layered driver.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _round(value: Any, digits: int = 3) -> float:
    return round(float(value), digits)


def _finite_ttc(value: Any) -> float:
    return min(float(value), 999.0)


def _random_seed() -> int:
    return 100_000 + secrets.randbelow(900_000)


@lru_cache(maxsize=1)
def resolve_artifacts() -> tuple[Path, Path]:
    """Resolve explicit local paths or immutable public Hub artifacts once."""
    local_base = os.getenv("SDR_BASE_MODEL")
    local_override = os.getenv("SDR_OVERRIDE_MODEL")
    if bool(local_base) != bool(local_override):
        raise RuntimeError("Set both SDR_BASE_MODEL and SDR_OVERRIDE_MODEL, or neither.")
    if local_base and local_override:
        base_path = Path(local_base).expanduser().resolve()
        override_path = Path(local_override).expanduser().resolve()
    else:
        with _artifact_lock:
            base_path = Path(
                hf_hub_download(repo_id=MODEL_REPOSITORY, filename="model.zip")
            )
            override_path = Path(
                hf_hub_download(repo_id=MODEL_REPOSITORY, filename="override_model.pt")
            )
    if not base_path.is_file() or not override_path.is_file():
        raise FileNotFoundError("The frozen v1.0 model artifacts are unavailable.")
    return base_path, override_path


def create_policy() -> LongitudinalIntentPolicy:
    base_path, override_path = resolve_artifacts()
    return LongitudinalIntentPolicy(
        load_override_policy(base_path, override_path, device="cpu")
    )


@dataclass
class SessionSettings:
    seed: int
    difficulty: str = "hard"
    dynamic_traffic: bool = True
    rate: float = 1.0
    paused: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "difficulty": self.difficulty,
            "dynamic_traffic": self.dynamic_traffic,
            "rate": self.rate,
            "paused": self.paused,
        }


class LiveDriveSession:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.settings = SessionSettings(seed=_random_seed())
        self.policy: LongitudinalIntentPolicy | None = None
        self.env: NeonHighwayEnv | None = None
        self.observation: Any = None
        self.info: dict[str, Any] = {}
        self.episode_number = 0
        self.restart_countdown = 0.0
        self.pending_reset: dict[str, Any] | None = None
        self.closed = False
        self._car_ids: dict[int, int] = {}
        self._next_car_id = 0
        self._receiver: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()

    async def send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_text(
                json.dumps(payload, separators=(",", ":"), allow_nan=False)
            )

    def _car_id(self, car: Any) -> int:
        identity = id(car)
        if identity not in self._car_ids:
            self._car_ids[identity] = self._next_car_id
            self._next_car_id += 1
        return self._car_ids[identity]

    def _frame(self, inference_ms: float) -> dict[str, Any]:
        assert self.env is not None and self.policy is not None
        env = self.env
        hud = self.policy.hud_data
        sensors = [
            [_round(ahead), _round(ahead_rel), _round(behind), _round(behind_rel)]
            for ahead, ahead_rel, behind, behind_rel in env.lane_sensors()
        ]
        cars = [
            [
                self._car_id(car),
                _round(car.position - env.ego_position),
                _round(env.traffic_lateral_position(car)),
                _round(car.speed),
                int(car.braking),
                int(car.color_index),
                int(car.style),
            ]
            for car in env.traffic
        ]
        cars.sort(key=lambda item: item[0])
        info = self.info
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
            "c": cars,
            "s": sensors,
            "t": {
                "action_index": int(env.last_action),
                "action": ACTION_NAMES[int(env.last_action)],
                "episode_return": _round(env.episode_return),
                "intent": str(hud.get("driving_intent", "CRUISE")),
                "desired_speed": _round(hud.get("desired_speed", env.target_speed)),
                "reason": str(hud.get("speed_reason", "open-road cruise")),
                "braking_mode": str(hud.get("braking_mode", "COAST")),
                "preference_decision": str(hud.get("preference_decision", "V5 BASE")),
                "lane_intervened": bool(hud.get("lane_intervened", False)),
                "lane_veto_reason": str(hud.get("lane_veto_reason", "")),
                "challenge": str(info.get("challenge", env.challenge_name)),
                "challenge_active": bool(
                    info.get("challenge_active", env.challenge_active)
                ),
                "overtakes": int(info.get("overtakes", env.overtakes)),
                "passed_by_traffic": int(
                    info.get("passed_by_traffic", env.passed_by_traffic)
                ),
                "lane_changes": int(info.get("lane_changes", env.lane_changes)),
                "near_misses": int(info.get("near_misses", env.near_misses)),
                "traffic_lane_changes": int(
                    info.get("traffic_lane_changes", env.traffic_lane_changes)
                ),
                "ttc": _round(_finite_ttc(info.get("ttc", 999.0))),
                "rear_ttc": _round(_finite_ttc(info.get("rear_ttc", 999.0))),
                "threat": str(info.get("threat_level", "clear")),
                "elapsed_seconds": _round(env.elapsed_seconds, 1),
                "inference_ms": _round(inference_ms),
            },
        }

    async def reset(self, requested: dict[str, Any] | None = None) -> None:
        requested = requested or {}
        seed = requested.get("seed", self.settings.seed)
        if seed is None:
            seed = _random_seed()
        seed = max(0, min(int(seed), 2_147_483_647))
        difficulty = str(requested.get("difficulty", self.settings.difficulty))
        if difficulty not in NeonHighwayEnv.DIFFICULTY_MODES:
            difficulty = "hard"
        dynamic = requested.get("dynamic_traffic", self.settings.dynamic_traffic)
        self.settings.seed = seed
        self.settings.difficulty = difficulty
        self.settings.dynamic_traffic = bool(dynamic)
        if self.env is not None:
            self.env.close()
        self.env = NeonHighwayEnv(
            difficulty_mode=difficulty,
            dynamic_traffic=self.settings.dynamic_traffic,
            endless=True,
        )
        self.observation, self.info = self.env.reset(seed=seed)
        assert self.policy is not None
        self.policy.reset()
        self._car_ids.clear()
        self._next_car_id = 0
        self.episode_number += 1
        self.restart_countdown = 0.0

    def apply_control(self, message: dict[str, Any]) -> None:
        command = message.get("command")
        if command == "pause":
            self.settings.paused = True
        elif command == "resume":
            self.settings.paused = False
        elif command == "rate":
            rate = float(message.get("value", 1.0))
            self.settings.rate = rate if rate in VALID_RATES else 1.0
        elif command == "reset":
            self.pending_reset = {
                "seed": message.get("seed", self.settings.seed),
                "difficulty": message.get("difficulty", self.settings.difficulty),
                "dynamic_traffic": message.get(
                    "dynamic_traffic", self.settings.dynamic_traffic
                ),
            }

    async def receive_controls(self) -> None:
        try:
            while True:
                message = await self.websocket.receive_json()
                if isinstance(message, dict) and message.get("type") == "control":
                    self.apply_control(message)
                    await self.send(
                        {"type": "status", "settings": self.settings.public()}
                    )
        except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
            self.closed = True

    async def run(self) -> None:
        await self.send(
            {"type": "loading", "message": "Loading the frozen v1.0 policy…"}
        )
        load_started = perf_counter()
        self.policy = await asyncio.to_thread(create_policy)
        load_seconds = perf_counter() - load_started
        await self.reset()
        await self.send(
            {
                "type": "hello",
                "schema": 1,
                "driver": "Frozen v1.0 layered driver",
                "environment": NeonHighwayEnv.VERSION,
                "decision_hz": int(round(1 / NeonHighwayEnv.DT)),
                "model_load_seconds": _round(load_seconds),
                "model_repository": MODEL_REPOSITORY,
                "model_hashes": {
                    "base": BASE_SHA256,
                    "override": OVERRIDE_SHA256,
                },
                "road": {
                    "lanes": NeonHighwayEnv.LANES,
                    "lane_width_m": NeonHighwayEnv.LANE_WIDTH,
                    "car_length_m": NeonHighwayEnv.CAR_LENGTH,
                    "car_width_m": NeonHighwayEnv.CAR_WIDTH,
                    "sensor_range_m": NeonHighwayEnv.SENSOR_DISTANCE,
                },
                "settings": self.settings.public(),
            }
        )
        await self.send(
            {
                "type": "state",
                "frame": self._frame(0.0),
                "settings": self.settings.public(),
                "episode": self.episode_number,
            }
        )
        self._receiver = asyncio.create_task(self.receive_controls())

        try:
            while not self.closed:
                loop_started = perf_counter()
                if self.pending_reset is not None:
                    requested, self.pending_reset = self.pending_reset, None
                    await self.reset(requested)
                    await self.send(
                        {
                            "type": "state",
                            "frame": self._frame(0.0),
                            "settings": self.settings.public(),
                            "episode": self.episode_number,
                        }
                    )

                if self.settings.paused:
                    await asyncio.sleep(0.05)
                    continue

                if self.restart_countdown > 0:
                    self.restart_countdown -= 0.05
                    if self.restart_countdown <= 0:
                        await self.reset({"seed": self.settings.seed + 1})
                    await asyncio.sleep(0.05)
                    continue

                assert self.policy is not None and self.env is not None
                inference_started = perf_counter()
                action = int(self.policy(self.observation))
                inference_ms = (perf_counter() - inference_started) * 1000
                (
                    self.observation,
                    _,
                    terminated,
                    truncated,
                    self.info,
                ) = self.env.step(action)
                await self.send(
                    {
                        "type": "state",
                        "frame": self._frame(inference_ms),
                        "settings": self.settings.public(),
                        "episode": self.episode_number,
                    }
                )
                if terminated or truncated:
                    outcome = "crashed" if self.info.get("crashed") else "timed_out"
                    await self.send(
                        {
                            "type": "outcome",
                            "outcome": outcome,
                            "seed": self.settings.seed,
                            "elapsed_seconds": _round(self.env.elapsed_seconds, 1),
                            "distance_km": _round(self.env.ego_position / 1000),
                            "net_overtakes": int(
                                self.info.get(
                                    "net_overtakes",
                                    self.env.overtakes - self.env.passed_by_traffic,
                                )
                            ),
                            "collision": self.info.get("collision"),
                            "next_seed": self.settings.seed + 1,
                        }
                    )
                    self.restart_countdown = 1.5

                target_interval = NeonHighwayEnv.DT / self.settings.rate
                await asyncio.sleep(max(0.0, target_interval - (perf_counter() - loop_started)))
        finally:
            self.closed = True
            if self._receiver is not None:
                self._receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._receiver
            if self.env is not None:
                self.env.close()


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "driver": "frozen-v1.0-layered",
            "model_repository": MODEL_REPOSITORY,
            "active_sessions": _active_sessions,
            "max_sessions": MAX_SESSIONS,
        }
    )


@app.websocket("/ws")
async def live_drive(websocket: WebSocket) -> None:
    global _active_sessions
    await websocket.accept()
    async with _session_lock:
        if _active_sessions >= MAX_SESSIONS:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "busy",
                    "message": "The live demo is full. Please try again in a moment.",
                }
            )
            await websocket.close(code=1013)
            return
        _active_sessions += 1
    try:
        await LiveDriveSession(websocket).run()
    except WebSocketDisconnect:
        pass
    except Exception as error:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "server_error",
                    "message": f"The live driver stopped: {type(error).__name__}",
                }
            )
    finally:
        async with _session_lock:
            _active_sessions = max(0, _active_sessions - 1)
