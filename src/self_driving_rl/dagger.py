"""Human-in-the-loop DAgger for the Neon Highway driving policy.

The human teaches high-level lane and speed intent while the V7 longitudinal
planner continues to own throttle and braking. A small confidence-gated residual
learns when to replace LEFT / KEEP / RIGHT and when the road permits faster or
slower progress. Autonomous lane overrides retain an independent merge shield;
human teaching commands are authoritative inside the simulator.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from numpy.typing import NDArray
from torch import nn

from self_driving_rl.game import run_policy
from self_driving_rl.game_env import (
    ACTION_NAMES,
    STEER_KEEP,
    STEER_LEFT,
    STEER_RIGHT,
    NeonHighwayEnv,
    decode_action,
    encode_action,
)
from self_driving_rl.longitudinal import (
    LongitudinalIntentPolicy,
    SpeedGuidance,
    observed_lane_reading,
)
from self_driving_rl.metrics import evaluate_in_env
from self_driving_rl.rlaif import load_override_policy

SCHEMA_VERSION = 2
STEER_NAMES = {STEER_LEFT: "LEFT", STEER_KEEP: "KEEP", STEER_RIGHT: "RIGHT"}
SPEED_GUIDANCE_NAMES = {
    SpeedGuidance.BASE: "BASE SPEED",
    SpeedGuidance.FASTER: "FASTER",
    SpeedGuidance.SLOWER: "SLOWER",
}
DEFAULT_BASE_MODEL = Path("runs/game/v5-good-driver-2p5m-restart/model.zip")
DEFAULT_OVERRIDE_MODEL = Path("runs/rlaif/v6-good-driver/override_model.pt")
DEFAULT_DATASET = Path("runs/dagger/human-v1/demonstrations.npz")
DEFAULT_DAGGER_MODEL = Path("runs/dagger/human-v1/dagger_model.pt")


class DaggerCorrectionNet(nn.Module):
    """Predict whether to intervene and which steering decision the human wants."""

    def __init__(self, observation_size: int) -> None:
        super().__init__()
        self.observation_size = int(observation_size)
        self.trunk = nn.Sequential(
            nn.Linear(self.observation_size + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.intervention_head = nn.Linear(64, 2)
        self.steer_head = nn.Linear(64, 3)
        self.speed_intervention_head = nn.Linear(64, 2)
        self.speed_head = nn.Linear(64, 3)

    def forward(
        self,
        observations: th.Tensor,
        base_steers: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        one_hot = th.nn.functional.one_hot(base_steers.long(), num_classes=3).to(
            dtype=observations.dtype
        )
        hidden = self.trunk(th.cat((observations, one_hot), dim=1))
        return (
            self.intervention_head(hidden),
            self.steer_head(hidden),
            self.speed_intervention_head(hidden),
            self.speed_head(hidden),
        )


def projected_merge_is_safe(
    observation: NDArray[np.floating[Any]], steer: int
) -> bool:
    """Apply the V7 projected-gap rule to an outer DAgger lane correction."""
    if steer == STEER_KEEP:
        return True
    if float(observation[4]) >= 0.05:
        return False

    direction = -1 if steer == STEER_LEFT else 1
    target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
    candidate_lane = target_lane + direction
    if not 0 <= candidate_lane < NeonHighwayEnv.LANES:
        return False

    ahead_gap, ahead_relative, _, behind_gap, behind_relative, _ = (
        observed_lane_reading(observation, candidate_lane)
    )
    projected_ahead = ahead_gap + min(ahead_relative, 0.0) * 0.6
    projected_behind = behind_gap - max(behind_relative, 0.0) * 0.6
    rear_closing = max(behind_relative, 0.0)
    rear_ttc = behind_gap / rear_closing if rear_closing > 0.1 else float("inf")
    if not (
        ahead_gap > NeonHighwayEnv.SAFE_LANE_CHANGE_FRONT_GAP + 3.0
        and projected_ahead > NeonHighwayEnv.SAFE_LANE_CHANGE_FRONT_GAP
        and behind_gap > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_GAP + 3.0
        and projected_behind > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_GAP
        and rear_ttc > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_TTC
    ):
        return False

    origin_gap, origin_relative, _, _, _, _ = observed_lane_reading(
        observation, target_lane
    )
    projected_origin = origin_gap + min(origin_relative, 0.0) * 0.3
    return projected_origin > NeonHighwayEnv.CAR_LENGTH + 1.5


def teacher_lane_command_is_available(
    observation: NDArray[np.floating[Any]], steer: int
) -> tuple[bool, str]:
    """Reject only lane commands the simulator physically cannot start now."""
    if steer == STEER_KEEP:
        return True, ""
    if float(observation[4]) >= 0.05:
        return False, "finish the current lane change first"

    direction = -1 if steer == STEER_LEFT else 1
    target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
    candidate_lane = target_lane + direction
    if not 0 <= candidate_lane < NeonHighwayEnv.LANES:
        return False, "road boundary"
    return True, ""


def apply_dagger_gate(
    observations: NDArray[np.float32],
    base_actions: NDArray[np.int64],
    intervention_probabilities: NDArray[np.float32],
    proposed_steers: NDArray[np.int64],
    threshold: float,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Apply a confidence gate and the independent projected-gap shield."""
    final_actions = np.asarray(base_actions, dtype=np.int64).copy()
    applied = np.zeros(len(final_actions), dtype=bool)
    for index, (observation, action, probability, steer) in enumerate(
        zip(
            observations,
            base_actions,
            intervention_probabilities,
            proposed_steers,
            strict=True,
        )
    ):
        base_steer, pedal = decode_action(int(action))
        if (
            float(probability) >= threshold
            and int(steer) != base_steer
            and projected_merge_is_safe(observation, int(steer))
        ):
            final_actions[index] = encode_action(int(steer), pedal)
            applied[index] = True
    return final_actions, applied


def speed_guidance_is_safe(
    observation: NDArray[np.floating[Any]], guidance: SpeedGuidance | int
) -> bool:
    """Whether faster guidance can be acted on immediately, not whether it is valid."""
    selected = SpeedGuidance(int(guidance))
    if selected != SpeedGuidance.FASTER:
        return True
    lane_position = float(observation[2]) * (NeonHighwayEnv.LANES - 1)
    target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
    relevant_lanes = {
        lane
        for lane in range(NeonHighwayEnv.LANES)
        if abs(lane_position - lane) <= NeonHighwayEnv.LANE_COLLISION_WIDTH
    }
    relevant_lanes.add(target_lane)
    front_urgency = max(
        observed_lane_reading(observation, lane)[2] for lane in relevant_lanes
    )
    return front_urgency < 0.42


def apply_speed_gate(
    observations: NDArray[np.float32],
    intervention_probabilities: NDArray[np.float32],
    proposed_guidance: NDArray[np.int64],
    threshold: float,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    final = np.full(len(observations), int(SpeedGuidance.BASE), dtype=np.int64)
    applied = np.zeros(len(observations), dtype=bool)
    for index, (_observation, probability, guidance) in enumerate(
        zip(
            observations,
            intervention_probabilities,
            proposed_guidance,
            strict=True,
        )
    ):
        # This is persistent high-level guidance rather than a raw pedal
        # command. The longitudinal planner still owns following, braking, and
        # emergency response, so rejecting FASTER here only throws away useful
        # human intent and duplicates the planner's safety logic.
        if float(probability) >= threshold and int(guidance) != int(
            SpeedGuidance.BASE
        ):
            final[index] = int(guidance)
            applied[index] = True
    return final, applied


class DaggerCorrectionPolicy:
    """Frozen V7 driver plus a confidence-gated, safety-shielded human residual."""

    def __init__(
        self,
        base_policy: Any,
        network: DaggerCorrectionNet,
        lane_threshold: float,
        speed_threshold: float,
        *,
        device: str | th.device = "cpu",
        lane_residual_enabled: bool = True,
        faster_only: bool = False,
    ) -> None:
        self.base_policy = base_policy
        self.network = network
        self.lane_threshold = float(lane_threshold)
        self.speed_threshold = float(speed_threshold)
        self.device = th.device(device)
        self.lane_residual_enabled = bool(lane_residual_enabled)
        self.faster_only = bool(faster_only)
        self.network.to(self.device).eval()
        self.total_overrides = 0
        self.total_rejected = 0
        self.total_deferred = 0
        self.total_speed_overrides = 0
        self.total_speed_rejected = 0
        self.total_speed_deferred = 0
        self._steps_since_override = NeonHighwayEnv.RAPID_LANE_CHANGE_STEPS
        self._speed_guidance_steps_remaining = 0
        self.last_speed_guidance = SpeedGuidance.BASE
        self.last_base_action = encode_action(STEER_KEEP, 1)
        self.last_action = self.last_base_action
        self.last_decision = "V7 BASE"
        self.last_speed_decision = "V7 SPEED BASE"

    def reset(self) -> None:
        reset_policy = getattr(self.base_policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        self.last_decision = "V7 BASE"
        self.last_speed_decision = "V7 SPEED BASE"
        self._steps_since_override = NeonHighwayEnv.RAPID_LANE_CHANGE_STEPS
        self._speed_guidance_steps_remaining = 0
        self.last_speed_guidance = SpeedGuidance.BASE

    @property
    def hud_data(self) -> dict[str, Any]:
        nested = getattr(self.base_policy, "hud_data", {})
        data = dict(nested) if isinstance(nested, dict) else {}
        data.update(
            {
                "dagger_decision": self.last_decision,
                "dagger_overrides": self.total_overrides,
                "dagger_rejected": self.total_rejected,
                "dagger_deferred": self.total_deferred,
                "dagger_speed_decision": self.last_speed_decision,
                "dagger_speed_overrides": self.total_speed_overrides,
                "dagger_speed_rejected": self.total_speed_rejected,
                "dagger_speed_deferred": self.total_speed_deferred,
            }
        )
        return data

    def __call__(self, observation: NDArray[np.floating[Any]]) -> int:
        self._steps_since_override += 1
        if self._speed_guidance_steps_remaining > 0:
            self._speed_guidance_steps_remaining -= 1
            if self._speed_guidance_steps_remaining == 0:
                self.last_speed_guidance = SpeedGuidance.BASE
        base_action = int(self.base_policy(observation))
        base_steer, _ = decode_action(base_action)
        with th.no_grad():
            observation_tensor = th.as_tensor(
                np.asarray(observation, dtype=np.float32)[None, :], device=self.device
            )
            steer_tensor = th.as_tensor([base_steer], dtype=th.long, device=self.device)
            (
                intervention_logits,
                steer_logits,
                speed_intervention_logits,
                speed_logits,
            ) = self.network(observation_tensor, steer_tensor)
            probability = float(
                th.softmax(intervention_logits, dim=1)[0, 1].cpu().item()
            )
            proposed_steer = int(steer_logits.argmax(dim=1).cpu().item())
            speed_probability = float(
                th.softmax(speed_intervention_logits, dim=1)[0, 1].cpu().item()
            )
            proposed_speed_guidance = SpeedGuidance(
                int(speed_logits.argmax(dim=1).cpu().item())
            )

        final, applied = apply_dagger_gate(
            np.asarray(observation, dtype=np.float32)[None, :],
            np.asarray([base_action], dtype=np.int64),
            np.asarray([probability], dtype=np.float32),
            np.asarray([proposed_steer], dtype=np.int64),
            self.lane_threshold,
        )
        wanted_override = (
            probability >= self.lane_threshold and proposed_steer != base_steer
        )
        if not self.lane_residual_enabled:
            final[0] = base_action
            applied[0] = False
            wanted_override = False
        repeated_lane_request = (
            bool(applied[0])
            and proposed_steer != STEER_KEEP
            and self._steps_since_override < NeonHighwayEnv.RAPID_LANE_CHANGE_STEPS
        )
        if repeated_lane_request:
            final[0] = base_action
            applied[0] = False
            self.total_deferred += 1
            self.last_decision = "DAGGER HELD FOR CALM"
        elif bool(applied[0]):
            self.total_overrides += 1
            self._steps_since_override = 0
            self.last_decision = f"HUMAN RESIDUAL: {STEER_NAMES[proposed_steer]}"
        elif wanted_override:
            self.total_rejected += 1
            self.last_decision = "DAGGER VETOED BY SAFETY"
        else:
            self.last_decision = "V7 BASE"

        speed_final, speed_applied = apply_speed_gate(
            np.asarray(observation, dtype=np.float32)[None, :],
            np.asarray([speed_probability], dtype=np.float32),
            np.asarray([int(proposed_speed_guidance)], dtype=np.int64),
            self.speed_threshold,
        )
        wanted_speed_override = (
            speed_probability >= self.speed_threshold
            and proposed_speed_guidance != SpeedGuidance.BASE
        )
        if self.faster_only and proposed_speed_guidance != SpeedGuidance.FASTER:
            speed_final[0] = int(SpeedGuidance.BASE)
            speed_applied[0] = False
            wanted_speed_override = False
        speed_cooldown = self._speed_guidance_steps_remaining > 0
        safety_slowdown = (
            proposed_speed_guidance == SpeedGuidance.SLOWER
            and self.last_speed_guidance == SpeedGuidance.FASTER
        )
        if bool(speed_applied[0]) and speed_cooldown and not safety_slowdown:
            self.total_speed_deferred += 1
            self.last_speed_decision = "SPEED GUIDANCE HELD"
        elif bool(speed_applied[0]):
            set_guidance = getattr(self.base_policy, "set_speed_guidance", None)
            if callable(set_guidance):
                speed_span = NeonHighwayEnv.MAX_SPEED - NeonHighwayEnv.MIN_SPEED
                current_speed = (
                    NeonHighwayEnv.MIN_SPEED + float(observation[0]) * speed_span
                )
                selected_guidance = SpeedGuidance(int(speed_final[0]))
                set_guidance(selected_guidance, current_speed=current_speed)
                self.last_speed_guidance = selected_guidance
                self._speed_guidance_steps_remaining = 30
                self.total_speed_overrides += 1
                self.last_speed_decision = (
                    f"HUMAN SPEED: {SPEED_GUIDANCE_NAMES[selected_guidance]}"
                )
            else:
                self.total_speed_rejected += 1
                self.last_speed_decision = "SPEED GUIDANCE UNAVAILABLE"
        elif wanted_speed_override:
            self.total_speed_rejected += 1
            self.last_speed_decision = "SPEED VETOED BY SAFETY"
        else:
            self.last_speed_decision = "V7 SPEED BASE"
        self.last_base_action = base_action
        self.last_action = int(final[0])
        return self.last_action


def _base_driver(base_model: Path, override_model: Path, device: str) -> Any:
    preference_policy = load_override_policy(
        base_model, override_model, device=device
    )
    return LongitudinalIntentPolicy(preference_policy)


def load_dagger_policy(
    base_model: Path,
    override_model: Path,
    dagger_model: Path,
    *,
    device: str = "cpu",
) -> DaggerCorrectionPolicy:
    selected_device = _device(device)
    payload = th.load(dagger_model, map_location=selected_device, weights_only=True)
    network = DaggerCorrectionNet(int(payload["observation_size"]))
    network.load_state_dict(payload["state_dict"])
    return DaggerCorrectionPolicy(
        _base_driver(base_model, override_model, str(selected_device)),
        network,
        float(payload["lane_threshold"]),
        float(payload["speed_threshold"]),
        device=selected_device,
        lane_residual_enabled=bool(payload.get("lane_residual_enabled", True)),
        faster_only=bool(payload.get("faster_only", False)),
    )


DATA_KEYS = (
    "observations",
    "proposed_actions",
    "teacher_steers",
    "lane_labelled",
    "lane_corrections",
    "teacher_speed_guidance",
    "speed_labelled",
    "speed_corrections",
    "seeds",
    "episode_steps",
    "session_ids",
)


def load_dataset(path: Path, observation_size: int = 33) -> dict[str, NDArray[Any]]:
    if not path.exists():
        return {
            "observations": np.empty((0, observation_size), dtype=np.float32),
            "proposed_actions": np.empty(0, dtype=np.int64),
            "teacher_steers": np.empty(0, dtype=np.int64),
            "lane_labelled": np.empty(0, dtype=np.int8),
            "lane_corrections": np.empty(0, dtype=np.int8),
            "teacher_speed_guidance": np.empty(0, dtype=np.int64),
            "speed_labelled": np.empty(0, dtype=np.int8),
            "speed_corrections": np.empty(0, dtype=np.int8),
            "seeds": np.empty(0, dtype=np.int64),
            "episode_steps": np.empty(0, dtype=np.int64),
            "session_ids": np.empty(0, dtype=np.int64),
        }
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["schema_version"][0])
        if version != SCHEMA_VERSION:
            raise SystemExit(
                f"Unsupported DAgger dataset schema {version}; expected {SCHEMA_VERSION}"
            )
        data = {key: np.asarray(archive[key]) for key in DATA_KEYS}
    if data["observations"].ndim != 2:
        raise SystemExit(f"Invalid observation array in {path}")
    if data["observations"].shape[1] != observation_size:
        raise SystemExit(
            f"Dataset observations are shaped {data['observations'].shape[1:]}, but "
            f"this environment expects ({observation_size},)"
        )
    lengths = {len(value) for value in data.values()}
    if len(lengths) != 1:
        raise SystemExit(f"Mismatched array lengths in {path}")
    return data


def save_dataset(path: Path, records: dict[str, list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    observations = np.asarray(records["observations"], dtype=np.float32)
    if observations.size == 0:
        observations = observations.reshape(0, 33)
    payload = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
        "observations": observations,
        "proposed_actions": np.asarray(records["proposed_actions"], dtype=np.int64),
        "teacher_steers": np.asarray(records["teacher_steers"], dtype=np.int64),
        "lane_labelled": np.asarray(records["lane_labelled"], dtype=np.int8),
        "lane_corrections": np.asarray(records["lane_corrections"], dtype=np.int8),
        "teacher_speed_guidance": np.asarray(
            records["teacher_speed_guidance"], dtype=np.int64
        ),
        "speed_labelled": np.asarray(records["speed_labelled"], dtype=np.int8),
        "speed_corrections": np.asarray(
            records["speed_corrections"], dtype=np.int8
        ),
        "seeds": np.asarray(records["seeds"], dtype=np.int64),
        "episode_steps": np.asarray(records["episode_steps"], dtype=np.int64),
        "session_ids": np.asarray(records["session_ids"], dtype=np.int64),
    }
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _records(data: dict[str, NDArray[Any]]) -> dict[str, list[Any]]:
    return {key: list(data[key]) for key in DATA_KEYS}


def _teacher_event(env: NeonHighwayEnv) -> str | None:
    renderer = env.renderer
    pop_event = getattr(renderer, "pop_teacher_event", None)
    return pop_event() if callable(pop_event) else None


def _longitudinal_policy(driver: Any) -> LongitudinalIntentPolicy | None:
    if isinstance(driver, LongitudinalIntentPolicy):
        return driver
    nested = getattr(driver, "base_policy", None)
    return nested if isinstance(nested, LongitudinalIntentPolicy) else None


def collect(args: argparse.Namespace) -> None:
    """Collect sparse human corrections on states visited by the current policy."""
    driver: Any = _base_driver(args.base_model, args.override_model, args.device)
    if args.dagger_model is not None:
        driver = load_dagger_policy(
            args.base_model,
            args.override_model,
            args.dagger_model,
            device=args.device,
        )
    env = NeonHighwayEnv(
        render_mode="human",
        render_fps=args.fps,
        render_speed=args.speed,
        difficulty_mode=args.difficulty,
        episode_seconds=args.seconds,
        endless=args.endless,
        dynamic_traffic=args.dynamic_traffic,
    )
    data = load_dataset(args.dataset, env.observation_space.shape[0])
    records = _records(data)
    session_id = (
        int(np.max(data["session_ids"])) + 1 if len(data["session_ids"]) else 0
    )
    session_start = len(records["observations"])
    completed = 0
    last_label = "waiting for your first label"
    print("Human DAgger collection started.")
    print("  A/LEFT = left, K = keep lane, D/RIGHT = right")
    print("  W/UP = faster progress, S/DOWN = slower progress")
    print("  ENTER = explicitly approve, U/BACKSPACE = undo, ESC = save and exit")
    print("  Only deliberate key presses become labels; silence records nothing.\n")
    try:
        while completed < args.episodes and not env.quit_requested:
            episode_seed = args.seed + completed
            observation, _ = env.reset(seed=episode_seed)
            reset = getattr(driver, "reset", None)
            if callable(reset):
                reset()
            terminated = truncated = False
            while not (terminated or truncated) and not env.quit_requested:
                proposed_action = int(driver(observation))
                network_base_action = int(
                    getattr(driver, "last_base_action", proposed_action)
                )
                event = _teacher_event(env)
                executed_action = proposed_action

                if event == "undo":
                    if len(records["observations"]) > session_start:
                        for key in DATA_KEYS:
                            records[key].pop()
                        save_dataset(args.dataset, records)
                        last_label = "undid previous label"
                    else:
                        last_label = "nothing from this session to undo"
                elif event is not None:
                    proposed_steer, proposed_pedal = decode_action(proposed_action)
                    base_steer, _ = decode_action(network_base_action)
                    teacher_steer = base_steer
                    teacher_speed = SpeedGuidance.BASE
                    lane_labelled = event in {"approve", "left", "keep", "right"}
                    speed_labelled = event in {"approve", "faster", "slower"}
                    if event == "approve":
                        teacher_steer = proposed_steer
                        teacher_speed = SpeedGuidance(
                            int(
                                getattr(
                                    driver,
                                    "last_speed_guidance",
                                    getattr(
                                        _longitudinal_policy(driver),
                                        "speed_guidance",
                                        SpeedGuidance.BASE,
                                    ),
                                )
                            )
                        )
                        label_name = "approved current lane and speed plan"
                    elif lane_labelled:
                        teacher_steer = {
                            "left": STEER_LEFT,
                            "keep": STEER_KEEP,
                            "right": STEER_RIGHT,
                        }[event]
                        label_name = f"taught {STEER_NAMES[teacher_steer]}"
                    else:
                        teacher_speed = {
                            "faster": SpeedGuidance.FASTER,
                            "slower": SpeedGuidance.SLOWER,
                        }[event]
                        label_name = f"taught {SPEED_GUIDANCE_NAMES[teacher_speed]}"

                    lane_available, unavailable_reason = (
                        teacher_lane_command_is_available(
                            observation, teacher_steer
                        )
                        if lane_labelled
                        else (True, "")
                    )
                    if lane_available:
                        merge_outside_autonomous_envelope = (
                            lane_labelled
                            and not projected_merge_is_safe(
                                observation, teacher_steer
                            )
                        )
                        faster_waits_for_gap = (
                            speed_labelled
                            and teacher_speed == SpeedGuidance.FASTER
                            and not speed_guidance_is_safe(
                                observation, teacher_speed
                            )
                        )
                        if lane_labelled:
                            executed_action = encode_action(
                                teacher_steer, proposed_pedal
                            )
                        if speed_labelled and teacher_speed != SpeedGuidance.BASE:
                            longitudinal = _longitudinal_policy(driver)
                            if longitudinal is not None:
                                speed_span = (
                                    NeonHighwayEnv.MAX_SPEED
                                    - NeonHighwayEnv.MIN_SPEED
                                )
                                current_speed = (
                                    NeonHighwayEnv.MIN_SPEED
                                    + float(observation[0]) * speed_span
                                )
                                longitudinal.set_speed_guidance(
                                    teacher_speed, current_speed=current_speed
                                )
                        records["observations"].append(observation.copy())
                        records["proposed_actions"].append(network_base_action)
                        records["teacher_steers"].append(teacher_steer)
                        records["lane_labelled"].append(int(lane_labelled))
                        records["lane_corrections"].append(
                            int(lane_labelled and teacher_steer != base_steer)
                        )
                        records["teacher_speed_guidance"].append(
                            int(teacher_speed)
                        )
                        records["speed_labelled"].append(int(speed_labelled))
                        records["speed_corrections"].append(
                            int(
                                speed_labelled
                                and teacher_speed != SpeedGuidance.BASE
                            )
                        )
                        records["seeds"].append(episode_seed)
                        records["episode_steps"].append(env.step_count)
                        records["session_ids"].append(session_id)
                        save_dataset(args.dataset, records)
                        if merge_outside_autonomous_envelope:
                            last_label = (
                                f"{label_name}; teacher override accepted"
                            )
                        elif faster_waits_for_gap:
                            last_label = (
                                f"{label_name}; planner will use the next safe gap"
                            )
                        else:
                            last_label = label_name
                    else:
                        last_label = (
                            f"{STEER_NAMES[teacher_steer]} unavailable: "
                            f"{unavailable_reason}"
                        )

                labels = len(records["observations"])
                lane_corrections = int(np.sum(records["lane_corrections"]))
                speed_corrections = int(np.sum(records["speed_corrections"]))
                driver_hud = getattr(driver, "hud_data", None)
                if isinstance(driver_hud, dict):
                    env.hud_data.update(driver_hud)
                env.hud_data.update(
                    {
                        "mode": "HUMAN DAGGER TEACHING",
                        "dagger_collecting": True,
                        "dagger_proposal": ACTION_NAMES[proposed_action],
                        "dagger_last_label": last_label,
                        "dagger_labels": labels,
                        "dagger_corrections": lane_corrections + speed_corrections,
                        "dagger_lane_corrections": lane_corrections,
                        "dagger_speed_corrections": speed_corrections,
                    }
                )
                observation, _, terminated, truncated, _ = env.step(executed_action)
            if not env.quit_requested:
                completed += 1
    finally:
        save_dataset(args.dataset, records)
        env.close()

    added = len(records["observations"]) - session_start
    total = len(records["observations"])
    lane_labels = int(np.sum(records["lane_labelled"]))
    speed_labels = int(np.sum(records["speed_labelled"]))
    lane_corrections = int(np.sum(records["lane_corrections"]))
    speed_corrections = int(np.sum(records["speed_corrections"]))
    print(f"\nSaved {added} new labels ({total} total) to {args.dataset.resolve()}")
    print(
        f"Dataset balance: lane {lane_corrections}/{lane_labels} corrections, "
        f"speed {speed_corrections}/{speed_labels} corrections."
    )


@dataclass(frozen=True)
class GateMetrics:
    score: float
    threshold: float
    overall_accuracy: float
    correction_accuracy: float
    approval_preservation: float
    override_precision: float
    override_rate: float
    overrides: int


@dataclass(frozen=True)
class OfflineMetrics:
    score: float
    lane: GateMetrics
    speed: GateMetrics


def _predict(
    network: DaggerCorrectionNet,
    data: dict[str, NDArray[Any]],
    indices: NDArray[np.int64],
    *,
    device: th.device,
    batch_size: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.int64],
]:
    lane_probabilities: list[NDArray[np.float32]] = []
    steers: list[NDArray[np.int64]] = []
    speed_probabilities: list[NDArray[np.float32]] = []
    speed_guidance: list[NDArray[np.int64]] = []
    network.eval()
    with th.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            observations = th.as_tensor(
                data["observations"][batch], dtype=th.float32, device=device
            )
            base_steers = th.as_tensor(
                [
                    decode_action(int(action))[0]
                    for action in data["proposed_actions"][batch]
                ],
                dtype=th.long,
                device=device,
            )
            (
                lane_intervention_logits,
                steer_logits,
                speed_intervention_logits,
                speed_logits,
            ) = network(observations, base_steers)
            lane_probabilities.append(
                th.softmax(lane_intervention_logits, dim=1)[:, 1].cpu().numpy()
            )
            steers.append(steer_logits.argmax(dim=1).cpu().numpy())
            speed_probabilities.append(
                th.softmax(speed_intervention_logits, dim=1)[:, 1].cpu().numpy()
            )
            speed_guidance.append(speed_logits.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(lane_probabilities),
        np.concatenate(steers),
        np.concatenate(speed_probabilities),
        np.concatenate(speed_guidance),
    )


def _gate_metrics(
    expected: NDArray[np.int64],
    corrections: NDArray[np.bool_],
    final: NDArray[np.int64],
    applied: NDArray[np.bool_],
    threshold: float,
) -> GateMetrics:
    correct = final == expected
    approval_mask = ~corrections
    correction_accuracy = (
        float(np.mean(correct[corrections])) if np.any(corrections) else 0.0
    )
    preservation = (
        float(np.mean(correct[approval_mask])) if np.any(approval_mask) else 1.0
    )
    precision = float(np.mean(correct[applied])) if np.any(applied) else 1.0
    score = 0.65 * correction_accuracy + 0.20 * precision + 0.15 * preservation
    return GateMetrics(
        score=score,
        threshold=float(threshold),
        overall_accuracy=float(np.mean(correct)),
        correction_accuracy=correction_accuracy,
        approval_preservation=preservation,
        override_precision=precision,
        override_rate=float(np.mean(applied)),
        overrides=int(np.sum(applied)),
    )


def lane_offline_metrics(
    data: dict[str, NDArray[Any]],
    indices: NDArray[np.int64],
    probabilities: NDArray[np.float32],
    steers: NDArray[np.int64],
    threshold: float,
) -> GateMetrics:
    observations = np.asarray(data["observations"][indices], dtype=np.float32)
    base_actions = np.asarray(data["proposed_actions"][indices], dtype=np.int64)
    expected = np.asarray(data["teacher_steers"][indices], dtype=np.int64)
    corrections = np.asarray(data["lane_corrections"][indices], dtype=bool)
    final_actions, applied = apply_dagger_gate(
        observations, base_actions, probabilities, steers, threshold
    )
    final = np.asarray(
        [decode_action(int(action))[0] for action in final_actions], dtype=np.int64
    )
    return _gate_metrics(expected, corrections, final, applied, threshold)


def speed_offline_metrics(
    data: dict[str, NDArray[Any]],
    indices: NDArray[np.int64],
    probabilities: NDArray[np.float32],
    guidance: NDArray[np.int64],
    threshold: float,
) -> GateMetrics:
    observations = np.asarray(data["observations"][indices], dtype=np.float32)
    expected = np.asarray(
        data["teacher_speed_guidance"][indices], dtype=np.int64
    )
    corrections = np.asarray(data["speed_corrections"][indices], dtype=bool)
    final, applied = apply_speed_gate(
        observations, probabilities, guidance, threshold
    )
    return _gate_metrics(expected, corrections, final, applied, threshold)


def _threshold_candidates(
    probabilities: NDArray[np.float32], minimum: float
) -> NDArray[np.float64]:
    return np.unique(
        np.concatenate(
            (
                np.linspace(minimum, 0.995, 100),
                np.clip(
                    np.quantile(probabilities, np.linspace(0.0, 1.0, 50)),
                    minimum,
                    0.995,
                ),
            )
        )
    )


def _reliable_gate(
    candidates: list[GateMetrics],
    *,
    minimum_preservation: float,
    minimum_precision: float,
) -> GateMetrics | None:
    reliable = [
        metrics
        for metrics in candidates
        if metrics.approval_preservation >= minimum_preservation
        and metrics.override_precision >= minimum_precision
        and metrics.overrides > 0
    ]
    if not reliable:
        return None
    return max(
        reliable,
        key=lambda metrics: (
            metrics.score,
            metrics.override_precision,
            metrics.approval_preservation,
            metrics.threshold,
        ),
    )


def tune_thresholds(
    data: dict[str, NDArray[Any]],
    validation_indices: NDArray[np.int64],
    lane_probabilities: NDArray[np.float32],
    steers: NDArray[np.int64],
    speed_probabilities: NDArray[np.float32],
    speed_guidance: NDArray[np.int64],
    *,
    minimum_threshold: float,
    minimum_preservation: float,
    minimum_precision: float,
) -> OfflineMetrics | None:
    lane_mask = np.asarray(data["lane_labelled"][validation_indices], dtype=bool)
    speed_mask = np.asarray(data["speed_labelled"][validation_indices], dtype=bool)
    lane_indices = validation_indices[lane_mask]
    speed_indices = validation_indices[speed_mask]
    lane_probabilities = lane_probabilities[lane_mask]
    steers = steers[lane_mask]
    speed_probabilities = speed_probabilities[speed_mask]
    speed_guidance = speed_guidance[speed_mask]
    lane = _reliable_gate(
        [
            lane_offline_metrics(
                data, lane_indices, lane_probabilities, steers, float(threshold)
            )
            for threshold in _threshold_candidates(
                lane_probabilities, minimum_threshold
            )
        ],
        minimum_preservation=minimum_preservation,
        minimum_precision=minimum_precision,
    )
    speed = _reliable_gate(
        [
            speed_offline_metrics(
                data,
                speed_indices,
                speed_probabilities,
                speed_guidance,
                float(threshold),
            )
            for threshold in _threshold_candidates(
                speed_probabilities, minimum_threshold
            )
        ],
        minimum_preservation=minimum_preservation,
        minimum_precision=minimum_precision,
    )
    if lane is None or speed is None:
        return None
    return OfflineMetrics(score=0.5 * (lane.score + speed.score), lane=lane, speed=speed)


def _split_indices(
    data: dict[str, NDArray[Any]], seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Hold out complete traffic seeds so validation cannot see a training route."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(data["seeds"], dtype=np.int64)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise SystemExit(
            "DAgger needs labels from at least two traffic seeds for route-level "
            "validation."
        )

    lane_labelled = np.asarray(data["lane_labelled"], dtype=bool)
    lane_corrections = np.asarray(data["lane_corrections"], dtype=bool)
    teacher_steers = np.asarray(data["teacher_steers"], dtype=np.int64)
    speed_labelled = np.asarray(data["speed_labelled"], dtype=bool)
    speed_corrections = np.asarray(data["speed_corrections"], dtype=bool)
    speed_guidance = np.asarray(data["teacher_speed_guidance"], dtype=np.int64)
    # Each column is a behavior that should be represented on unseen routes.
    features = np.column_stack(
        (
            lane_labelled & ~lane_corrections,
            lane_labelled & lane_corrections & (teacher_steers == STEER_LEFT),
            lane_labelled & lane_corrections & (teacher_steers == STEER_KEEP),
            lane_labelled & lane_corrections & (teacher_steers == STEER_RIGHT),
            speed_labelled & ~speed_corrections,
            speed_labelled
            & speed_corrections
            & (speed_guidance == int(SpeedGuidance.FASTER)),
            speed_labelled
            & speed_corrections
            & (speed_guidance == int(SpeedGuidance.SLOWER)),
        )
    )
    feature_totals = features.sum(axis=0).astype(np.float64)
    feature_group_support = np.asarray(
        [
            sum(bool(np.any(features[groups == group, column])) for group in unique_groups)
            for column in range(features.shape[1])
        ],
        dtype=np.int64,
    )
    validation_group_count = int(
        np.clip(round(0.2 * len(unique_groups)), 1, len(unique_groups) - 1)
    )
    target_fraction = 0.2
    best_validation_groups: NDArray[np.int64] | None = None
    best_score = float("inf")
    # Search many deterministic group splits. This is cheap for a human-sized
    # dataset and handles routes with very different numbers of key presses.
    for _ in range(max(512, 128 * len(unique_groups))):
        candidate = rng.choice(
            unique_groups, size=validation_group_count, replace=False
        )
        validation_mask = np.isin(groups, candidate)
        validation_totals = features[validation_mask].sum(axis=0)
        training_totals = feature_totals - validation_totals
        coverable = feature_group_support >= 2
        coverage_failures = np.sum(
            coverable & ((validation_totals == 0) | (training_totals == 0))
        )
        represented = feature_totals > 0
        feature_fraction_error = float(
            np.mean(
                np.abs(
                    validation_totals[represented]
                    / feature_totals[represented]
                    - target_fraction
                )
            )
        )
        row_fraction_error = abs(float(np.mean(validation_mask)) - target_fraction)
        score = 100.0 * float(coverage_failures) + feature_fraction_error + row_fraction_error
        if score < best_score:
            best_score = score
            best_validation_groups = candidate.copy()

    assert best_validation_groups is not None
    validation_mask = np.isin(groups, best_validation_groups)
    validation = np.flatnonzero(validation_mask).astype(np.int64)
    train = np.flatnonzero(~validation_mask).astype(np.int64)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def _temporal_sample_weights(
    data: dict[str, NDArray[Any]], indices: NDArray[np.int64]
) -> NDArray[np.float32]:
    """Downweight repeated copies of one persistent intent without deleting data."""
    weights = np.ones(len(data["observations"]), dtype=np.float32)
    sessions = np.asarray(data["session_ids"], dtype=np.int64)
    seeds = np.asarray(data["seeds"], dtype=np.int64)
    steps = np.asarray(data["episode_steps"], dtype=np.int64)
    signatures = list(
        zip(
            np.asarray(data["lane_labelled"], dtype=np.int8),
            np.asarray(data["teacher_steers"], dtype=np.int64),
            np.asarray(data["speed_labelled"], dtype=np.int8),
            np.asarray(data["teacher_speed_guidance"], dtype=np.int64),
            strict=True,
        )
    )
    selected = np.zeros(len(weights), dtype=bool)
    selected[indices] = True
    for session, route_seed in sorted(set(zip(sessions[indices], seeds[indices], strict=True))):
        group = np.flatnonzero(
            selected & (sessions == session) & (seeds == route_seed)
        )
        group = group[np.argsort(steps[group])]
        start = 0
        for end in range(1, len(group) + 1):
            cluster_ends = (
                end == len(group)
                or signatures[group[end]] != signatures[group[end - 1]]
                or steps[group[end]] - steps[group[end - 1]] >= 30
            )
            if cluster_ends:
                cluster = group[start:end]
                weights[cluster] = 1.0 / np.sqrt(float(len(cluster)))
                start = end
    return weights


def _balanced_class_weights(
    targets: NDArray[Any],
    labelled: NDArray[Any],
    indices: NDArray[np.int64],
    sample_weights: NDArray[np.float32],
    *,
    classes: int,
    power: float = 1.0,
) -> NDArray[np.float32]:
    """Inverse-frequency weights from training routes after temporal weighting."""
    mask = np.asarray(labelled[indices], dtype=bool)
    selected_targets = np.asarray(targets[indices][mask], dtype=np.int64)
    selected_weights = np.asarray(sample_weights[indices][mask], dtype=np.float64)
    counts = np.bincount(
        selected_targets, weights=selected_weights, minlength=classes
    )[:classes]
    present = counts > 0.0
    result = np.ones(classes, dtype=np.float32)
    if np.any(present):
        total = float(np.sum(counts[present]))
        inverse = total / (float(np.sum(present)) * counts[present])
        result[present] = np.asarray(inverse**power, dtype=np.float32)
        weighted_mean = float(
            np.sum(counts[present] * result[present]) / total
        )
        result[present] /= weighted_mean
        result[present] = np.clip(result[present], 0.20, 8.0)
    return result


def _weighted_cross_entropy(
    logits: th.Tensor,
    targets: th.Tensor,
    sample_weights: th.Tensor,
    class_weights: th.Tensor,
    *,
    label_smoothing: float,
) -> th.Tensor:
    losses = th.nn.functional.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    return th.sum(losses * sample_weights) / th.clamp(
        th.sum(sample_weights), min=1e-6
    )


def _device(name: str) -> th.device:
    if name == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    device = th.device(name)
    if device.type == "cuda" and not th.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot access the GPU")
    return device


def train(args: argparse.Namespace) -> None:
    data = load_dataset(args.dataset)
    total = len(data["observations"])
    lane_correction_count = int(np.sum(data["lane_corrections"]))
    speed_correction_count = int(np.sum(data["speed_corrections"]))
    guidance = np.asarray(data["teacher_speed_guidance"], dtype=np.int64)
    faster_count = int(
        np.sum(
            np.asarray(data["speed_corrections"], dtype=bool)
            & (guidance == int(SpeedGuidance.FASTER))
        )
    )
    slower_count = int(
        np.sum(
            np.asarray(data["speed_corrections"], dtype=bool)
            & (guidance == int(SpeedGuidance.SLOWER))
        )
    )
    approval_count = int(
        np.sum(
            np.asarray(data["lane_labelled"], dtype=bool)
            & np.asarray(data["speed_labelled"], dtype=bool)
            & ~np.asarray(data["lane_corrections"], dtype=bool)
            & ~np.asarray(data["speed_corrections"], dtype=bool)
        )
    )
    if total < args.minimum_labels:
        raise SystemExit(
            f"Need at least {args.minimum_labels} labels; dataset contains {total}."
        )
    if min(lane_correction_count, speed_correction_count, approval_count) < args.minimum_each:
        raise SystemExit(
            f"Need at least {args.minimum_each} lane corrections, speed corrections, "
            f"and complete approvals; found {lane_correction_count}, "
            f"{speed_correction_count}, and {approval_count}."
        )
    if min(faster_count, slower_count) < args.minimum_speed_direction:
        raise SystemExit(
            f"Need at least {args.minimum_speed_direction} faster and slower labels; "
            f"found {faster_count} and {slower_count}."
        )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} already exists; pass --overwrite to replace it")

    device = _device(args.device)
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_indices, validation_indices = _split_indices(data, args.seed)
    temporal_weights = _temporal_sample_weights(data, train_indices)
    lane_action_labelled = np.asarray(data["lane_labelled"], dtype=bool) & np.asarray(
        data["lane_corrections"], dtype=bool
    )
    speed_action_labelled = np.asarray(
        data["speed_labelled"], dtype=bool
    ) & np.asarray(data["speed_corrections"], dtype=bool)
    class_weight_arrays = {
        "lane_intervention": _balanced_class_weights(
            data["lane_corrections"],
            data["lane_labelled"],
            train_indices,
            temporal_weights,
            classes=2,
        ),
        "steer": _balanced_class_weights(
            data["teacher_steers"],
            lane_action_labelled,
            train_indices,
            temporal_weights,
            classes=3,
            power=0.5,
        ),
        "speed_intervention": _balanced_class_weights(
            data["speed_corrections"],
            data["speed_labelled"],
            train_indices,
            temporal_weights,
            classes=2,
        ),
        "speed_guidance": _balanced_class_weights(
            data["teacher_speed_guidance"],
            speed_action_labelled,
            train_indices,
            temporal_weights,
            classes=3,
            power=0.5,
        ),
    }
    class_weights = {
        name: th.as_tensor(values, dtype=th.float32, device=device)
        for name, values in class_weight_arrays.items()
    }
    observation_size = int(data["observations"].shape[1])
    network = DaggerCorrectionNet(observation_size).to(device)
    optimizer = th.optim.AdamW(
        network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    any_correction = (
        np.asarray(data["lane_corrections"], dtype=bool)
        | np.asarray(data["speed_corrections"], dtype=bool)
    )
    train_corrections = train_indices[any_correction[train_indices]]
    history: list[dict[str, Any]] = []
    best_state: dict[str, th.Tensor] | None = None
    best_metrics: OfflineMetrics | None = None

    train_seeds = sorted(set(map(int, data["seeds"][train_indices])))
    validation_seeds = sorted(set(map(int, data["seeds"][validation_indices])))
    print(
        f"Training DAgger residual on {len(train_indices)} labels; validating on "
        f"{len(validation_indices)} labels from {len(validation_seeds)} entirely "
        f"held-out routes ({device.type})."
    )
    print(f"Validation seeds: {validation_seeds}")
    print(
        "Balanced class weights: "
        + ", ".join(
            f"{name}={np.round(values, 3).tolist()}"
            for name, values in class_weight_arrays.items()
        )
    )
    for epoch in range(1, args.epochs + 1):
        epoch_indices = np.concatenate(
            (
                train_indices,
                np.repeat(train_corrections, max(0, args.correction_repeats - 1)),
            )
        )
        rng.shuffle(epoch_indices)
        network.train()
        losses: list[float] = []
        for start in range(0, len(epoch_indices), args.batch_size):
            indices = epoch_indices[start : start + args.batch_size]
            observations = th.as_tensor(
                data["observations"][indices], dtype=th.float32, device=device
            )
            base_steers = th.as_tensor(
                [
                    decode_action(int(action))[0]
                    for action in data["proposed_actions"][indices]
                ],
                dtype=th.long,
                device=device,
            )
            (
                lane_intervention_logits,
                steer_logits,
                speed_intervention_logits,
                speed_logits,
            ) = network(observations, base_steers)
            lane_mask = th.as_tensor(
                data["lane_labelled"][indices], dtype=th.bool, device=device
            )
            speed_mask = th.as_tensor(
                data["speed_labelled"][indices], dtype=th.bool, device=device
            )
            batch_weights = th.as_tensor(
                temporal_weights[indices], dtype=th.float32, device=device
            )
            lane_targets = th.as_tensor(
                data["lane_corrections"][indices], dtype=th.long, device=device
            )
            steer_targets = th.as_tensor(
                data["teacher_steers"][indices], dtype=th.long, device=device
            )
            speed_targets = th.as_tensor(
                data["speed_corrections"][indices], dtype=th.long, device=device
            )
            guidance_targets = th.as_tensor(
                data["teacher_speed_guidance"][indices],
                dtype=th.long,
                device=device,
            )
            loss = lane_intervention_logits.sum() * 0.0
            if bool(lane_mask.any()):
                loss = loss + _weighted_cross_entropy(
                    lane_intervention_logits[lane_mask],
                    lane_targets[lane_mask],
                    batch_weights[lane_mask],
                    class_weights["lane_intervention"],
                    label_smoothing=args.label_smoothing,
                )
                lane_action_mask = lane_mask & lane_targets.bool()
                if bool(lane_action_mask.any()):
                    loss = loss + args.steer_loss_weight * _weighted_cross_entropy(
                        steer_logits[lane_action_mask],
                        steer_targets[lane_action_mask],
                        batch_weights[lane_action_mask],
                        class_weights["steer"],
                        label_smoothing=args.label_smoothing,
                    )
            if bool(speed_mask.any()):
                loss = loss + _weighted_cross_entropy(
                    speed_intervention_logits[speed_mask],
                    speed_targets[speed_mask],
                    batch_weights[speed_mask],
                    class_weights["speed_intervention"],
                    label_smoothing=args.label_smoothing,
                )
                speed_action_mask = speed_mask & speed_targets.bool()
                if bool(speed_action_mask.any()):
                    loss = loss + args.speed_loss_weight * _weighted_cross_entropy(
                        speed_logits[speed_action_mask],
                        guidance_targets[speed_action_mask],
                        batch_weights[speed_action_mask],
                        class_weights["speed_guidance"],
                        label_smoothing=args.label_smoothing,
                    )
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        lane_probabilities, steers, speed_probabilities, speed_guidance = _predict(
            network,
            data,
            validation_indices,
            device=device,
            batch_size=args.batch_size,
        )
        metrics = tune_thresholds(
            data,
            validation_indices,
            lane_probabilities,
            steers,
            speed_probabilities,
            speed_guidance,
            minimum_threshold=args.minimum_threshold,
            minimum_preservation=args.minimum_preservation,
            minimum_precision=args.minimum_precision,
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "valid_gate": metrics is not None,
        }
        if metrics is not None:
            record["validation"] = asdict(metrics)
            if best_metrics is None or (
                metrics.score,
                metrics.lane.override_precision + metrics.speed.override_precision,
                metrics.lane.approval_preservation
                + metrics.speed.approval_preservation,
                metrics.lane.threshold + metrics.speed.threshold,
            ) > (
                best_metrics.score,
                best_metrics.lane.override_precision
                + best_metrics.speed.override_precision,
                best_metrics.lane.approval_preservation
                + best_metrics.speed.approval_preservation,
                best_metrics.lane.threshold + best_metrics.speed.threshold,
            ):
                best_metrics = metrics
                best_state = copy.deepcopy(network.state_dict())
            print(
                f"Epoch {epoch:02d}: lane {metrics.lane.correction_accuracy:.0%} "
                f"@ {metrics.lane.threshold:.3f}, speed "
                f"{metrics.speed.correction_accuracy:.0%} @ "
                f"{metrics.speed.threshold:.3f}"
            )
        else:
            print(f"Epoch {epoch:02d}: no gate meets the reliability constraints yet")
        history.append(record)

    if best_state is None or best_metrics is None:
        raise SystemExit(
            "No reliable DAgger gate was found. Collect more explicit approvals and "
            "corrections, then train again."
        )
    network.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    th.save(
        {
            "schema_version": SCHEMA_VERSION,
            "state_dict": network.cpu().state_dict(),
            "observation_size": observation_size,
            "lane_threshold": best_metrics.lane.threshold,
            "speed_threshold": best_metrics.speed.threshold,
            "base_model": str(args.base_model),
            "override_model": str(args.override_model),
            "dataset": str(args.dataset),
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "class_weights": {
                name: values.tolist() for name, values in class_weight_arrays.items()
            },
            "temporal_weighting": (
                "inverse_sqrt_cluster_for_repeated_intent_under_30_steps"
            ),
            "correction_repeats": args.correction_repeats,
            "lane_residual_enabled": not args.disable_lane_residual,
            "faster_only": args.faster_only,
            "validation": asdict(best_metrics),
        },
        args.output,
    )
    history_path = args.output.with_suffix(".training.json")
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved DAgger residual to {args.output.resolve()}")
    print(json.dumps(asdict(best_metrics), indent=2))


def promotion_gate(
    base: dict[str, Any], candidate: dict[str, Any], *, episodes: int
) -> dict[str, Any]:
    """Require matched-seed safety and driving quality before promoting DAgger."""
    checks = {
        "completion_preserved": (
            float(candidate["completion_rate"])
            >= float(base["completion_rate"]) - 0.01
        ),
        "crash_rate_preserved": (
            float(candidate["crash_rate"]) <= float(base["crash_rate"]) + 0.01
        ),
        "unsafe_merges_preserved": (
            float(candidate["mean_unsafe_lane_changes"])
            <= float(base["mean_unsafe_lane_changes"]) + 0.02
        ),
        "net_overtakes_not_regressed": (
            float(candidate["mean_net_overtakes"])
            >= float(base["mean_net_overtakes"]) - 0.5
        ),
        "passing_response_not_regressed": (
            float(candidate["passing_response_rate"])
            >= float(base["passing_response_rate"]) - 0.05
        ),
        "avoidable_following_not_regressed": (
            float(candidate["avoidable_following_rate"])
            <= float(base["avoidable_following_rate"]) + 0.01
        ),
        "clear_road_stalls_not_regressed": (
            float(candidate["clear_road_stall_rate"])
            <= float(base["clear_road_stall_rate"]) + 0.005
        ),
        "lane_reversals_not_regressed": (
            float(candidate["mean_lane_reversals"])
            <= float(base["mean_lane_reversals"]) + 0.5
        ),
        "unjustified_braking_not_regressed": (
            float(candidate["unjustified_brakes_per_1000_steps"])
            <= float(base["unjustified_brakes_per_1000_steps"]) + 0.5
        ),
    }
    meaningful_gain = any(
        (
            float(candidate["mean_net_overtakes"])
            >= float(base["mean_net_overtakes"]) + 0.25,
            float(candidate["avoidable_following_rate"])
            <= float(base["avoidable_following_rate"]) - 0.005,
            float(candidate["mean_lane_reversals"])
            <= float(base["mean_lane_reversals"]) - 0.25,
        )
    )
    enough_routes = episodes >= 50
    passed = enough_routes and all(checks.values()) and meaningful_gain
    return {
        "status": "PROMOTE" if passed else "HOLD" if enough_routes else "INCONCLUSIVE",
        "passed": passed,
        "minimum_routes_met": enough_routes,
        "meaningful_gain": meaningful_gain,
        "checks": checks,
        "note": (
            "Promotion requires at least 50 matched routes, every reliability check, "
            "and a measurable gain in passing, avoidable following, or reversals."
        ),
    }


def evaluate(args: argparse.Namespace) -> None:
    base_env = NeonHighwayEnv(
        difficulty_mode=args.difficulty, dynamic_traffic=args.dynamic_traffic
    )
    dagger_env = NeonHighwayEnv(
        difficulty_mode=args.difficulty, dynamic_traffic=args.dynamic_traffic
    )
    base = _base_driver(args.base_model, args.override_model, args.device)
    dagger = load_dagger_policy(
        args.base_model,
        args.override_model,
        args.dagger_model,
        device=args.device,
    )
    try:
        base_result = evaluate_in_env(
            base_env, base, episodes=args.episodes, seed=args.seed
        ).to_dict()
        dagger_result = evaluate_in_env(
            dagger_env, dagger, episodes=args.episodes, seed=args.seed
        ).to_dict()
    finally:
        base_env.close()
        dagger_env.close()
    result = {
        "seed": args.seed,
        "episodes": args.episodes,
        "difficulty": args.difficulty,
        "dynamic_traffic": args.dynamic_traffic,
        "base": base_result,
        "dagger": dagger_result,
        "dagger_overrides": dagger.total_overrides,
        "dagger_safety_vetoes": dagger.total_rejected,
        "dagger_cooldown_deferrals": dagger.total_deferred,
        "dagger_speed_overrides": dagger.total_speed_overrides,
        "dagger_speed_safety_vetoes": dagger.total_speed_rejected,
        "dagger_speed_cooldown_deferrals": dagger.total_speed_deferred,
        "promotion_gate": promotion_gate(
            base_result, dagger_result, episodes=args.episodes
        ),
    }
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def watch(args: argparse.Namespace) -> None:
    policy = load_dagger_policy(
        args.base_model,
        args.override_model,
        args.dagger_model,
        device=args.device,
    )
    env = NeonHighwayEnv(
        render_mode="human",
        render_fps=args.fps,
        render_speed=args.speed,
        difficulty_mode=args.difficulty,
        episode_seconds=args.seconds,
        endless=args.endless,
        dynamic_traffic=args.dynamic_traffic,
    )
    try:
        run_policy(
            env,
            policy,
            episodes=args.episodes,
            seed=args.seed,
            mode="V8 HUMAN-TAUGHT DAGGER",
            epsilon=0.0,
        )
    finally:
        env.close()


def _driver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--override-model", type=Path, default=DEFAULT_OVERRIDE_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")


def _traffic_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--difficulty", choices=sorted(NeonHighwayEnv.DIFFICULTY_MODES), default="hard"
    )
    parser.add_argument(
        "--no-dynamic-traffic",
        dest="dynamic_traffic",
        action="store_false",
        help="Disable traffic-car lane changes (enabled by default for DAgger)",
    )
    parser.set_defaults(dynamic_traffic=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="Teach lane and speed decisions while the current driver runs"
    )
    _driver_arguments(collect_parser)
    _traffic_arguments(collect_parser)
    collect_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    collect_parser.add_argument(
        "--dagger-model",
        type=Path,
        default=None,
        help="Optional previous DAgger model for the next aggregation round",
    )
    collect_parser.add_argument("--episodes", type=int, default=10)
    collect_parser.add_argument("--seed", type=int, default=510_000)
    collect_parser.add_argument("--fps", type=int, default=60)
    collect_parser.add_argument("--speed", type=float, default=1.0)
    collect_parser.add_argument("--seconds", type=float, default=120.0)
    collect_parser.add_argument("--endless", action="store_true")
    collect_parser.set_defaults(handler=collect)

    train_parser = subparsers.add_parser(
        "train", help="Train and calibrate a safety-gated human residual"
    )
    _driver_arguments(train_parser)
    train_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train_parser.add_argument("--output", type=Path, default=DEFAULT_DAGGER_MODEL)
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.add_argument("--seed", type=int, default=8128)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--label-smoothing", type=float, default=0.02)
    train_parser.add_argument("--steer-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--speed-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--correction-repeats", type=int, default=1)
    train_parser.add_argument("--minimum-labels", type=int, default=30)
    train_parser.add_argument("--minimum-each", type=int, default=8)
    train_parser.add_argument("--minimum-speed-direction", type=int, default=4)
    train_parser.add_argument("--minimum-threshold", type=float, default=0.70)
    train_parser.add_argument("--disable-lane-residual", action="store_true")
    train_parser.add_argument("--faster-only", action="store_true")
    train_parser.add_argument("--minimum-preservation", type=float, default=0.98)
    train_parser.add_argument("--minimum-precision", type=float, default=0.80)
    train_parser.set_defaults(handler=train)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Compare the frozen V7 driver and human residual on matched seeds"
    )
    _driver_arguments(evaluate_parser)
    _traffic_arguments(evaluate_parser)
    evaluate_parser.add_argument("--dagger-model", type=Path, default=DEFAULT_DAGGER_MODEL)
    evaluate_parser.add_argument("--episodes", type=int, default=100)
    evaluate_parser.add_argument("--seed", type=int, default=520_000)
    evaluate_parser.add_argument("--output", type=Path, default=None)
    evaluate_parser.set_defaults(handler=evaluate)

    watch_parser = subparsers.add_parser("watch", help="Watch the human-taught driver")
    _driver_arguments(watch_parser)
    _traffic_arguments(watch_parser)
    watch_parser.add_argument("--dagger-model", type=Path, default=DEFAULT_DAGGER_MODEL)
    watch_parser.add_argument("--episodes", type=int, default=10)
    watch_parser.add_argument("--seed", type=int, default=42)
    watch_parser.add_argument("--fps", type=int, default=60)
    watch_parser.add_argument("--speed", type=float, default=1.0)
    watch_parser.add_argument("--seconds", type=float, default=None)
    watch_parser.add_argument("--endless", action="store_true")
    watch_parser.set_defaults(handler=watch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "episodes", 1) < 1:
        raise SystemExit("--episodes must be at least 1")
    if not 0.0 <= getattr(args, "minimum_threshold", 0.7) < 1.0:
        raise SystemExit("--minimum-threshold must be in [0, 1)")
    args.handler(args)


if __name__ == "__main__":
    main()
