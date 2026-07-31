"""Preference learning and guarded fine-tuning for Neon Highway.

The current V5 reward is deliberately kept as the safety foundation. RLAIF
adds a small, learned style reward trained from pairwise trajectory choices:
which of two drives was safer, more decisive, smoother, and less wasteful?

The first reward model is linear on purpose. Its weights can be inspected,
tested, and converted exactly into per-step feedback; a black-box sequence
model would make reward hacking much harder to diagnose at this project size.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from numpy.typing import NDArray
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import ConstantSchedule, LinearSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn

from self_driving_rl.game import _evaluation, load_model, run_policy
from self_driving_rl.game_env import (
    ACTION_COUNT,
    PEDAL_COAST,
    STEER_KEEP,
    STEER_LEFT,
    STEER_RIGHT,
    NeonHighwayEnv,
    decode_action,
    encode_action,
)
from self_driving_rl.longitudinal import LongitudinalIntentPolicy
from self_driving_rl.metrics import evaluate_in_env
from self_driving_rl.symmetry import MirrorSymmetry

FEATURE_NAMES = (
    "completion",
    "crash",
    "distance_km",
    "cruise_quality_seconds",
    "overtakes",
    "passed_by_traffic",
    "lane_changes",
    "rapid_lane_changes",
    "lane_reversals",
    "unsafe_lane_changes",
    "unproductive_lane_changes",
    "missed_passing_opportunities",
    "blocked_seconds",
    "front_risk_seconds",
    "rear_risk_seconds",
    "control_jerk",
    "near_misses",
    "challenges_resolved",
)

# Typical full-route magnitudes. Fixed scales avoid leaking validation/test
# comparisons into preprocessing and keep learned coefficients interpretable.
FEATURE_SCALES = np.asarray(
    [1.0, 1.0, 1.0, 40.0, 10.0, 5.0, 15.0, 10.0, 8.0, 4.0, 4.0, 5.0,
     5.0, 4.0, 4.0, 12.0, 3.0, 3.0],
    dtype=np.float32,
)

COUNTER_FEATURES = {
    "overtakes": "overtakes",
    "passed_by_traffic": "passed_by_traffic",
    "lane_changes": "lane_changes",
    "rapid_lane_changes": "rapid_lane_changes",
    "lane_reversals": "lane_reversals",
    "unsafe_lane_changes": "unsafe_lane_changes",
    "unproductive_lane_changes": "unproductive_lane_changes",
    "missed_passing_opportunities": "missed_passing_opportunities",
    "near_misses": "near_misses",
    "challenges_resolved": "challenges_resolved",
}

POSITIVE_PREFERENCE_FEATURES = {
    "completion",
    "distance_km",
    "cruise_quality_seconds",
    "overtakes",
    "challenges_resolved",
}
NEGATIVE_PREFERENCE_FEATURES = set(FEATURE_NAMES) - POSITIVE_PREFERENCE_FEATURES


class PreferenceFeatureAccumulator:
    """Convert an episode's telemetry into additive preference features."""

    def __init__(self) -> None:
        self.values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self._previous_counters = {info_name: 0 for info_name in COUNTER_FEATURES.values()}
        self._previous_distance = 0.0
        self._previous_acceleration = 0.0

    def observe(self, info: dict[str, Any]) -> NDArray[np.float32]:
        delta = np.zeros_like(self.values)

        def add(name: str, value: float) -> None:
            delta[FEATURE_NAMES.index(name)] += float(value)

        if info.get("completed", False):
            add("completion", 1.0)
        if info.get("crashed", False):
            add("crash", 1.0)

        distance = float(info.get("distance_m", self._previous_distance))
        add("distance_km", max(distance - self._previous_distance, 0.0) / 1000.0)
        self._previous_distance = distance

        speed = float(info.get("speed", NeonHighwayEnv.MIN_SPEED))
        speed_fraction = (speed - NeonHighwayEnv.MIN_SPEED) / (
            NeonHighwayEnv.CRUISE_SPEED - NeonHighwayEnv.MIN_SPEED
        )
        add("cruise_quality_seconds", np.clip(speed_fraction, 0.0, 1.0) * NeonHighwayEnv.DT)

        for feature_name, info_name in COUNTER_FEATURES.items():
            current = int(info.get(info_name, self._previous_counters[info_name]))
            change = max(current - self._previous_counters[info_name], 0)
            add(feature_name, float(change))
            self._previous_counters[info_name] = current

        blocked_steps = int(info.get("blocked_steps", 0))
        previous_blocked = int(getattr(self, "_previous_blocked_steps", 0))
        add("blocked_seconds", max(blocked_steps - previous_blocked, 0) * NeonHighwayEnv.DT)
        self._previous_blocked_steps = blocked_steps

        add("front_risk_seconds", float(info.get("threat_level", 0.0)) * NeonHighwayEnv.DT)
        add("rear_risk_seconds", float(info.get("rear_threat_level", 0.0)) * NeonHighwayEnv.DT)

        acceleration = float(info.get("acceleration", 0.0))
        acceleration_span = NeonHighwayEnv.MAX_ACCELERATION + NeonHighwayEnv.MAX_BRAKING
        add("control_jerk", abs(acceleration - self._previous_acceleration) / acceleration_span)
        self._previous_acceleration = acceleration

        self.values += delta
        return delta

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(FEATURE_NAMES, self.values, strict=True)}


@dataclass(frozen=True)
class PreferenceRewardModel:
    feature_names: tuple[str, ...]
    feature_scales: tuple[float, ...]
    weights: tuple[float, ...]
    episode_reward_scale: float
    clip_per_step: float

    @classmethod
    def load(cls, path: Path) -> PreferenceRewardModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(
            feature_names=tuple(payload["feature_names"]),
            feature_scales=tuple(float(value) for value in payload["feature_scales"]),
            weights=tuple(float(value) for value in payload["weights"]),
            episode_reward_scale=float(payload["episode_reward_scale"]),
            clip_per_step=float(payload.get("clip_per_step", 1.0)),
        )
        if model.feature_names != FEATURE_NAMES:
            raise ValueError("Preference model feature schema does not match this environment")
        return model

    @property
    def effective_weights(self) -> NDArray[np.float32]:
        weights = np.asarray(self.weights, dtype=np.float32)
        scales = np.asarray(self.feature_scales, dtype=np.float32)
        return self.episode_reward_scale * weights / scales

    def reward(self, feature_delta: NDArray[np.float32]) -> float:
        raw = float(np.dot(self.effective_weights, feature_delta))
        return float(np.clip(raw, -self.clip_per_step, self.clip_per_step))


class PreferenceRewardWrapper(gym.Wrapper):
    """Add a bounded, frozen learned reward while retaining V5's base reward."""

    def __init__(self, env: gym.Env, reward_model: PreferenceRewardModel) -> None:
        super().__init__(env)
        self.reward_model = reward_model
        self.features = PreferenceFeatureAccumulator()
        self.preference_episode_reward = 0.0

    def reset(self, **kwargs: Any) -> tuple[NDArray[np.float32], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.features = PreferenceFeatureAccumulator()
        self.preference_episode_reward = 0.0
        info["preference_episode_reward"] = 0.0
        return observation, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        feature_delta = self.features.observe(info)
        preference_reward = self.reward_model.reward(feature_delta)
        self.preference_episode_reward += preference_reward
        info["base_reward"] = float(base_reward)
        info["preference_reward"] = preference_reward
        info["preference_episode_reward"] = self.preference_episode_reward
        if terminated or truncated:
            info["preference_features"] = self.features.to_dict()
        return observation, float(base_reward) + preference_reward, terminated, truncated, info


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    model_path: str
    variant: str = "raw"
    epsilon: float = 0.0


class CandidateController:
    """A saved policy plus a small data-generation perturbation.

    Variants only create diverse comparison examples. Fine-tuning never uses
    these rules; it must recover preferred behavior from the learned reward.
    """

    def __init__(self, spec: CandidateSpec, model: Any) -> None:
        self.spec = spec
        self.model = model

    def choose(
        self,
        observation: NDArray[np.float32],
        env: NeonHighwayEnv,
        rng: np.random.Generator,
    ) -> int:
        if rng.random() < self.spec.epsilon:
            return int(rng.integers(0, env.action_space.n))

        action = int(self.model.predict(observation, deterministic=True)[0])
        return self.adjust(action, env)

    def adjust(self, action: int, env: NeonHighwayEnv) -> int:
        """Apply this comparison variant to an already chosen base action."""
        steer, pedal = decode_action(action)
        lane_change_finished = abs(env.lane_position - env.target_lane) < 0.05
        if not lane_change_finished:
            return action

        since_change = env.step_count - env._last_lane_change_step
        threat = max(env.current_threat()["level"], env.rear_threat()["level"])
        if self.spec.variant in {"calm", "balanced"}:
            if steer != STEER_KEEP and since_change < env.RAPID_LANE_CHANGE_STEPS and threat < 0.6:
                steer = STEER_KEEP

        if self.spec.variant in {"proactive", "balanced"}:
            options = env.passing_lane_options()
            chosen_lane = env.target_lane + (-1 if steer == STEER_LEFT else 1)
            chosen_is_option = steer != STEER_KEEP and chosen_lane in options
            ready = since_change >= 20 or threat >= 0.5
            if options and not chosen_is_option and ready:
                sensors = env.lane_sensors()
                best_lane = max(options, key=lambda lane: sensors[lane][0])
                steer = STEER_LEFT if best_lane < env.target_lane else STEER_RIGHT
                if pedal not in range(3):
                    pedal = PEDAL_COAST
        return encode_action(steer, pedal)


DEFAULT_COMPARISON_PAIRS = (
    ("safe", "assertive"),
    ("safe", "pilot"),
    ("safe", "calm"),
    ("safe", "proactive"),
    ("safe", "balanced"),
    ("balanced", "assertive"),
    ("balanced", "proactive"),
    ("calm", "proactive"),
    ("calm", "balanced"),
    ("pilot", "balanced"),
)


def _episode_record(
    env: NeonHighwayEnv,
    controller: CandidateController,
    *,
    seed: int,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 17_003)
    features = PreferenceFeatureAccumulator()
    action_counts: Counter[str] = Counter()
    timeline: list[dict[str, Any]] = []
    previous_events = {
        name: 0
        for name in (
            "overtakes",
            "passed_by_traffic",
            "lane_changes",
            "rapid_lane_changes",
            "lane_reversals",
            "missed_passing_opportunities",
        )
    }
    terminated = truncated = False
    final_info: dict[str, Any] = {}

    while not (terminated or truncated):
        action = controller.choose(observation, env, rng)
        observation, _, terminated, truncated, final_info = env.step(action)
        features.observe(final_info)
        action_counts[str(action)] += 1

        changed = {
            name: int(final_info.get(name, 0))
            for name, previous in previous_events.items()
            if int(final_info.get(name, 0)) != previous
        }
        if changed or final_info.get("crashed", False):
            timeline.append(
                {
                    "second": round(float(final_info.get("elapsed_seconds", 0.0)), 1),
                    "lane": round(float(final_info.get("lane", 0.0)), 2),
                    "speed_kmh": round(float(final_info.get("speed", 0.0)) * 3.6, 1),
                    "ttc": round(float(final_info.get("ttc", 99.0)), 2),
                    "events": changed,
                }
            )
        previous_events.update(
            {name: int(final_info.get(name, 0)) for name in previous_events}
        )

    record_id = f"{controller.spec.name}:{seed}"
    summary_keys = (
        "crashed",
        "completed",
        "timed_out",
        "elapsed_seconds",
        "distance_m",
        "overtakes",
        "passed_by_traffic",
        "net_overtakes",
        "lane_changes",
        "rapid_lane_changes",
        "lane_reversals",
        "unsafe_lane_changes",
        "unproductive_lane_changes",
        "passing_opportunities",
        "passing_actions",
        "missed_passing_opportunities",
        "blocked_steps",
        "near_misses",
        "challenges_presented",
        "challenges_resolved",
    )
    summary = {key: final_info.get(key, 0) for key in summary_keys}
    summary["mean_speed_kmh"] = round(
        3.6
        * float(features.values[FEATURE_NAMES.index("distance_km")])
        * 1000.0
        / max(float(final_info.get("elapsed_seconds", NeonHighwayEnv.DT)), NeonHighwayEnv.DT),
        2,
    )
    collision = final_info.get("collision")
    summary["collision"] = collision if isinstance(collision, dict) else None
    summary["passing_response_rate"] = (
        float(summary["passing_actions"]) / float(summary["passing_opportunities"])
        if summary["passing_opportunities"]
        else 0.0
    )
    summary["blocked_seconds"] = float(summary["blocked_steps"]) * NeonHighwayEnv.DT
    summary["lane_changes_per_overtake"] = (
        float(summary["lane_changes"]) / float(summary["overtakes"])
        if summary["overtakes"]
        else None
    )
    return {
        "id": record_id,
        "seed": seed,
        "candidate": controller.spec.name,
        "model_path": controller.spec.model_path,
        "variant": controller.spec.variant,
        "epsilon": controller.spec.epsilon,
        "features": features.to_dict(),
        "summary": summary,
        "action_counts": dict(sorted(action_counts.items())),
        "timeline": timeline,
    }


def collect_preferences(args: argparse.Namespace) -> None:
    safe_path = str(args.safe_model)
    specs = [
        CandidateSpec("safe", safe_path),
        CandidateSpec("assertive", str(args.assertive_model)),
        CandidateSpec("pilot", str(args.pilot_model)),
        CandidateSpec("calm", safe_path, variant="calm"),
        CandidateSpec("proactive", safe_path, variant="proactive"),
        CandidateSpec("balanced", safe_path, variant="balanced"),
    ]
    models: dict[str, Any] = {}
    controllers: list[CandidateController] = []
    for spec in specs:
        if spec.model_path not in models:
            models[spec.model_path] = load_model(Path(spec.model_path), device=args.device)
        controllers.append(CandidateController(spec, models[spec.model_path]))

    env = NeonHighwayEnv(difficulty_mode=args.difficulty)
    records: list[dict[str, Any]] = []
    try:
        for offset in range(args.episodes):
            episode_seed = args.seed + offset
            for controller in controllers:
                records.append(_episode_record(env, controller, seed=episode_seed))
            print(f"Collected matched seed {episode_seed:,} ({offset + 1}/{args.episodes})")
    finally:
        env.close()

    record_ids = {record["id"] for record in records}
    comparisons = []
    for offset in range(args.episodes):
        episode_seed = args.seed + offset
        for candidate_a, candidate_b in DEFAULT_COMPARISON_PAIRS:
            record_a = f"{candidate_a}:{episode_seed}"
            record_b = f"{candidate_b}:{episode_seed}"
            if record_a in record_ids and record_b in record_ids:
                comparisons.append(
                    {
                        "id": f"{episode_seed}:{candidate_a}-vs-{candidate_b}",
                        "seed": episode_seed,
                        "a": record_a,
                        "b": record_b,
                        "preference": None,
                        "reason": None,
                    }
                )

    payload = {
        "schema_version": 1,
        "environment": NeonHighwayEnv.VERSION,
        "teacher_rubric": {
            "priority_order": [
                "avoid crashes and dangerous TTC",
                "complete staged challenges",
                "take clear safe passing opportunities",
                "make net progress through traffic",
                "avoid rapid reversals and unproductive lane changes",
                "maintain smooth sensible speed and control",
            ],
            "hard_rule": "A crash normally loses to a completed route on the same seed.",
        },
        "seed_start": args.seed,
        "episodes_per_candidate": args.episodes,
        "feature_names": list(FEATURE_NAMES),
        "feature_scales": FEATURE_SCALES.tolist(),
        "candidates": [asdict(spec) for spec in specs],
        "records": records,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} trajectories and {len(comparisons)} pairs to {args.output}")


def _label_value(preference: str) -> float:
    if preference == "a":
        return 1.0
    if preference == "b":
        return 0.0
    if preference == "tie":
        return 0.5
    raise ValueError(f"Unsupported preference label: {preference}")


def _split_seeds(seeds: list[int], split_seed: int) -> dict[str, set[int]]:
    shuffled = list(sorted(set(seeds)))
    np.random.default_rng(split_seed).shuffle(shuffled)
    train_end = max(1, int(0.70 * len(shuffled)))
    validation_end = max(train_end + 1, int(0.85 * len(shuffled)))
    validation_end = min(validation_end, len(shuffled) - 1)
    return {
        "train": set(shuffled[:train_end]),
        "validation": set(shuffled[train_end:validation_end]),
        "test": set(shuffled[validation_end:]),
    }


def _preference_metrics(logits: th.Tensor, targets: th.Tensor) -> dict[str, float]:
    probabilities = th.sigmoid(logits)
    loss = th.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    non_ties = targets != 0.5
    if bool(non_ties.any()):
        predicted = probabilities[non_ties] >= 0.5
        expected = targets[non_ties] >= 0.5
        accuracy = float((predicted == expected).float().mean().item())
    else:
        accuracy = float("nan")
    return {"loss": float(loss.item()), "non_tie_accuracy": accuracy}


def train_reward_model(args: argparse.Namespace) -> None:
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.labels is not None:
        label_payload = json.loads(args.labels.read_text(encoding="utf-8"))
        labels = label_payload["labels"]
        for comparison in payload["comparisons"]:
            label = labels.get(comparison["id"])
            if label is not None:
                comparison.update(label)
    records = {record["id"]: record for record in payload["records"]}
    labelled = [item for item in payload["comparisons"] if item.get("preference")]
    if len(labelled) < 20:
        raise SystemExit("At least 20 AI-labelled comparisons are required")

    splits = _split_seeds([int(item["seed"]) for item in labelled], args.split_seed)
    examples: dict[str, tuple[th.Tensor, th.Tensor]] = {}
    for split_name, split_seeds in splits.items():
        split_items = [item for item in labelled if int(item["seed"]) in split_seeds]
        differences = []
        targets = []
        for item in split_items:
            values_a = np.asarray(
                [records[item["a"]]["features"][name] for name in FEATURE_NAMES],
                dtype=np.float32,
            )
            values_b = np.asarray(
                [records[item["b"]]["features"][name] for name in FEATURE_NAMES],
                dtype=np.float32,
            )
            differences.append((values_a - values_b) / FEATURE_SCALES)
            targets.append(_label_value(str(item["preference"])))
        examples[split_name] = (
            th.as_tensor(np.asarray(differences), dtype=th.float32),
            th.as_tensor(np.asarray(targets), dtype=th.float32),
        )

    weights = th.zeros(len(FEATURE_NAMES), dtype=th.float32, requires_grad=True)
    optimizer = th.optim.Adam([weights], lr=args.learning_rate)
    train_x, train_y = examples["train"]
    best_weights = weights.detach().clone()
    best_validation = float("inf")
    patience = 0
    for _ in range(args.epochs):
        optimizer.zero_grad()
        logits = train_x @ weights
        data_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        loss = data_loss + args.l2 * weights.square().mean()
        loss.backward()
        optimizer.step()
        # Preference data can contain incidental correlations (our faster
        # candidates also happened to jerk more, for example). Encode only the
        # direction of indisputable driving values; magnitudes remain learned.
        with th.no_grad():
            for index, name in enumerate(FEATURE_NAMES):
                if name in POSITIVE_PREFERENCE_FEATURES:
                    weights[index].clamp_(min=0.0)
                elif name in NEGATIVE_PREFERENCE_FEATURES:
                    weights[index].clamp_(max=0.0)

        validation_x, validation_y = examples["validation"]
        with th.no_grad():
            validation_loss = float(
                th.nn.functional.binary_cross_entropy_with_logits(
                    validation_x @ weights, validation_y
                ).item()
            )
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_weights = weights.detach().clone()
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    weights = best_weights
    metrics = {
        split_name: _preference_metrics(values @ weights, targets)
        for split_name, (values, targets) in examples.items()
    }

    all_values = np.asarray(
        [
            [record["features"][name] for name in FEATURE_NAMES]
            for record in records.values()
            if int(record["seed"]) in splits["train"]
        ],
        dtype=np.float32,
    )
    raw_episode_scores = (all_values / FEATURE_SCALES) @ weights.numpy()
    score_std = max(float(np.std(raw_episode_scores)), 1e-3)
    episode_reward_scale = args.target_reward_std / score_std

    model_payload = {
        "schema_version": 1,
        "source_dataset": str(args.dataset),
        "teacher": "Codex AI driving rubric with pair-specific reasons",
        "feature_names": list(FEATURE_NAMES),
        "feature_scales": FEATURE_SCALES.tolist(),
        "weights": [float(value) for value in weights.tolist()],
        "episode_reward_scale": episode_reward_scale,
        "clip_per_step": args.clip_per_step,
        "split_seed": args.split_seed,
        "split_seeds": {name: sorted(values) for name, values in splits.items()},
        "labelled_comparisons": len(labelled),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nLearned standardized preference weights:")
    for name, weight in sorted(
        zip(FEATURE_NAMES, weights.tolist(), strict=True),
        key=lambda item: abs(item[1]),
        reverse=True,
    ):
        print(f"  {name:34s} {weight:+.3f}")
    print(f"\nWrote preference reward model to {args.output}")


def overall_driving_score(result: dict[str, Any]) -> float:
    """Safety-constrained score for balanced checkpoint selection."""
    if float(result["completion_rate"]) < 0.90:
        return float("-inf")
    return (
        100.0 * float(result["completion_rate"])
        + 1.5 * float(result["mean_net_overtakes"])
        - 0.30 * float(result["mean_lane_changes"])
        - 0.80 * float(result["mean_lane_reversals"])
        - 0.50 * float(result["mean_missed_passing_opportunities"])
        - 20.0 * float(result["blocked_step_rate"])
    )


class RLAIFEvalCallback(BaseCallback):
    """Save every checkpoint plus independent safety and balanced champions."""

    def __init__(
        self,
        run_dir: Path,
        *,
        eval_freq: int,
        episodes: int,
        seed: int,
        difficulty_mode: str,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.seed = seed
        self.difficulty_mode = difficulty_mode
        self.next_evaluation = eval_freq
        self.best_safety = (float("-inf"), float("-inf"))
        self.best_balanced = float("-inf")
        self.history: list[dict[str, Any]] = []
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_evaluation:
            return True
        result = _evaluation(
            self.model,
            episodes=self.episodes,
            seed=self.seed,
            difficulty_mode=self.difficulty_mode,
        )
        record = {
            "timesteps": self.num_timesteps,
            "overall_driving_score": overall_driving_score(result),
            **result,
        }
        self.history.append(record)
        (self.run_dir / "validation_history.json").write_text(
            json.dumps(self.history, indent=2) + "\n", encoding="utf-8"
        )
        self.model.save(self.run_dir / "checkpoints" / f"model_{self.num_timesteps:09d}")

        safety_score = (float(result["completion_rate"]), float(result["mean_return"]))
        if safety_score > self.best_safety:
            self.best_safety = safety_score
            self.model.save(self.run_dir / "best_safety_model")
            (self.run_dir / "best_safety_validation.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )

        balanced_score = float(record["overall_driving_score"])
        if balanced_score > self.best_balanced:
            self.best_balanced = balanced_score
            self.model.save(self.run_dir / "best_balanced_model")
            (self.run_dir / "best_balanced_validation.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )

        print(
            f"RLAIF validation at {self.num_timesteps:,}: "
            f"completion {result['completion_rate']:.0%}, "
            f"net passes {result['mean_net_overtakes']:+.1f}, "
            f"lane changes {result['mean_lane_changes']:.1f}, "
            f"missed chances {result['mean_missed_passing_opportunities']:.1f}"
        )
        self.next_evaluation += self.eval_freq
        return True


def make_preference_env(
    reward_model_path: str,
    difficulty_mode: str,
    seed: int,
    index: int,
    mirror: bool,
) -> Any:
    """Picklable factory for preference-reward subprocess environments."""

    def _init() -> Monitor:
        base: gym.Env = NeonHighwayEnv(difficulty_mode=difficulty_mode)
        preferred: gym.Env = PreferenceRewardWrapper(
            base, PreferenceRewardModel.load(Path(reward_model_path))
        )
        if mirror:
            preferred = MirrorSymmetry(preferred)
        preferred.reset(seed=seed + 1_000 * index)
        return Monitor(preferred)

    return _init


def _configure_fine_tuning(model: Any, args: argparse.Namespace) -> None:
    model.learning_rate = args.learning_rate
    model.lr_schedule = ConstantSchedule(args.learning_rate)
    for group in model.policy.optimizer.param_groups:
        group["lr"] = args.learning_rate
    model.learning_starts = args.learning_starts
    model.exploration_initial_eps = args.exploration_initial
    model.exploration_final_eps = args.exploration_final
    model.exploration_fraction = args.exploration_fraction
    model.exploration_schedule = LinearSchedule(
        args.exploration_initial,
        args.exploration_final,
        args.exploration_fraction,
    )
    model.exploration_rate = args.exploration_initial


def prefill_balanced_demonstrations(
    model: Any,
    reward_model: PreferenceRewardModel,
    *,
    transitions: int,
    seed: int,
) -> dict[str, int]:
    """Seed replay with preference-selected balanced trajectories.

    This is offline demonstration data, not a runtime safety shield. Once
    learning starts, the neural policy chooses every action itself.
    """
    replay_buffer = model.replay_buffer
    if replay_buffer is None:
        raise RuntimeError("The loaded off-policy model has no replay buffer")
    env_count = int(replay_buffer.n_envs)
    teacher = CandidateController(
        CandidateSpec("balanced-demo", "in-memory-base-model", variant="balanced"),
        model,
    )
    base_envs = [NeonHighwayEnv(difficulty_mode="hard") for _ in range(env_count)]
    wrapped_envs = [PreferenceRewardWrapper(env, reward_model) for env in base_envs]
    rngs = [np.random.default_rng(seed + 10_000 + index) for index in range(env_count)]
    observations = [
        wrapped.reset(seed=seed + index)[0] for index, wrapped in enumerate(wrapped_envs)
    ]
    episodes = completions = crashes = collected = 0
    try:
        while collected < transitions:
            actions = [
                teacher.choose(observation, base, rng)
                for observation, base, rng in zip(observations, base_envs, rngs, strict=True)
            ]
            results = [
                wrapped.step(action)
                for wrapped, action in zip(wrapped_envs, actions, strict=True)
            ]
            next_observations = [result[0] for result in results]
            rewards = [float(result[1]) for result in results]
            dones = [bool(result[2] or result[3]) for result in results]
            infos = [dict(result[4]) for result in results]
            replay_buffer.add(
                np.asarray(observations, dtype=np.float32),
                np.asarray(next_observations, dtype=np.float32),
                np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=np.float32),
                infos,
            )
            collected += env_count
            for index, done in enumerate(dones):
                if done:
                    episodes += 1
                    completions += int(bool(infos[index].get("completed", False)))
                    crashes += int(bool(infos[index].get("crashed", False)))
                    observations[index] = wrapped_envs[index].reset(
                        seed=seed + episodes * env_count + index
                    )[0]
                else:
                    observations[index] = next_observations[index]
    finally:
        for wrapped in wrapped_envs:
            wrapped.close()
    return {
        "transitions": collected,
        "episodes": episodes,
        "completions": completions,
        "crashes": crashes,
    }


def finetune(args: argparse.Namespace) -> None:
    run_name = args.run_name or datetime.now().strftime("rlaif-%Y%m%d-%H%M%S")
    run_dir = Path("runs") / "rlaif" / run_name
    if run_dir.exists():
        raise SystemExit(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    config = {
        "environment": NeonHighwayEnv.VERSION,
        "base_model": str(args.base_model),
        "preference_reward_model": str(args.reward_model),
        "timesteps": args.timesteps,
        "seed": args.seed,
        "parallel_envs": args.envs,
        "difficulty": args.difficulty,
        "mirror_augmentation": args.mirror,
        "learning_rate": args.learning_rate,
        "learning_starts": args.learning_starts,
        "exploration_initial": args.exploration_initial,
        "exploration_final": args.exploration_final,
        "exploration_fraction": args.exploration_fraction,
        "demonstration_transitions": args.demo_transitions,
        "demonstration_seed": args.demo_seed,
        "validation_frequency": args.validation_freq,
        "validation_episodes": args.validation_episodes,
        "validation_seed": args.validation_seed,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["gymnasium", "numpy", "sb3-contrib", "stable-baselines3", "torch"]
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if args.envs > 1:
        training_env: Any = SubprocVecEnv(
            [
                make_preference_env(
                    str(args.reward_model),
                    args.difficulty,
                    args.seed,
                    index,
                    args.mirror,
                )
                for index in range(args.envs)
            ]
        )
        training_env = VecMonitor(training_env, str(run_dir / "monitor.csv"))
    else:
        base: gym.Env = NeonHighwayEnv(difficulty_mode=args.difficulty)
        preferred: gym.Env = PreferenceRewardWrapper(
            base, PreferenceRewardModel.load(args.reward_model)
        )
        if args.mirror:
            preferred = MirrorSymmetry(preferred)
        training_env = Monitor(preferred, str(run_dir / "monitor.csv"))

    model = load_model(args.base_model, device=args.device, env=training_env)
    model.tensorboard_log = str(run_dir / "tensorboard")
    _configure_fine_tuning(model, args)
    if args.demo_transitions > 0:
        demonstration_summary = prefill_balanced_demonstrations(
            model,
            PreferenceRewardModel.load(args.reward_model),
            transitions=args.demo_transitions,
            seed=args.demo_seed,
        )
        model.learning_starts = 0
        (run_dir / "demonstrations.json").write_text(
            json.dumps(demonstration_summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Prefilled replay with {demonstration_summary['transitions']:,} demonstrations")
    callback = RLAIFEvalCallback(
        run_dir,
        eval_freq=args.validation_freq,
        episodes=args.validation_episodes,
        seed=args.validation_seed,
        difficulty_mode=args.difficulty,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nRLAIF fine-tuning interrupted; saving current model.")
    finally:
        model.save(run_dir / "last_model")
        training_env.close()

    balanced_path = run_dir / "best_balanced_model.zip"
    safety_path = run_dir / "best_safety_model.zip"
    recommended_path = balanced_path if balanced_path.exists() else safety_path
    recommended = load_model(recommended_path, device=args.device)
    recommended.save(run_dir / "model")
    evaluation = _evaluation(
        recommended,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        difficulty_mode=args.difficulty,
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved RLAIF agent to {run_dir.resolve()}")
    print(
        f"Evaluation: completion {evaluation['completion_rate']:.0%}, "
        f"net passes {evaluation['mean_net_overtakes']:+.1f}, "
        f"lane changes {evaluation['mean_lane_changes']:.1f}"
    )


def collect_distillation_dataset(
    model: Any,
    *,
    episodes: int,
    seed: int,
) -> dict[str, NDArray[Any]]:
    """Collect only completed trajectories from the AI-selected balanced teacher."""
    controller = CandidateController(
        CandidateSpec("balanced-distillation", "in-memory-base-model", variant="balanced"),
        model,
    )
    env = NeonHighwayEnv(difficulty_mode="hard")
    observations: list[NDArray[np.float32]] = []
    base_actions: list[int] = []
    teacher_actions: list[int] = []
    intervention_kinds: list[int] = []  # 0 agreement, 1 calm correction, 2 pass correction
    temporal_contexts: list[tuple[float, float]] = []
    sample_seeds: list[int] = []
    completed = attempts = crashes = 0
    try:
        while completed < episodes and attempts < 2 * episodes:
            episode_seed = seed + attempts
            observation, _ = env.reset(seed=episode_seed)
            episode_samples: list[
                tuple[NDArray[np.float32], int, int, int, tuple[float, float]]
            ] = []
            terminated = truncated = False
            final_info: dict[str, Any] = {}
            while not (terminated or truncated):
                base_action = int(model.predict(observation, deterministic=True)[0])
                passing_options = env.passing_lane_options()
                teacher_action = controller.adjust(base_action, env)
                if teacher_action == base_action:
                    kind = 0
                else:
                    teacher_steer, _ = decode_action(teacher_action)
                    teacher_lane = env.target_lane + (-1 if teacher_steer == STEER_LEFT else 1)
                    kind = (
                        2
                        if teacher_steer != STEER_KEEP and teacher_lane in passing_options
                        else 1
                    )
                episode_samples.append(
                    (
                        observation.copy(),
                        base_action,
                        teacher_action,
                        kind,
                        (
                            float(
                                np.clip(
                                    (env.step_count - env._last_lane_change_step)
                                    / env.RAPID_LANE_CHANGE_STEPS,
                                    0.0,
                                    1.0,
                                )
                            ),
                            float(env._last_lane_change_direction),
                        ),
                    )
                )
                observation, _, terminated, truncated, final_info = env.step(teacher_action)
            attempts += 1
            if not final_info.get("completed", False):
                crashes += int(bool(final_info.get("crashed", False)))
                continue
            completed += 1
            for (
                sample_observation,
                base_action,
                teacher_action,
                kind,
                temporal_context,
            ) in episode_samples:
                observations.append(sample_observation)
                base_actions.append(base_action)
                teacher_actions.append(teacher_action)
                intervention_kinds.append(kind)
                temporal_contexts.append(temporal_context)
                sample_seeds.append(episode_seed)
    finally:
        env.close()
    if completed < episodes:
        raise RuntimeError(f"Only collected {completed}/{episodes} completed teacher episodes")
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "base_actions": np.asarray(base_actions, dtype=np.int64),
        "teacher_actions": np.asarray(teacher_actions, dtype=np.int64),
        "intervention_kinds": np.asarray(intervention_kinds, dtype=np.int8),
        "temporal_contexts": np.asarray(temporal_contexts, dtype=np.float32),
        "seeds": np.asarray(sample_seeds, dtype=np.int64),
        "attempts": np.asarray([attempts], dtype=np.int64),
        "crashes_discarded": np.asarray([crashes], dtype=np.int64),
    }


class PreferenceOverrideNet(nn.Module):
    """Small residual policy that leaves the safety policy frozen by default."""

    def __init__(self, observation_size: int, action_count: int = ACTION_COUNT) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.action_count = action_count
        self.trunk = nn.Sequential(
            nn.Linear(observation_size + action_count, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.kind_head = nn.Linear(64, 3)
        self.action_head = nn.Linear(64, action_count)

    def forward(
        self,
        observations: th.Tensor,
        base_actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        action_one_hot = th.nn.functional.one_hot(
            base_actions.long(), num_classes=self.action_count
        ).to(dtype=observations.dtype)
        hidden = self.trunk(th.cat((observations, action_one_hot), dim=1))
        return self.kind_head(hidden), self.action_head(hidden)


def _network_observations(
    dataset: dict[str, NDArray[Any]],
) -> NDArray[np.float32]:
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    contexts = dataset.get("temporal_contexts")
    if contexts is None:
        return observations
    return np.concatenate(
        (observations, np.asarray(contexts, dtype=np.float32)), axis=1
    )


@dataclass(frozen=True)
class OverrideThresholds:
    calm: float
    passing: float


def _override_predictions(
    network: PreferenceOverrideNet,
    dataset: dict[str, NDArray[Any]],
    *,
    device: th.device,
    batch_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    probabilities: list[NDArray[np.float32]] = []
    actions: list[NDArray[np.int64]] = []
    network.eval()
    network_observations = _network_observations(dataset)
    with th.no_grad():
        for start in range(0, len(network_observations), batch_size):
            stop = start + batch_size
            observations = th.as_tensor(
                network_observations[start:stop], dtype=th.float32, device=device
            )
            base_actions = th.as_tensor(
                dataset["base_actions"][start:stop], dtype=th.long, device=device
            )
            kind_logits, action_logits = network(observations, base_actions)
            probabilities.append(th.softmax(kind_logits, dim=1).cpu().numpy())
            actions.append(action_logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(actions)


def _apply_override_gate(
    base_actions: NDArray[np.int64],
    kind_probabilities: NDArray[np.float32],
    proposed_actions: NDArray[np.int64],
    thresholds: OverrideThresholds,
) -> tuple[NDArray[np.int64], NDArray[np.int8]]:
    """Choose at most one confident residual correction per observation."""
    calm_strength = kind_probabilities[:, 1] / max(thresholds.calm, 1e-6)
    passing_strength = kind_probabilities[:, 2] / max(thresholds.passing, 1e-6)
    selected_kinds = np.zeros(len(base_actions), dtype=np.int8)
    calm_mask = (kind_probabilities[:, 1] >= thresholds.calm) & (
        calm_strength >= passing_strength
    )
    passing_mask = (kind_probabilities[:, 2] >= thresholds.passing) & (
        passing_strength > calm_strength
    )
    selected_kinds[calm_mask] = 1
    selected_kinds[passing_mask] = 2
    final_actions = np.asarray(base_actions, dtype=np.int64).copy()
    override_mask = selected_kinds > 0
    final_actions[override_mask] = proposed_actions[override_mask]
    selected_kinds[final_actions == base_actions] = 0
    return final_actions, selected_kinds


def _override_metrics(
    dataset: dict[str, NDArray[Any]],
    kind_probabilities: NDArray[np.float32],
    proposed_actions: NDArray[np.int64],
    thresholds: OverrideThresholds,
) -> dict[str, float | int]:
    base_actions = dataset["base_actions"]
    expected = dataset["teacher_actions"]
    kinds = dataset["intervention_kinds"]
    final_actions, selected_kinds = _apply_override_gate(
        base_actions, kind_probabilities, proposed_actions, thresholds
    )
    override_mask = final_actions != base_actions

    def accuracy(mask: NDArray[np.bool_]) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(final_actions[mask] == expected[mask]))

    agreement_mask = kinds == 0
    calm_mask = kinds == 1
    passing_mask = kinds == 2
    correct_override = override_mask & (final_actions == expected)
    return {
        "overall_accuracy": accuracy(np.ones(len(expected), dtype=bool)),
        "agreement_preservation": accuracy(agreement_mask),
        "calm_correction_accuracy": accuracy(calm_mask),
        "passing_correction_accuracy": accuracy(passing_mask),
        "override_precision": (
            float(np.sum(correct_override) / np.sum(override_mask))
            if np.any(override_mask)
            else 1.0
        ),
        "override_rate": float(np.mean(override_mask)),
        "overrides": int(np.sum(override_mask)),
        "calm_overrides": int(np.sum(selected_kinds == 1)),
        "passing_overrides": int(np.sum(selected_kinds == 2)),
        "samples": int(len(expected)),
        "calm_examples": int(np.sum(calm_mask)),
        "passing_examples": int(np.sum(passing_mask)),
    }


def _tune_override_thresholds(
    dataset: dict[str, NDArray[Any]],
    kind_probabilities: NDArray[np.float32],
    proposed_actions: NDArray[np.int64],
    *,
    minimum_preservation: float,
    minimum_precision: float,
) -> tuple[OverrideThresholds, dict[str, float | int]]:
    """Tune confidence gates without consulting an on-policy test seed."""
    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(0.50, 0.99, 26),
                np.quantile(kind_probabilities[:, 1], np.linspace(0.50, 0.999, 24)),
                np.quantile(kind_probabilities[:, 2], np.linspace(0.50, 0.999, 24)),
            )
        )
    )
    candidates = candidates[(candidates >= 0.45) & (candidates <= 0.9999)]
    best: tuple[float, OverrideThresholds, dict[str, float | int]] | None = None
    for calm_threshold in candidates:
        for passing_threshold in candidates:
            thresholds = OverrideThresholds(
                calm=float(calm_threshold), passing=float(passing_threshold)
            )
            metrics = _override_metrics(
                dataset, kind_probabilities, proposed_actions, thresholds
            )
            if metrics["agreement_preservation"] < minimum_preservation:
                continue
            if metrics["override_precision"] < minimum_precision:
                continue
            if metrics["passing_overrides"] == 0:
                continue
            score = (
                0.48 * float(metrics["passing_correction_accuracy"])
                + 0.17 * float(metrics["calm_correction_accuracy"])
                + 0.20 * float(metrics["overall_accuracy"])
                + 0.10 * float(metrics["override_precision"])
                + 0.05 * float(metrics["agreement_preservation"])
            )
            if best is None or score > best[0]:
                best = score, thresholds, metrics
    if best is None:
        raise RuntimeError(
            "No override gate met the requested preservation and precision constraints"
        )
    return best[1], best[2]


class PreferenceOverridePolicy:
    """Callable composite policy: frozen V5 base plus confidence-gated residual."""

    def __init__(
        self,
        base_model: Any,
        network: PreferenceOverrideNet,
        thresholds: OverrideThresholds,
        *,
        device: str | th.device = "cpu",
    ) -> None:
        self.base_model = base_model
        self.network = network
        self.thresholds = thresholds
        self.device = th.device(device)
        self.network.to(self.device).eval()
        self.last_decision = "V5 BASE"
        self._previous_target_lane: int | None = None
        self._previous_route_remaining: float | None = None
        self._steps_since_lane_change = 1_000_000
        self._last_lane_change_direction = 0

    def reset(self) -> None:
        """Clear episode memory; `run_policy` calls this after each environment reset."""
        self._previous_target_lane = None
        self._previous_route_remaining = None
        self._steps_since_lane_change = 1_000_000
        self._last_lane_change_direction = 0
        self.last_decision = "V5 BASE"

    def _temporal_context(
        self, observation: NDArray[np.floating[Any]]
    ) -> NDArray[np.float32]:
        target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
        route_remaining = float(observation[6])
        reset = (
            self._previous_route_remaining is not None
            and route_remaining > self._previous_route_remaining + 0.25
        )
        if self._previous_target_lane is None or reset:
            self._steps_since_lane_change = 1_000_000
            self._last_lane_change_direction = 0
        elif target_lane != self._previous_target_lane:
            self._steps_since_lane_change = 1
            self._last_lane_change_direction = int(
                np.sign(target_lane - self._previous_target_lane)
            )
        else:
            self._steps_since_lane_change += 1
        self._previous_target_lane = target_lane
        self._previous_route_remaining = route_remaining
        return np.asarray(
            [
                min(
                    self._steps_since_lane_change
                    / NeonHighwayEnv.RAPID_LANE_CHANGE_STEPS,
                    1.0,
                ),
                float(self._last_lane_change_direction),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _passing_options(
        observation: NDArray[np.floating[Any]],
    ) -> tuple[int, ...]:
        """Reconstruct the environment's safe/useful pass check from observation."""
        if float(observation[4]) >= 0.05:
            return ()
        target_lane = int(round(float(observation[3]) * (NeonHighwayEnv.LANES - 1)))
        speed_span = NeonHighwayEnv.MAX_SPEED - NeonHighwayEnv.MIN_SPEED

        def lane_reading(lane: int) -> tuple[float, float, float, float]:
            offset = 9 + 6 * lane
            return (
                float(observation[offset]) * NeonHighwayEnv.SENSOR_DISTANCE,
                float(observation[offset + 1]) * speed_span,
                float(observation[offset + 3]) * NeonHighwayEnv.SENSOR_DISTANCE,
                float(observation[offset + 4]) * speed_span,
            )

        front_gap, front_relative, _, _ = lane_reading(target_lane)
        if (
            front_gap >= NeonHighwayEnv.PASSING_TRIGGER_GAP
            or front_relative >= -NeonHighwayEnv.PASSING_MIN_CLOSING_SPEED
        ):
            return ()
        options: list[int] = []
        for candidate in (target_lane - 1, target_lane + 1):
            if not 0 <= candidate < NeonHighwayEnv.LANES:
                continue
            ahead_gap, ahead_relative, behind_gap, behind_relative = lane_reading(candidate)
            rear_closing_speed = max(behind_relative, 0.0)
            rear_ttc = (
                behind_gap / rear_closing_speed
                if rear_closing_speed > 0.1
                else float("inf")
            )
            safe = (
                ahead_gap > NeonHighwayEnv.SAFE_LANE_CHANGE_FRONT_GAP
                and behind_gap > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_GAP
                and rear_ttc > NeonHighwayEnv.SAFE_LANE_CHANGE_REAR_TTC
            )
            materially_better = (
                ahead_gap > front_gap + NeonHighwayEnv.PASSING_CLEARANCE_GAIN
                and (
                    ahead_relative
                    > front_relative + NeonHighwayEnv.PASSING_MIN_CLOSING_SPEED
                    or ahead_gap
                    > front_gap + 2.0 * NeonHighwayEnv.PASSING_CLEARANCE_GAIN
                )
            )
            if safe and materially_better:
                options.append(candidate)
        return tuple(options)

    @staticmethod
    def _current_threat_level(
        observation: NDArray[np.floating[Any]],
    ) -> float:
        current_lane = int(round(float(observation[2]) * (NeonHighwayEnv.LANES - 1)))
        offset = 9 + 6 * current_lane
        return max(float(observation[offset + 2]), float(observation[offset + 5]))

    def _shield_override(
        self,
        observation: NDArray[np.floating[Any]],
        context: NDArray[np.float32],
        base_action: int,
        proposed_action: int,
        kind: int,
    ) -> tuple[int, int]:
        base_steer, base_pedal = decode_action(base_action)
        proposed_steer, proposed_pedal = decode_action(proposed_action)
        same_pedal = proposed_pedal == base_pedal
        if kind == 1:
            base_direction = -1 if base_steer == STEER_LEFT else 1
            valid = (
                base_steer != STEER_KEEP
                and proposed_steer == STEER_KEEP
                and same_pedal
                and float(context[0]) < 1.0
                and base_direction == -int(context[1])
                and float(observation[4]) < 0.05
                and self._current_threat_level(observation) < 0.6
            )
        elif kind == 2:
            target_lane = int(
                round(float(observation[3]) * (NeonHighwayEnv.LANES - 1))
            )
            proposed_lane = target_lane + (-1 if proposed_steer == STEER_LEFT else 1)
            valid = (
                proposed_steer != STEER_KEEP
                and same_pedal
                and proposed_lane in self._passing_options(observation)
            )
        else:
            valid = False
        return (proposed_action, kind) if valid else (base_action, 0)

    def __call__(self, observation: NDArray[np.floating[Any]]) -> int:
        base_action = int(self.base_model.predict(observation, deterministic=True)[0])
        context = self._temporal_context(observation)
        network_observation = np.concatenate(
            (np.asarray(observation, dtype=np.float32), context)
        )
        with th.no_grad():
            observation_tensor = th.as_tensor(
                network_observation[None, :], device=self.device
            )
            base_tensor = th.as_tensor([base_action], dtype=th.long, device=self.device)
            kind_logits, action_logits = self.network(observation_tensor, base_tensor)
            probabilities = th.softmax(kind_logits, dim=1).cpu().numpy()
            proposed = action_logits.argmax(dim=1).cpu().numpy()
        final_actions, selected_kinds = _apply_override_gate(
            np.asarray([base_action], dtype=np.int64),
            probabilities,
            proposed,
            self.thresholds,
        )
        final_action, selected_kind = self._shield_override(
            observation,
            context,
            base_action,
            int(final_actions[0]),
            int(selected_kinds[0]),
        )
        self.last_decision = {0: "V5 BASE", 1: "CALM", 2: "PASS"}[
            selected_kind
        ]
        return final_action


def load_override_policy(
    base_model_path: Path,
    override_model_path: Path,
    *,
    device: str = "cpu",
    thresholds: OverrideThresholds | None = None,
) -> PreferenceOverridePolicy:
    payload = th.load(override_model_path, map_location=device, weights_only=True)
    network = PreferenceOverrideNet(
        observation_size=int(payload["observation_size"]),
        action_count=int(payload["action_count"]),
    )
    network.load_state_dict(payload["state_dict"])
    selected_thresholds = thresholds or OverrideThresholds(**payload["thresholds"])
    base_model = load_model(base_model_path, device=device)
    return PreferenceOverridePolicy(
        base_model, network, selected_thresholds, device=device
    )


def _distillation_metrics(
    network: Any,
    dataset: dict[str, NDArray[Any]],
    *,
    device: th.device,
    batch_size: int,
) -> dict[str, float]:
    predictions: list[NDArray[np.int64]] = []
    observations = dataset["observations"]
    network.eval()
    with th.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = th.as_tensor(observations[start : start + batch_size], device=device)
            predictions.append(network(batch).mean(dim=1).argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)
    expected = dataset["teacher_actions"]
    kinds = dataset["intervention_kinds"]

    def accuracy(mask: NDArray[np.bool_]) -> float:
        return float(np.mean(predicted[mask] == expected[mask])) if np.any(mask) else float("nan")

    return {
        "overall_accuracy": accuracy(np.ones(len(expected), dtype=bool)),
        "agreement_accuracy": accuracy(kinds == 0),
        "intervention_accuracy": accuracy(kinds > 0),
        "calm_intervention_accuracy": accuracy(kinds == 1),
        "passing_intervention_accuracy": accuracy(kinds == 2),
        "samples": int(len(expected)),
        "interventions": int(np.sum(kinds > 0)),
        "passing_interventions": int(np.sum(kinds == 2)),
    }


def distill(args: argparse.Namespace) -> None:
    """Fine-tune only the action-value head toward preferred corrections."""
    run_name = args.run_name or datetime.now().strftime("distill-%Y%m%d-%H%M%S")
    run_dir = Path("runs") / "rlaif" / run_name
    if run_dir.exists():
        raise SystemExit(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    teacher = load_model(args.base_model, device="cpu")
    train_data = collect_distillation_dataset(
        teacher, episodes=args.train_episodes, seed=args.train_seed
    )
    validation_data = collect_distillation_dataset(
        teacher, episodes=args.validation_episodes, seed=args.validation_seed
    )
    np.savez_compressed(run_dir / "train_demonstrations.npz", **train_data)
    np.savez_compressed(run_dir / "validation_demonstrations.npz", **validation_data)

    student = load_model(args.base_model, device=args.device)
    network = student.quantile_net
    original = copy.deepcopy(network).to(student.device).eval()
    for parameter in network.parameters():
        parameter.requires_grad = False
    output_layer = network.quantile_net[-1]
    for parameter in output_layer.parameters():
        parameter.requires_grad = True
    optimizer = th.optim.Adam(output_layer.parameters(), lr=args.learning_rate)

    calm_indices = np.flatnonzero(train_data["intervention_kinds"] == 1)
    passing_indices = np.flatnonzero(train_data["intervention_kinds"] == 2)
    intervention_indices = np.concatenate([calm_indices, passing_indices])
    agreement_indices = np.flatnonzero(train_data["intervention_kinds"] == 0)
    if not len(intervention_indices):
        raise RuntimeError("Balanced teacher produced no action corrections")
    rng = np.random.default_rng(args.seed)
    device = student.device
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_state = copy.deepcopy(network.state_dict())

    for epoch in range(1, args.epochs + 1):
        sampled_agreements = rng.choice(
            agreement_indices,
            size=min(len(agreement_indices), args.agreement_ratio * len(intervention_indices)),
            replace=False,
        )
        epoch_indices = np.concatenate(
            [
                np.repeat(calm_indices, args.calm_repeats),
                np.repeat(passing_indices, args.passing_repeats),
                sampled_agreements,
            ]
        )
        rng.shuffle(epoch_indices)
        network.train()
        losses = []
        for start in range(0, len(epoch_indices), args.batch_size):
            indices = epoch_indices[start : start + args.batch_size]
            observations = th.as_tensor(train_data["observations"][indices], device=device)
            target_actions = th.as_tensor(
                train_data["teacher_actions"][indices], dtype=th.long, device=device
            )
            quantiles = network(observations)
            q_values = quantiles.mean(dim=1)
            with th.no_grad():
                original_q = original(observations).mean(dim=1)
            teacher_q = q_values.gather(1, target_actions[:, None]).squeeze(1)
            other_q = q_values.clone()
            other_q.scatter_(1, target_actions[:, None], float("-inf"))
            ranking_loss = th.relu(args.margin + other_q.max(dim=1).values - teacher_q).mean()
            anchor_loss = th.nn.functional.mse_loss(q_values, original_q)
            loss = ranking_loss + args.anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_metrics = _distillation_metrics(
            network, train_data, device=device, batch_size=args.batch_size
        )
        validation_metrics = _distillation_metrics(
            network, validation_data, device=device, batch_size=args.batch_size
        )
        score = (
            0.50 * validation_metrics["passing_intervention_accuracy"]
            + 0.20 * validation_metrics["calm_intervention_accuracy"]
            + 0.30 * validation_metrics["overall_accuracy"]
        )
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "selection_score": score,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(network.state_dict())
        print(
            f"Epoch {epoch:02d}: overall {validation_metrics['overall_accuracy']:.1%}, "
            f"pass corrections {validation_metrics['passing_intervention_accuracy']:.1%}, "
            f"calm corrections {validation_metrics['calm_intervention_accuracy']:.1%}"
        )

    network.load_state_dict(best_state)
    student.quantile_net_target.load_state_dict(best_state)
    for parameter in network.parameters():
        parameter.requires_grad = True
    student.save(run_dir / "model")
    (run_dir / "distillation_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "base_model": str(args.base_model),
        "preference_reward_model": str(args.reward_model),
        "teacher": "AI-preferred balanced V5 corrections",
        "train_seed": args.train_seed,
        "train_episodes": args.train_episodes,
        "validation_seed": args.validation_seed,
        "validation_episodes": args.validation_episodes,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "anchor_weight": args.anchor_weight,
        "calm_repeats": args.calm_repeats,
        "passing_repeats": args.passing_repeats,
        "agreement_ratio": args.agreement_ratio,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    evaluation = _evaluation(
        student,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        difficulty_mode="hard",
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Distilled evaluation: completion {evaluation['completion_rate']:.0%}, "
        f"net passes {evaluation['mean_net_overtakes']:+.1f}, "
        f"lane changes {evaluation['mean_lane_changes']:.1f}, "
        f"passing response {evaluation['passing_response_rate']:.0%}"
    )


def _training_device(name: str) -> th.device:
    if name == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    device = th.device(name)
    if device.type == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU")
    return device


def train_override(args: argparse.Namespace) -> None:
    """Train a high-precision residual policy while keeping V5 frozen."""
    run_name = args.run_name or datetime.now().strftime("override-%Y%m%d-%H%M%S")
    run_dir = Path("runs") / "rlaif" / run_name
    if run_dir.exists():
        raise SystemExit(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    print("Collecting completed preference-teacher drives...")
    base_model = load_model(args.base_model, device="cpu")
    train_data = collect_distillation_dataset(
        base_model, episodes=args.train_episodes, seed=args.train_seed
    )
    validation_data = collect_distillation_dataset(
        base_model, episodes=args.validation_episodes, seed=args.validation_seed
    )
    np.savez_compressed(run_dir / "train_demonstrations.npz", **train_data)
    np.savez_compressed(run_dir / "validation_demonstrations.npz", **validation_data)

    device = _training_device(args.device)
    train_network_observations = _network_observations(train_data)
    observation_size = int(train_network_observations.shape[1])
    network = PreferenceOverrideNet(observation_size).to(device)
    optimizer = th.optim.AdamW(
        network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    th.manual_seed(args.seed)
    if device.type == "cuda":
        th.cuda.manual_seed_all(args.seed)

    kinds = train_data["intervention_kinds"]
    agreement_indices = np.flatnonzero(kinds == 0)
    calm_indices = np.flatnonzero(kinds == 1)
    passing_indices = np.flatnonzero(kinds == 2)
    if not len(calm_indices) or not len(passing_indices):
        raise RuntimeError("Teacher data needs both calm and passing corrections")

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_state: dict[str, th.Tensor] | None = None
    best_thresholds: OverrideThresholds | None = None
    best_validation: dict[str, float | int] | None = None
    print(
        f"Training on {len(kinds):,} states: {len(passing_indices):,} pass, "
        f"{len(calm_indices):,} calm, {len(agreement_indices):,} unchanged."
    )

    for epoch in range(1, args.epochs + 1):
        intervention_count = len(calm_indices) + len(passing_indices)
        agreement_count = min(
            len(agreement_indices), args.agreement_ratio * intervention_count
        )
        sampled_agreements = rng.choice(
            agreement_indices, size=agreement_count, replace=False
        )
        epoch_indices = np.concatenate(
            (
                np.repeat(calm_indices, args.calm_repeats),
                np.repeat(passing_indices, args.passing_repeats),
                sampled_agreements,
            )
        )
        rng.shuffle(epoch_indices)
        network.train()
        losses: list[float] = []
        for start in range(0, len(epoch_indices), args.batch_size):
            indices = epoch_indices[start : start + args.batch_size]
            observations = th.as_tensor(
                train_network_observations[indices], dtype=th.float32, device=device
            )
            base_actions = th.as_tensor(
                train_data["base_actions"][indices], dtype=th.long, device=device
            )
            target_kinds = th.as_tensor(
                kinds[indices], dtype=th.long, device=device
            )
            target_actions = th.as_tensor(
                train_data["teacher_actions"][indices], dtype=th.long, device=device
            )
            kind_logits, action_logits = network(observations, base_actions)
            kind_loss = th.nn.functional.cross_entropy(
                kind_logits, target_kinds, label_smoothing=args.label_smoothing
            )
            action_loss = th.nn.functional.cross_entropy(
                action_logits, target_actions, label_smoothing=args.label_smoothing
            )
            loss = kind_loss + args.action_loss_weight * action_loss
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        probabilities, proposed_actions = _override_predictions(
            network, validation_data, device=device, batch_size=args.batch_size
        )
        try:
            thresholds, validation_metrics = _tune_override_thresholds(
                validation_data,
                probabilities,
                proposed_actions,
                minimum_preservation=args.minimum_preservation,
                minimum_precision=args.minimum_precision,
            )
        except RuntimeError:
            validation_metrics = None
        if validation_metrics is None:
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid": False})
            print(f"Epoch {epoch:02d}: no safe confidence gate yet")
            continue

        score = (
            0.48 * float(validation_metrics["passing_correction_accuracy"])
            + 0.17 * float(validation_metrics["calm_correction_accuracy"])
            + 0.20 * float(validation_metrics["overall_accuracy"])
            + 0.10 * float(validation_metrics["override_precision"])
            + 0.05 * float(validation_metrics["agreement_preservation"])
        )
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "valid": True,
            "selection_score": score,
            "thresholds": asdict(thresholds),
            "validation": validation_metrics,
        }
        history.append(record)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(network.state_dict())
            best_thresholds = thresholds
            best_validation = validation_metrics
        print(
            f"Epoch {epoch:02d}: pass {validation_metrics['passing_correction_accuracy']:.1%}, "
            f"calm {validation_metrics['calm_correction_accuracy']:.1%}, "
            f"preserve {validation_metrics['agreement_preservation']:.2%}, "
            f"precision {validation_metrics['override_precision']:.1%}"
        )

    if best_state is None or best_thresholds is None or best_validation is None:
        raise RuntimeError("Training never produced a residual that passed the safety gate")
    network.load_state_dict(best_state)
    artifact_path = run_dir / "override_model.pt"
    th.save(
        {
            "state_dict": network.cpu().state_dict(),
            "observation_size": observation_size,
            "action_count": ACTION_COUNT,
            "thresholds": asdict(best_thresholds),
            "base_model": str(args.base_model),
            "preference_reward_model": str(args.reward_model),
        },
        artifact_path,
    )
    (run_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "method": "confidence-gated residual preference distillation",
        "base_model": str(args.base_model),
        "preference_reward_model": str(args.reward_model),
        "train_seed": args.train_seed,
        "train_episodes": args.train_episodes,
        "validation_seed": args.validation_seed,
        "validation_episodes": args.validation_episodes,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "thresholds": asdict(best_thresholds),
        "offline_validation": best_validation,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    policy = load_override_policy(args.base_model, artifact_path, device="cpu")
    env = NeonHighwayEnv(difficulty_mode="hard")
    try:
        evaluation = evaluate_in_env(
            env, policy, episodes=args.eval_episodes, seed=args.eval_seed
        ).to_dict()
    finally:
        env.close()
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved gated RLAIF driver to {artifact_path.resolve()}")
    print(
        f"Evaluation: completion {evaluation['completion_rate']:.0%}, "
        f"net passes {evaluation['mean_net_overtakes']:+.1f}, "
        f"lane changes {evaluation['mean_lane_changes']:.1f}, "
        f"passing response {evaluation['passing_response_rate']:.0%}"
    )


def evaluate_override(args: argparse.Namespace) -> None:
    thresholds = _optional_thresholds(args)
    policy = load_override_policy(
        args.base_model, args.override_model, device=args.device, thresholds=thresholds
    )
    if args.longitudinal_intent:
        policy = LongitudinalIntentPolicy(policy)
    env = NeonHighwayEnv(difficulty_mode=args.difficulty)
    try:
        result = evaluate_in_env(
            env, policy, episodes=args.episodes, seed=args.seed
        ).to_dict()
    finally:
        env.close()
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def watch_override(args: argparse.Namespace) -> None:
    thresholds = _optional_thresholds(args)
    policy = load_override_policy(
        args.base_model, args.override_model, device=args.device, thresholds=thresholds
    )
    if args.longitudinal_intent:
        policy = LongitudinalIntentPolicy(policy)
    env = NeonHighwayEnv(
        render_mode="human",
        render_fps=args.fps,
        render_speed=args.speed,
        difficulty_mode=args.difficulty,
        episode_seconds=args.seconds,
        endless=args.endless,
    )
    try:
        run_policy(
            env,
            policy,
            episodes=args.episodes,
            seed=args.seed,
            mode="RLAIF V6 · GATED DRIVER",
            epsilon=0.0,
        )
    finally:
        env.close()


def _optional_thresholds(args: argparse.Namespace) -> OverrideThresholds | None:
    calm = args.calm_threshold
    passing = args.passing_threshold
    if (calm is None) != (passing is None):
        raise SystemExit("Provide both --calm-threshold and --passing-threshold")
    if calm is None:
        return None
    if not 0.0 <= calm <= 1.1 or not 0.0 <= passing <= 1.1:
        raise SystemExit("Override thresholds must be between 0.0 and 1.1")
    return OverrideThresholds(calm=float(calm), passing=float(passing))


def calibrate_override(args: argparse.Namespace) -> None:
    """Package validation-selected gates into a deployable override artifact."""
    payload = th.load(args.input, map_location="cpu", weights_only=True)
    payload["thresholds"] = asdict(
        OverrideThresholds(calm=args.calm_threshold, passing=args.passing_threshold)
    )
    payload["calibration"] = {
        "method": "matched on-road development seeds; safety confirmed on held-out seeds",
        "development_seed": args.development_seed,
        "heldout_seed": args.heldout_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    th.save(payload, args.output)
    print(f"Saved calibrated override to {args.output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="Collect matched trajectories for AI preference labelling"
    )
    collect_parser.add_argument(
        "--safe-model",
        type=Path,
        default=Path("runs/game/v5-good-driver-2p5m-restart/model.zip"),
    )
    collect_parser.add_argument(
        "--assertive-model",
        type=Path,
        default=Path("runs/game/v5-good-driver-2p5m-restart/last_model.zip"),
    )
    collect_parser.add_argument(
        "--pilot-model",
        type=Path,
        default=Path("runs/game/v5-safe-driver-pilot-300k/best_model.zip"),
    )
    collect_parser.add_argument("--episodes", type=int, default=16)
    collect_parser.add_argument("--seed", type=int, default=70_000)
    collect_parser.add_argument("--difficulty", choices=["standard", "hard"], default="hard")
    collect_parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    collect_parser.add_argument(
        "--output", type=Path, default=Path("runs/rlaif/v1/preferences.json")
    )
    collect_parser.set_defaults(handler=collect_preferences)

    reward_parser = subparsers.add_parser(
        "train-reward", help="Fit an interpretable Bradley-Terry preference model"
    )
    reward_parser.add_argument("--dataset", type=Path, required=True)
    reward_parser.add_argument("--labels", type=Path, default=None)
    reward_parser.add_argument("--output", type=Path, required=True)
    reward_parser.add_argument("--split-seed", type=int, default=1701)
    reward_parser.add_argument("--learning-rate", type=float, default=0.03)
    reward_parser.add_argument("--l2", type=float, default=0.02)
    reward_parser.add_argument("--epochs", type=int, default=4_000)
    reward_parser.add_argument("--patience", type=int, default=250)
    reward_parser.add_argument("--target-reward-std", type=float, default=4.0)
    reward_parser.add_argument("--clip-per-step", type=float, default=1.0)
    reward_parser.set_defaults(handler=train_reward_model)

    fine_parser = subparsers.add_parser(
        "finetune", help="Fine-tune a frozen V5 clone with learned preference reward"
    )
    fine_parser.add_argument("--base-model", type=Path, required=True)
    fine_parser.add_argument("--reward-model", type=Path, required=True)
    fine_parser.add_argument("--run-name", type=str, default=None)
    fine_parser.add_argument("--timesteps", type=int, default=300_000)
    fine_parser.add_argument("--seed", type=int, default=31)
    fine_parser.add_argument("--envs", type=int, default=8)
    fine_parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    fine_parser.add_argument("--difficulty", choices=["standard", "hard"], default="hard")
    fine_parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    fine_parser.set_defaults(mirror=True)
    fine_parser.add_argument("--learning-rate", type=float, default=5e-5)
    fine_parser.add_argument("--learning-starts", type=int, default=5_000)
    fine_parser.add_argument("--exploration-initial", type=float, default=0.08)
    fine_parser.add_argument("--exploration-final", type=float, default=0.01)
    fine_parser.add_argument("--exploration-fraction", type=float, default=0.20)
    fine_parser.add_argument("--demo-transitions", type=int, default=20_000)
    fine_parser.add_argument("--demo-seed", type=int, default=75_000)
    fine_parser.add_argument("--validation-freq", type=int, default=50_000)
    fine_parser.add_argument("--validation-episodes", type=int, default=60)
    fine_parser.add_argument("--validation-seed", type=int, default=80_000)
    fine_parser.add_argument("--eval-episodes", type=int, default=100)
    fine_parser.add_argument("--eval-seed", type=int, default=90_000)
    fine_parser.set_defaults(handler=finetune)

    distill_parser = subparsers.add_parser(
        "distill", help="Distill AI-preferred passing and calm-control corrections"
    )
    distill_parser.add_argument("--base-model", type=Path, required=True)
    distill_parser.add_argument("--reward-model", type=Path, required=True)
    distill_parser.add_argument("--run-name", type=str, default=None)
    distill_parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    distill_parser.add_argument("--seed", type=int, default=47)
    distill_parser.add_argument("--train-seed", type=int, default=72_000)
    distill_parser.add_argument("--train-episodes", type=int, default=120)
    distill_parser.add_argument("--validation-seed", type=int, default=73_000)
    distill_parser.add_argument("--validation-episodes", type=int, default=30)
    distill_parser.add_argument("--eval-seed", type=int, default=90_000)
    distill_parser.add_argument("--eval-episodes", type=int, default=100)
    distill_parser.add_argument("--epochs", type=int, default=10)
    distill_parser.add_argument("--learning-rate", type=float, default=1e-4)
    distill_parser.add_argument("--batch-size", type=int, default=512)
    distill_parser.add_argument("--margin", type=float, default=0.02)
    distill_parser.add_argument("--anchor-weight", type=float, default=10.0)
    distill_parser.add_argument("--calm-repeats", type=int, default=5)
    distill_parser.add_argument("--passing-repeats", type=int, default=30)
    distill_parser.add_argument("--agreement-ratio", type=int, default=2)
    distill_parser.set_defaults(handler=distill)

    override_parser = subparsers.add_parser(
        "override-train",
        help="Train a confidence-gated preference residual over the frozen V5 driver",
    )
    override_parser.add_argument("--base-model", type=Path, required=True)
    override_parser.add_argument("--reward-model", type=Path, required=True)
    override_parser.add_argument("--run-name", type=str, default=None)
    override_parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    override_parser.add_argument("--seed", type=int, default=53)
    override_parser.add_argument("--train-seed", type=int, default=72_000)
    override_parser.add_argument("--train-episodes", type=int, default=120)
    override_parser.add_argument("--validation-seed", type=int, default=73_000)
    override_parser.add_argument("--validation-episodes", type=int, default=30)
    override_parser.add_argument("--eval-seed", type=int, default=90_000)
    override_parser.add_argument("--eval-episodes", type=int, default=100)
    override_parser.add_argument("--epochs", type=int, default=25)
    override_parser.add_argument("--learning-rate", type=float, default=3e-4)
    override_parser.add_argument("--weight-decay", type=float, default=1e-4)
    override_parser.add_argument("--batch-size", type=int, default=512)
    override_parser.add_argument("--label-smoothing", type=float, default=0.02)
    override_parser.add_argument("--action-loss-weight", type=float, default=0.8)
    override_parser.add_argument("--calm-repeats", type=int, default=3)
    override_parser.add_argument("--passing-repeats", type=int, default=100)
    override_parser.add_argument("--agreement-ratio", type=int, default=3)
    override_parser.add_argument("--minimum-preservation", type=float, default=0.995)
    override_parser.add_argument("--minimum-precision", type=float, default=0.80)
    override_parser.set_defaults(handler=train_override)

    evaluate_parser = subparsers.add_parser(
        "override-evaluate", help="Evaluate a frozen-base gated RLAIF driver"
    )
    evaluate_parser.add_argument("--base-model", type=Path, required=True)
    evaluate_parser.add_argument("--override-model", type=Path, required=True)
    evaluate_parser.add_argument("--episodes", type=int, default=100)
    evaluate_parser.add_argument("--seed", type=int, default=90_000)
    evaluate_parser.add_argument(
        "--difficulty", choices=["standard", "hard"], default="hard"
    )
    evaluate_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    evaluate_parser.add_argument("--calm-threshold", type=float, default=None)
    evaluate_parser.add_argument("--passing-threshold", type=float, default=None)
    evaluate_parser.add_argument(
        "--longitudinal-intent",
        action="store_true",
        help="Use the V7 persistent speed-intent controller",
    )
    evaluate_parser.add_argument("--output", type=Path, default=None)
    evaluate_parser.set_defaults(handler=evaluate_override)

    watch_parser = subparsers.add_parser(
        "watch", help="Watch the gated RLAIF driver in Neon Highway"
    )
    watch_parser.add_argument("--base-model", type=Path, required=True)
    watch_parser.add_argument("--override-model", type=Path, required=True)
    watch_parser.add_argument("--episodes", type=int, default=10)
    watch_parser.add_argument("--seed", type=int, default=42)
    watch_parser.add_argument(
        "--difficulty", choices=["standard", "hard"], default="hard"
    )
    watch_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    watch_parser.add_argument("--calm-threshold", type=float, default=None)
    watch_parser.add_argument("--passing-threshold", type=float, default=None)
    watch_parser.add_argument(
        "--longitudinal-intent",
        action="store_true",
        help="Use the V7 persistent speed-intent controller",
    )
    watch_parser.add_argument("--fps", type=int, default=60)
    watch_parser.add_argument("--speed", type=float, default=1.0)
    watch_parser.add_argument("--seconds", type=float, default=None)
    watch_parser.add_argument("--endless", action="store_true")
    watch_parser.set_defaults(handler=watch_override)

    calibrate_parser = subparsers.add_parser(
        "override-calibrate", help="Package validation-selected confidence gates"
    )
    calibrate_parser.add_argument("--input", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    calibrate_parser.add_argument("--calm-threshold", type=float, required=True)
    calibrate_parser.add_argument("--passing-threshold", type=float, required=True)
    calibrate_parser.add_argument("--development-seed", type=int, required=True)
    calibrate_parser.add_argument("--heldout-seed", type=int, required=True)
    calibrate_parser.set_defaults(handler=calibrate_override)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
