"""Export the frozen V5/V6 learned layers to browser-compatible ONNX files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch as th
from torch import nn

from self_driving_rl.game import load_model
from self_driving_rl.game_env import NeonHighwayEnv
from self_driving_rl.longitudinal import LongitudinalIntentPolicy
from self_driving_rl.rlaif import PreferenceOverrideNet, load_override_policy

OPSET = 18


class BrowserQRDQN(nn.Module):
    """Expose mean action values from QR-DQN's quantile network."""

    def __init__(self, quantile_network: nn.Module) -> None:
        super().__init__()
        self.quantile_network = quantile_network

    def forward(self, observation: th.Tensor) -> th.Tensor:
        return self.quantile_network(observation).mean(dim=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _collect_observations(
    base_model: Path,
    override_model: Path,
    *,
    seed: int,
    samples: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    policy = LongitudinalIntentPolicy(
        load_override_policy(base_model, override_model, device="cpu")
    )
    observations: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    episode = 0
    while len(observations) < samples:
        env = NeonHighwayEnv(
            difficulty_mode="hard",
            dynamic_traffic=True,
            endless=True,
        )
        observation, _ = env.reset(seed=seed + episode)
        policy.reset()
        try:
            for _ in range(min(250, samples - len(observations))):
                observations.append(np.asarray(observation, dtype=np.float32))
                action = int(policy(observation))
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
        episode += 1
    return np.asarray(observations[:samples], dtype=np.float32)


def _export_base(base_model_path: Path, output: Path) -> BrowserQRDQN:
    model = load_model(base_model_path, device="cpu")
    wrapper = BrowserQRDQN(model.quantile_net).eval()
    example = th.zeros((1, 33), dtype=th.float32)
    th.onnx.export(
        wrapper,
        (example,),
        output,
        input_names=["observation"],
        output_names=["q_values"],
        dynamic_axes={"observation": {0: "batch"}, "q_values": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    return wrapper


def _export_override(override_model_path: Path, output: Path) -> PreferenceOverrideNet:
    payload = th.load(override_model_path, map_location="cpu", weights_only=True)
    network = PreferenceOverrideNet(
        observation_size=int(payload["observation_size"]),
        action_count=int(payload["action_count"]),
    )
    network.load_state_dict(payload["state_dict"])
    network.eval()
    observation = th.zeros((1, int(payload["observation_size"])), dtype=th.float32)
    base_action = th.zeros((1,), dtype=th.int64)
    th.onnx.export(
        network,
        (observation, base_action),
        output,
        input_names=["observation", "base_action"],
        output_names=["kind_logits", "action_logits"],
        dynamic_axes={
            "observation": {0: "batch"},
            "base_action": {0: "batch"},
            "kind_logits": {0: "batch"},
            "action_logits": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    return network


def _parity(
    base_network: BrowserQRDQN,
    override_network: PreferenceOverrideNet,
    observations: np.ndarray[Any, np.dtype[np.float32]],
    base_path: Path,
    override_path: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    base_session = ort.InferenceSession(base_path, providers=["CPUExecutionProvider"])
    override_session = ort.InferenceSession(
        override_path, providers=["CPUExecutionProvider"]
    )
    with th.no_grad():
        base_expected = base_network(th.from_numpy(observations)).numpy()
    base_actual = base_session.run(None, {"observation": observations})[0]
    base_actions = base_expected.argmax(axis=1).astype(np.int64)

    rng = np.random.default_rng(seed)
    contexts = np.column_stack(
        (
            rng.uniform(0.0, 1.0, len(observations)),
            rng.choice([-1.0, 0.0, 1.0], len(observations)),
        )
    ).astype(np.float32)
    override_observations = np.concatenate((observations, contexts), axis=1)
    with th.no_grad():
        kind_expected, action_expected = override_network(
            th.from_numpy(override_observations), th.from_numpy(base_actions)
        )
    kind_expected_array = kind_expected.numpy()
    action_expected_array = action_expected.numpy()
    kind_actual, action_actual = override_session.run(
        None,
        {"observation": override_observations, "base_action": base_actions},
    )

    metrics = {
        "samples": len(observations),
        "base_max_absolute_error": float(np.max(np.abs(base_expected - base_actual))),
        "base_action_agreement": float(
            np.mean(base_expected.argmax(axis=1) == base_actual.argmax(axis=1))
        ),
        "override_kind_max_absolute_error": float(
            np.max(np.abs(kind_expected_array - kind_actual))
        ),
        "override_action_max_absolute_error": float(
            np.max(np.abs(action_expected_array - action_actual))
        ),
        "override_kind_argmax_agreement": float(
            np.mean(kind_expected_array.argmax(axis=1) == kind_actual.argmax(axis=1))
        ),
        "override_action_argmax_agreement": float(
            np.mean(action_expected_array.argmax(axis=1) == action_actual.argmax(axis=1))
        ),
    }
    agreements = (
        metrics["base_action_agreement"],
        metrics["override_kind_argmax_agreement"],
        metrics["override_action_argmax_agreement"],
    )
    if min(agreements) != 1.0:
        raise RuntimeError(f"ONNX action parity failed: {metrics}")
    if max(
        metrics["base_max_absolute_error"],
        metrics["override_kind_max_absolute_error"],
        metrics["override_action_max_absolute_error"],
    ) > 1e-4:
        raise RuntimeError(f"ONNX numeric parity failed: {metrics}")
    return metrics


def export(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    base_output = args.output / "v5-qrdqn.onnx"
    override_output = args.output / "v6-rlaif-override.onnx"
    base_network = _export_base(args.base_model, base_output)
    override_network = _export_override(args.override_model, override_output)
    observations = _collect_observations(
        args.base_model,
        args.override_model,
        seed=args.seed,
        samples=args.samples,
    )
    parity = _parity(
        base_network,
        override_network,
        observations,
        base_output,
        override_output,
        seed=args.seed,
    )
    manifest = {
        "schema": 1,
        "driver": "Self-Driving RL v1.0 browser learned layers",
        "source_tag": "v1.0.0",
        "opset": OPSET,
        "inputs": {
            "base_observation": ["batch", 33],
            "override_observation": ["batch", 35],
            "base_action": ["batch"],
        },
        "artifacts": {
            base_output.name: {
                "bytes": base_output.stat().st_size,
                "sha256": _sha256(base_output),
            },
            override_output.name: {
                "bytes": override_output.stat().st_size,
                "sha256": _sha256(override_output),
            },
        },
        "parity": parity,
    }
    manifest_path = args.output / "browser-policy-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-model", type=Path, required=True)
    result.add_argument("--override-model", type=Path, required=True)
    result.add_argument("--output", type=Path, default=Path("artifacts/browser-policy-v1"))
    result.add_argument("--samples", type=int, default=512)
    result.add_argument("--seed", type=int, default=720_000)
    return result


if __name__ == "__main__":
    export(parser().parse_args())
