from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th

from self_driving_rl.dagger import (
    DATA_KEYS,
    DaggerCorrectionNet,
    DaggerCorrectionPolicy,
    _balanced_class_weights,
    _split_indices,
    _temporal_sample_weights,
    apply_dagger_gate,
    apply_speed_gate,
    load_dataset,
    projected_merge_is_safe,
    promotion_gate,
    save_dataset,
)
from self_driving_rl.game_env import (
    PEDAL_GAS,
    STEER_KEEP,
    STEER_LEFT,
    STEER_RIGHT,
    NeonHighwayEnv,
    decode_action,
    encode_action,
)
from self_driving_rl.longitudinal import SpeedGuidance


def clear_observation() -> np.ndarray:
    observation = np.zeros(33, dtype=np.float32)
    observation[2] = 1.0 / 3.0
    observation[3] = 1.0 / 3.0
    for lane in range(NeonHighwayEnv.LANES):
        offset = 9 + 6 * lane
        observation[offset] = 1.0
        observation[offset + 3] = 1.0
    return observation


def test_dagger_network_has_intervention_and_steering_heads() -> None:
    network = DaggerCorrectionNet(33)
    lane_intervention, steering, speed_intervention, speed = network(
        th.zeros((4, 33)), th.tensor([STEER_LEFT, STEER_KEEP, STEER_RIGHT, STEER_KEEP])
    )

    assert lane_intervention.shape == (4, 2)
    assert steering.shape == (4, 3)
    assert speed_intervention.shape == (4, 2)
    assert speed.shape == (4, 3)


def test_projected_merge_shield_accepts_clear_gap_and_rejects_rear_blind_spot() -> None:
    observation = clear_observation()
    assert projected_merge_is_safe(observation, STEER_LEFT)
    assert projected_merge_is_safe(observation, STEER_KEEP)

    left_lane_offset = 9
    observation[left_lane_offset + 3] = 0.05
    assert not projected_merge_is_safe(observation, STEER_LEFT)


def test_dagger_gate_preserves_pedal_and_vetoes_unsafe_override() -> None:
    clear = clear_observation()
    unsafe = clear.copy()
    unsafe[9 + 3] = 0.05
    base_action = encode_action(STEER_KEEP, PEDAL_GAS)

    actions, applied = apply_dagger_gate(
        np.stack((clear, unsafe)),
        np.asarray([base_action, base_action], dtype=np.int64),
        np.asarray([0.95, 0.95], dtype=np.float32),
        np.asarray([STEER_LEFT, STEER_LEFT], dtype=np.int64),
        0.8,
    )

    assert decode_action(int(actions[0])) == (STEER_LEFT, PEDAL_GAS)
    assert int(actions[1]) == base_action
    np.testing.assert_array_equal(applied, [True, False])


def test_speed_gate_delegates_faster_guidance_safety_to_longitudinal_planner() -> None:
    clear = clear_observation()
    risky = clear.copy()
    current_lane_offset = 9 + 6
    risky[current_lane_offset + 2] = 0.8

    guidance, applied = apply_speed_gate(
        np.stack((clear, risky)),
        np.asarray([0.95, 0.95], dtype=np.float32),
        np.asarray([SpeedGuidance.FASTER, SpeedGuidance.FASTER], dtype=np.int64),
        0.8,
    )

    np.testing.assert_array_equal(
        guidance, [SpeedGuidance.FASTER, SpeedGuidance.FASTER]
    )
    np.testing.assert_array_equal(applied, [True, True])


def test_dagger_policy_uses_confident_safe_human_residual() -> None:
    class BasePolicy:
        hud_data = {"driving_intent": "CRUISE"}

        def __init__(self) -> None:
            self.guidance: list[SpeedGuidance] = []

        def __call__(self, _observation: np.ndarray) -> int:
            return encode_action(STEER_KEEP, PEDAL_GAS)

        def set_speed_guidance(
            self, guidance: SpeedGuidance, *, current_speed: float
        ) -> None:
            del current_speed
            self.guidance.append(guidance)

    network = DaggerCorrectionNet(33)
    with th.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.intervention_head.bias[:] = th.tensor([0.0, 10.0])
        network.steer_head.bias[:] = th.tensor([10.0, 0.0, 0.0])
        network.speed_intervention_head.bias[:] = th.tensor([0.0, 10.0])
        network.speed_head.bias[:] = th.tensor([0.0, 10.0, 0.0])
    base = BasePolicy()
    policy = DaggerCorrectionPolicy(
        base, network, lane_threshold=0.8, speed_threshold=0.8
    )

    action = policy(clear_observation())

    assert decode_action(action) == (STEER_LEFT, PEDAL_GAS)
    assert policy.total_overrides == 1
    assert policy.total_speed_overrides == 1
    assert base.guidance == [SpeedGuidance.FASTER]
    assert policy.hud_data["driving_intent"] == "CRUISE"

    second_action = policy(clear_observation())
    assert decode_action(second_action) == (STEER_KEEP, PEDAL_GAS)
    assert policy.total_deferred == 1
    assert policy.total_speed_deferred == 1


def test_dagger_policy_can_deploy_only_faster_progress_guidance() -> None:
    class BasePolicy:
        def __init__(self) -> None:
            self.guidance: list[SpeedGuidance] = []

        def __call__(self, _observation: np.ndarray) -> int:
            return encode_action(STEER_KEEP, PEDAL_GAS)

        def set_speed_guidance(
            self, guidance: SpeedGuidance, *, current_speed: float
        ) -> None:
            del current_speed
            self.guidance.append(guidance)

    network = DaggerCorrectionNet(33)
    with th.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.intervention_head.bias[:] = th.tensor([0.0, 10.0])
        network.steer_head.bias[:] = th.tensor([10.0, 0.0, 0.0])
        network.speed_intervention_head.bias[:] = th.tensor([0.0, 10.0])
        network.speed_head.bias[:] = th.tensor([0.0, 0.0, 10.0])
    base = BasePolicy()
    policy = DaggerCorrectionPolicy(
        base,
        network,
        lane_threshold=0.8,
        speed_threshold=0.8,
        lane_residual_enabled=False,
        faster_only=True,
    )

    action = policy(clear_observation())

    assert decode_action(action) == (STEER_KEEP, PEDAL_GAS)
    assert base.guidance == []
    assert policy.total_overrides == 0
    assert policy.total_speed_overrides == 0


def test_dagger_dataset_round_trip(tmp_path: Path) -> None:
    records: dict[str, list[object]] = {key: [] for key in DATA_KEYS}
    records["observations"].append(clear_observation())
    records["proposed_actions"].append(encode_action(STEER_KEEP, PEDAL_GAS))
    records["teacher_steers"].append(STEER_LEFT)
    records["lane_labelled"].append(1)
    records["lane_corrections"].append(1)
    records["teacher_speed_guidance"].append(SpeedGuidance.FASTER)
    records["speed_labelled"].append(1)
    records["speed_corrections"].append(1)
    records["seeds"].append(123)
    records["episode_steps"].append(45)
    records["session_ids"].append(0)
    path = tmp_path / "demonstrations.npz"

    save_dataset(path, records)
    loaded = load_dataset(path)

    assert loaded["observations"].shape == (1, 33)
    assert int(loaded["teacher_steers"][0]) == STEER_LEFT
    assert not path.with_suffix(".npz.tmp.npz").exists()


def test_dagger_split_holds_out_complete_traffic_routes() -> None:
    count = 36
    data = {
        "observations": np.zeros((count, 33), dtype=np.float32),
        "proposed_actions": np.zeros(count, dtype=np.int64),
        "teacher_steers": np.tile([STEER_LEFT, STEER_KEEP, STEER_RIGHT], 12),
        "lane_labelled": np.ones(count, dtype=np.int8),
        "lane_corrections": np.tile([1, 0, 1], 12).astype(np.int8),
        "teacher_speed_guidance": np.tile(
            [SpeedGuidance.FASTER, SpeedGuidance.BASE, SpeedGuidance.SLOWER], 12
        ).astype(np.int64),
        "speed_labelled": np.ones(count, dtype=np.int8),
        "speed_corrections": np.tile([1, 0, 1], 12).astype(np.int8),
        "seeds": np.repeat(np.arange(100, 106), 6),
        "episode_steps": np.tile(np.arange(6), 6),
        "session_ids": np.zeros(count, dtype=np.int64),
    }

    train, validation = _split_indices(data, seed=8128)

    assert len(train) + len(validation) == count
    assert set(data["seeds"][train]).isdisjoint(data["seeds"][validation])


def test_temporal_and_class_weights_reduce_repeated_label_dominance() -> None:
    data = {
        "observations": np.zeros((4, 33), dtype=np.float32),
        "lane_labelled": np.zeros(4, dtype=np.int8),
        "teacher_steers": np.full(4, STEER_KEEP, dtype=np.int64),
        "speed_labelled": np.ones(4, dtype=np.int8),
        "teacher_speed_guidance": np.full(
            4, SpeedGuidance.FASTER, dtype=np.int64
        ),
        "session_ids": np.zeros(4, dtype=np.int64),
        "seeds": np.full(4, 123, dtype=np.int64),
        "episode_steps": np.asarray([1, 2, 3, 40], dtype=np.int64),
    }
    indices = np.arange(4, dtype=np.int64)

    temporal = _temporal_sample_weights(data, indices)
    classes = _balanced_class_weights(
        np.asarray([0, 1, 1, 1]),
        np.ones(4, dtype=np.int8),
        indices,
        np.ones(4, dtype=np.float32),
        classes=2,
    )

    assert np.all(temporal[:3] < temporal[3])
    assert classes[0] > classes[1]


def test_promotion_gate_requires_enough_routes_and_no_regression() -> None:
    base = {
        "completion_rate": 1.0,
        "crash_rate": 0.0,
        "mean_unsafe_lane_changes": 0.0,
        "mean_net_overtakes": 7.0,
        "passing_response_rate": 0.90,
        "avoidable_following_rate": 0.08,
        "clear_road_stall_rate": 0.01,
        "mean_lane_reversals": 2.0,
        "unjustified_brakes_per_1000_steps": 1.0,
    }
    better = {
        **base,
        "mean_net_overtakes": 7.5,
        "avoidable_following_rate": 0.06,
    }
    worse = {**better, "crash_rate": 0.05}

    assert promotion_gate(base, better, episodes=20)["status"] == "INCONCLUSIVE"
    assert promotion_gate(base, better, episodes=100)["status"] == "PROMOTE"
    assert promotion_gate(base, worse, episodes=100)["status"] == "HOLD"
