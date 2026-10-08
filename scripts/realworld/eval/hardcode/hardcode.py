from __future__ import annotations

from typing import Any, Mapping

import numpy as np


# Hardcode phase1 is a client-side bypass for Evo-1: the UI prompt is converted
# directly into sample_kwargs["phase1_actions"], then the server's phase2 policy
# consumes it through infer_direct instead of running the steerer backend.
#
# The UI global/local values are interpreted as normalized action values in
# [-1, 1]. Before sending phase1_actions to the server, this module
# denormalizes each xyz dimension with the downstream action norm stats so the
# phase2 policy receives raw physical-scale actions. If hardcode actions are
# still too large, tune the normalized UI composition or clip bound below; do
# not reintroduce a fixed raw-space scale unless the action stats are known to
# be wrong.
DEFAULT_PHASE1_SOURCE = "evo"
HARDCODE_PHASE1_SOURCE = "hardcode"
XYZ_ACTION_DIMS = (0, 1, 2)
STATE_TCP_QUAT_XYZW = slice(3, 7)


def prompt_phase1_source(prompt_payload: Mapping[str, Any] | None) -> str:
    # Missing payload/source keeps the original behavior: use Evo-1 when a
    # steerer is available.
    if prompt_payload is None:
        return DEFAULT_PHASE1_SOURCE
    source = str(prompt_payload.get("prompt_phase1_source", DEFAULT_PHASE1_SOURCE)).strip().lower()
    if source not in {DEFAULT_PHASE1_SOURCE, HARDCODE_PHASE1_SOURCE}:
        raise ValueError(
            f"Unsupported prompt_phase1_source {source!r}. "
            f"Expected {DEFAULT_PHASE1_SOURCE!r} or {HARDCODE_PHASE1_SOURCE!r}."
        )
    return source


def should_use_hardcode_phase1(*, steer_mode: str | None, prompt_payload: Mapping[str, Any] | None) -> bool:
    # The command-line --steer=hardcode is a global override; otherwise the UI
    # per-prompt source selector decides between Evo-1 and hardcode.
    return steer_mode == HARDCODE_PHASE1_SOURCE or (
        steer_mode is not None and prompt_phase1_source(prompt_payload) == HARDCODE_PHASE1_SOURCE
    )


def has_2d_prompt(prompt_payload: Mapping[str, Any] | None) -> bool:
    if prompt_payload is None:
        return False

    prompt_2d_drag_mask = prompt_payload.get("prompt_2d_drag_mask", False)
    if bool(np.any(np.asarray(prompt_2d_drag_mask, dtype=np.bool_))):
        return True

    prompt_image_masks = prompt_payload.get("prompt_image_masks")
    if isinstance(prompt_image_masks, Mapping):
        prompt_mask = prompt_image_masks.get("prompt_0", False)
        return bool(np.any(np.asarray(prompt_mask, dtype=np.bool_)))
    return False


def build_phase1_action_chunk(
    *,
    prompt_payload: Mapping[str, Any] | None,
    observation_state: np.ndarray,
    metadata: Mapping[str, Any],
    action_horizon: int,
    action_dim: int = 32,
) -> np.ndarray | None:
    if prompt_payload is None:
        return None

    # Compose normalized xyz action from the structured UI prompt. Global and
    # local motion both keep their slider magnitudes.
    # 2D drag/image prompts are intentionally not projected here.
    phase1_xyz_norm = np.zeros((3,), dtype=np.float32)
    has_phase1_xyz = False

    global_motion = prompt_payload.get("prompt_global_motion")
    global_motion_mask = prompt_payload.get("prompt_global_motion_mask", False)
    if global_motion is not None and bool(np.any(np.asarray(global_motion_mask, dtype=np.bool_))):
        global_motion = np.asarray(global_motion, dtype=np.float32)
        if global_motion.shape != (3,):
            raise ValueError(f"Expected prompt_global_motion shape (3,), got {global_motion.shape}.")
        phase1_xyz_norm += np.clip(global_motion[:3], -1.0, 1.0).astype(np.float32)
        has_phase1_xyz = True

    local_motion = prompt_payload.get("prompt_local_motion")
    local_motion_mask = prompt_payload.get("prompt_local_motion_mask", False)
    if local_motion is not None and bool(np.any(np.asarray(local_motion_mask, dtype=np.bool_))):
        local_motion = np.asarray(local_motion, dtype=np.float32)
        if local_motion.shape != (3,):
            raise ValueError(f"Expected prompt_local_motion shape (3,), got {local_motion.shape}.")
        phase1_xyz_norm += wrist_local_xyz_to_base_xyz(local_motion[:3], observation_state)
        has_phase1_xyz = True

    if not has_phase1_xyz:
        return None

    # Repeat the same hardcoded command over the action horizon. The gripper
    # channel is held at the current raw gripper state so phase2 does not receive
    # a spurious open/close command from the hardcode chunk.
    phase1_actions = np.zeros((int(action_horizon), int(action_dim)), dtype=np.float32)
    # The clip bound determines the normalized command sent into denormalization.
    # With default bounds, each active xyz dimension maps to that dimension's
    # action-stat min/max (or q01/q99) instead of an arbitrary fixed scale.
    phase1_xyz_norm = np.clip(phase1_xyz_norm, -1.0, 1.0)
    phase1_actions[:, :3] = denormalize_action_values(phase1_xyz_norm, XYZ_ACTION_DIMS, metadata)[None, :]
    if action_dim > 6:
        current_gripper = float(np.asarray(observation_state, dtype=np.float32)[-1])
        phase1_actions[:, 6] = current_gripper
    return phase1_actions


def with_hardcode_phase1_sample_kwargs(
    sample_kwargs: Mapping[str, Any],
    *,
    prompt_payload: Mapping[str, Any] | None,
    observation_state: np.ndarray,
    metadata: Mapping[str, Any],
    action_horizon: int,
    enabled: bool,
    action_dim: int = 32,
) -> dict[str, Any]:
    sample_kwargs = dict(sample_kwargs)
    if not enabled:
        return sample_kwargs

    # This is the only handoff the server needs: SteeredPhase2Runtime sees the
    # steerer disabled, and DiffusionPolicyPhase2Policy.infer_direct consumes
    # phase1_actions as past_action.
    phase1_actions = build_phase1_action_chunk(
        prompt_payload=prompt_payload,
        observation_state=observation_state,
        metadata=metadata,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    if phase1_actions is not None:
        sample_kwargs["phase1_actions"] = phase1_actions
    return sample_kwargs


def denormalize_action_values(
    normalized_values: np.ndarray,
    action_dims: tuple[int, ...],
    metadata: Mapping[str, Any],
) -> np.ndarray:
    return np.asarray(
        [
            denormalize_action_value(float(value), action_dim, metadata)
            for value, action_dim in zip(normalized_values, action_dims, strict=True)
        ],
        dtype=np.float32,
    )


def wrist_local_xyz_to_base_xyz(local_xyz: np.ndarray, observation_state: np.ndarray) -> np.ndarray:
    state = np.asarray(observation_state, dtype=np.float32).reshape(-1)
    local_xyz = np.asarray(local_xyz, dtype=np.float32)
    if local_xyz.shape != (3,):
        raise ValueError(f"Expected local_xyz shape (3,), got {local_xyz.shape}.")
    if state.shape[0] < 7:
        return local_xyz
    tcp_rot = quat_xyzw_to_mat(state[STATE_TCP_QUAT_XYZW])
    return (tcp_rot @ local_xyz.astype(np.float64)).astype(np.float32)


def quat_xyzw_to_mat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = quat / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def denormalize_action_value(normalized_value: float, action_dim: int, metadata: Mapping[str, Any]) -> float:
    # Match the downstream policy's action normalization metadata when present.
    # The server may expose quantile stats (q01/q99), min/max stats from a
    # Diffusion Policy limits normalizer, or Gaussian stats (mean/std).
    action_stats = metadata.get("action_norm_stats")
    if not isinstance(action_stats, Mapping):
        raise ValueError("Hardcode phase1 requires metadata['action_norm_stats'] to denormalize UI actions.")

    use_quantile_norm = bool(metadata.get("use_quantile_norm", False))
    if use_quantile_norm:
        q01 = np.asarray(action_stats.get("q01"), dtype=np.float32)
        q99 = np.asarray(action_stats.get("q99"), dtype=np.float32)
        if q01.ndim != 1 or q99.ndim != 1 or action_dim >= len(q01) or action_dim >= len(q99):
            raise ValueError(f"Missing q01/q99 action stats for dim {action_dim}.")
        return float((normalized_value + 1.0) * 0.5 * (q99[action_dim] - q01[action_dim]) + q01[action_dim])

    min_values = np.asarray(action_stats.get("min"), dtype=np.float32)
    max_values = np.asarray(action_stats.get("max"), dtype=np.float32)
    if min_values.ndim == 1 and max_values.ndim == 1 and action_dim < len(min_values) and action_dim < len(max_values):
        return float(
            (normalized_value + 1.0) * 0.5 * (max_values[action_dim] - min_values[action_dim])
            + min_values[action_dim]
        )

    mean = np.asarray(action_stats.get("mean"), dtype=np.float32)
    std = np.asarray(action_stats.get("std"), dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1 or action_dim >= len(mean) or action_dim >= len(std):
        raise ValueError(f"Missing usable action norm stats for dim {action_dim}.")
    return float(normalized_value * std[action_dim] + mean[action_dim])
