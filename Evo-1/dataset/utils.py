from __future__ import annotations

import functools
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HARDWARE_ROOT = REPO_ROOT / "hardware"
if HARDWARE_ROOT.is_dir() and str(HARDWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(HARDWARE_ROOT))

from scripts.utils.draw_overlay import (
    draw_prompt_overlays,
    randomize_prompt_point,
)

try:
    from my_device.macros import Gripper_TCP_T
except ImportError:
    Gripper_TCP_T = np.eye(4, dtype=np.float64)

TRAINING_CACHE_VERSION = "rp_v2"
VALID_CMD_TYPES = {"sigma", "state"}
PROMPT_GLOBAL_MOTION_APPLY_PROB = 0.6
PROMPT_LOCAL_MOTION_APPLY_PROB = 0.0
VISUAL_PROMPT_APPLY_PROB = 0.5
VISUAL_PROMPT_TRAJ_APPLY_PROB = VISUAL_PROMPT_APPLY_PROB
VISUAL_PROMPT_POINT_APPLY_PROB = VISUAL_PROMPT_APPLY_PROB
VISUAL_PROMPT_COLOR = (255, 80, 0)
VISUAL_PROMPT_POINT_NOISE_STD = 3.0
DEFAULT_PROMPT_MOTION_NOISE_STD = 0.005
PROMPT_MOTION_TEXT_THRESHOLD = 0.1
PROMPT_GLOBAL_MOTION_TEXT_THRESHOLD = 0.1

STATE_TCP_QUAT_XYZW = slice(3, 7)
ACTION_POSITION = slice(0, 3)


def normalize_cmd_type(cmd_type: str | None) -> str:
    value = "sigma" if cmd_type is None else str(cmd_type).strip().lower()
    if value not in VALID_CMD_TYPES:
        raise ValueError(f"cmd_type must be one of {sorted(VALID_CMD_TYPES)}, got {cmd_type!r}")
    return value


def training_cache_version_for_cmd_type(cmd_type: str | None) -> str:
    cmd_type = normalize_cmd_type(cmd_type)
    if cmd_type == "sigma":
        return TRAINING_CACHE_VERSION
    return f"{TRAINING_CACHE_VERSION}_cmd-{cmd_type}"


def parse_episode_index_for_sample(parquet_path: Path) -> int:
    """Extract the episode index from a standard LeRobot parquet filename."""
    parquet_digits = "".join(ch for ch in parquet_path.stem if ch.isdigit())
    if not parquet_digits:
        raise ValueError(f"Cannot parse episode index from parquet filename: {parquet_path}")
    return int(parquet_digits)


def _nonzero_motion_axes(vector: np.ndarray, threshold: float = PROMPT_MOTION_TEXT_THRESHOLD) -> list[int]:
    vector = np.asarray(vector, dtype=np.float32)[:3]
    return [idx for idx, value in enumerate(vector) if np.isfinite(value) and abs(float(value)) >= threshold]


def _sample_motion_axis_subset(
    vector: np.ndarray,
    threshold: float = PROMPT_MOTION_TEXT_THRESHOLD,
) -> list[int]:
    axes = _nonzero_motion_axes(vector, threshold)
    if not axes:
        return []
    subset_size = random.randint(1, len(axes))
    return sorted(random.sample(axes, subset_size))


def _mask_motion_axes(vector: np.ndarray, axis_indices: list[int]) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)[:3]
    masked = np.zeros((3,), dtype=np.float32)
    if axis_indices:
        masked[np.asarray(axis_indices, dtype=np.int64)] = vector[np.asarray(axis_indices, dtype=np.int64)]
    return masked


def _format_motion_axis_values(vector: np.ndarray, axis_indices: list[int] | None = None) -> str:
    axis_names = ("x", "y", "z")
    vector = np.asarray(vector, dtype=np.float32)[:3]
    if axis_indices is None:
        axis_indices = _nonzero_motion_axes(vector)
    return ", ".join(f"{axis_names[idx]}:{float(vector[idx]):.3f}" for idx in axis_indices)


def _add_prompt_motion_noise(vector: np.ndarray, noise_std: float = DEFAULT_PROMPT_MOTION_NOISE_STD) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)[:3]
    if noise_std <= 0:
        return vector.astype(np.float32)
    noise = np.random.normal(0.0, float(noise_std), size=vector.shape)
    return (vector + noise.astype(np.float32)).astype(np.float32)


def _sample_prompt_motion(
    vector: np.ndarray,
    noise_std: float,
    threshold: float = PROMPT_MOTION_TEXT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    axis_indices = _sample_motion_axis_subset(vector, threshold)
    if not axis_indices:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.bool_), "", False
    noisy_motion = _add_prompt_motion_noise(vector, noise_std)
    axis_indices = [
        idx
        for idx in axis_indices
        if np.isfinite(noisy_motion[idx]) and abs(float(noisy_motion[idx])) >= threshold
    ]
    if not axis_indices:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.bool_), "", False
    prompt_motion = _mask_motion_axes(noisy_motion, axis_indices)
    axis_mask = np.zeros((3,), dtype=np.bool_)
    axis_mask[np.asarray(axis_indices, dtype=np.int64)] = True
    motion_text = _format_motion_axis_values(prompt_motion, axis_indices)
    return prompt_motion, axis_mask, motion_text, True


@functools.lru_cache(maxsize=4096)
def load_numpy_array(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Required numpy array not found: {path}")
    return np.load(path)


def load_episode_target_pixels(dataset_path: Path, trajectory_id: int):
    target_pixels_path = dataset_path / "extras" / f"episode_{int(trajectory_id):06d}" / "target_pixels.npy"
    if not target_pixels_path.exists():
        raise FileNotFoundError(f"Required target pixels not found: {target_pixels_path}")
    pixels = load_numpy_array(str(target_pixels_path))
    if pixels.size == 0:
        raise ValueError(f"Required target pixels array is empty: {target_pixels_path}")
    return np.asarray(pixels, dtype=np.int32)


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


def compute_gripper_tip_world_position(state: np.ndarray, gripper_tcp_t: np.ndarray | None = None) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] < 3:
        raise ValueError(f"Expected state with at least 3 values, got shape {state.shape}.")
    tcp_position = state[:3]
    if state.shape[0] < 7:
        return tcp_position.astype(np.float32)

    tcp_rot = quat_xyzw_to_mat(state[STATE_TCP_QUAT_XYZW])
    tcp_t_tip = np.asarray(Gripper_TCP_T if gripper_tcp_t is None else gripper_tcp_t, dtype=np.float64)
    tip_offset_tcp = tcp_t_tip[:3, 3]
    return (tcp_position + tcp_rot @ tip_offset_tcp).astype(np.float32)


def compute_raw_wrist_relative_xyz(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    frame_count = min(len(states), len(actions))
    if frame_count == 0:
        return np.zeros((0, 3), dtype=np.float32)

    states = np.asarray(states[:frame_count], dtype=np.float64)
    actions = np.asarray(actions[:frame_count], dtype=np.float64)
    local_xyz = np.zeros((frame_count, 3), dtype=np.float64)

    for frame_idx in range(frame_count):
        tcp_rot = quat_xyzw_to_mat(states[frame_idx, STATE_TCP_QUAT_XYZW])
        local_xyz[frame_idx] = tcp_rot.T @ actions[frame_idx, ACTION_POSITION]

    return local_xyz.astype(np.float32)


def find_episode_parquet_path(dataset_path: Path, trajectory_id: int) -> Path:
    episode_name = f"episode_{int(trajectory_id):06d}"
    matches = sorted(dataset_path.glob(f"data/*/{episode_name}.parquet"))
    if not matches:
        raise FileNotFoundError(f"Cannot find parquet for {episode_name} under {dataset_path / 'data'}")
    return matches[0]


@functools.lru_cache(maxsize=4096)
def load_raw_wrist_relative_xyz(dataset_path_str: str, trajectory_id: int) -> np.ndarray:
    dataset_path = Path(dataset_path_str)
    raw_path = dataset_path / "extras" / f"episode_{int(trajectory_id):06d}" / "wrist_relative_action_raw.npy"
    if raw_path.exists():
        raw_actions = load_numpy_array(str(raw_path))
        if raw_actions.size == 0:
            raise ValueError(f"Required raw wrist relative action array is empty: {raw_path}")
        raw_actions = np.asarray(raw_actions, dtype=np.float32)
        if raw_actions.ndim == 1:
            raw_actions = raw_actions[None, :]
        if raw_actions.shape[-1] < 3:
            raise ValueError(f"Expected raw wrist relative action dim >= 3, got {raw_actions.shape[-1]} at {raw_path}")
        return raw_actions[:, :3]

    parquet_path = find_episode_parquet_path(dataset_path, trajectory_id)
    df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
    states = np.stack(df["observation.state"].to_list()).astype(np.float64)
    actions = np.stack(df["action"].to_list()).astype(np.float64)
    return compute_raw_wrist_relative_xyz(states, actions)


@functools.lru_cache(maxsize=4096)
def load_raw_global_xyz(dataset_path_str: str, trajectory_id: int) -> np.ndarray:
    dataset_path = Path(dataset_path_str)
    parquet_path = find_episode_parquet_path(dataset_path, trajectory_id)
    df = pd.read_parquet(parquet_path, columns=["action"])
    actions = np.stack(df["action"].to_list()).astype(np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.shape[-1] < 3:
        raise ValueError(f"Expected action dim >= 3, got {actions.shape[-1]} at {parquet_path}")
    return actions[:, :3]


def _sum_xyz_over_horizon(raw_xyz: np.ndarray, frame_index: int, horizon: int, *, label: str) -> np.ndarray:
    raw_xyz = np.asarray(raw_xyz, dtype=np.float32)
    if raw_xyz.ndim == 1:
        raw_xyz = raw_xyz[None, :]
    if raw_xyz.size == 0:
        return np.zeros((3,), dtype=np.float32)
    if raw_xyz.shape[-1] < 3:
        raise ValueError(f"Expected {label} dim >= 3, got {raw_xyz.shape[-1]}")

    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= len(raw_xyz):
        raise IndexError(f"Frame index {frame_index} out of range for {label} length {len(raw_xyz)}")

    horizon = max(int(horizon), 1)
    end_index = frame_index + horizon
    horizon_xyz = raw_xyz[frame_index:min(end_index, len(raw_xyz)), :3]
    if end_index > len(raw_xyz):
        pad_count = end_index - len(raw_xyz)
        horizon_xyz = np.concatenate(
            [horizon_xyz, np.repeat(raw_xyz[-1:, :3], pad_count, axis=0)],
            axis=0,
        )
    return np.sum(horizon_xyz, axis=0).astype(np.float32)


@functools.lru_cache(maxsize=128)
def load_wrist_horizon_sum_quantile_stats(dataset_path_str: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(dataset_path_str)
    horizon = max(int(horizon), 1)
    trajectory_ids = [parse_episode_index_for_sample(path) for path in sorted(dataset_path.glob("data/*/episode_*.parquet"))]
    if not trajectory_ids:
        raw_paths = sorted(dataset_path.glob("extras/episode_*/wrist_relative_action_raw.npy"))
        trajectory_ids = [int("".join(ch for ch in raw_path.parent.name if ch.isdigit())) for raw_path in raw_paths]
    if not trajectory_ids:
        raise FileNotFoundError(f"No raw wrist relative action files or episode parquet files found under {dataset_path}")

    horizon_sums = []
    for trajectory_id in trajectory_ids:
        raw_actions = load_raw_wrist_relative_xyz(str(dataset_path), int(trajectory_id))
        if raw_actions.size == 0:
            continue
        for frame_index in range(len(raw_actions)):
            horizon_sums.append(_sum_xyz_over_horizon(raw_actions, frame_index, horizon, label="raw wrist relative action"))
    if not horizon_sums:
        raise ValueError(f"No valid raw wrist relative actions found under {dataset_path}")

    stacked = np.stack(horizon_sums, axis=0)
    q01 = np.quantile(stacked, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(stacked, 0.99, axis=0).astype(np.float32)
    return q01, q99


def load_wrist_relative_action_horizon_sum(
    dataset_path: Path,
    trajectory_id: int,
    frame_index: int,
    horizon: int,
) -> np.ndarray:
    raw_actions = load_raw_wrist_relative_xyz(str(dataset_path), int(trajectory_id))
    raw_horizon_sum = _sum_xyz_over_horizon(
        raw_actions,
        int(frame_index),
        int(horizon),
        label="raw wrist relative action",
    )
    q01, q99 = load_wrist_horizon_sum_quantile_stats(str(dataset_path), int(horizon))
    normalized = normalize_signed_quantile_range(raw_horizon_sum, q01, q99)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


@functools.lru_cache(maxsize=1)
def load_default_env_camera_params() -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        from my_device.macros import E2H_CAM_T, E2H_INTRINSIC
    except Exception:
        return None, None
    return np.asarray(E2H_CAM_T, dtype=np.float32), np.asarray(E2H_INTRINSIC, dtype=np.float32)


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def get_env_camera_params(dataset_config: dict | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    dataset_config = dataset_config or {}
    camera_config = dataset_config.get("env_camera", {}) or {}
    base_t_camera = _first_present(
        camera_config.get("base_t_camera"),
        camera_config.get("base_to_camera"),
        dataset_config.get("base_t_camera"),
        dataset_config.get("base_to_camera"),
    )
    intrinsic = _first_present(
        camera_config.get("intrinsic"),
        camera_config.get("camera_intrinsic"),
        dataset_config.get("intrinsic"),
        dataset_config.get("camera_intrinsic"),
    )
    if base_t_camera is None or intrinsic is None:
        default_base_t_camera, default_intrinsic = load_default_env_camera_params()
        base_t_camera = default_base_t_camera if base_t_camera is None else base_t_camera
        intrinsic = default_intrinsic if intrinsic is None else intrinsic
    if base_t_camera is None or intrinsic is None:
        return np.eye(4, dtype=np.float32), np.eye(3, dtype=np.float32), False
    return np.asarray(base_t_camera, dtype=np.float32), np.asarray(intrinsic, dtype=np.float32), True


def normalize_signed_quantile_range(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)[:3]
    q01 = np.asarray(q01, dtype=np.float32)[:3]
    q99 = np.asarray(q99, dtype=np.float32)[:3]
    scale = np.maximum(np.maximum(np.abs(q01), np.abs(q99)), 1e-6)
    return (values / scale).astype(np.float32)


def normalize_signed_directional_range(
    values: np.ndarray, action_min: np.ndarray, action_max: np.ndarray
) -> np.ndarray:
    """Map negative/positive raw values independently to [-1, 0] / [0, 1]."""
    values = np.asarray(values, dtype=np.float32)[:3]
    action_min = np.asarray(action_min, dtype=np.float32)[:3]
    action_max = np.asarray(action_max, dtype=np.float32)[:3]
    negative_scale = np.maximum(np.abs(action_min), 1e-6)
    positive_scale = np.maximum(np.abs(action_max), 1e-6)
    scale = np.where(values >= 0.0, positive_scale, negative_scale)
    return (values / scale).astype(np.float32)


def build_global_motion(
    actions: np.ndarray,
    action_mask: np.ndarray,
    horizon: int,
    global_motion_min: np.ndarray,
    global_motion_max: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Compute sign-preserving global motion from raw XYZ actions.

    The raw XYZ actions are first summed over the horizon so opposing motions
    cancel in physical space. The signed sum is then normalized directly with
    the dataset-level positive/negative horizon-sum bounds.
    No thresholding or sign discretisation is applied.
    """
    actions = np.asarray(actions, dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=bool)
    if actions.ndim != 2 or action_mask.ndim != 2 or actions.shape[-1] < 3 or action_mask.shape[-1] < 3:
        return np.zeros((3,), dtype=np.float32), False
    if not bool(np.all(action_mask[:, :3])):
        return np.zeros((3,), dtype=np.float32), False

    primitive_horizon = min(int(horizon), len(actions))
    if primitive_horizon <= 0:
        return np.zeros((3,), dtype=np.float32), False
    raw_action_sum = np.sum(actions[:primitive_horizon, :3], axis=0)
    directional_action_sum = normalize_signed_directional_range(
        raw_action_sum, global_motion_min, global_motion_max
    )
    return np.clip(directional_action_sum, -1.0, 1.0).astype(np.float32), True



def _default_visual_prompt_info() -> dict:
    return {
        "visual_prompt_applied": np.bool_(False),
        "visual_prompt_type": "none",
        "visual_prompt_traj_applied": np.bool_(False),
        "visual_prompt_point_applied": np.bool_(False),
        "visual_prompt_frame_start": np.int32(-1),
        "visual_prompt_frame_end": np.int32(-1),
        "prompt_point": np.zeros((2,), dtype=np.float32),
    }


def _visual_prompt_type(traj_applied: bool, point_applied: bool) -> str:
    if traj_applied and point_applied:
        return "traj+point"
    if traj_applied:
        return "traj"
    if point_applied:
        return "point"
    return "none"


def _prepare_visual_prompt(image: Image.Image, item: dict, horizon: int) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    info = _default_visual_prompt_info()

    want_traj = random.random() < VISUAL_PROMPT_TRAJ_APPLY_PROB
    want_point = random.random() < VISUAL_PROMPT_POINT_APPLY_PROB
    if not (want_traj or want_point):
        return None, None, info

    dataset_path = item.get("dataset_path")
    trajectory_id = item.get("trajectory_id")
    frame_index = item.get("frame_index")
    if dataset_path is None or trajectory_id is None or frame_index is None:
        return None, None, info

    target_pixels = load_episode_target_pixels(Path(dataset_path), int(trajectory_id))
    if target_pixels is None:
        return None, None, info

    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= len(target_pixels):
        return None, None, info

    points_hw = None
    point_hw = None
    traj_applied = False
    point_applied = False
    frame_end = frame_index

    if want_traj:
        future_end = min(len(target_pixels), frame_index + int(horizon))
        if future_end > frame_index:
            points_hw = np.asarray(target_pixels[frame_index:future_end], dtype=np.float32)
            traj_applied = len(points_hw) > 0
            frame_end = future_end - 1

    if want_point:
        point_frame_index = min(len(target_pixels) - 1, frame_index + max(int(horizon), 1) - 1)
        point_hw = randomize_prompt_point(
            np.asarray(target_pixels[point_frame_index], dtype=np.float32),
            image.size[::-1],
            noise_std=VISUAL_PROMPT_POINT_NOISE_STD,
        )
        point_applied = True
        frame_end = max(frame_end, point_frame_index)

    if not (traj_applied or point_applied):
        return None, None, info

    info.update(
        {
            "visual_prompt_applied": np.bool_(traj_applied or point_applied),
            "visual_prompt_type": _visual_prompt_type(traj_applied, point_applied),
            "visual_prompt_traj_applied": np.bool_(traj_applied),
            "visual_prompt_point_applied": np.bool_(point_applied),
            "visual_prompt_frame_start": np.int32(frame_index),
            "visual_prompt_frame_end": np.int32(frame_end),
            "prompt_point": (
                point_hw.astype(np.float32) if point_hw is not None else np.zeros((2,), dtype=np.float32)
            ),
        }
    )
    return points_hw, point_hw, info


def apply_visual_prompt(image: Image.Image, item: dict, horizon: int) -> tuple[Image.Image, dict]:
    points_hw, point_hw, info = _prepare_visual_prompt(image, item, horizon)
    if not bool(info["visual_prompt_applied"]):
        return image, info

    rendered = np.asarray(image.convert("RGB")).copy()
    rendered = draw_prompt_overlays(
        rendered,
        trajectory_points_hw=points_hw,
        point_hw=point_hw,
        draw_trajectory=bool(info["visual_prompt_traj_applied"]),
        draw_point=bool(info["visual_prompt_point_applied"]),
        trajectory_color=VISUAL_PROMPT_COLOR,
        point_color=VISUAL_PROMPT_COLOR,
    )
    return Image.fromarray(rendered), info


def build_visual_prompt_inputs(image: Image.Image, item: dict, horizon: int) -> tuple[Image.Image, Image.Image | None, dict]:
    points_hw, point_hw, info = _prepare_visual_prompt(image, item, horizon)
    if not bool(info["visual_prompt_applied"]):
        return image, None, info

    base_rendered = np.asarray(image.convert("RGB")).copy()
    prompt_rendered = np.zeros_like(base_rendered)
    base_rendered = draw_prompt_overlays(
        base_rendered,
        trajectory_points_hw=points_hw,
        point_hw=point_hw,
        draw_trajectory=bool(info["visual_prompt_traj_applied"]),
        draw_point=bool(info["visual_prompt_point_applied"]),
        trajectory_color=VISUAL_PROMPT_COLOR,
        point_color=VISUAL_PROMPT_COLOR,
    )
    prompt_rendered = draw_prompt_overlays(
        prompt_rendered,
        trajectory_points_hw=points_hw,
        point_hw=point_hw,
        draw_trajectory=bool(info["visual_prompt_traj_applied"]),
        draw_point=bool(info["visual_prompt_point_applied"]),
        trajectory_color=VISUAL_PROMPT_COLOR,
        point_color=VISUAL_PROMPT_COLOR,
    )
    return Image.fromarray(base_rendered), Image.fromarray(prompt_rendered), info


def _build_prompt_global_motion(
    actions: np.ndarray,
    action_mask: np.ndarray,
    horizon: int,
    global_motion_min: np.ndarray,
    global_motion_max: np.ndarray,
) -> tuple[np.ndarray, bool]:
    return build_global_motion(actions, action_mask, horizon, global_motion_min, global_motion_max)


def build_rp_prompt_fields(
    task_description: str,
    *,
    item: dict,
    actions: np.ndarray,
    action_mask: np.ndarray,
    horizon: int,
    global_motion_min: np.ndarray | None = None,
    global_motion_max: np.ndarray | None = None,
    task_dropout_prob: float = 0.0,
    prompt_motion_noise_std: float = DEFAULT_PROMPT_MOTION_NOISE_STD,
) -> dict:
    """Build prompt fields for a single sample.

    Args:
        horizon: Length of the action sequence.
        actions: Per-step raw actions in physical action units. Global motion is
            summed in raw physical space, then normalized with the dataset-level
            positive/negative horizon-sum bounds.
        global_motion_min/global_motion_max: Per-axis raw horizon-sum bounds used
            for directional, sign-preserving normalization.
        task_dropout_prob: Probability of dropping the task description from
            the text prompt (set to empty string). Default 0.0.
    """
    if global_motion_min is None or global_motion_max is None:
        raise ValueError("global_motion_min and global_motion_max are required to build global motion prompts.")

    prompt_parts = []
    if random.random() >= task_dropout_prob:
        prompt_parts.append(task_description.strip())
    prompt_global_motion = np.zeros((3,), dtype=np.float32)
    prompt_global_motion_axis_mask = np.zeros((3,), dtype=np.bool_)
    prompt_global_motion_mask = np.bool_(False)
    prompt_local_motion = np.zeros((3,), dtype=np.float32)
    prompt_local_motion_mask = np.bool_(False)

    # ------------------------------------------------------------------
    # Determine eligibility for each prompt type
    # ------------------------------------------------------------------
    global_eligible = False
    if actions.ndim == 2 and actions.shape[-1] >= 3:
        global_motion_test, global_mask_test = _build_prompt_global_motion(
            actions, action_mask, horizon, global_motion_min, global_motion_max
        )
        global_eligible = global_mask_test and bool(
            _nonzero_motion_axes(global_motion_test, PROMPT_GLOBAL_MOTION_TEXT_THRESHOLD)
        )

    local_eligible = False
    local_motion_test = np.zeros((3,), dtype=np.float32)
    local_action_mask = np.asarray(action_mask, dtype=bool)
    if (
        local_action_mask.ndim == 2
        and local_action_mask.shape[-1] >= 3
        and bool(np.all(local_action_mask[:, :3]))
    ):
        dataset_path = item.get("dataset_path")
        trajectory_id = item.get("trajectory_id")
        frame_index = item.get("frame_index")
        if dataset_path is not None and trajectory_id is not None and frame_index is not None:
            local_motion_test = load_wrist_relative_action_horizon_sum(
                Path(dataset_path),
                int(trajectory_id),
                int(frame_index),
                horizon,
            )
            local_eligible = bool(_nonzero_motion_axes(local_motion_test))

    # ------------------------------------------------------------------
    # Randomly select a subset from {global, local}.
    # Visual prompt dropout is handled by apply_visual_prompt().
    # ------------------------------------------------------------------
    candidates = []
    if global_eligible and random.random() < PROMPT_GLOBAL_MOTION_APPLY_PROB:
        candidates.append("global")
    if local_eligible and random.random() < PROMPT_LOCAL_MOTION_APPLY_PROB:
        candidates.append("local")

    # ------------------------------------------------------------------
    # Build tensors / prompt text for selected candidates
    # ------------------------------------------------------------------
    for candidate in candidates:
        if candidate == "global":
            signed_global_motion, global_mask = _build_prompt_global_motion(
                actions,
                action_mask,
                horizon,
                global_motion_min,
                global_motion_max,
            )
            if global_mask:
                prompt_global_motion, prompt_global_motion_axis_mask, motion_text, motion_applied = _sample_prompt_motion(
                    signed_global_motion,
                    prompt_motion_noise_std,
                    PROMPT_GLOBAL_MOTION_TEXT_THRESHOLD,
                )
                if motion_applied:
                    prompt_global_motion_mask = np.bool_(True)
                    prompt_parts.append(f"move {motion_text} in global frame")
        elif candidate == "local":
            if not np.allclose(local_motion_test, 0.0):
                prompt_local_motion, _, motion_text, motion_applied = _sample_prompt_motion(
                    local_motion_test,
                    prompt_motion_noise_std,
                )
                if motion_applied:
                    prompt_local_motion_mask = np.bool_(True)
                    prompt_parts.append(f"move {motion_text} in wrist local frame")

    return {
        "prompt": ", ".join(part for part in prompt_parts if part),
        "prompt_global_motion": prompt_global_motion,
        "prompt_global_motion_axis_mask": prompt_global_motion_axis_mask,
        "prompt_global_motion_mask": prompt_global_motion_mask,
        "prompt_local_motion": prompt_local_motion,
        "prompt_local_motion_mask": prompt_local_motion_mask,
        "visual_prompt_applied": np.bool_(False),
        "visual_prompt_type": "none",
        "visual_prompt_frame_start": np.int32(-1),
        "visual_prompt_frame_end": np.int32(-1),
    }


def get_left_image_index(video_paths: dict) -> int:
    for idx, (view_key, view_path) in enumerate(video_paths.items()):
        view_text = f"{view_key} {view_path}".lower()
        if "left" in view_text:
            return idx
    return 0


def compute_lerobot_normalization_stats_from_minmax(jsonl_path):
    state_mins, state_maxs = [], []
    action_mins, action_maxs = [], []

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Extracting min/max"):
            obj = json.loads(line)
            stats = obj.get("stats", {})
            try:
                state_mins.append(stats["observation.state"]["min"])
                state_maxs.append(stats["observation.state"]["max"])
                action_mins.append(stats["action"]["min"])
                action_maxs.append(stats["action"]["max"])
            except Exception as e:
                print(f"skipping abnormal line: {e}")

    state_min_global = np.min(np.array(state_mins), axis=0).tolist()
    state_max_global = np.max(np.array(state_maxs), axis=0).tolist()
    action_min_global = np.min(np.array(action_mins), axis=0).tolist()
    action_max_global = np.max(np.array(action_maxs), axis=0).tolist()

    return {
        "observation.state": {"min": state_min_global, "max": state_max_global},
        "action": {"min": action_min_global, "max": action_max_global},
    }


def compute_lerobot_state_delta_action_stats(dataset_path: Path) -> Dict[str, List[float]]:
    parquet_paths = sorted(dataset_path.glob("data/*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_path / 'data'}")

    delta_mins = []
    delta_maxs = []
    delta_dim = None
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path, columns=["observation.state"])
        states = np.stack(df["observation.state"].to_list()).astype(np.float32)
        if states.ndim == 1:
            states = states[:, None]
        if delta_dim is None:
            delta_dim = int(states.shape[-1])
        elif int(states.shape[-1]) != delta_dim:
            raise ValueError(
                f"Inconsistent observation.state dim under {dataset_path}: "
                f"expected {delta_dim}, got {states.shape[-1]} at {parquet_path}"
            )

        if len(states) >= 2:
            deltas = states[1:] - states[:-1]
        else:
            deltas = np.zeros((1, states.shape[-1]), dtype=np.float32)
        deltas = np.concatenate([deltas, np.zeros((1, states.shape[-1]), dtype=np.float32)], axis=0)
        delta_mins.append(np.min(deltas, axis=0))
        delta_maxs.append(np.max(deltas, axis=0))

    if not delta_mins:
        raise ValueError(f"No state-delta action stats could be computed under {dataset_path}")
    return {
        "min": np.min(np.stack(delta_mins), axis=0).astype(np.float32).tolist(),
        "max": np.max(np.stack(delta_maxs), axis=0).astype(np.float32).tolist(),
    }


def replace_action_stats_with_state_delta(stats: Dict, dataset_path: Path) -> Dict:
    updated = dict(stats)
    updated["action"] = compute_lerobot_state_delta_action_stats(dataset_path)
    return updated


def merge_lerobot_stats(stats_list: List[Dict[str, Dict[str, List[float]]]]) -> Dict:
    state_mins = [np.array(d["observation.state"]["min"]) for d in stats_list]
    state_maxs = [np.array(d["observation.state"]["max"]) for d in stats_list]
    action_mins = [np.array(d["action"]["min"]) for d in stats_list]
    action_maxs = [np.array(d["action"]["max"]) for d in stats_list]
    state_min_global = np.min(np.stack(state_mins), axis=0).tolist()
    state_max_global = np.max(np.stack(state_maxs), axis=0).tolist()
    action_min_global = np.min(np.stack(action_mins), axis=0).tolist()
    action_max_global = np.max(np.stack(action_maxs), axis=0).tolist()

    merged = {
        "observation.state": {"min": state_min_global, "max": state_max_global},
        "action": {"min": action_min_global, "max": action_max_global},
    }

    # Merge q01/q99 when available
    if any("q01" in d.get("observation.state", {}) for d in stats_list):
        state_q01s = [np.array(d["observation.state"]["q01"]) for d in stats_list if "q01" in d.get("observation.state", {})]
        state_q99s = [np.array(d["observation.state"]["q99"]) for d in stats_list if "q99" in d.get("observation.state", {})]
        if state_q01s and state_q99s:
            merged["observation.state"]["q01"] = np.min(np.stack(state_q01s), axis=0).tolist()
            merged["observation.state"]["q99"] = np.max(np.stack(state_q99s), axis=0).tolist()

    if any("q01" in d.get("action", {}) for d in stats_list):
        action_q01s = [np.array(d["action"]["q01"]) for d in stats_list if "q01" in d.get("action", {})]
        action_q99s = [np.array(d["action"]["q99"]) for d in stats_list if "q99" in d.get("action", {})]
        if action_q01s and action_q99s:
            merged["action"]["q01"] = np.min(np.stack(action_q01s), axis=0).tolist()
            merged["action"]["q99"] = np.max(np.stack(action_q99s), axis=0).tolist()

    return merged


def _compute_dataframe_horizon_sum_bounds(
    df: pd.DataFrame, start_count: int, horizon: int, cmd_type: str = "sigma"
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw XYZ horizon-sum bounds for all requested start frames."""
    cmd_type = normalize_cmd_type(cmd_type)
    horizon = max(int(horizon), 1)
    if cmd_type == "sigma":
        commands = np.stack(df["action"].to_list()).astype(np.float32)
    else:
        states = np.stack(df["observation.state"].to_list()).astype(np.float32)
        commands = states[1:] - states[:-1]

    if commands.ndim != 2 or commands.shape[-1] < 3:
        raise ValueError(f"Expected {cmd_type} command dim >= 3, got shape {commands.shape}")
    xyz = commands[:, :3]
    cumulative = np.concatenate(
        [np.zeros((1, 3), dtype=np.float64), np.cumsum(xyz, axis=0, dtype=np.float64)],
        axis=0,
    )
    starts = np.arange(int(start_count), dtype=np.int64)
    horizon_sums = cumulative[starts + horizon] - cumulative[starts]
    return horizon_sums.min(axis=0).astype(np.float32), horizon_sums.max(axis=0).astype(np.float32)


def process_parquet_file_worker(args):
    parquet_path, arm_name, dataset_name, dataset_config, dataset_path, task_mapping, action_horizon, max_samples_per_file, cache_dir, cmd_type = args
    cmd_type = normalize_cmd_type(cmd_type)

    view_map = dataset_config.get("view_map", None)
    if not view_map:
        raise KeyError(f"Missing required view_map for '{arm_name}-{dataset_name}'")

    df = pd.read_parquet(parquet_path)

    original_len = len(df)
    last_row = df.iloc[-1:]
    padding_rows = pd.concat([last_row] * max(int(action_horizon), 0), ignore_index=True)
    df = pd.concat([df, padding_rows], ignore_index=True)

    sample_count = original_len
    if max_samples_per_file is not None:
        sample_count = min(sample_count, int(max_samples_per_file))

    horizon_sum_min, horizon_sum_max = _compute_dataframe_horizon_sum_bounds(
        df, original_len, action_horizon, cmd_type
    )
    horizon_sum_stats = {
        "arm_name": arm_name,
        "min": horizon_sum_min,
        "max": horizon_sum_max,
    }

    cache_subdir = cache_dir / arm_name / dataset_name / parquet_path.parent.name / parquet_path.stem

    # Fast path: if all expected cache files already exist, return them directly
    cache_version = training_cache_version_for_cmd_type(cmd_type)
    expected_caches = [cache_subdir / f"{cache_version}_{i}_{i + action_horizon}.pkl" for i in range(sample_count)]
    if all(c.exists() for c in expected_caches):
        return [str(c) for c in expected_caches], None, horizon_sum_stats

    # Otherwise: remove stale cache files and rebuild
    if cache_subdir.exists():
        for stale in cache_subdir.glob(f"{cache_version}_*.pkl"):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

    episode_files = []
    for i in range(sample_count):
        start_idx = i
        end_idx = i + action_horizon

        cache_filename = f"{cache_version}_{start_idx}_{end_idx}.pkl"
        cache_filepath = cache_subdir / cache_filename

        logging.info(f"build {cache_filename}")
        sub_df = df.iloc[i : i + action_horizon]
        if cmd_type == "sigma":
            cmd_sequence = [row["action"] for _, row in sub_df.iterrows()]
        else:
            state_df = df.iloc[i : i + action_horizon + 1]
            states = np.stack(state_df["observation.state"].to_list()).astype(np.float32)
            cmd_sequence = list(states[1:] - states[:-1])
        video_paths = {}
        base_video_path = dataset_path / "videos" / parquet_path.parent.name

        for view_key, view_folder in view_map.items():
            # Standard LeRobot layout:
            # videos/chunk-*/view/episode_XXXXXX/episode_XXXXXX.mp4
            full_path = (
                base_video_path
                / view_folder
                / parquet_path.stem
                / f"{parquet_path.stem}.mp4"
            )
            if full_path.exists():
                video_paths[view_key] = str(full_path)
            else:
                raise FileNotFoundError(f"Required video file not found: {full_path}")

        task_index = sub_df.iloc[0].get("task_index", None)
        if task_index is not None and task_index in task_mapping:
            prompt = task_mapping[task_index]
        else:
            logging.info(f"cannot find task description from task_index={task_index}")
            prompt = ""

        episode = {
            "arm_key": arm_name,
            "dataset_key": dataset_name,
            "dataset_path": str(dataset_path),
            "trajectory_id": parse_episode_index_for_sample(parquet_path),
            "frame_index": start_idx,
            "prompt": prompt,
            "state": sub_df.iloc[0].get("observation.state", None),
            "action": cmd_sequence,
            "cmd_type": cmd_type,
            "video_paths": video_paths,
            "timestamp": sub_df.iloc[0].get("timestamp", None),
        }

        cache_subdir.mkdir(parents=True, exist_ok=True)
        with open(cache_filepath, "wb") as f:
            pickle.dump(episode, f)

        episode_files.append(str(cache_filepath))
    return episode_files, None, horizon_sum_stats
