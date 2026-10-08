"""Groot-LeRobot dataset implementation for openpi training."""

import functools
import glob
import json
import logging
import os
import random
from collections.abc import Iterator, Sequence
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
import openpi.shared.normalize as _normalize

import pathlib
from pathlib import Path

T_co = TypeVar("T_co", covariant=True)

BASE_MOTION_PROMPT_DISABLE_THRESHOLD = 1e-2

from robocasa.utils.groot_utils.groot_dataset import LeRobotSingleDataset, LeRobotMixtureDataset, LE_ROBOT_MODALITY_FILENAME, ModalityConfig, LE_ROBOT_EPISODE_FILENAME
from robocasa.utils.groot_utils.embodiment_tags import EmbodimentTag
from robocasa.utils.groot_utils.groot_video_utils import get_frames_by_indices


@functools.lru_cache(maxsize=128)
def _load_subtasks_by_episode(dataset_path_str: str) -> dict[int, list[dict]]:
    subtasks_path = Path(dataset_path_str) / "meta" / "subtasks.jsonl"
    if not subtasks_path.exists():
        return {}

    subtasks_by_episode: dict[int, list[dict]] = {}
    with open(subtasks_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            episode_index = payload.get("episode_index")
            phases = payload.get("segmentation", {}).get("phases", [])
            if episode_index is None:
                continue
            subtasks_by_episode[int(episode_index)] = phases
    return subtasks_by_episode


@functools.lru_cache(maxsize=4096)
def _load_wrist_relative_action_file(file_path_str: str) -> np.ndarray | None:
    file_path = Path(file_path_str)
    if not file_path.exists():
        return None
    return np.load(file_path).astype(np.float32)


def _load_prompt_wrist_relative_action(
    dataset_path: pathlib.Path,
    trajectory_id: int,
    frame_index: int,
) -> np.ndarray:
    """Load the wrist-relative-action vector aligned with the current frame."""
    wrist_relative_action_path = (
        Path(dataset_path)
        / "extras"
        / f"episode_{int(trajectory_id):06d}"
        / "wrist_relative_action.npy"
    )
    wrist_relative_actions = _load_wrist_relative_action_file(str(wrist_relative_action_path))
    if wrist_relative_actions is None or wrist_relative_actions.size == 0:
        print(f"Warning: Missing or empty wrist_relative_action for episode {trajectory_id} at {wrist_relative_action_path}")
        return np.zeros((3,), dtype=np.float32)

    clamped_frame_index = min(max(int(frame_index), 0), len(wrist_relative_actions) - 1)
    return np.asarray(wrist_relative_actions[clamped_frame_index], dtype=np.float32)


@functools.lru_cache(maxsize=128)
def _load_wrist_relative_action_norm_stats(dataset_path_str: str) -> _transforms.NormStats | None:
    dataset_path = pathlib.Path(dataset_path_str)
    wrist_relative_action_paths = sorted(dataset_path.glob("extras/episode_*/wrist_relative_action.npy"))
    if not wrist_relative_action_paths:
        return None

    running_stats = _normalize.RunningStats()
    updated = False
    for wrist_relative_action_path in wrist_relative_action_paths:
        wrist_relative_actions = _load_wrist_relative_action_file(str(wrist_relative_action_path))
        if wrist_relative_actions is None or wrist_relative_actions.size == 0:
            continue
        running_stats.update(np.asarray(wrist_relative_actions, dtype=np.float32))
        updated = True

    if not updated:
        return None

    return running_stats.get_statistics()


@functools.lru_cache(maxsize=128)
def _load_wrist_relative_action_quantile_stats(dataset_path_str: str) -> tuple[np.ndarray, np.ndarray]:
    stats = _load_wrist_relative_action_norm_stats(dataset_path_str)
    if stats is None or stats.q01 is None or stats.q99 is None:
        raise ValueError(f"Missing wrist_relative_action q01/q99 stats under {dataset_path_str}.")
    return np.asarray(stats.q01[:3], dtype=np.float32), np.asarray(stats.q99[:3], dtype=np.float32)


def _keep_with_probability(apply_prob: float) -> bool:
    """Return whether a prompt signal should be kept under the given probability."""
    return random.random() < apply_prob


PRIMITIVE_CMD_HORIZON = 25
PRIMITIVE_CMD_THRESHOLD = 0.2
DRAG_2D_HORIZON = 25
DRAG_2D_ACTION_SCALE = 0.05


def _has_meaningful_base_motion(base_motion: np.ndarray) -> bool:
    base_motion = np.asarray(base_motion, dtype=np.float32)
    if base_motion.ndim != 1:
        raise ValueError(f"Expected base_motion to be a vector, got shape {base_motion.shape}.")
    return bool(np.max(np.abs(base_motion)) > BASE_MOTION_PROMPT_DISABLE_THRESHOLD)


@functools.lru_cache(maxsize=128)
def _load_lerobot_info(dataset_path_str: str) -> dict:
    info_path = Path(dataset_path_str) / "meta" / "info.json"
    with open(info_path, "r") as f:
        return json.load(f)


@functools.lru_cache(maxsize=128)
def _load_lerobot_modality(dataset_path_str: str) -> dict:
    modality_path = Path(dataset_path_str) / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        return json.load(f)


@functools.lru_cache(maxsize=4096)
def _load_episode_meta(dataset_path_str: str, trajectory_id: int) -> dict:
    ep_meta_path = Path(dataset_path_str) / "extras" / f"episode_{int(trajectory_id):06d}" / "ep_meta.json"
    with open(ep_meta_path, "r") as f:
        return json.load(f)


@functools.lru_cache(maxsize=128)
def _load_action_quantile_stats(dataset_path_str: str, action_dim: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return q01/q99 in the action ordering consumed by openpi."""
    stats_path = Path(dataset_path_str) / "meta" / "stats.json"
    data = json.loads(stats_path.read_text())
    raw_actions_stats = data["action"]
    if "q01" not in raw_actions_stats or "q99" not in raw_actions_stats:
        raise ValueError(f"Missing action q01/q99 stats under {dataset_path_str}.")
    raw_actions_q01 = np.asarray(raw_actions_stats["q01"], dtype=np.float32)
    raw_actions_q99 = np.asarray(raw_actions_stats["q99"], dtype=np.float32)
    actions_indices = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
    q01 = raw_actions_q01[actions_indices]
    q99 = raw_actions_q99[actions_indices]
    if len(q01) < action_dim:
        q01 = np.concatenate([q01, np.zeros(action_dim - len(q01), dtype=np.float32)])
        q99 = np.concatenate([q99, np.zeros(action_dim - len(q99), dtype=np.float32)])
    return q01[:action_dim].astype(np.float32), q99[:action_dim].astype(np.float32)


def _replace_degenerate_quantiles_with_minmax(
    *,
    dataset_path: pathlib.Path,
    key: str,
    q01: np.ndarray,
    q99: np.ndarray,
    std: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use min/max when sparse nonzero values collapse the q01/q99 interval."""
    degenerate_quantile = np.isclose(q01, q99)
    non_constant = std > 1e-8
    has_minmax_range = max_values > min_values
    fallback_mask = degenerate_quantile & non_constant & has_minmax_range # q01 == q99, std is non-zero, min/max exists
    if np.any(fallback_mask):
        dims = np.flatnonzero(fallback_mask).tolist()
        logging.warning(
            "Using min/max quantile fallback for %s dims %s under %s because q01 == q99 but std is non-zero.",
            key,
            dims,
            dataset_path,
        )
        q01 = q01.copy()
        q99 = q99.copy()
        q01[fallback_mask] = min_values[fallback_mask]
        q99[fallback_mask] = max_values[fallback_mask]

    impossible_mask = degenerate_quantile & non_constant & ~has_minmax_range
    if np.any(impossible_mask):
        raise ValueError(
            f"Cannot repair degenerate quantile stats for {key} dims "
            f"{np.flatnonzero(impossible_mask).tolist()} under {dataset_path}: q01 == q99 and std is non-zero, "
            "but min/max do not provide a valid range."
        )

    return q01, q99


def _quat_xyzw_to_mat(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.asarray(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def _make_pose(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _pose_inv(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = pose[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ pose[:3, 3]
    return inv


CAMERA_AXIS_CORRECTION = np.asarray(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _build_prompt_projection_fields(
    *,
    dataset_path: pathlib.Path,
    trajectory_id: int,
    image_hw: tuple[int, int],
    eef_pos_rel: np.ndarray,
    eef_quat_xyzw: np.ndarray,
    base_pos: np.ndarray,
    base_quat_xyzw: np.ndarray,
    camera_name: str = "robot0_agentview_left",
) -> dict:
    """Build camera geometry for projecting local action chunks into prompt-image pixels."""
    h, w = image_hw
    q01, q99 = _load_action_quantile_stats(str(dataset_path))
    wrist_q01, wrist_q99 = _load_wrist_relative_action_quantile_stats(str(dataset_path))
    ep_meta = _load_episode_meta(str(dataset_path), int(trajectory_id))
    cam_config = ep_meta["cam_configs"][camera_name]

    base_pos = np.asarray(base_pos, dtype=np.float64)
    base_rot = _quat_xyzw_to_mat(np.asarray(base_quat_xyzw, dtype=np.float64))
    tcp_rot = _quat_xyzw_to_mat(np.asarray(eef_quat_xyzw, dtype=np.float64))
    tcp_world_pos = base_pos + base_rot @ np.asarray(eef_pos_rel, dtype=np.float64)

    local_pos = np.asarray(cam_config["pos"], dtype=np.float64)
    cam_quat_wxyz = np.asarray(cam_config["quat"], dtype=np.float64)
    local_quat_xyzw = np.asarray([cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]], dtype=np.float64)
    local_rot = _quat_xyzw_to_mat(local_quat_xyzw)

    parent_body_name = cam_config.get("parent_body")
    if parent_body_name:
        camera_pos = base_pos + base_rot @ local_pos
        camera_rot = base_rot @ local_rot
    else:
        camera_pos = local_pos
        camera_rot = local_rot

    camera_pose = _make_pose(camera_pos, camera_rot) @ CAMERA_AXIS_CORRECTION
    fovy = float(cam_config["camera_attribs"]["fovy"])
    focal = 0.5 * h / np.tan(fovy * np.pi / 360.0)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = np.asarray(
        [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    world_to_camera = world_to_camera @ _pose_inv(camera_pose)

    return {
        "prompt_world_to_camera": world_to_camera.astype(np.float32),
        "prompt_camera_resolution": np.asarray([h, w], dtype=np.float32),
        "prompt_tcp_world_pos": tcp_world_pos.astype(np.float32),
        "prompt_tcp_rot": tcp_rot.astype(np.float32),
        "prompt_base_rot": base_rot.astype(np.float32),
        "prompt_action_q01": q01,
        "prompt_action_q99": q99,
        "prompt_wrist_action_q01": wrist_q01,
        "prompt_wrist_action_q99": wrist_q99,
    }


def _get_video_path(
    dataset_path: pathlib.Path,
    trajectory_id: int,
    video_key: str,
) -> pathlib.Path:
    info = _load_lerobot_info(str(dataset_path))
    modality_meta = _load_lerobot_modality(str(dataset_path))
    original_key = modality_meta["video"].get(video_key, {}).get("original_key") or video_key
    chunk_size = int(info["chunks_size"])
    episode_chunk = int(trajectory_id) // chunk_size
    return dataset_path / info["video_path"].format(
        episode_chunk=episode_chunk,
        episode_index=int(trajectory_id),
        video_key=original_key,
    )


def _load_video_frame(
    dataset_path: pathlib.Path,
    trajectory_id: int,
    frame_index: int,
    *,
    video_key: str = "robot0_agentview_left",
) -> np.ndarray:
    video_path = _get_video_path(dataset_path, trajectory_id, video_key)
    frames = get_frames_by_indices(
        str(video_path),
        [max(0, int(frame_index))],
        video_backend="opencv",
        video_backend_kwargs={},
    )
    if frames.shape[0] != 1:
        raise ValueError(f"Expected one frame from {video_path}, got shape {frames.shape}")
    return np.asarray(frames[0])


def _build_prompt_primitive_cmd(*, actions: np.ndarray, dataset_path: pathlib.Path) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < 3:
        raise ValueError(f"Expected actions with shape [horizon, action_dim>=3], got {actions.shape}.")
    q01, q99 = _load_action_quantile_stats(str(dataset_path), action_dim=max(actions.shape[-1], 3))
    primitive_horizon = min(PRIMITIVE_CMD_HORIZON, len(actions))
    normalized_xyz = (
        (actions[:primitive_horizon, :3] - q01[:3])
        / (q99[:3] - q01[:3] + 1e-6)
        * 2.0
        - 1.0
    )
    cumulative_xyz = np.clip(np.sum(normalized_xyz, axis=0), -1.0, 1.0)
    return np.where(
        np.abs(cumulative_xyz) >= PRIMITIVE_CMD_THRESHOLD,
        np.sign(cumulative_xyz),
        0.0,
    ).astype(np.float32)


def _project_world_points_to_xy_unit(points_world: np.ndarray, projection_fields: dict) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    if points_world.ndim != 2 or points_world.shape[-1] != 3:
        raise ValueError(f"Expected world points with shape [n, 3], got {points_world.shape}.")
    world_to_camera = np.asarray(projection_fields["prompt_world_to_camera"], dtype=np.float32)
    camera_hw = np.asarray(projection_fields["prompt_camera_resolution"], dtype=np.float32)
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=-1)
    camera_points = points_h @ world_to_camera.T
    z = camera_points[:, 2:3]
    z = np.where(np.abs(z) > 1e-6, z, np.where(z >= 0.0, 1e-6, -1e-6))
    xy_pixels = camera_points[:, :2] / z
    denom_xy = np.maximum(np.asarray([camera_hw[1] - 1.0, camera_hw[0] - 1.0], dtype=np.float32), 1.0)
    xy_unit = xy_pixels / denom_xy
    xy_unit = np.nan_to_num(xy_unit, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(xy_unit, 0.0, 1.0).astype(np.float32)


def _build_prompt_2d_drag(
    *,
    actions: np.ndarray,
    projection_fields: dict,
    horizon: int = DRAG_2D_HORIZON,
    action_scale: float = DRAG_2D_ACTION_SCALE,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < 3:
        raise ValueError(f"Expected actions with shape [horizon, action_dim>=3], got {actions.shape}.")
    horizon = min(int(horizon), len(actions))
    tcp_world_pos = np.asarray(projection_fields["prompt_tcp_world_pos"], dtype=np.float32)
    base_rot = np.asarray(projection_fields["prompt_base_rot"], dtype=np.float32)
    world_delta = actions[:horizon, :3] @ base_rot.T * float(action_scale)
    end_world_pos = tcp_world_pos + np.sum(world_delta, axis=0)
    xy_points = _project_world_points_to_xy_unit(
        np.stack([tcp_world_pos, end_world_pos], axis=0),
        projection_fields,
    )
    return (xy_points[1] - xy_points[0]).astype(np.float32)


def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict


def _build_rp_prompt_fields(
    *,
    actions: np.ndarray,
    dataset_path: pathlib.Path,
    trajectory_id: int,
    frame_index: int,
    enable_overlay_prompt: bool,
    projection_fields: dict | None = None,
    wrist_relative_action_apply_prob: float = 0.5,
    primitive_cmd_apply_prob: float = 0.5,
    drag_2d_apply_prob: float = 0.5,
) -> dict:
    """Build short-term RoboPrompt fields for a sample."""
    prompt_source_frame_index = int(frame_index)
    prompt_local_motion = np.zeros((3,), dtype=np.float32)
    prompt_local_motion_mask = np.False_
    prompt_global_motion = np.zeros((3,), dtype=np.float32)
    prompt_global_motion_mask = np.False_
    prompt_2d_drag = np.zeros((2,), dtype=np.float32)
    prompt_2d_drag_mask = np.False_

    if enable_overlay_prompt:
        if _keep_with_probability(primitive_cmd_apply_prob):
            prompt_global_motion = _build_prompt_primitive_cmd(actions=actions, dataset_path=dataset_path)
            prompt_global_motion_mask = np.True_

        if _keep_with_probability(wrist_relative_action_apply_prob):
            prompt_local_motion = _load_prompt_wrist_relative_action(
                dataset_path=dataset_path,
                trajectory_id=trajectory_id,
                frame_index=prompt_source_frame_index,
            )
            prompt_local_motion_mask = np.bool_(not np.allclose(prompt_local_motion, 0.0))

        if projection_fields is not None and _keep_with_probability(drag_2d_apply_prob):
            prompt_2d_drag = _build_prompt_2d_drag(actions=actions, projection_fields=projection_fields)
            prompt_2d_drag_mask = np.bool_(not np.allclose(prompt_2d_drag, 0.0))

        if not bool(prompt_local_motion_mask) and not bool(prompt_global_motion_mask) and not bool(prompt_2d_drag_mask):
            if random.choice(["wrist", "primitive"]) == "wrist":
                prompt_local_motion = _load_prompt_wrist_relative_action(
                    dataset_path=dataset_path,
                    trajectory_id=trajectory_id,
                    frame_index=prompt_source_frame_index,
                )
                prompt_local_motion_mask = np.True_
            else:
                prompt_global_motion = _build_prompt_primitive_cmd(actions=actions, dataset_path=dataset_path)
                prompt_global_motion_mask = np.True_

    return {
        "prompt_local_motion": prompt_local_motion,
        "prompt_local_motion_mask": prompt_local_motion_mask,
        "prompt_global_motion": prompt_global_motion,
        "prompt_global_motion_mask": prompt_global_motion_mask,
        "prompt_2d_drag": prompt_2d_drag,
        "prompt_2d_drag_mask": prompt_2d_drag_mask,
        "prompt_source_frame_index": np.int32(prompt_source_frame_index),
    }


class GrootOpenpiSingleDataset(LeRobotSingleDataset):
    def __init__(
        self,
        dataset_meta: dict,
        action_horizon: int,
        add_promp: bool = False,
        primitive_cmd_apply_prob: float = 0.5,
        drag_2d_apply_prob: float = 0.5,
    ):        
        # this part copied from Abhi's DP codebasee
        dataset_path = dataset_meta["path"]
        dataset_path = pathlib.Path(dataset_path)
        filter_key = dataset_meta["filter_key"]
        self._add_promp = add_promp
        self._action_horizon = action_horizon
        self._primitive_cmd_apply_prob = float(primitive_cmd_apply_prob)
        self._drag_2d_apply_prob = float(drag_2d_apply_prob)
        delta_indices = list(range(0, action_horizon))
        delta_indices_obs = [0]
        modality_keys_dict = get_modality_keys(dataset_path)
        video_modality_keys = modality_keys_dict["video"]
        language_modality_keys = modality_keys_dict["annotation"]
        state_modality_keys = modality_keys_dict["state"]
        action_modality_keys = modality_keys_dict["action"]
        state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
        modality_configs = {
            "video": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=video_modality_keys,  # we will include all video modalities
            ),
            "state": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=state_modality_keys,
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=action_modality_keys,
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=language_modality_keys,
            ),
        }
        
        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            video_backend="opencv",
            video_backend_kwargs=None,
            transforms=None,
            filter_key=filter_key,
        )

    def __getitem__(self, index: SupportsIndex) -> dict:
        item = super().__getitem__(index)
        trajectory_id, base_index = self.all_steps[index.__index__()]
        has_base_motion = _has_meaningful_base_motion(item["action.base_motion"][0])
        enable_overlay_prompt = self._add_promp and not has_base_motion

        state = np.concatenate([
            item["state.end_effector_position_relative"],
            item["state.end_effector_rotation_relative"],
            item["state.base_position"],
            item["state.base_rotation"],
            item["state.gripper_qpos"],
        ], axis=1)
        actions = np.concatenate([
            item["action.end_effector_position"],
            item["action.end_effector_rotation"],
            item["action.gripper_close"],
            item["action.base_motion"],
            item["action.control_mode"],
        ], axis=1)

        # Get left camera image
        left_image = item["video.robot0_agentview_left"][0]

        new_item = {
            "observation/image": left_image,
            "observation/wrist_image": item["video.robot0_eye_in_hand"][0],
            "observation/state": state[0],
            "actions": actions,
            "prompt": item["annotation.human.task_description"][0],
        }
        projection_fields = _build_prompt_projection_fields(
            dataset_path=self.dataset_path,
            trajectory_id=trajectory_id,
            image_hw=left_image.shape[:2],
            eef_pos_rel=item["state.end_effector_position_relative"][0],
            eef_quat_xyzw=item["state.end_effector_rotation_relative"][0],
            base_pos=item["state.base_position"][0],
            base_quat_xyzw=item["state.base_rotation"][0],
        )
        new_item.update(
            _build_rp_prompt_fields(
                actions=actions,
                dataset_path=self.dataset_path,
                trajectory_id=trajectory_id,
                frame_index=base_index,
                enable_overlay_prompt=enable_overlay_prompt,
                projection_fields=projection_fields,
                primitive_cmd_apply_prob=self._primitive_cmd_apply_prob,
                drag_2d_apply_prob=self._drag_2d_apply_prob,
            )
        )
        new_item.update(projection_fields)
        return new_item
    

class GrootOpenpiMultiDataset(LeRobotMixtureDataset):
    def __init__(
            self,
            dataset_meta_list,
            dataset_weights,
            dataset_weights_alpha: float,
            action_horizon: int,
            metadata_config: dict = { # probably doesn't play a role here? TOOD double check
                "percentile_mixing_method": "weighted_average",
            },
            add_promp: bool = False,
            primitive_cmd_apply_prob: float = 0.5,
            drag_2d_apply_prob: float = 0.5,
        ):
        self._add_promp = add_promp
        self._action_horizon = action_horizon
        self._primitive_cmd_apply_prob = float(primitive_cmd_apply_prob)
        self._drag_2d_apply_prob = float(drag_2d_apply_prob)
        self._dataset_meta_list = dataset_meta_list
        datasets = []

        for ds_meta in dataset_meta_list:
            ds_path = ds_meta["path"]
            ds_path = pathlib.Path(ds_path)
            filter_key = ds_meta["filter_key"]
            delta_indices = list(range(0, action_horizon))
            delta_indices_obs = [0]
            modality_keys_dict = get_modality_keys(ds_path)
            video_modality_keys = modality_keys_dict["video"]
            language_modality_keys = modality_keys_dict["annotation"]
            state_modality_keys = modality_keys_dict["state"]
            action_modality_keys = modality_keys_dict["action"]
            state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
            modality_configs = {
                "video": ModalityConfig(
                    delta_indices=delta_indices_obs,
                    modality_keys=video_modality_keys,  # we will include all video modalities
                ),
                "state": ModalityConfig(
                    delta_indices=delta_indices_obs,
                    modality_keys=state_modality_keys,
                ),
                "action": ModalityConfig(
                    delta_indices=delta_indices,
                    modality_keys=action_modality_keys,
                ),
                "language": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=language_modality_keys,
                ),
            }
            this_dataset = LeRobotSingleDataset(
                dataset_path=ds_path,
                modality_configs=modality_configs,
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                video_backend="opencv",
                video_backend_kwargs=None,
                transforms=None,
                filter_key=filter_key,
            )
            datasets.append(this_dataset)

        if not dataset_weights:
            ds_weights = np.array([np.power(len(dataset), dataset_weights_alpha) for dataset in datasets])
            # the groot dataloader requires that at least one dataset has weight 1.0
            ds_weights = ds_weights / ds_weights[0]
        else:
            ds_weights = np.asarray(dataset_weights, dtype=np.float64)
            if len(ds_weights) != len(datasets):
                raise ValueError(
                    f"dataset_weights length ({len(ds_weights)}) must match dataset count ({len(datasets)})."
                )
            if np.any(ds_weights < 0) or not np.any(ds_weights > 0):
                raise ValueError("dataset_weights must contain at least one positive non-negative weight.")
            ds_weights = ds_weights / ds_weights.max()
        dataset_mixture = list(zip(datasets, ds_weights))
        # set balance_dataset_weights to False, since we are calculating weights ourselves
        super().__init__(
            data_mixture=dataset_mixture,
            mode="train", 
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            metadata_config=metadata_config,
        )

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """
        this code effectively ignores the index and samples randomly.
        we had to override this...
        """

        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        # Set seed
        # seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        # rng = np.random.default_rng(None)

        # Sample dataset
        dataset_index = np.random.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        trajectory_index = np.random.choice(
            len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        )
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = np.random.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index
    
    def __getitem__(self, index: SupportsIndex) -> dict:
        dataset, trajectory_id, base_index = self.sample_step(index.__index__())
        item = dataset.transforms(dataset.get_step_data(trajectory_id, base_index))
        has_base_motion = _has_meaningful_base_motion(item["action.base_motion"][0])
        enable_overlay_prompt = self._add_promp and not has_base_motion

        state = np.concatenate([
            item["state.end_effector_position_relative"],
            item["state.end_effector_rotation_relative"],
            item["state.base_position"],
            item["state.base_rotation"],
            item["state.gripper_qpos"],
        ], axis=1)
        actions = np.concatenate([
            item["action.end_effector_position"],
            item["action.end_effector_rotation"],
            item["action.gripper_close"],
            item["action.base_motion"],
            item["action.control_mode"],
        ], axis=1)

        # Get left camera image
        left_image = item["video.robot0_agentview_left"][0]

        new_item = {
            "observation/image": left_image,
            "observation/wrist_image": item["video.robot0_eye_in_hand"][0],
            "observation/state": state[0],
            "actions": actions,
            "prompt": item["annotation.human.task_description"][0], # TODO: Soroush change this later to task_description
            # "prompt": item["annotation.human.coarse_action"][0], # TODO: Soroush change this later to task_description
        }
        projection_fields = _build_prompt_projection_fields(
            dataset_path=dataset.dataset_path,
            trajectory_id=trajectory_id,
            image_hw=left_image.shape[:2],
            eef_pos_rel=item["state.end_effector_position_relative"][0],
            eef_quat_xyzw=item["state.end_effector_rotation_relative"][0],
            base_pos=item["state.base_position"][0],
            base_quat_xyzw=item["state.base_rotation"][0],
        )
        new_item.update(
            _build_rp_prompt_fields(
                actions=actions,
                dataset_path=dataset.dataset_path,
                trajectory_id=trajectory_id,
                frame_index=base_index,
                enable_overlay_prompt=enable_overlay_prompt,
                projection_fields=projection_fields,
                primitive_cmd_apply_prob=self._primitive_cmd_apply_prob,
                drag_2d_apply_prob=self._drag_2d_apply_prob,
            )
        )
        new_item.update(projection_fields)
        return new_item
    

def _load_norm_stats_from_groot_dataset(ds_meta: dict) -> dict[str, _transforms.NormStats] | None:
    def pad_zeros(input, targ_len):
        return np.concatenate([input, np.zeros(targ_len - len(input))])
    
    def pad_ones(input, targ_len):
        return np.concatenate([input, np.ones(targ_len - len(input))])
    
    dataset_path = ds_meta["path"]
    dataset_path = pathlib.Path(dataset_path)
    path = dataset_path / "meta" / "stats.json"
    data = json.loads(path.read_text())

    """
    the groot state ordering
    "state.base_position" 0, 1, 2
    "state.base_rotation" 3, 4, 5, 6
    "state.end_effector_position_relative" 7, 8, 9
    "state.end_effector_rotation_relative" 10, 11, 12, 13
    "state.gripper_qpos" 14, 15

    the desired state ordering
    "state.end_effector_position_relative" 7, 8, 9
    "state.end_effector_rotation_relative" 10, 11, 12, 13
    "state.base_position" 0, 1, 2
    "state.base_rotation" 3, 4, 5, 6
    "state.gripper_qpos" 14, 15
    """
    raw_states_stats = data["observation.state"]
    raw_states_mean = np.array(raw_states_stats["mean"])
    raw_states_std = np.array(raw_states_stats["std"])
    if "q01" not in raw_states_stats or "q99" not in raw_states_stats:
        raise ValueError(f"Missing observation.state q01/q99 stats under {dataset_path}.")
    raw_states_q01 = np.array(raw_states_stats["q01"])
    raw_states_q99 = np.array(raw_states_stats["q99"])
    raw_states_min = np.array(raw_states_stats["min"])
    raw_states_max = np.array(raw_states_stats["max"])

    # HACK: choose appropriate state indices
    states_indices = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15]
    states_mean = raw_states_mean[states_indices]
    states_std = raw_states_std[states_indices]
    states_q01 = raw_states_q01[states_indices]
    states_q99 = raw_states_q99[states_indices]
    states_min = raw_states_min[states_indices]
    states_max = raw_states_max[states_indices]
    states_q01, states_q99 = _replace_degenerate_quantiles_with_minmax(
        dataset_path=dataset_path,
        key="observation.state",
        q01=states_q01,
        q99=states_q99,
        std=states_std,
        min_values=states_min,
        max_values=states_max,
    )

    states_stats = _normalize.NormStats(
        mean=pad_zeros(states_mean, targ_len=32),
        std=pad_ones(states_std, targ_len=32),
        q01=pad_zeros(states_q01, targ_len=32),
        q99=pad_zeros(states_q99, targ_len=32),
    )

    """
    the groot action ordering
    "action.base_motion" 0, 1, 2, 3
    "action.control_mode" 4
    "action.end_effector_position" 5, 6, 7
    "action.end_effector_rotation" 8, 9, 10
    "action.gripper_close" 11

    the desired action ordering
    "action.end_effector_position" 5, 6, 7
    "action.end_effector_rotation" 8, 9, 10
    "action.gripper_close" 11
    "action.base_motion" 0, 1, 2, 3
    "action.control_mode" 4
    """
    raw_actions_stats = data["action"]
    raw_actions_mean = np.array(raw_actions_stats["mean"])
    raw_actions_std = np.array(raw_actions_stats["std"])
    if "q01" not in raw_actions_stats or "q99" not in raw_actions_stats:
        raise ValueError(f"Missing action q01/q99 stats under {dataset_path}.")
    raw_actions_q01 = np.array(raw_actions_stats["q01"])
    raw_actions_q99 = np.array(raw_actions_stats["q99"])
    raw_actions_min = np.array(raw_actions_stats["min"])
    raw_actions_max = np.array(raw_actions_stats["max"])
    
    # HACK: choose appropriate action indices
    actions_indices = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
    actions_mean = raw_actions_mean[actions_indices]
    actions_std = raw_actions_std[actions_indices]
    actions_q01 = raw_actions_q01[actions_indices]
    actions_q99 = raw_actions_q99[actions_indices]
    actions_min = raw_actions_min[actions_indices]
    actions_max = raw_actions_max[actions_indices]
    actions_q01, actions_q99 = _replace_degenerate_quantiles_with_minmax(
        dataset_path=dataset_path,
        key="action",
        q01=actions_q01,
        q99=actions_q99,
        std=actions_std,
        min_values=actions_min,
        max_values=actions_max,
    )

    actions_stats = _normalize.NormStats(
        mean=pad_zeros(actions_mean, targ_len=32),
        std=pad_ones(actions_std, targ_len=32),
        q01=pad_zeros(actions_q01, targ_len=32),
        q99=pad_zeros(actions_q99, targ_len=32),
    )

    norm_stats = {
        "state": states_stats,
        "actions": actions_stats,
    }

    if (wrist_relative_action_stats := _load_wrist_relative_action_norm_stats(str(dataset_path))) is not None:
        norm_stats["prompt_local_motion"] = wrist_relative_action_stats

    return norm_stats

def compute_overall_statistics(
    per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
    dataset_sampling_weights: list[float] | np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    """
    Computes overall statistics from per-task statistics using dataset sample weights.

    Args:
        per_task_stats: List of per-task statistics.
        Example format of one element in the per-task statistics list:
            {
                "state.gripper": {
                    "min": [...],
                    "max": [...],
                    "mean": [...],
                    "std": [...],
                    "q01": [...],
                    "q99": [...],
                },
                ...
            }
        dataset_sampling_weights: List of sample weights for each task.

    Returns:
        A dict of overall statistics per modality.
    """
    # Normalize the sample weights to sum to 1
    dataset_sampling_weights = np.array(dataset_sampling_weights)
    normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

    # Initialize overall statistics dict
    overall_stats: dict[str, dict[str, list[float]]] = {}

    # Get the list of modality keys
    modality_keys = per_task_stats[0].keys()

    for modality in modality_keys:
        # Number of dimensions (assuming consistent across tasks)
        num_dims = len(per_task_stats[0][modality].mean)

        # Initialize accumulators for means and variances
        weighted_means = np.zeros(num_dims)
        weighted_squares = np.zeros(num_dims)
        weighted_q01 = np.zeros(num_dims)
        weighted_q99 = np.zeros(num_dims)

        for task_idx, task_stats in enumerate(per_task_stats):
            w_i = normalized_weights[task_idx]
            stats = task_stats[modality]
            means = np.array(stats.mean)
            stds = np.array(stats.std)
            q01s = np.array(stats.q01) if stats.q01 is not None else means
            q99s = np.array(stats.q99) if stats.q99 is not None else means

            # Update weighted sums for mean and variance
            weighted_means += w_i * means
            weighted_squares += w_i * (stds**2 + means**2)
            weighted_q01 += w_i * q01s
            weighted_q99 += w_i * q99s
        
        # Compute overall mean
        overall_mean = weighted_means.tolist()

        # Compute overall variance and std deviation
        overall_variance = weighted_squares - weighted_means**2
        overall_std = np.sqrt(overall_variance).tolist()
        
        overall_q01 = weighted_q01.tolist()
        overall_q99 = weighted_q99.tolist()

        # Store the overall statistics for the modality
        overall_stats[modality] = _normalize.NormStats(
            mean=overall_mean,
            std=overall_std,
            q01=overall_q01,
            q99=overall_q99,
        )

    return overall_stats


def _load_norm_stats_from_groot_mixture_dataset(dataset_meta_list) -> dict[str, _transforms.NormStats] | None:
    # Merge the dataset statistics
    per_dataset_norm_stats = []
    for ds_meta in dataset_meta_list:
        per_dataset_norm_stats.append(_load_norm_stats_from_groot_dataset(ds_meta))
    
    return compute_overall_statistics(
        per_dataset_norm_stats,
        dataset_sampling_weights=np.ones(len(dataset_meta_list)),
    )
