from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from scripts.utils.draw_overlay import draw_future_trajectory
from scripts.utils.draw_overlay import make_trajectory_gradient

STATE_TCP_POSITION = slice(0, 3)
STATE_TCP_QUAT_XYZW = slice(3, 7)


def quat_xyzw_to_mat(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError(f"Invalid near-zero quaternion: {quat_xyzw}")
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_state_to_transform(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.shape[0] < 7:
        raise ValueError(f"Expected state with at least 7 values, got shape {state.shape}.")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_xyzw_to_mat(state[STATE_TCP_QUAT_XYZW])
    transform[:3, 3] = state[STATE_TCP_POSITION]
    return transform


def compute_gripper_tip_points_base(states: np.ndarray, gripper_tcp_t: np.ndarray) -> np.ndarray:
    points = []
    gripper_tip_local = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for state in np.asarray(states, dtype=np.float64):
        base_t_tcp = pose_state_to_transform(state)
        base_tip = base_t_tcp @ np.asarray(gripper_tcp_t, dtype=np.float64) @ gripper_tip_local
        points.append(base_tip[:3])
    return np.asarray(points, dtype=np.float64)


def project_base_points_to_camera_pixels(
    points_base: np.ndarray,
    base_t_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_base = np.asarray(points_base, dtype=np.float64)
    camera_t_base = np.linalg.inv(np.asarray(base_t_camera, dtype=np.float64))
    points_h = np.concatenate([points_base, np.ones((len(points_base), 1), dtype=np.float64)], axis=1)
    points_camera = (camera_t_base @ points_h.T).T[:, :3]
    z = points_camera[:, 2]
    if np.any(np.isclose(z, 0.0)):
        raise ValueError("Cannot project points with near-zero camera depth.")
    uvw = (np.asarray(intrinsic, dtype=np.float64) @ points_camera.T).T
    pixels_xy = uvw[:, :2] / z[:, None]
    return pixels_xy, points_camera


def project_base_points_to_camera_pixels_safe(
    points_base: np.ndarray,
    base_t_camera: np.ndarray,
    intrinsic: np.ndarray,
    *,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_base = np.asarray(points_base, dtype=np.float64)
    if points_base.ndim != 2 or points_base.shape[1] != 3:
        raise ValueError(f"Expected points_base shape [N, 3], got {points_base.shape}.")
    if len(points_base) == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=bool),
        )

    camera_t_base = np.linalg.inv(np.asarray(base_t_camera, dtype=np.float64))
    points_h = np.concatenate([points_base, np.ones((len(points_base), 1), dtype=np.float64)], axis=1)
    points_camera = (camera_t_base @ points_h.T).T[:, :3]
    z = points_camera[:, 2]
    valid_mask = np.isfinite(points_camera).all(axis=1) & (z > min_depth)

    pixels_xy = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    if np.any(valid_mask):
        uvw = (np.asarray(intrinsic, dtype=np.float64) @ points_camera[valid_mask].T).T
        pixels_xy[valid_mask] = uvw[:, :2] / z[valid_mask, None]
    return pixels_xy, points_camera, valid_mask


def map_camera_pixels_to_video_pixels(
    pixels_xy: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    video_width: int,
    video_height: int,
    processed_padding: int,
) -> np.ndarray:
    resized_width = video_width + processed_padding
    resized_height = video_height + processed_padding
    crop_x = (resized_width - video_width) / 2.0
    crop_y = (resized_height - video_height) / 2.0
    pixels_xy = np.asarray(pixels_xy, dtype=np.float64).copy()
    pixels_xy[:, 0] = pixels_xy[:, 0] * resized_width / source_width - crop_x
    pixels_xy[:, 1] = pixels_xy[:, 1] * resized_height / source_height - crop_y
    return np.rint(pixels_xy).astype(np.int32)


def build_action_horizon_points_base(
    current_tcp_position: np.ndarray,
    action_chunk: np.ndarray,
    *,
    include_current: bool = True,
) -> np.ndarray:
    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None, :]
    if action_chunk.ndim != 2 or action_chunk.shape[1] < 3:
        raise ValueError(f"Expected action chunk with shape [T, D>=3], got {action_chunk.shape}.")

    current_tcp_position = np.asarray(current_tcp_position, dtype=np.float64)
    if current_tcp_position.shape != (3,):
        raise ValueError(f"Expected current_tcp_position shape (3,), got {current_tcp_position.shape}.")

    deltas = action_chunk[:, :3]
    points = current_tcp_position[None, :] + np.cumsum(deltas, axis=0)
    if include_current:
        points = np.concatenate([current_tcp_position[None, :], points], axis=0)
    return points


def draw_projected_action_horizon(
    image_rgb: np.ndarray,
    *,
    current_tcp_position: np.ndarray,
    action_chunk: np.ndarray,
    base_t_camera: np.ndarray,
    intrinsic: np.ndarray,
    source_width: int | None = None,
    source_height: int | None = None,
    processed_padding: int = 8,
    color: Sequence[int] = (0, 180, 255),
) -> np.ndarray:
    points_base = build_action_horizon_points_base(current_tcp_position, action_chunk)
    pixels_xy, _, valid_mask = project_base_points_to_camera_pixels_safe(
        points_base,
        base_t_camera=base_t_camera,
        intrinsic=intrinsic,
    )

    image = image_rgb.copy()
    height, width = image.shape[:2]
    if source_width is not None and source_height is not None:
        mapped_pixels_xy = np.full_like(pixels_xy, np.nan, dtype=np.float64)
        finite_rows = np.isfinite(pixels_xy).all(axis=1)
        if np.any(finite_rows):
            mapped_pixels_xy[finite_rows] = map_camera_pixels_to_video_pixels(
                pixels_xy[finite_rows],
                source_width=int(source_width),
                source_height=int(source_height),
                video_width=width,
                video_height=height,
                processed_padding=int(processed_padding),
            ).astype(np.float64)
        pixels_xy = mapped_pixels_xy

    finite_mask = valid_mask & np.isfinite(pixels_xy).all(axis=1)
    safe_pixels_xy = np.where(np.isfinite(pixels_xy), pixels_xy, 0.0)
    points_hw = np.rint(safe_pixels_xy[:, [1, 0]]).astype(np.int32, copy=False)

    start = 0
    while start < len(points_hw):
        while start < len(points_hw) and not finite_mask[start]:
            start += 1
        end = start
        while end < len(points_hw) and finite_mask[end]:
            end += 1
        if end - start >= 2:
            draw_future_trajectory(image, points_hw[start:end], color=make_trajectory_gradient(color), thickness=1)
        start = end

    if len(points_hw) > 0 and finite_mask[0]:
        cy, cx = int(points_hw[0][0]), int(points_hw[0][1])
        cv2.circle(image, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 2, tuple(int(value) for value in color), -1, cv2.LINE_AA)
    return image


def project_action_horizon_to_pixels(
    *,
    image_shape: tuple[int, int] | tuple[int, int, int],
    current_tcp_position: np.ndarray,
    action_chunk: np.ndarray,
    base_t_camera: np.ndarray,
    intrinsic: np.ndarray,
    source_width: int | None = None,
    source_height: int | None = None,
    processed_padding: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    points_base = build_action_horizon_points_base(current_tcp_position, action_chunk)
    pixels_xy, _, valid_mask = project_base_points_to_camera_pixels_safe(
        points_base,
        base_t_camera=base_t_camera,
        intrinsic=intrinsic,
    )

    height, width = int(image_shape[0]), int(image_shape[1])
    if source_width is not None and source_height is not None:
        mapped_pixels_xy = np.full_like(pixels_xy, np.nan, dtype=np.float64)
        finite_rows = np.isfinite(pixels_xy).all(axis=1)
        if np.any(finite_rows):
            mapped_pixels_xy[finite_rows] = map_camera_pixels_to_video_pixels(
                pixels_xy[finite_rows],
                source_width=int(source_width),
                source_height=int(source_height),
                video_width=width,
                video_height=height,
                processed_padding=int(processed_padding),
            ).astype(np.float64)
        pixels_xy = mapped_pixels_xy

    finite_mask = valid_mask & np.isfinite(pixels_xy).all(axis=1)
    safe_pixels_xy = np.where(np.isfinite(pixels_xy), pixels_xy, 0.0)
    points_hw = np.rint(safe_pixels_xy[:, [1, 0]]).astype(np.int32, copy=False)
    return points_hw, finite_mask


def draw_cached_projected_horizon(
    image_rgb: np.ndarray,
    points_hw: np.ndarray,
    valid_mask: np.ndarray,
    *,
    color: Sequence[int] = (0, 180, 255),
) -> np.ndarray:
    image = image_rgb.copy()
    points_hw = np.asarray(points_hw, dtype=np.int32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    start = 0
    while start < len(points_hw):
        while start < len(points_hw) and not valid_mask[start]:
            start += 1
        end = start
        while end < len(points_hw) and valid_mask[end]:
            end += 1
        if end - start >= 2:
            draw_future_trajectory(image, points_hw[start:end], color=make_trajectory_gradient(color), thickness=1)
        start = end

    if len(points_hw) > 0 and bool(valid_mask[0]):
        cy, cx = int(points_hw[0][0]), int(points_hw[0][1])
        cv2.circle(image, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 2, tuple(int(value) for value in color), -1, cv2.LINE_AA)
    return image
