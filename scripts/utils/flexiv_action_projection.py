from __future__ import annotations

from typing import Any

import numpy as np

from my_device.macros import E2H_CAM_T
from my_device.macros import E2H_INTRINSIC
from my_device.macros import Gripper_TCP_T
from scripts.utils.realworld_projection import compute_gripper_tip_points_base
from scripts.utils.realworld_projection import project_action_horizon_to_pixels


def project_flexiv_action_chunk(
    *,
    image_shape: tuple[int, int, int],
    observation: dict[str, Any],
    action_chunk: np.ndarray,
    color: tuple[int, int, int],
    source_image_shape: tuple[int, int, int] | None = None,
) -> dict[str, np.ndarray | tuple[int, int, int]]:
    source_height = source_width = None
    if source_image_shape is not None:
        source_height = int(source_image_shape[0])
        source_width = int(source_image_shape[1])
    observation_state = np.asarray(observation["observation/state"], dtype=np.float64).reshape(-1)
    current_tcp_position = observation_state[:3]
    if observation_state.shape[0] >= 7:
        current_tcp_position = compute_gripper_tip_points_base(observation_state[None], Gripper_TCP_T)[0]

    points_hw, valid_mask = project_action_horizon_to_pixels(
        image_shape=image_shape,
        current_tcp_position=current_tcp_position,
        action_chunk=action_chunk,
        base_t_camera=E2H_CAM_T,
        intrinsic=E2H_INTRINSIC,
        source_width=source_width,
        source_height=source_height,
    )
    return {
        "points_hw": points_hw,
        "valid_mask": valid_mask,
        "color": color,
    }
