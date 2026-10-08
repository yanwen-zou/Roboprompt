import dataclasses
from collections.abc import Sequence

import cv2
import numpy as np
from openpi_client import image_tools
from robosuite.utils import transform_utils as T
from robosuite.utils.camera_utils import get_camera_transform_matrix


PHASE2_PROMPT_MEM_STEP_FRACTION = 0.05


def policy_inference_steps_from_metadata(metadata: dict | None, default: float = 10.0) -> float:
    if not isinstance(metadata, dict):
        return float(default)
    candidates = [
        metadata.get("max_phase2_steps"),
        metadata.get("phase2_max_steps"),
        metadata.get("policy_inference_steps"),
        metadata.get("num_steps"),
        metadata.get("num_inference_steps"),
    ]
    for nested_key in ("diffusion_policy", "fastwam", "openpi"):
        nested = metadata.get(nested_key)
        if isinstance(nested, dict):
            candidates.extend(
                [
                    nested.get("max_phase2_steps"),
                    nested.get("phase2_max_steps"),
                    nested.get("policy_inference_steps"),
                    nested.get("num_steps"),
                    nested.get("num_inference_steps"),
                ]
            )
    for value in candidates:
        if value is None:
            continue
        try:
            steps = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(steps) and steps > 0:
            return steps
    return float(default)


def phase2_prompt_mem_step(num_steps: float) -> float:
    return float(num_steps) * PHASE2_PROMPT_MEM_STEP_FRACTION


def render_rollout_window(
    image_bgr: np.ndarray,
    env_name: str,
    episode_idx: int,
    step_idx: int,
    *,
    enable_prompt_hotkey: bool = False,
    denoise_step: float | None = None,
) -> None:
    header = np.full((44, image_bgr.shape[1], 3), 245, dtype=np.uint8)
    hotkeys = "p: prompt  q: abort episode" if enable_prompt_hotkey else "q: abort episode"
    denoise_text = "" if denoise_step is None else f" | denoise step {float(denoise_step):.1f}"
    cv2.putText(
        header,
        f"{env_name} | episode {episode_idx} | step {step_idx}{denoise_text} | {hotkeys}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("RoboCasa Rollout", np.concatenate([header, image_bgr], axis=0))


def _get_tcp_world_position(obs: dict) -> np.ndarray:
    base_pos = np.asarray(obs["state.base_position"], dtype=np.float64)
    base_rot = T.quat2mat(np.asarray(obs["state.base_rotation"], dtype=np.float64))
    eef_pos_rel = np.asarray(obs["state.end_effector_position_relative"], dtype=np.float64)
    return base_pos + base_rot @ eef_pos_rel


def _project_points_from_world_to_camera_unclipped(
    points: np.ndarray,
    world_to_camera_transform: np.ndarray,
) -> np.ndarray:
    """Project world points to integer (height, width) pixels, matching vis_train.py."""
    points = np.asarray(points)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with last dimension 3, got shape {points.shape}.")
    world_to_camera_transform = np.asarray(world_to_camera_transform)
    if world_to_camera_transform.shape != (4, 4):
        raise ValueError(f"Expected world_to_camera_transform shape (4, 4), got {world_to_camera_transform.shape}.")

    ones_pad = np.ones(points.shape[:-1] + (1,), dtype=points.dtype)
    points_h = np.concatenate((points, ones_pad), axis=-1)
    mat_reshape = [1] * len(points.shape[:-1]) + [4, 4]
    projected = np.matmul(world_to_camera_transform.reshape(mat_reshape), points_h[..., None])[..., 0]
    z = projected[..., 2:3]
    if np.any(np.abs(z) <= 1e-12):
        raise ValueError("Cannot project points with near-zero camera depth.")
    projected = projected[..., :2] / z
    pixels_xy = np.rint(projected).astype(np.int32)
    return np.concatenate((pixels_xy[..., 1:2], pixels_xy[..., 0:1]), axis=-1)


def _build_action_overlay_projection_fields(
    *,
    image_bgr: np.ndarray,
    env,
    obs: dict,
    camera_name: str,
) -> dict:
    height, width = image_bgr.shape[:2]
    return {
        "prompt_base_rot": T.quat2mat(np.asarray(obs["state.base_rotation"], dtype=np.float64)).astype(np.float32),
        "prompt_tcp_world_pos": _get_tcp_world_position(obs).astype(np.float32),
        "prompt_world_to_camera": get_camera_transform_matrix(
            sim=env.sim,
            camera_name=camera_name,
            camera_height=height,
            camera_width=width,
        ).astype(np.float32),
        "prompt_camera_resolution": np.asarray([height, width], dtype=np.float32),
    }


def _project_action_chunk_to_left_pixels(
    projection_fields: dict,
    actions: np.ndarray,
    action_scale: float,
) -> np.ndarray:
    """Project policy actions with the same geometry path used by vis_train.py."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < 3:
        raise ValueError(f"Expected action chunk with shape [horizon, action_dim>=3], got {actions.shape}.")

    required_keys = (
        "prompt_base_rot",
        "prompt_tcp_world_pos",
        "prompt_world_to_camera",
        "prompt_camera_resolution",
    )
    missing_keys = [key for key in required_keys if key not in projection_fields]
    if missing_keys:
        raise KeyError(f"projection_fields missing keys required for action projection: {missing_keys}")

    action_xyz = actions[:, :3]
    base_rot = np.asarray(projection_fields["prompt_base_rot"], dtype=np.float32)
    world_step_delta = action_xyz @ base_rot.T * float(action_scale)
    world_traj = np.asarray(projection_fields["prompt_tcp_world_pos"], dtype=np.float32)[None, :] + np.cumsum(
        world_step_delta,
        axis=0,
    )
    return _project_points_from_world_to_camera_unclipped(
        world_traj,
        np.asarray(projection_fields["prompt_world_to_camera"], dtype=np.float32),
    )


def _draw_plain_action_trajectory(
    image: np.ndarray,
    points_hw: np.ndarray,
    color: Sequence[int],
    *,
    thickness: int = 1,
) -> None:
    if len(points_hw) < 2:
        return
    points_hw = np.asarray(points_hw)
    points_xy = np.stack((points_hw[:, 1], points_hw[:, 0]), axis=-1).astype(np.int32)
    cv2.polylines(
        image,
        [points_xy],
        isClosed=False,
        color=tuple(int(channel) for channel in color),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def overlay_predicted_action_chunk(
    *,
    image_bgr: np.ndarray,
    env,
    obs: dict,
    action_chunk: np.ndarray,
    camera_name: str,
    action_scale: float,
    color: Sequence[int] = (0, 220, 255),
) -> np.ndarray:
    if action_scale <= 0.0:
        raise ValueError(f"overlay_action_scale must be positive, got {action_scale}.")
    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk with shape [T, D] for overlay, got {action_chunk.shape}.")
    if action_chunk.shape[-1] < 3:
        raise ValueError(f"Expected action dimension >= 3 for overlay, got {action_chunk.shape[-1]}.")
    if len(action_chunk) == 0:
        return image_bgr

    image = image_bgr.copy()
    projection_fields = _build_action_overlay_projection_fields(
        image_bgr=image,
        env=env,
        obs=obs,
        camera_name=camera_name,
    )
    points_hw = _project_action_chunk_to_left_pixels(
        projection_fields,
        action_chunk,
        action_scale=action_scale,
    )
    color = [int(channel) for channel in color]
    _draw_plain_action_trajectory(
        image,
        points_hw,
        color=color,
        thickness=1,
    )
    if len(points_hw) > 0:
        cv2.circle(image, (int(points_hw[0][1]), int(points_hw[0][0])), 3, (40, 40, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (int(points_hw[-1][1]), int(points_hw[-1][0])), 3, tuple(color), -1, cv2.LINE_AA)
    return image


def overlay_predicted_action_chunks(
    *,
    image_bgr: np.ndarray,
    env,
    obs: dict,
    action_chunks: Sequence[np.ndarray],
    camera_name: str,
    action_scale: float,
) -> np.ndarray:
    if action_scale <= 0.0:
        raise ValueError(f"overlay_action_scale must be positive, got {action_scale}.")

    colors = (
        (48, 130, 255),
        (80, 190, 255),
        (80, 220, 180),
        (80, 210, 90),
        (180, 210, 70),
        (230, 180, 60),
        (245, 120, 70),
        (235, 80, 120),
        (180, 90, 230),
        (0, 220, 255),
    )
    image = image_bgr.copy()
    projection_fields = _build_action_overlay_projection_fields(
        image_bgr=image,
        env=env,
        obs=obs,
        camera_name=camera_name,
    )
    for chunk_idx, action_chunk in enumerate(action_chunks):
        action_chunk = np.asarray(action_chunk, dtype=np.float64)
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with shape [T, D] for overlay, got {action_chunk.shape}.")
        if action_chunk.shape[-1] < 3:
            raise ValueError(f"Expected action dimension >= 3 for overlay, got {action_chunk.shape[-1]}.")
        if len(action_chunk) == 0:
            continue

        points_hw = _project_action_chunk_to_left_pixels(
            projection_fields,
            action_chunk,
            action_scale=action_scale,
        )
        color = colors[chunk_idx % len(colors)]
        _draw_plain_action_trajectory(
            image,
            points_hw,
            color=color,
            thickness=1,
        )
        if len(points_hw) > 0:
            start_xy = (int(points_hw[0][1]), int(points_hw[0][0]))
            end_xy = (int(points_hw[-1][1]), int(points_hw[-1][0]))
            cv2.circle(image, start_xy, 2, (40, 40, 255), -1, cv2.LINE_AA)
            cv2.circle(image, end_xy, 3, color, -1, cv2.LINE_AA)
    return image


def has_active_motion_prompt(prompt_inputs: dict) -> bool:
    local_motion = prompt_inputs.get("prompt_local_motion", prompt_inputs.get("prompt_wrist_relative_action"))
    if local_motion is not None:
        local_mask = bool(
            np.any(
                np.asarray(
                    prompt_inputs.get(
                        "prompt_local_motion_mask",
                        prompt_inputs.get("prompt_wrist_relative_action_mask", True),
                    ),
                    dtype=np.bool_,
                )
            )
        )
        if local_mask and not np.allclose(np.asarray(local_motion, dtype=np.float32), 0.0):
            return True

    global_motion = prompt_inputs.get("prompt_global_motion", prompt_inputs.get("prompt_primitive_cmd"))
    if global_motion is not None:
        global_motion = _as_prompt_global_motion(global_motion)
        global_mask = prompt_inputs.get("prompt_global_motion_mask", prompt_inputs.get("prompt_primitive_cmd_mask"))
        if global_mask is None:
            raise ValueError(
                "prompt_global_motion_mask must be provided with prompt_global_motion."
            )
        global_mask = bool(np.any(np.asarray(global_mask, dtype=np.bool_)))
        if global_mask and not np.allclose(global_motion, 0.0):
            return True

    drag = prompt_inputs.get("prompt_2d_drag")
    if drag is not None:
        drag_mask = bool(np.any(np.asarray(prompt_inputs.get("prompt_2d_drag_mask", True), dtype=np.bool_)))
        if drag_mask and not np.allclose(np.asarray(drag, dtype=np.float32), 0.0):
            return True

    return False


def select_replan_steps_for_prompt(
    *,
    prompt_inputs: dict,
    replan_steps: int,
    prompt_replan_steps: int,
) -> int:
    if replan_steps < 1:
        raise ValueError(f"replan_steps must be at least 1, got {replan_steps}.")
    if prompt_replan_steps < 1:
        raise ValueError(f"prompt_replan_steps must be at least 1, got {prompt_replan_steps}.")
    return prompt_replan_steps if has_active_motion_prompt(prompt_inputs) else replan_steps


def format_prompt_for_log(prompt_inputs: dict) -> str:
    parts = []

    global_motion = prompt_inputs.get("prompt_global_motion", prompt_inputs.get("prompt_primitive_cmd"))
    if global_motion is not None:
        global_motion = _as_prompt_global_motion(global_motion)
        global_text = _format_prompt_global_motion(global_motion)
        global_mask = prompt_inputs.get("prompt_global_motion_mask", prompt_inputs.get("prompt_primitive_cmd_mask"))
        if global_mask is None:
            raise ValueError(
                "prompt_global_motion_mask must be provided with prompt_global_motion."
            )
        if not bool(np.any(np.asarray(global_mask, dtype=np.bool_))):
            global_text = f"{global_text} masked"
        parts.append(f"global {global_text}")

    drag = prompt_inputs.get("prompt_2d_drag")
    if drag is not None:
        drag = np.asarray(drag, dtype=np.float32)
        if drag.shape != (2,):
            raise ValueError(f"Expected prompt_2d_drag shape (2,), got {drag.shape}.")
        drag_text = " ".join(f"{axis}={value:+.2f}" for axis, value in zip("xy", drag, strict=True))
        if not bool(np.any(np.asarray(prompt_inputs.get("prompt_2d_drag_mask", True), dtype=np.bool_))):
            drag_text = f"{drag_text} masked"
        parts.append(f"drag {drag_text}")

    local_motion = prompt_inputs.get("prompt_local_motion", prompt_inputs.get("prompt_wrist_relative_action"))
    if local_motion is None:
        return " | ".join(parts) if parts else "<none>"

    local_motion = np.asarray(local_motion, dtype=np.float32)
    if local_motion.shape != (3,):
        raise ValueError(
            f"Expected prompt_local_motion shape (3,), got {local_motion.shape}."
        )
    local_text = " ".join(
        f"{axis}={value:+.2f}" for axis, value in zip("xyz", local_motion, strict=True)
    )
    parts.append(f"local {local_text}")
    return " | ".join(parts)


def format_sample_kwargs_for_log(sample_kwargs: dict) -> str:
    if not sample_kwargs:
        return "<default>"
    parts = [f"{key}={value}" for key, value in sorted(sample_kwargs.items())]
    return " ".join(parts)


def _as_prompt_global_motion(value) -> np.ndarray:
    global_motion = np.asarray(value, dtype=np.float32)
    if global_motion.shape != (3,):
        raise ValueError(
            f"Expected prompt_global_motion shape (3,), got {global_motion.shape}."
        )
    if not np.all(np.isfinite(global_motion)) or np.any(global_motion < -1.0) or np.any(global_motion > 1.0):
        raise ValueError(
            "prompt_global_motion must contain finite xyz values in [-1, 1], "
            f"got {global_motion.tolist()}."
        )
    return global_motion


def _format_prompt_global_motion(global_motion: np.ndarray) -> str:
    values = np.asarray(global_motion, dtype=np.float32).reshape(3)
    return f"[{values[0]:+.2f}, {values[1]:+.2f}, {values[2]:+.2f}]"


def render_prompt_window(prompt_inputs: dict, env_name: str, episode_idx: int, infer_idx: int) -> None:
    prompt_image = None
    prompt_images = prompt_inputs.get("prompt_images")
    if isinstance(prompt_images, dict):
        prompt_image = prompt_images.get("prompt_0")

    if prompt_image is None:
        prompt_canvas = np.full((320, 320, 3), 245, dtype=np.uint8)
        cv2.putText(
            prompt_canvas,
            "No prompt image",
            (36, 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
    else:
        prompt_canvas = np.ascontiguousarray(prompt_image)
        if prompt_canvas.dtype != np.uint8:
            prompt_canvas = image_tools.convert_to_uint8(prompt_canvas)
        prompt_canvas = cv2.cvtColor(prompt_canvas, cv2.COLOR_RGB2BGR)
        prompt_canvas = cv2.resize(prompt_canvas, (420, 420), interpolation=cv2.INTER_NEAREST)

    sample_kwargs = prompt_inputs.get("sample_kwargs")
    header = np.full((120, prompt_canvas.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        f"{env_name} | episode {episode_idx} | infer {infer_idx}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"Prompt: {format_prompt_for_log(prompt_inputs)}",
        (12, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    if isinstance(sample_kwargs, dict):
        cv2.putText(
            header,
            f"Sampling: {format_sample_kwargs_for_log(sample_kwargs)}",
            (12, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow("RoboCasa Prompt", np.concatenate([header, prompt_canvas], axis=0))


@dataclasses.dataclass
class InteractivePromptState:
    payload: dict | None = None
    prompt_mem: int = 0
    prompt_update_count: int = 0

    def update(self, payload: dict) -> None:
        self.prompt_update_count += 1
        self.prompt_mem = 0
        self.payload = dict(payload)

    def clear(self) -> None:
        self.payload = None
        self.prompt_mem = 0
        self.prompt_update_count = 0

    def build_model_inputs(self) -> dict:
        if self.payload is None:
            return {}
        excluded_keys = {
            "interactive_prompt_result",
            "sample_kwargs",
            "prompt_effect_mode",
            "prompt_phase1_source",
            "prompt_images",
            "prompt_image_masks",
        }
        return {
            key: value
            for key, value in self.payload.items()
            if key not in excluded_keys
        }

    def build_display_inputs(self) -> dict:
        if self.payload is None:
            return {}
        return {
            key: value
            for key, value in self.payload.items()
            if key not in {"interactive_prompt_result"}
        }

    def add_prompt_mem(self) -> None:
        if self.payload is not None:
            self.prompt_mem += 1

    def build_sample_kwargs(self, *, policy_inference_steps: float | None = None, emit_num_steps: bool = False) -> dict:
        if self.payload is None:
            return {}
        sample_kwargs = self.payload.get("sample_kwargs")
        sample_kwargs = dict(sample_kwargs) if isinstance(sample_kwargs, dict) else {}
        num_steps = float(
            sample_kwargs.get(
                "num_steps",
                10.0 if policy_inference_steps is None else policy_inference_steps,
            )
        )
        if emit_num_steps:
            sample_kwargs.setdefault("num_steps", num_steps)
        if self._has_short_term_prompt():
            if self.prompt_mem > 0:
                sample_kwargs["phase2_steps"] = num_steps
        elif self._has_long_term_prompt():
            base_phase2_steps = float(sample_kwargs.get("phase2_steps", 0.0))
            sample_kwargs["phase2_steps"] = min(
                num_steps,
                base_phase2_steps + self.prompt_mem * phase2_prompt_mem_step(num_steps),
            )
        return sample_kwargs

    def _has_short_term_prompt(self) -> bool:
        if self.payload is None:
            return False
        return self._prompt_effect_mode(self.payload) == "short_term"

    def _has_long_term_prompt(self) -> bool:
        if self.payload is None:
            return False
        return self._prompt_effect_mode(self.payload) == "long_term"

    def _prompt_effect_mode(self, payload: dict) -> str:
        return str(payload.get("prompt_effect_mode", "long_term")).strip().lower()


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = image_tools.convert_to_uint8(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}.")
    return np.ascontiguousarray(image)


def _prompt_vector_is_active(prompt_payload: dict, value_key: str, mask_key: str) -> bool:
    if value_key not in prompt_payload:
        return False
    return bool(np.any(np.asarray(prompt_payload.get(mask_key, True), dtype=np.bool_)))


def _format_prompt_vector(prompt_payload: dict, value_key: str, mask_key: str, axes: str) -> str:
    if not _prompt_vector_is_active(prompt_payload, value_key, mask_key):
        return "<none>"
    values = np.asarray(prompt_payload[value_key], dtype=np.float32).reshape(-1)
    if value_key == "prompt_global_motion":
        return " ".join(f"{axis}={value:+.2f}" for axis, value in zip(axes, values, strict=False))
    return " ".join(f"{axis}={value:+.2f}" for axis, value in zip(axes, values, strict=False))


def _draw_prompt_action_text(frame_rgb: np.ndarray, prompt_payload: dict | None) -> np.ndarray:
    frame = frame_rgb.copy()
    if prompt_payload is None:
        lines = ["Wrist action: <none>", "Global action: <none>"]
    else:
        wrist_text = _format_prompt_vector(
            prompt_payload,
            "prompt_local_motion",
            "prompt_local_motion_mask",
            "xyz",
        )
        global_text = _format_prompt_vector(
            prompt_payload,
            "prompt_global_motion",
            "prompt_global_motion_mask",
            "xyz",
        )
        lines = [f"Wrist action: {wrist_text}", f"Global action: {global_text}"]

    pad = 10
    line_height = 24
    panel_height = pad * 2 + line_height * len(lines)
    panel_width = min(frame.shape[1], max(360, int(frame.shape[1] * 0.62)))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, dst=frame)
    for line_idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (pad, pad + 17 + line_idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def _make_prompt_drag_panel(prompt_payload: dict | None, height: int) -> np.ndarray:
    panel_width = height
    panel = np.full((height, panel_width, 3), 245, dtype=np.uint8)
    header_height = 34
    cv2.putText(
        panel,
        "2D drag prompt",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )

    prompt_image = None
    if prompt_payload is not None and _prompt_vector_is_active(
        prompt_payload,
        "prompt_2d_drag",
        "prompt_2d_drag_mask",
    ):
        prompt_images = prompt_payload.get("prompt_images")
        if isinstance(prompt_images, dict):
            prompt_image = prompt_images.get("prompt_0")

    if prompt_image is None:
        cv2.putText(
            panel,
            "No active drag",
            (18, header_height + max(34, (height - header_height) // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        return panel

    prompt_image = _as_uint8_rgb(prompt_image)
    image_area_height = height - header_height
    scale = min(panel_width / prompt_image.shape[1], image_area_height / prompt_image.shape[0])
    resized_width = max(1, int(round(prompt_image.shape[1] * scale)))
    resized_height = max(1, int(round(prompt_image.shape[0] * scale)))
    resized = cv2.resize(prompt_image, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    x0 = (panel_width - resized_width) // 2
    y0 = header_height + (image_area_height - resized_height) // 2
    panel[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return panel


def compose_interactive_replay_frame(frame_rgb: np.ndarray, prompt_state: InteractivePromptState) -> np.ndarray:
    frame_rgb = _as_uint8_rgb(frame_rgb)
    prompt_payload = prompt_state.build_display_inputs() if prompt_state.payload is not None else None
    annotated_frame = _draw_prompt_action_text(frame_rgb, prompt_payload)
    drag_panel = _make_prompt_drag_panel(prompt_payload, annotated_frame.shape[0])
    gap = np.full((annotated_frame.shape[0], 8, 3), 235, dtype=np.uint8)
    return np.concatenate([annotated_frame, gap, drag_panel], axis=1)


def compose_rollout_replay_frame(
    *,
    image_bgr: np.ndarray,
    prompt_state: InteractivePromptState,
    interactive_prompt: bool,
) -> np.ndarray:
    frame_rgb = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2RGB)
    if interactive_prompt:
        return compose_interactive_replay_frame(frame_rgb, prompt_state)
    return _as_uint8_rgb(frame_rgb)
