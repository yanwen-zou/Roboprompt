from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from scripts.realworld.eval.eval_ui import render_utils

if TYPE_CHECKING:
    import jax.numpy as jnp
    from openpi.models import model as _model


LOGGER = logging.getLogger("openpi.interactive_labeling")
PHASE2_STEP_SIZE = 0.2


def _get_jnp():
    import jax.numpy as jnp

    return jnp


@dataclasses.dataclass(frozen=True)
class InteractiveLabelConfig:
    enabled: bool = False
    every_n_steps: int = 1
    batch_index: int = 0
    output_dir: str | None = None
    wrist_action_dim: int = 3
    prompt_chunk_ttl: int | None = 4
    wrist_action_chunk_ttl: int | None = None

    @classmethod
    def from_train_config(cls, config: Any) -> "InteractiveLabelConfig":
        return cls(
            enabled=bool(getattr(config, "interactive_labeling", False)),
            every_n_steps=max(1, int(getattr(config, "interactive_label_every", 1))),
            batch_index=max(0, int(getattr(config, "interactive_label_batch_index", 0))),
            output_dir=getattr(config, "interactive_label_output_dir", None),
            wrist_action_dim=max(1, int(getattr(config, "interactive_label_wrist_action_dim", 3))),
            prompt_chunk_ttl=(
                None
                if getattr(config, "interactive_label_prompt_chunk_ttl", 4) is None
                else max(1, int(getattr(config, "interactive_label_prompt_chunk_ttl", 4)))
            ),
            wrist_action_chunk_ttl=(
                None
                if getattr(config, "interactive_label_wrist_action_chunk_ttl", None) is None
                else max(0, int(getattr(config, "interactive_label_wrist_action_chunk_ttl")))
            ),
        )


@dataclasses.dataclass(frozen=True)
class InteractivePromptResult:
    prompt_image_bgr: np.ndarray
    prompt_local_motion: np.ndarray
    prompt_global_motion: np.ndarray
    prompt_2d_drag: np.ndarray
    phase2_steps: float
    prompt_effect_mode: str
    prompt_phase1_source: str
    draw_ops: list[dict[str, Any]]
    prompt_overlay_bgr: np.ndarray | None = None


class InteractiveStepLabeler:
    def __init__(self, config: InteractiveLabelConfig, checkpoint_dir: Path):
        self._config = config
        self._checkpoint_dir = checkpoint_dir
        self._output_dir = Path(config.output_dir) if config.output_dir else checkpoint_dir / "interactive_labels"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._cached_prompt_result: InteractivePromptResult | None = None
        self._remaining_prompt_chunks: int | None = None
        self._remaining_wrist_action_chunks: int | None = None

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def should_label(self, step: int) -> bool:
        return self.enabled and step % self._config.every_n_steps == 0

    def apply_step(self, step: int, observation: _model.Observation) -> _model.Observation:
        if not self.enabled:
            return observation

        if self._cached_prompt_result is None or self.should_label(step):
            self._cached_prompt_result = self.collect_prompt(step, observation)
            self._remaining_prompt_chunks = self._config.prompt_chunk_ttl
            self._remaining_wrist_action_chunks = (
                self._config.prompt_chunk_ttl
                if self._config.wrist_action_chunk_ttl is None
                else self._config.wrist_action_chunk_ttl
            )

        apply_prompt = self._should_apply_prompt()
        apply_wrist_prompt = apply_prompt and self._should_apply_wrist_prompt()
        if not apply_prompt:
            return observation
        updated_observation = self._inject_prompt_inputs(
            observation,
            self._cached_prompt_result,
            apply_wrist_prompt=apply_wrist_prompt,
        )
        self._consume_prompt_chunk(applied=True)
        self._consume_wrist_prompt_chunk(apply_wrist_prompt)
        return updated_observation

    def label_step(self, step: int, observation: _model.Observation) -> _model.Observation:
        prompt_result = self.collect_prompt(step, observation)
        self._cached_prompt_result = prompt_result
        self._remaining_prompt_chunks = self._config.prompt_chunk_ttl
        self._remaining_wrist_action_chunks = (
            self._config.prompt_chunk_ttl
            if self._config.wrist_action_chunk_ttl is None
            else self._config.wrist_action_chunk_ttl
        )
        return self._inject_prompt_inputs(
            observation,
            prompt_result,
            apply_wrist_prompt=self._should_apply_wrist_prompt(),
        )

    def clear_cached_prompt(self) -> None:
        self._cached_prompt_result = None
        self._remaining_prompt_chunks = None
        self._remaining_wrist_action_chunks = None

    def collect_prompt(self, step: int, observation: _model.Observation) -> InteractivePromptResult:
        base_image, wrist_image = self._extract_images(observation, self._config.batch_index)
        step_dir = self._output_dir / f"step_{step:08d}" / f"sample_{self._config.batch_index:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(step_dir / "base_reference.png"), base_image)
        cv2.imwrite(str(step_dir / "wrist_reference.png"), wrist_image)
        LOGGER.info("Interactive labeling opened for step %d at %s", step, step_dir)

        prompt_result = _collect_prompt_with_opencv(
            base_image=base_image,
            wrist_image=wrist_image,
            wrist_action_dim=self._config.wrist_action_dim,
        )

        cv2.imwrite(str(step_dir / "prompt_overlay.png"), prompt_result.prompt_image_bgr)
        (step_dir / "label.json").write_text(
            json.dumps(
                {
                    "mode": "opencv_interactive_prompt",
                    "step": int(step),
                    "batch_index": int(self._config.batch_index),
                    "prompt_local_motion": prompt_result.prompt_local_motion.astype(float).tolist(),
                    "prompt_global_motion": prompt_result.prompt_global_motion.astype(float).tolist(),
                    "prompt_2d_drag": prompt_result.prompt_2d_drag.astype(float).tolist(),
                    "prompt_effect_mode": prompt_result.prompt_effect_mode,
                    "prompt_phase1_source": prompt_result.prompt_phase1_source,
                    "draw_ops": prompt_result.draw_ops,
                    "base_reference": "base_reference.png",
                    "wrist_reference": "wrist_reference.png",
                    "prompt_overlay": "prompt_overlay.png",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return prompt_result

    def _extract_images(self, observation: _model.Observation, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_size = int(np.asarray(observation.state).shape[0])
        if batch_index >= batch_size:
            raise IndexError(
                f"interactive_label_batch_index={batch_index} is out of range for batch size {batch_size}"
            )
        base_image = np.asarray(observation.images["base_0_rgb"][batch_index])
        wrist_image = np.asarray(observation.images["left_wrist_0_rgb"][batch_index])
        return _to_uint8_bgr_image(base_image), _to_uint8_bgr_image(wrist_image)

    def _inject_prompt_inputs(
        self,
        observation: _model.Observation,
        prompt_result: InteractivePromptResult,
        *,
        apply_wrist_prompt: bool = True,
    ) -> _model.Observation:
        jnp = _get_jnp()
        batch_index = self._config.batch_index
        batch_size = int(np.asarray(observation.state).shape[0])

        prompt_image_rgb = cv2.cvtColor(prompt_result.prompt_image_bgr, cv2.COLOR_BGR2RGB)
        prompt_image_model = _uint8_rgb_to_model_range(prompt_image_rgb)

        if observation.prompt_images is None:
            prompt_images = {
                "prompt_0": jnp.zeros_like(observation.images["base_0_rgb"]),
            }
        else:
            prompt_images = dict(observation.prompt_images)
            if "prompt_0" not in prompt_images:
                prompt_images["prompt_0"] = jnp.zeros_like(observation.images["base_0_rgb"])

        if observation.prompt_image_masks is None:
            prompt_image_masks = {
                "prompt_0": jnp.zeros((batch_size,), dtype=jnp.bool_),
            }
        else:
            prompt_image_masks = dict(observation.prompt_image_masks)
            if "prompt_0" not in prompt_image_masks:
                prompt_image_masks["prompt_0"] = jnp.zeros((batch_size,), dtype=jnp.bool_)

        prompt_dtype = prompt_images["prompt_0"].dtype
        prompt_images["prompt_0"] = prompt_images["prompt_0"].at[batch_index].set(
            jnp.asarray(prompt_image_model, dtype=prompt_dtype)
        )
        prompt_image_masks["prompt_0"] = prompt_image_masks["prompt_0"].at[batch_index].set(True)

        prompt_local_motion = observation.prompt_local_motion
        prompt_local_motion_mask = observation.prompt_local_motion_mask
        if apply_wrist_prompt:
            prompt_local_motion = _ensure_vector_buffer(
                observation.prompt_local_motion,
                batch_size=batch_size,
                vector_dim=prompt_result.prompt_local_motion.shape[0],
                dtype=jnp.float32,
            )
            prompt_local_motion = prompt_local_motion.at[batch_index].set(
                jnp.asarray(prompt_result.prompt_local_motion, dtype=jnp.float32)
            )
            if prompt_local_motion_mask is None:
                prompt_local_motion_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
            prompt_local_motion_mask = prompt_local_motion_mask.at[batch_index].set(
                jnp.bool_(not np.allclose(prompt_result.prompt_local_motion, 0.0))
            )

        prompt_global_motion = _ensure_vector_buffer(
            observation.prompt_global_motion,
            batch_size=batch_size,
            vector_dim=3,
            dtype=jnp.float32,
        )
        prompt_global_motion = prompt_global_motion.at[batch_index].set(
            jnp.asarray(prompt_result.prompt_global_motion, dtype=jnp.float32)
        )

        prompt_global_motion_mask = observation.prompt_global_motion_mask
        if prompt_global_motion_mask is None:
            prompt_global_motion_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
        prompt_global_motion_mask = prompt_global_motion_mask.at[batch_index].set(
            jnp.bool_(not np.allclose(prompt_result.prompt_global_motion, 0.0))
        )

        prompt_2d_drag = _ensure_vector_buffer(
            observation.prompt_2d_drag,
            batch_size=batch_size,
            vector_dim=2,
            dtype=jnp.float32,
        )
        prompt_2d_drag = prompt_2d_drag.at[batch_index].set(
            jnp.asarray(prompt_result.prompt_2d_drag, dtype=jnp.float32)
        )
        prompt_2d_drag_mask = observation.prompt_2d_drag_mask
        if prompt_2d_drag_mask is None:
            prompt_2d_drag_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
        prompt_2d_drag_mask = prompt_2d_drag_mask.at[batch_index].set(
            jnp.bool_(not np.allclose(prompt_result.prompt_2d_drag, 0.0))
        )

        return dataclasses.replace(
            observation,
            prompt_images=prompt_images,
            prompt_image_masks=prompt_image_masks,
            prompt_local_motion=prompt_local_motion,
            prompt_local_motion_mask=prompt_local_motion_mask,
            prompt_global_motion=prompt_global_motion,
            prompt_global_motion_mask=prompt_global_motion_mask,
            prompt_2d_drag=prompt_2d_drag,
            prompt_2d_drag_mask=prompt_2d_drag_mask,
        )

    def _should_apply_prompt(self) -> bool:
        return self._remaining_prompt_chunks is None or self._remaining_prompt_chunks > 0

    def _consume_prompt_chunk(self, applied: bool) -> None:
        if not applied or self._remaining_prompt_chunks is None:
            return
        self._remaining_prompt_chunks = max(0, self._remaining_prompt_chunks - 1)

    def _should_apply_wrist_prompt(self) -> bool:
        return self._remaining_wrist_action_chunks is None or self._remaining_wrist_action_chunks > 0

    def _consume_wrist_prompt_chunk(self, applied: bool) -> None:
        if not applied or self._remaining_wrist_action_chunks is None:
            return
        self._remaining_wrist_action_chunks = max(0, self._remaining_wrist_action_chunks - 1)

def _ensure_vector_buffer(existing: Any, *, batch_size: int, vector_dim: int, dtype: Any) -> "jnp.ndarray":
    jnp = _get_jnp()
    if existing is None:
        return jnp.zeros((batch_size, vector_dim), dtype=dtype)
    array = jnp.asarray(existing, dtype=dtype)
    if array.shape != (batch_size, vector_dim):
        raise ValueError(
            f"Expected vector buffer shape {(batch_size, vector_dim)}, got {array.shape}."
        )
    return array


def _to_uint8_bgr_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if min_value >= -1.01 and max_value <= 1.01:
            image = ((image + 1.0) * 127.5).clip(0, 255)
        elif min_value >= -0.01 and max_value <= 1.01:
            image = (image * 255.0).clip(0, 255)
        else:
            image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _uint8_rgb_to_model_range(image_rgb: np.ndarray) -> np.ndarray:
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    return image_rgb / 255.0 * 2.0 - 1.0


def collect_prompt_payload_from_images(
    *,
    base_image_rgb: np.ndarray,
    wrist_image_rgb: np.ndarray | None = None,
    observation_state: np.ndarray | None = None,
    source_image_shape: tuple[int, int, int] | None = None,
    wrist_action_dim: int = 3,
    phase2_steps: float = 0.0,
    max_phase2_steps: float = 10.0,
) -> dict[str, Any]:
    if wrist_image_rgb is None:
        wrist_image_rgb = base_image_rgb
    prompt_result = _collect_prompt_with_opencv(
        base_image=_to_uint8_bgr_image(base_image_rgb),
        wrist_image=_to_uint8_bgr_image(wrist_image_rgb),
        wrist_action_dim=wrist_action_dim,
        phase2_steps=phase2_steps,
        max_phase2_steps=max_phase2_steps,
        tcp_point_hw=_project_tcp_point_to_prompt_image(
            observation_state=observation_state,
            image_shape=base_image_rgb.shape,
            source_image_shape=source_image_shape,
        ),
    )
    prompt_image_rgb = cv2.cvtColor(prompt_result.prompt_image_bgr, cv2.COLOR_BGR2RGB)
    has_draw_overlay = bool(prompt_result.draw_ops)
    payload = {
        "prompt_images": {
            "prompt_0": prompt_image_rgb,
        },
        "prompt_image_masks": {
            "prompt_0": np.bool_(has_draw_overlay),
        },
        "prompt_global_motion": prompt_result.prompt_global_motion,
        "prompt_global_motion_mask": np.bool_(
            not np.allclose(prompt_result.prompt_global_motion, 0.0)
        ),
        "prompt_local_motion": np.asarray(
            prompt_result.prompt_local_motion, dtype=np.float32
        ),
        "prompt_local_motion_mask": np.bool_(
            not np.allclose(prompt_result.prompt_local_motion, 0.0)
        ),
        "prompt_2d_drag": np.asarray(prompt_result.prompt_2d_drag, dtype=np.float32),
        "prompt_2d_drag_mask": np.bool_(
            not np.allclose(prompt_result.prompt_2d_drag, 0.0)
        ),
        "sample_kwargs": {
            "phase2_steps": float(prompt_result.phase2_steps),
        },
        "prompt_effect_mode": prompt_result.prompt_effect_mode,
        "prompt_phase1_source": prompt_result.prompt_phase1_source,
        "interactive_prompt_result": prompt_result,
    }
    if prompt_result.prompt_overlay_bgr is not None:
        payload["prompt_images"]["prompt_overlay_0"] = cv2.cvtColor(
            prompt_result.prompt_overlay_bgr,
            cv2.COLOR_BGR2RGB,
        )
    return payload


def _drag_vector_from_draw_ops(draw_ops: list[dict[str, Any]], image_hw: tuple[int, int]) -> np.ndarray:
    if not draw_ops:
        return np.zeros((2,), dtype=np.float32)
    points_hw = np.asarray(draw_ops[-1].get("points_hw", []), dtype=np.float32)
    if points_hw.ndim != 2 or points_hw.shape[0] < 2 or points_hw.shape[-1] != 2:
        return np.zeros((2,), dtype=np.float32)
    height, width = image_hw
    denom_xy = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    start_xy = np.asarray([points_hw[0, 1], points_hw[0, 0]], dtype=np.float32) / denom_xy
    end_xy = np.asarray([points_hw[-1, 1], points_hw[-1, 0]], dtype=np.float32) / denom_xy
    return (end_xy - start_xy).astype(np.float32)


def _project_tcp_point_to_prompt_image(
    *,
    observation_state: np.ndarray | None,
    image_shape: tuple[int, int, int],
    source_image_shape: tuple[int, int, int] | None,
) -> np.ndarray | None:
    if observation_state is None:
        return None
    try:
        from my_device.macros import E2H_CAM_T, E2H_INTRINSIC, Gripper_TCP_T
        from scripts.utils.realworld_projection import compute_gripper_tip_points_base
        from scripts.utils.realworld_projection import map_camera_pixels_to_video_pixels
        from scripts.utils.realworld_projection import project_base_points_to_camera_pixels_safe
    except ImportError as exc:
        LOGGER.debug("TCP target prompt unavailable: failed to import camera calibration: %s", exc)
        return None

    state = np.asarray(observation_state, dtype=np.float64).reshape(-1)
    if state.shape[0] < 3:
        return None
    point_base = state[:3][None]
    if state.shape[0] >= 7:
        point_base = compute_gripper_tip_points_base(state[None], Gripper_TCP_T)
    source_height, source_width = (
        (int(source_image_shape[0]), int(source_image_shape[1]))
        if source_image_shape is not None
        else (int(image_shape[0]), int(image_shape[1]))
    )
    try:
        pixels_xy, _, valid_mask = project_base_points_to_camera_pixels_safe(
            point_base,
            base_t_camera=E2H_CAM_T,
            intrinsic=E2H_INTRINSIC,
        )
    except ValueError as exc:
        LOGGER.debug("TCP target prompt unavailable: failed to project TCP point: %s", exc)
        return None
    if not bool(valid_mask[0]):
        return None
    mapped_xy = map_camera_pixels_to_video_pixels(
        pixels_xy,
        source_width=source_width,
        source_height=source_height,
        video_width=int(image_shape[1]),
        video_height=int(image_shape[0]),
        processed_padding=8,
    )[0]
    if not np.all(np.isfinite(mapped_xy)):
        return None
    return np.asarray([mapped_xy[1], mapped_xy[0]], dtype=np.float32)


def _scale_point_hw(
    point_hw: np.ndarray | None,
    *,
    from_hw: tuple[int, int],
    to_hw: tuple[int, int],
) -> np.ndarray | None:
    if point_hw is None:
        return None
    point = np.asarray(point_hw, dtype=np.float32).reshape(2)
    scale = np.asarray(
        [
            float(to_hw[0]) / max(float(from_hw[0]), 1.0),
            float(to_hw[1]) / max(float(from_hw[1]), 1.0),
        ],
        dtype=np.float32,
    )
    return point * scale


def _current_screen_size() -> tuple[int, int]:
    width_override = os.environ.get("INTERACTIVE_UI_SCREEN_WIDTH")
    height_override = os.environ.get("INTERACTIVE_UI_SCREEN_HEIGHT")
    if width_override and height_override:
        try:
            return max(1, int(width_override)), max(1, int(height_override))
        except ValueError:
            LOGGER.warning(
                "Ignoring invalid INTERACTIVE_UI_SCREEN_WIDTH/HEIGHT=%r/%r.",
                width_override,
                height_override,
            )

    try:
        result = subprocess.run(
            ["xrandr", "--current"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return (1920, 1080)

    match = re.search(r"\bcurrent\s+(\d+)\s+x\s+(\d+)\b", result.stdout)
    if match is None:
        return (1920, 1080)
    return int(match.group(1)), int(match.group(2))


def _bounded_preview_layout(
    base_shape: tuple[int, int],
    wrist_shape: tuple[int, int],
    *,
    gap: int,
) -> tuple[int, int, int, int]:
    screen_width, screen_height = _current_screen_size()
    max_width = int(os.environ.get("INTERACTIVE_UI_MAX_WIDTH", max(900, screen_width - 120)))
    max_height = int(os.environ.get("INTERACTIVE_UI_MAX_HEIGHT", max(680, screen_height - 120)))
    max_width = max(640, max_width)
    max_height = max(560, max_height)

    controls_height = render_utils.CONTROL_PANEL_HEIGHT
    header_height = render_utils.PANEL_HEADER_HEIGHT
    available_image_height = max_height - controls_height - header_height - gap
    target_height = min(
        max(render_utils.PREVIEW_IMAGE_TARGET_HEIGHT, int(base_shape[0])),
        max(220, available_image_height),
    )

    base_ratio = float(base_shape[1]) / max(float(base_shape[0]), 1.0)
    wrist_ratio = float(wrist_shape[1]) / max(float(wrist_shape[0]), 1.0)
    max_image_height_by_width = int((max_width - gap) / max(base_ratio + wrist_ratio, 1e-6))
    target_height = min(target_height, max(220, max_image_height_by_width))

    top_width = int(round(base_ratio * target_height)) + gap + int(round(wrist_ratio * target_height))
    preview_width = min(max(900, top_width + 160), max_width)
    return target_height, preview_width, max_width, max_height


def _collect_prompt_with_opencv(
    *,
    base_image: np.ndarray,
    wrist_image: np.ndarray,
    wrist_action_dim: int,
    phase2_steps: float = 0.0,
    max_phase2_steps: float = 10.0,
    tcp_point_hw: np.ndarray | None = None,
) -> InteractivePromptResult:
    if max_phase2_steps < 0:
        raise ValueError(f"max_phase2_steps must be non-negative, got {max_phase2_steps}.")
    window_name = "Interactive Prompt Labeler"
    original_base_shape = base_image.shape[:2]
    gap = render_utils.PREVIEW_GAP
    target_height, preview_width, _, _ = _bounded_preview_layout(
        base_image.shape[:2],
        wrist_image.shape[:2],
        gap=gap,
    )
    base_image = render_utils.resize_to_height(base_image, target_height)
    wrist_image = render_utils.resize_to_height(wrist_image, target_height)
    tcp_point_hw = _scale_point_hw(
        tcp_point_hw,
        from_hw=original_base_shape,
        to_hw=base_image.shape[:2],
    )

    draw_ops: list[dict[str, Any]] = []
    primitive_cmd_values = np.zeros((3,), dtype=np.float32)
    phase2_steps = float(np.clip(phase2_steps, 0.0, max_phase2_steps))
    prompt_effect_mode = "long_term"
    prompt_phase1_source = "evo"
    state: dict[str, Any] = {
        "stroke_points": None,
        "wrist_drag_index": None,
        "primitive_drag_index": None,
    }

    prompt_canvas = base_image.copy()
    wrist_values = np.zeros((wrist_action_dim,), dtype=np.float32)
    left_panel_width = base_image.shape[1]
    wrist_panel_x0 = left_panel_width + gap
    image_y0 = render_utils.PANEL_HEADER_HEIGHT
    controls_y0 = image_y0 + max(base_image.shape[0], wrist_image.shape[0]) + gap

    def set_prompt_effect_mode(mode: str) -> None:
        nonlocal prompt_effect_mode
        if mode not in {"short_term", "long_term"}:
            raise ValueError(f"Unsupported prompt effect mode: {mode!r}")
        prompt_effect_mode = mode

    def set_prompt_phase1_source(source: str) -> None:
        nonlocal prompt_phase1_source
        if source not in {"evo", "hardcode"}:
            raise ValueError(f"Unsupported prompt phase1 source: {source!r}")
        prompt_phase1_source = source

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if x < 0 or y < 0:
            return
        if state["primitive_drag_index"] is not None and event in (
            cv2.EVENT_MOUSEMOVE,
            cv2.EVENT_LBUTTONUP,
        ):
            primitive_index = int(state["primitive_drag_index"])
            primitive_regions = render_utils.get_global_bar_regions(preview_width)
            primitive_cmd_values[primitive_index] = render_utils.slider_value_from_x(
                primitive_regions[primitive_index],
                x,
            )
            if event == cv2.EVENT_LBUTTONUP:
                state["primitive_drag_index"] = None
            return

        local_y = y - image_y0
        if 0 <= x < base_image.shape[1] and 0 <= local_y < base_image.shape[0]:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["stroke_points"] = [(x, local_y)]
            elif event == cv2.EVENT_MOUSEMOVE and state["stroke_points"] is not None:
                state["stroke_points"].append((x, local_y))
            elif event == cv2.EVENT_LBUTTONUP and state["stroke_points"] is not None:
                state["stroke_points"].append((x, local_y))
                op = render_utils.shape_from_stroke(state["stroke_points"])
                draw_ops.append(op)
                state["stroke_points"] = None
            elif event == cv2.EVENT_RBUTTONDOWN:
                draw_ops.append(
                    {
                        "type": "point",
                        "point_hw": [float(local_y), float(x)],
                        "source": "right_click_point",
                    }
                )
            return

        wrist_local_x = x - wrist_panel_x0
        if 0 <= wrist_local_x < wrist_image.shape[1] and 0 <= local_y < wrist_image.shape[0]:
            bar_regions = render_utils.get_wrist_bar_regions(
                wrist_image.shape[0],
                wrist_image.shape[1],
                wrist_action_dim,
            )
            if event == cv2.EVENT_LBUTTONDOWN:
                bar_index = render_utils.find_bar_at_point(bar_regions, wrist_local_x, local_y)
                if bar_index is not None:
                    state["wrist_drag_index"] = bar_index
                    wrist_values[bar_index] = render_utils.slider_value_from_x(bar_regions[bar_index], wrist_local_x)
            elif event == cv2.EVENT_MOUSEMOVE and state["wrist_drag_index"] is not None:
                bar_index = int(state["wrist_drag_index"])
                wrist_values[bar_index] = render_utils.slider_value_from_x(bar_regions[bar_index], wrist_local_x)
            elif event == cv2.EVENT_LBUTTONUP and state["wrist_drag_index"] is not None:
                bar_index = int(state["wrist_drag_index"])
                wrist_values[bar_index] = render_utils.slider_value_from_x(bar_regions[bar_index], wrist_local_x)
                state["wrist_drag_index"] = None
            return

        if event == cv2.EVENT_LBUTTONUP:
            state["wrist_drag_index"] = None
            state["primitive_drag_index"] = None

        control_local_y = y - controls_y0
        if control_local_y >= 0:
            primitive_regions = render_utils.get_global_bar_regions(preview_width)
            if event == cv2.EVENT_LBUTTONDOWN:
                primitive_index = render_utils.find_bar_at_point(primitive_regions, x, control_local_y)
                if primitive_index is not None:
                    state["primitive_drag_index"] = primitive_index
                    primitive_cmd_values[primitive_index] = render_utils.slider_value_from_x(
                        primitive_regions[primitive_index],
                        x,
                    )
                    return
            elif event == cv2.EVENT_MOUSEMOVE and state["primitive_drag_index"] is not None:
                primitive_index = int(state["primitive_drag_index"])
                primitive_cmd_values[primitive_index] = render_utils.slider_value_from_x(
                    primitive_regions[primitive_index],
                    x,
                )
                return

        if event == cv2.EVENT_LBUTTONUP:
            control_local_y = y - controls_y0
            if 224 <= control_local_y <= 266:
                effect_button_width = 164
                effect_gap = 12
                effect_start_x = 180
                for mode_index, mode in enumerate(("short_term", "long_term")):
                    x0 = effect_start_x + mode_index * (effect_button_width + effect_gap)
                    x1 = x0 + effect_button_width
                    if x0 <= x <= x1:
                        set_prompt_effect_mode(mode)
                        break
            if 276 <= control_local_y <= 318:
                source_button_width = 164
                source_gap = 12
                source_start_x = 180
                for source_index, source in enumerate(("evo", "hardcode")):
                    x0 = source_start_x + source_index * (source_button_width + source_gap)
                    x1 = x0 + source_button_width
                    if x0 <= x <= x1:
                        set_prompt_phase1_source(source)
                        break

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_width = max(preview_width, base_image.shape[1] + wrist_image.shape[1] + gap)
    initial_height = image_y0 + max(base_image.shape[0], wrist_image.shape[0]) + gap + render_utils.CONTROL_PANEL_HEIGHT
    cv2.resizeWindow(window_name, initial_width, initial_height)
    cv2.moveWindow(window_name, 20, 20)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            rendered_prompt = render_utils.render_draw_ops(prompt_canvas, draw_ops)
            if tcp_point_hw is not None:
                rendered_prompt = render_utils.render_draw_ops(
                    rendered_prompt,
                    [{"type": "point", "point_hw": tcp_point_hw.tolist()}],
                )
            if state["stroke_points"] is not None:
                rendered_prompt = render_utils.render_draw_ops(
                    rendered_prompt,
                    [render_utils.shape_from_stroke(state["stroke_points"])],
                )

            prompt_controls = render_utils.render_prompt_controls(
                wrist_values=wrist_values,
                primitive_values=primitive_cmd_values,
                phase2_steps=phase2_steps,
                max_phase2_steps=max_phase2_steps,
                prompt_effect_mode=prompt_effect_mode,
                prompt_phase1_source=prompt_phase1_source,
                width=preview_width,
            )
            preview = render_utils.compose_preview(
                left_image=rendered_prompt,
                wrist_image=render_utils.render_wrist_controls(
                    wrist_image,
                    wrist_values,
                    active_index=state["wrist_drag_index"],
                ),
                control_panel=prompt_controls,
                gap=gap,
            )

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue
            if key in (13, 10):
                break
            if key == 27:
                raise SystemExit("Canceled by user.")
            if key in (8, 127):
                if draw_ops:
                    draw_ops.pop()
                continue
            if key in (ord("["), ord(",")):
                phase2_steps = max(0.0, round(phase2_steps - PHASE2_STEP_SIZE, 6))
                continue
            if key in (ord("]"), ord(".")):
                phase2_steps = min(max_phase2_steps, round(phase2_steps + PHASE2_STEP_SIZE, 6))
                continue
    finally:
        cv2.destroyWindow(window_name)

    prompt_image_bgr = render_utils.render_draw_ops(prompt_canvas, draw_ops)
    prompt_overlay_bgr = render_utils.render_draw_ops(np.zeros_like(prompt_canvas), draw_ops)
    if prompt_image_bgr.shape[:2] != original_base_shape:
        prompt_image_bgr = cv2.resize(
            prompt_image_bgr,
            (original_base_shape[1], original_base_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        prompt_overlay_bgr = cv2.resize(
            prompt_overlay_bgr,
            (original_base_shape[1], original_base_shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    return InteractivePromptResult(
        prompt_image_bgr=prompt_image_bgr,
        prompt_local_motion=np.asarray(wrist_values, dtype=np.float32),
        prompt_global_motion=np.asarray(primitive_cmd_values, dtype=np.float32),
        prompt_2d_drag=_drag_vector_from_draw_ops(draw_ops, base_image.shape[:2]),
        phase2_steps=float(phase2_steps),
        prompt_effect_mode=prompt_effect_mode,
        prompt_phase1_source=prompt_phase1_source,
        draw_ops=draw_ops,
        prompt_overlay_bgr=prompt_overlay_bgr,
    )
