from __future__ import annotations

import dataclasses

import cv2
import numpy as np
from openpi_client import image_tools


PHASE2_PROMPT_MEM_STEP_FRACTION = 0.05
PROMPT_RANDOM_NOISE_RATIO_STEP = 0.2


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


def denoise_step_from_sample_kwargs(
    sample_kwargs: dict | None,
    *,
    policy_inference_steps: float | None = None,
    phase2_policy_type: str | None = None,
) -> float | None:
    del policy_inference_steps, phase2_policy_type
    if not isinstance(sample_kwargs, dict):
        return None
    for key in ("phase2_steps", "steerer_phase2_steps"):
        if key not in sample_kwargs:
            continue
        phase2_steps = float(sample_kwargs[key])
        return phase2_steps
    return None


def _sample_num_steps(sample_kwargs: dict, *, policy_inference_steps: float | None = None) -> float:
    fallback = 10.0 if policy_inference_steps is None else float(policy_inference_steps)
    try:
        num_steps = float(sample_kwargs.get("num_inference_steps", sample_kwargs.get("num_steps", fallback)))
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(num_steps) or num_steps <= 0:
        return fallback
    return num_steps


def format_prompt_for_log(prompt_inputs: dict) -> str:
    parts = []

    global_motion = prompt_inputs.get("prompt_global_motion", prompt_inputs.get("prompt_primitive_cmd"))
    if global_motion is not None:
        global_motion = _as_prompt_global_motion(global_motion)
        global_text = _format_prompt_global_motion(global_motion)
        global_mask = prompt_inputs.get("prompt_global_motion_mask", prompt_inputs.get("prompt_primitive_cmd_mask"))
        if global_mask is None:
            raise ValueError("prompt_global_motion_mask must be provided with prompt_global_motion.")
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

    phase1_source = prompt_inputs.get("prompt_phase1_source")
    if phase1_source is not None:
        parts.append(f"source {phase1_source}")

    local_motion = prompt_inputs.get("prompt_local_motion", prompt_inputs.get("prompt_wrist_relative_action"))
    if local_motion is None:
        return " | ".join(parts) if parts else "<none>"

    local_motion = np.asarray(local_motion, dtype=np.float32)
    if local_motion.shape != (3,):
        raise ValueError(f"Expected prompt_local_motion shape (3,), got {local_motion.shape}.")
    local_text = " ".join(f"{axis}={value:+.2f}" for axis, value in zip("xyz", local_motion, strict=True))
    parts.append(f"local {local_text}")
    return " | ".join(parts)


def format_sample_kwargs_for_log(sample_kwargs: dict) -> str:
    if not sample_kwargs:
        return "<default>"
    parts = [f"{key}={value}" for key, value in sorted(sample_kwargs.items())]
    return " ".join(parts)


def render_prompt_window(prompt_inputs: dict, env_name: str, episode_idx: int, infer_idx: int) -> None:
    prompt_image = None
    prompt_images = prompt_inputs.get("prompt_images")
    prompt_image_masks = prompt_inputs.get("prompt_image_masks")
    prompt_image_active = True
    if isinstance(prompt_image_masks, dict):
        prompt_image_active = bool(np.any(np.asarray(prompt_image_masks.get("prompt_0", False), dtype=np.bool_)))
    if isinstance(prompt_images, dict) and prompt_image_active:
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
    image_overlay_update_count: int = 0
    global_action_update_count: int = 0
    local_action_update_count: int = 0

    def update(self, payload: dict) -> None:
        self.prompt_update_count += 1
        self._update_prompt_type_counts(payload)
        self.prompt_mem = 0
        self.payload = dict(payload)

    def clear(self) -> None:
        self.payload = None
        self.prompt_mem = 0
        self.prompt_update_count = 0
        self.image_overlay_update_count = 0
        self.global_action_update_count = 0
        self.local_action_update_count = 0

    def build_model_inputs(self) -> dict:
        if self.payload is None:
            return {}
        excluded_keys = {
            "interactive_prompt_result",
            "prompt_effect_mode",
            "prompt_phase1_source",
            "sample_kwargs",
        }
        return {key: value for key, value in self.payload.items() if key not in excluded_keys}

    def build_display_inputs(self) -> dict:
        if self.payload is None:
            return {}
        return {key: value for key, value in self.payload.items() if key not in {"interactive_prompt_result"}}

    def add_prompt_mem(self) -> None:
        if self.payload is not None:
            self.prompt_mem += 1

    def build_sample_kwargs(
        self,
        *,
        policy_inference_steps: float | None = None,
        emit_num_steps: bool = False,
        phase2_policy_type: str | None = None,
        random_noise_ratio_step: float = PROMPT_RANDOM_NOISE_RATIO_STEP,
    ) -> dict:
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
                if random_noise_ratio_step > 0.0:
                    sample_kwargs["random_noise_ratio"] = 1.0
                sample_kwargs["phase2_steps"] = num_steps
        elif self._has_long_term_prompt():
            base_phase2_steps = float(sample_kwargs.get("phase2_steps", 0.0))
            sample_kwargs["phase2_steps"] = min(
                num_steps,
                base_phase2_steps + self.prompt_mem * phase2_prompt_mem_step(num_steps),
            )
            if random_noise_ratio_step > 0.0:
                sample_kwargs["random_noise_ratio"] = min(1.0, self.prompt_mem * float(random_noise_ratio_step))
        return sample_kwargs

    def prompt_count_summary(self) -> dict[str, int]:
        return {
            "total": int(self.prompt_update_count),
            "img_overlay": int(self.image_overlay_update_count),
            "global_action": int(self.global_action_update_count),
            "local_action": int(self.local_action_update_count),
        }

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

    def _update_prompt_type_counts(self, payload: dict) -> None:
        prompt_image_masks = payload.get("prompt_image_masks")
        image_overlay_active = False
        if isinstance(prompt_image_masks, dict):
            image_overlay_active = bool(np.any(np.asarray(prompt_image_masks.get("prompt_0", False), dtype=np.bool_)))
        if image_overlay_active:
            self.image_overlay_update_count += 1

        global_active = bool(
            np.any(
                np.asarray(
                    payload.get("prompt_global_motion_mask", payload.get("prompt_primitive_cmd_mask", False)),
                    dtype=np.bool_,
                )
            )
        )
        if global_active:
            self.global_action_update_count += 1

        local_active = bool(
            np.any(
                np.asarray(
                    payload.get("prompt_local_motion_mask", payload.get("prompt_wrist_relative_action_mask", False)),
                    dtype=np.bool_,
                )
            )
        )
        if local_active:
            self.local_action_update_count += 1

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


def overlay_prompt_info_frame(
    frame_rgb: np.ndarray,
    prompt_payload: dict | None,
    *,
    denoise_step: float | None = None,
) -> np.ndarray:
    frame_rgb = _as_uint8_rgb(frame_rgb)
    if prompt_payload is None and denoise_step is None:
        return frame_rgb
    return _draw_prompt_action_text(frame_rgb, prompt_payload, denoise_step=denoise_step)


def compose_prompt_overlay_frame(
    frame_rgb: np.ndarray,
    prompt_payload: dict | None,
    *,
    denoise_step: float | None = None,
) -> np.ndarray:
    annotated_frame = overlay_prompt_info_frame(frame_rgb, prompt_payload, denoise_step=denoise_step)
    prompt_panel = _make_prompt_image_panel(prompt_payload, annotated_frame.shape[0])
    gap = np.full((annotated_frame.shape[0], 8, 3), 235, dtype=np.uint8)
    return np.concatenate([annotated_frame, gap, prompt_panel], axis=1)


def _as_prompt_global_motion(value) -> np.ndarray:
    global_motion = np.asarray(value, dtype=np.float32)
    if global_motion.shape != (3,):
        raise ValueError(f"Expected prompt_global_motion shape (3,), got {global_motion.shape}.")
    if not np.all(np.isfinite(global_motion)) or np.any(global_motion < -1.0) or np.any(global_motion > 1.0):
        raise ValueError(
            "prompt_global_motion must contain finite xyz values in [-1, 1], "
            f"got {global_motion.tolist()}."
        )
    return global_motion


def _format_prompt_global_motion(global_motion: np.ndarray) -> str:
    values = np.asarray(global_motion, dtype=np.float32).reshape(3)
    return f"[{values[0]:+.2f}, {values[1]:+.2f}, {values[2]:+.2f}]"


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


def _draw_prompt_action_text(
    frame_rgb: np.ndarray,
    prompt_payload: dict | None,
    *,
    denoise_step: float | None = None,
) -> np.ndarray:
    frame = frame_rgb.copy()
    if prompt_payload is None:
        lines = ["Wrist action: <none>", "Global action: <none>"]
    else:
        wrist_text = _format_prompt_vector(prompt_payload, "prompt_local_motion", "prompt_local_motion_mask", "xyz")
        global_text = _format_prompt_vector(
            prompt_payload,
            "prompt_global_motion",
            "prompt_global_motion_mask",
            "xyz",
        )
        lines = [f"Wrist action: {wrist_text}", f"Global action: {global_text}"]
    if denoise_step is not None:
        lines.append(f"Denoise steps: {float(denoise_step):.1f}")

    frame_h, frame_w = frame.shape[:2]
    font_scale = max(0.24, min(0.48, frame_h / 480.0))
    thickness = 1
    text_gap = max(6, int(round(frame_h * 0.018)))
    pad_x = max(6, int(round(frame_w * 0.03)))
    pad_y = max(6, int(round(frame_h * 0.03)))
    max_panel_width = max(40, int(round(frame_w * 0.94)))
    text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for line in lines]
    max_text_width = max(size[0] for size in text_sizes)
    available_text_width = max(10, max_panel_width - pad_x * 2)
    if max_text_width > available_text_width:
        shrink_ratio = available_text_width / max_text_width
        font_scale = max(0.20, font_scale * shrink_ratio)
        text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for line in lines]
    line_height = max(size[1] for size in text_sizes)
    panel_height = pad_y * 2 + line_height * len(lines) + text_gap * max(0, len(lines) - 1)
    max_text_width = max(size[0] for size in text_sizes)
    panel_width = min(max_panel_width, max_text_width + pad_x * 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, dst=frame)
    for line_idx, line in enumerate(lines):
        baseline_y = pad_y + line_height + line_idx * (line_height + text_gap)
        cv2.putText(
            frame,
            line,
            (pad_x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame


def _make_prompt_image_panel(prompt_payload: dict | None, height: int) -> np.ndarray:
    panel_width = height
    panel = np.full((height, panel_width, 3), 245, dtype=np.uint8)
    header_height = 34
    cv2.putText(
        panel,
        "Evo1 prompt image",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )

    prompt_image = None
    if prompt_payload is not None:
        prompt_images = prompt_payload.get("prompt_images")
        prompt_image_masks = prompt_payload.get("prompt_image_masks")
        prompt_image_active = True
        if isinstance(prompt_image_masks, dict):
            prompt_image_active = bool(np.any(np.asarray(prompt_image_masks.get("prompt_0", False), dtype=np.bool_)))
        if isinstance(prompt_images, dict) and prompt_image_active:
            prompt_image = prompt_images.get("prompt_0")

    if prompt_image is None:
        cv2.putText(
            panel,
            "No prompt image",
            (18, header_height + max(34, (height - header_height) // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
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
    if prompt_payload is not None and _prompt_vector_is_active(prompt_payload, "prompt_2d_drag", "prompt_2d_drag_mask"):
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
