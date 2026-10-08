import dataclasses
from datetime import datetime
import logging
import pathlib
import sys
from typing import Any

import cv2
import numpy as np
import tyro

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_SRC_ROOT = OPENPI_ROOT / "src"
OPENPI_CLIENT_ROOT = OPENPI_ROOT / "packages" / "openpi-client" / "src"
HARDWARE_ROOT = REPO_ROOT / "hardware"
preferred_paths = [str(REPO_ROOT), str(HARDWARE_ROOT), str(OPENPI_ROOT), str(OPENPI_SRC_ROOT), str(OPENPI_CLIENT_ROOT)]
sys.path[:] = [path for path in sys.path if path not in preferred_paths]
sys.path[:0] = preferred_paths

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy

from examples.flexiv_real import flexiv_env as _flexiv_env
from examples.flexiv_real import recorder as _recorder
from examples.robocasa.main_utils import render_rollout_window
from scripts.realworld.eval.eval_ui.interactive_labeling import collect_prompt_payload_from_images
from scripts.realworld.eval.hardcode import hardcode as _hardcode
from scripts.utils.flexiv_action_projection import project_flexiv_action_chunk
from scripts.utils.interactive_prompt import InteractivePromptState
from scripts.utils.interactive_prompt import PROMPT_RANDOM_NOISE_RATIO_STEP
from scripts.utils.interactive_prompt import compose_prompt_overlay_frame
from scripts.utils.interactive_prompt import denoise_step_from_sample_kwargs
from scripts.utils.interactive_prompt import policy_inference_steps_from_metadata
from scripts.utils.interactive_prompt import render_prompt_window
from scripts.utils.realworld_projection import draw_cached_projected_horizon
from scripts.utils.web_steer_client import WebSteerPromptClient


@dataclasses.dataclass(frozen=True)
class EpisodeEnd:
    outcome: str
    reason: str
    save: bool = True


def _format_motion_vector(vector: np.ndarray) -> str:
    vector = np.asarray(vector, dtype=np.float32)[:3]
    return "[" + ",".join(f"{float(value):.3f}" for value in vector) + "]"


class ObservationTargetDeltaPolicy:
    """Converts observation-to-target delta actions into absolute target poses."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def infer(self, obs: dict, *, sample_kwargs: dict[str, Any] | None = None) -> dict:
        result = self._policy.infer(obs, sample_kwargs=sample_kwargs)
        observation_state = obs.get("observation/state")
        if observation_state is None:
            raise KeyError("Missing observation/state; cannot convert observation-target delta actions.")
        return _flexiv_env.add_observation_target_poses(result, observation_state)

    def reset(self) -> None:
        self._policy.reset()


def _format_motion_axis_values(vector: np.ndarray) -> str:
    vector = np.asarray(vector, dtype=np.float32)[:3]
    return ", ".join(
        f"{axis}:{float(value):.3f}"
        for axis, value in zip(("x", "y", "z"), vector, strict=True)
        if np.isfinite(value) and not np.isclose(value, 0.0, atol=1e-6)
    )


def _normalize_action_value(raw_value: float, action_dim: int, metadata: dict) -> float:
    action_stats = metadata.get("action_norm_stats")
    if not isinstance(action_stats, dict):
        return 0.0

    use_quantile_norm = bool(metadata.get("use_quantile_norm", False))
    if use_quantile_norm:
        q01 = np.asarray(action_stats.get("q01"), dtype=np.float32)
        q99 = np.asarray(action_stats.get("q99"), dtype=np.float32)
        if q01.ndim != 1 or q99.ndim != 1 or action_dim >= len(q01) or action_dim >= len(q99):
            return 0.0
        return float((raw_value - q01[action_dim]) / (q99[action_dim] - q01[action_dim] + 1e-6) * 2.0 - 1.0)

    mean = np.asarray(action_stats.get("mean"), dtype=np.float32)
    std = np.asarray(action_stats.get("std"), dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1 or action_dim >= len(mean) or action_dim >= len(std):
        return 0.0
    return float((raw_value - mean[action_dim]) / (std[action_dim] + 1e-6))


def _as_uint8_rgb_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image.shape}.")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image.shape}.")
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
    return np.ascontiguousarray(image)


def _build_steer_sample_kwargs(
    prompt_state: InteractivePromptState,
    *,
    observation_state: np.ndarray,
    metadata: dict,
    action_horizon: int,
    enable_hardcode_phase1: bool = False,
    action_dim: int = 32,
) -> dict:
    sample_kwargs = prompt_state.build_sample_kwargs(
        policy_inference_steps=policy_inference_steps_from_metadata(metadata),
        emit_num_steps=_phase2_policy_type(metadata) == "openpi",
        phase2_policy_type=_phase2_policy_type(metadata),
        random_noise_ratio_step=_random_noise_ratio_step(metadata),
    )
    sample_kwargs = _hardcode.with_hardcode_phase1_sample_kwargs(
        sample_kwargs,
        prompt_payload=prompt_state.payload,
        observation_state=observation_state,
        metadata=metadata,
        action_horizon=action_horizon,
        enabled=enable_hardcode_phase1,
        action_dim=action_dim,
    )
    return _set_action_horizon(sample_kwargs, metadata=metadata, action_horizon=action_horizon)


def _normalize_steer_mode(steer: str | None) -> str | None:
    if steer is None:
        return None
    mode = str(steer).strip().lower()
    if mode in {"", "none", "false", "0", "off"}:
        return None
    if mode in {"true", "1", "yes", "on"}:
        return "hardcode"
    if mode not in {"hardcode", "evo"}:
        raise ValueError(f"Unsupported --steer value '{steer}'. Expected 'hardcode', 'evo', or omit --steer.")
    return mode


def _use_hardcode_phase1(*, steer_mode: str | None, prompt_state: InteractivePromptState) -> bool:
    return _hardcode.should_use_hardcode_phase1(steer_mode=steer_mode, prompt_payload=prompt_state.payload)


def _use_evo_2d_phase1_with_hardcode(*, metadata: dict, prompt_state: InteractivePromptState) -> bool:
    return _hardcode.has_2d_prompt(prompt_state.payload) and _server_supports_steerer(metadata, "evo")


def _set_steerer_enabled(sample_kwargs: dict, *, mode: str | None, enable: bool) -> dict:
    sample_kwargs = dict(sample_kwargs or {})
    sample_kwargs["enable_steerer"] = bool(enable)
    if mode == "evo":
        sample_kwargs["enable_evo1_steerer"] = bool(enable)
    return sample_kwargs


def _set_action_horizon(sample_kwargs: dict, *, metadata: dict, action_horizon: int) -> dict:
    sample_kwargs = dict(sample_kwargs or {})
    if _phase2_policy_type(metadata) in {"fastwam", "diffusion_policy"}:
        sample_kwargs["action_horizon"] = int(action_horizon)
    return sample_kwargs


def _server_supports_steerer(metadata: dict, mode: str | None) -> bool:
    if mode == "hardcode":
        return False
    steerer_metadata = metadata.get("steerer")
    if isinstance(steerer_metadata, dict) and bool(steerer_metadata.get("enabled", False)):
        return mode is None or steerer_metadata.get("mode") in {None, mode}
    if mode == "evo":
        steerer_metadata = metadata.get("evo1_steerer")
        return isinstance(steerer_metadata, dict) and bool(steerer_metadata.get("enabled", False))
    return False


def _server_has_any_steerer(metadata: dict) -> bool:
    return any(_server_supports_steerer(metadata, mode) for mode in (None, "evo"))


def _random_noise_ratio_step(metadata: dict) -> float:
    try:
        value = float(metadata.get("random_noise_ratio_step", PROMPT_RANDOM_NOISE_RATIO_STEP))
    except (TypeError, ValueError):
        return PROMPT_RANDOM_NOISE_RATIO_STEP
    if not np.isfinite(value) or value < 0.0:
        return PROMPT_RANDOM_NOISE_RATIO_STEP
    return value


def _phase2_policy_type(metadata: dict) -> str:
    return str(metadata.get("phase2_policy_type", "")).strip().lower()


def _build_steerer_text_prompt(task: str | None, prompt_state: InteractivePromptState) -> str | None:
    prompt_parts = []
    if task:
        prompt_parts.append(task.strip())

    payload = prompt_state.payload
    if payload is None:
        return ", ".join(part for part in prompt_parts if part) or None

    global_motion = payload.get("prompt_global_motion")
    global_motion_mask = bool(np.any(np.asarray(payload.get("prompt_global_motion_mask", False), dtype=np.bool_)))
    if global_motion is not None and global_motion_mask:
        motion_text = _format_motion_axis_values(np.asarray(global_motion, dtype=np.float32)[:3])
        if motion_text:
            prompt_parts.append(f"move {motion_text} in global frame")

    local_motion = payload.get("prompt_local_motion")
    local_motion_mask = bool(np.any(np.asarray(payload.get("prompt_local_motion_mask", False), dtype=np.bool_)))
    if local_motion is not None and local_motion_mask:
        motion_text = _format_motion_axis_values(np.asarray(local_motion, dtype=np.float32)[:3])
        if motion_text:
            prompt_parts.append(f"move {motion_text} in wrist local frame")

    return ", ".join(part for part in prompt_parts if part) or None


def _action_chunk_for_overlay(broker, action: dict) -> np.ndarray | None:
    last_results = broker.last_results()
    if last_results is not None and "actions" in last_results:
        return np.asarray(last_results["actions"], dtype=np.float64)
    if "actions" in action:
        return np.asarray(action["actions"], dtype=np.float64)[None]
    return None


def _phase1_action_chunk_for_overlay(broker) -> np.ndarray | None:
    last_results = broker.last_results()
    if last_results is None:
        return None
    phase1_key = next(
        (
            key
            for key in (
                "phase1_actions_raw",
                "steerer_phase1_actions_raw",
                "evo1_phase1_actions_raw",
                "hit_phase1_actions_raw",
            )
            if key in last_results
        ),
        None,
    )
    if phase1_key is None:
        return None
    phase1_actions = np.asarray(last_results[phase1_key], dtype=np.float64)
    if phase1_actions.ndim != 2 or phase1_actions.shape[-1] < 3:
        return None
    return phase1_actions


def _phase1_actions_for_reuse(broker) -> np.ndarray | None:
    last_results = broker.last_results()
    if last_results is None:
        return None
    phase1_key = next(
        (
            key
            for key in (
                "phase1_actions_raw",
                "steerer_phase1_actions_raw",
                "evo1_phase1_actions_raw",
                "hit_phase1_actions_raw",
            )
            if key in last_results
        ),
        None,
    )
    if phase1_key is None:
        return None
    phase1_actions = np.asarray(last_results[phase1_key], dtype=np.float32)
    if phase1_actions.ndim != 2:
        return None
    return phase1_actions


def _format_action_debug(name: str, actions: np.ndarray | None) -> str:
    if actions is None:
        return f"{name}: <none>"
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        return f"{name}: shape={actions.shape}"
    xyz = actions[:, : min(3, actions.shape[-1])]
    xyz_norm = np.linalg.norm(xyz, axis=-1) if xyz.shape[-1] else np.zeros(actions.shape[0], dtype=np.float32)
    xyz_adj = np.diff(xyz, axis=0) if len(xyz) >= 2 else np.zeros((0, xyz.shape[-1]), dtype=np.float32)
    xyz_second = np.diff(xyz, n=2, axis=0) if len(xyz) >= 3 else np.zeros((0, xyz.shape[-1]), dtype=np.float32)
    first_xyz = np.array2string(xyz[:5], precision=4, suppress_small=False)
    parts = [
        f"{name}: shape={actions.shape}",
        f"xyz_min={np.min(xyz, axis=0).tolist() if xyz.size else []}",
        f"xyz_max={np.max(xyz, axis=0).tolist() if xyz.size else []}",
        f"xyz_norm_mean={float(np.mean(xyz_norm)):.6f}",
        f"xyz_norm_max={float(np.max(xyz_norm)):.6f}",
        f"xyz_adj_abs_mean={float(np.mean(np.abs(xyz_adj))):.6f}" if xyz_adj.size else "xyz_adj_abs_mean=0.000000",
        f"xyz_second_abs_mean={float(np.mean(np.abs(xyz_second))):.6f}" if xyz_second.size else "xyz_second_abs_mean=0.000000",
        f"first5_xyz={first_xyz}",
    ]
    if actions.shape[-1] > 6:
        parts.append(f"gripper_minmax=({float(np.min(actions[:, 6])):.6f}, {float(np.max(actions[:, 6])):.6f})")
    return " ".join(parts)


def _save_action_chunk_log(logs: list[dict], output_dir: pathlib.Path) -> None:
    if not logs:
        return

    max_horizon = max(int(entry["actions"].shape[0]) for entry in logs)
    max_action_dim = max(int(entry["actions"].shape[1]) for entry in logs)
    action_chunks = np.full((len(logs), max_horizon, max_action_dim), np.nan, dtype=np.float32)
    valid_lengths = np.zeros((len(logs),), dtype=np.int32)
    episode_indices = np.zeros((len(logs),), dtype=np.int32)
    chunk_indices = np.zeros((len(logs),), dtype=np.int32)
    start_steps = np.zeros((len(logs),), dtype=np.int32)
    executed_steps = np.zeros((len(logs),), dtype=np.int32)
    denoise_steps = np.full((len(logs),), np.nan, dtype=np.float32)
    client_roundtrip_ms = np.full((len(logs),), np.nan, dtype=np.float32)
    server_infer_ms = np.full((len(logs),), np.nan, dtype=np.float32)
    server_prev_total_ms = np.full((len(logs),), np.nan, dtype=np.float32)
    policy_infer_ms = np.full((len(logs),), np.nan, dtype=np.float32)
    openpi_rtc_enabled = np.zeros((len(logs),), dtype=np.bool_)
    openpi_rtc_time_base = np.full((len(logs),), -1, dtype=np.int32)
    openpi_rtc_inference_delay = np.full((len(logs),), -1, dtype=np.int32)
    openpi_rtc_prefix_attention_horizon = np.full((len(logs),), -1, dtype=np.int32)

    for idx, entry in enumerate(logs):
        actions = np.asarray(entry["actions"], dtype=np.float32)
        horizon, action_dim = actions.shape
        action_chunks[idx, :horizon, :action_dim] = actions
        valid_lengths[idx] = horizon
        episode_indices[idx] = int(entry["episode_idx"])
        chunk_indices[idx] = int(entry["chunk_idx"])
        start_steps[idx] = int(entry["start_step"])
        executed_steps[idx] = int(entry["executed_steps"])
        denoise_step = entry.get("denoise_step")
        if denoise_step is not None:
            denoise_steps[idx] = float(denoise_step)
        for key, target in (
            ("client_roundtrip_ms", client_roundtrip_ms),
            ("server_infer_ms", server_infer_ms),
            ("server_prev_total_ms", server_prev_total_ms),
            ("policy_infer_ms", policy_infer_ms),
        ):
            value = entry.get(key)
            if value is not None:
                target[idx] = float(value)
        rtc_info = entry.get("openpi_rtc")
        if isinstance(rtc_info, dict):
            openpi_rtc_enabled[idx] = bool(rtc_info.get("enabled", False))
            if rtc_info.get("time_base") is not None:
                openpi_rtc_time_base[idx] = int(rtc_info["time_base"])
            if rtc_info.get("inference_delay") is not None:
                openpi_rtc_inference_delay[idx] = int(rtc_info["inference_delay"])
            if rtc_info.get("prefix_attention_horizon") is not None:
                openpi_rtc_prefix_attention_horizon[idx] = int(rtc_info["prefix_attention_horizon"])

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "action_chunks.npz",
        action_chunks=action_chunks,
        valid_lengths=valid_lengths,
        episode_indices=episode_indices,
        chunk_indices=chunk_indices,
        start_steps=start_steps,
        executed_steps=executed_steps,
        denoise_steps=denoise_steps,
        client_roundtrip_ms=client_roundtrip_ms,
        server_infer_ms=server_infer_ms,
        server_prev_total_ms=server_prev_total_ms,
        policy_infer_ms=policy_infer_ms,
        openpi_rtc_enabled=openpi_rtc_enabled,
        openpi_rtc_time_base=openpi_rtc_time_base,
        openpi_rtc_inference_delay=openpi_rtc_inference_delay,
        openpi_rtc_prefix_attention_horizon=openpi_rtc_prefix_attention_horizon,
    )


def _append_action_chunk_log(
    logs: list[dict],
    *,
    output_dir: pathlib.Path,
    episode_idx: int,
    chunk_idx: int,
    start_step: int,
    executed_steps: int,
    action_chunk: np.ndarray | None,
    denoise_step: float | None,
    timing: dict | None = None,
    openpi_rtc: dict | None = None,
) -> None:
    if action_chunk is None:
        return
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2:
        logging.warning("Skipping action chunk log with unexpected shape %s", actions.shape)
        return
    logs.append(
        {
            "episode_idx": int(episode_idx),
            "chunk_idx": int(chunk_idx),
            "start_step": int(start_step),
            "executed_steps": int(max(1, min(executed_steps, actions.shape[0]))),
            "denoise_step": denoise_step,
            "actions": actions.copy(),
            **(timing or {}),
            "openpi_rtc": openpi_rtc,
        }
    )
    _save_action_chunk_log(logs, output_dir)


def _timing_value_ms(results: dict | None, section: str, key: str) -> float | None:
    if not isinstance(results, dict):
        return None
    timing = results.get(section)
    if not isinstance(timing, dict) or key not in timing:
        return None
    try:
        value = float(timing[key])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _inference_timing_for_log(results: dict | None) -> dict[str, float]:
    timing = {}
    for output_key, section, timing_key in (
        ("client_roundtrip_ms", "client_timing", "roundtrip_ms"),
        ("server_infer_ms", "server_timing", "infer_ms"),
        ("server_prev_total_ms", "server_timing", "prev_total_ms"),
        ("policy_infer_ms", "policy_timing", "infer_ms"),
    ):
        value = _timing_value_ms(results, section, timing_key)
        if value is not None:
            timing[output_key] = value
    return timing


def _log_inference_timing(
    *,
    episode_idx: int,
    chunk_idx: int,
    step_idx: int,
    timing: dict[str, float],
    openpi_rtc: dict | None,
) -> None:
    rtc_text = ""
    if isinstance(openpi_rtc, dict):
        rtc_text = (
            f" rtc_enabled={bool(openpi_rtc.get('enabled', False))}"
            f" time_base={openpi_rtc.get('time_base')}"
            f" delay={openpi_rtc.get('inference_delay')}"
        )
    logging.info(
        "Inference timing | episode=%d chunk=%d step=%d client_roundtrip_ms=%.2f "
        "server_infer_ms=%.2f policy_infer_ms=%.2f%s",
        episode_idx,
        chunk_idx,
        step_idx,
        float(timing.get("client_roundtrip_ms", np.nan)),
        float(timing.get("server_infer_ms", np.nan)),
        float(timing.get("policy_infer_ms", np.nan)),
        rtc_text,
    )


def _log_steerer_action_debug(broker, *, mode: str | None, episode_idx: int, step_idx: int) -> None:
    last_results = broker.last_results()
    if last_results is None:
        logging.info("Steerer action debug | mode=%s episode=%d step=%d no server result", mode, episode_idx, step_idx)
        return
    raw_actions = next(
        (
            np.asarray(last_results[key], dtype=np.float32)
            for key in ("phase1_actions_raw", "steerer_phase1_actions_raw", "hit_phase1_actions_raw", "evo1_phase1_actions_raw")
            if key in last_results
        ),
        None,
    )
    model_actions = next(
        (
            np.asarray(last_results[key], dtype=np.float32)
            for key in ("phase1_actions_model", "steerer_phase1_actions", "hit_phase1_actions", "evo1_phase1_actions")
            if key in last_results
        ),
        None,
    )
    logging.info(
        "Steerer action debug | mode=%s episode=%d step=%d | %s | %s",
        mode,
        episode_idx,
        step_idx,
        _format_action_debug("raw_denormalized", raw_actions),
        _format_action_debug("model", model_actions),
    )


def _set_reused_phase1_actions(sample_kwargs: dict, phase1_actions: np.ndarray | None) -> dict:
    sample_kwargs = dict(sample_kwargs or {})
    if phase1_actions is not None:
        sample_kwargs["phase1_actions"] = np.asarray(phase1_actions, dtype=np.float32)
    return sample_kwargs


def _set_short_term_direct_denoise(sample_kwargs: dict, *, max_phase2_steps: float) -> dict:
    sample_kwargs = dict(sample_kwargs or {})
    sample_kwargs.pop("phase1_actions", None)
    sample_kwargs["phase2_steps"] = float(sample_kwargs.get("num_steps", max_phase2_steps))
    return sample_kwargs


def _current_denoise_step_from_sample_kwargs(
    sample_kwargs: dict | None,
    *,
    policy_inference_steps: float,
    phase2_policy_type: str | None,
) -> float:
    denoise_step = denoise_step_from_sample_kwargs(
        sample_kwargs,
        policy_inference_steps=policy_inference_steps,
        phase2_policy_type=phase2_policy_type,
    )
    return float(policy_inference_steps if denoise_step is None else denoise_step)


def _project_overlay_chunk(
    *,
    image_shape: tuple[int, int, int],
    observation: dict,
    action_chunk: np.ndarray,
    color: tuple[int, int, int],
    source_image_shape: tuple[int, int, int] | None = None,
) -> dict:
    return project_flexiv_action_chunk(
        image_shape=image_shape,
        observation=observation,
        action_chunk=action_chunk,
        color=color,
        source_image_shape=source_image_shape,
    )


def _build_overlay_cache(
    *,
    image_shape: tuple[int, int, int],
    observation: dict,
    policy_action_chunk: np.ndarray | None,
    phase1_action_chunk: np.ndarray | None,
    source_image_shape: tuple[int, int, int] | None = None,
) -> list[dict]:
    overlays = []
    if policy_action_chunk is not None:
        overlays.append(
            _project_overlay_chunk(
                image_shape=image_shape,
                observation=observation,
                action_chunk=policy_action_chunk,
                color=(0, 180, 255),
                source_image_shape=source_image_shape,
            )
        )
    if phase1_action_chunk is not None:
        overlays.append(
            _project_overlay_chunk(
                image_shape=image_shape,
                observation=observation,
                action_chunk=phase1_action_chunk,
                color=(255, 80, 0),
                source_image_shape=source_image_shape,
            )
        )
    return overlays


def _draw_overlay_cache(image_rgb: np.ndarray, overlay_cache: list[dict]) -> np.ndarray:
    image_rgb = _as_uint8_rgb_image(image_rgb)
    for overlay in overlay_cache:
        image_rgb = draw_cached_projected_horizon(
            image_rgb,
            overlay["points_hw"],
            overlay["valid_mask"],
            color=overlay["color"],
        )
    return image_rgb


def _build_prompt_overlay_frame(
    image_rgb: np.ndarray,
    overlay_cache: list[dict],
    prompt_payload: dict | None,
    *,
    denoise_step: float | None = None,
) -> np.ndarray:
    return compose_prompt_overlay_frame(
        _draw_overlay_cache(image_rgb, overlay_cache),
        prompt_payload,
        denoise_step=denoise_step,
    )


def _render_env_camera_window(
    *,
    env: _flexiv_env.FlexivRealEnv,
    overlay_cache: list[dict],
    args: "Args",
    episode_idx: int,
    step_idx: int,
    enable_prompt_hotkey: bool,
    denoise_step: float | None = None,
) -> int:
    if not args.on_screen:
        return -1

    image_rgb = _draw_overlay_cache(env.get_latest_env_image(raw=True), overlay_cache)

    render_rollout_window(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        "flexiv_real",
        episode_idx,
        step_idx,
        enable_prompt_hotkey=enable_prompt_hotkey,
        denoise_step=denoise_step,
    )
    return cv2.waitKey(1) & 0xFF


def _episode_end_from_key(key: int) -> EpisodeEnd | None:
    if key in (ord("f"), ord("F")):
        return EpisodeEnd(outcome="success", reason="success_key")
    if key in (ord("d"), ord("D")):
        return EpisodeEnd(outcome="failure", reason="failure_key")
    if key in (ord("q"), ord("Q"), 27):
        return EpisodeEnd(outcome="aborted", reason="abort_key", save=False)
    return None


def _drain_rollout_window_keys() -> None:
    try:
        for _ in range(50):
            cv2.waitKey(1)
    except cv2.error:
        logging.debug("OpenCV GUI backend unavailable; skipping key drain.")


def _is_pat_subtract_key(key: int) -> bool:
    return key in (ord("w"), ord("W"))


def _read_task_progress(episode_idx: int) -> str | None:
    while True:
        try:
            raw_value = input(f"Episode {episode_idx} task progress (%): ").strip()
        except EOFError:
            logging.warning("No stdin available for task progress; recording null task_progress.")
            return None

        value_text = raw_value[:-1].strip() if raw_value.endswith("%") else raw_value
        try:
            value = float(value_text)
        except ValueError:
            print("Please enter a number, for example 33.", flush=True)
            continue

        if value.is_integer():
            return f"{int(value)}%"
        return f"{value:g}%"


def _print_episode_prompt_count(*, episode_idx: int, prompt_counts: dict[str, int]) -> None:
    message = (
        f"Episode {episode_idx} interactive UI prompt count: "
        f"total={prompt_counts['total']} "
        f"img_overlay={prompt_counts['img_overlay']} "
        f"global_action={prompt_counts['global_action']} "
        f"local_action={prompt_counts['local_action']}"
    )
    logging.info(message)
    print(message, flush=True)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 15
    fps: float = 5.0

    num_episodes: int = 1
    max_episode_steps: int = 5000

    render_height: int = 224
    render_width: int = 224
    task: str | None = None
    output_dir: str = "output"
    steer: str | None = None
    prompt_ui: str = "local"
    web_steer_url: str = "http://127.0.0.1:8765"
    web_steer_publish_hz: float = 5.0
    on_screen: bool = False
    enable_rtc: bool = False
    execute_observation_target_delta_actions: bool = False
    rtc_default_delay_steps: int = 4
    rtc_latency_percentile: float = 0.95
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "exp"


def _allocate_run_dir(output_root: pathlib.Path) -> pathlib.Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / timestamp
    if not candidate.exists():
        return candidate
    for suffix in range(1, 1000):
        candidate = output_root / f"{timestamp}_{suffix:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Failed to allocate a unique run directory under {output_root}.")


def _rewrite_legacy_steer_flag(argv: list[str]) -> list[str]:
    rewritten = list(argv)
    steer_flags = {"--args.steer", "--steer"}
    rtc_aliases = {
        "--rtc_default_delay_steps": "--args.rtc-default-delay-steps",
        "--rtc-default-delay-steps": "--args.rtc-default-delay-steps",
        "--rtc_latency_percentile": "--args.rtc-latency-percentile",
        "--rtc-latency-percentile": "--args.rtc-latency-percentile",
        "--rtc_max_guidance_weight": "--args.rtc-max-guidance-weight",
        "--rtc-max-guidance-weight": "--args.rtc-max-guidance-weight",
        "--rtc_prefix_attention_schedule": "--args.rtc-prefix-attention-schedule",
        "--rtc-prefix-attention-schedule": "--args.rtc-prefix-attention-schedule",
    }
    for idx, value in enumerate(rewritten):
        if value in {"--enable_rtc", "--enable-rtc"}:
            rewritten[idx] = "--args.enable-rtc"
            continue
        if value.startswith("--enable_rtc="):
            rewritten[idx] = "--args.enable-rtc=" + value.split("=", 1)[1]
            continue
        if value.startswith("--enable-rtc="):
            rewritten[idx] = "--args.enable-rtc=" + value.split("=", 1)[1]
            continue
        if value in {"--disable_rtc", "--disable-rtc", "--no-enable_rtc", "--no-enable-rtc"}:
            rewritten[idx] = "--args.no-enable-rtc"
            continue
        if value in rtc_aliases:
            rewritten[idx] = rtc_aliases[value]
            continue
        alias_name, has_equals, alias_value = value.partition("=")
        if has_equals and alias_name in rtc_aliases:
            rewritten[idx] = f"{rtc_aliases[alias_name]}={alias_value}"
            continue
        if value not in steer_flags:
            continue
        next_value = rewritten[idx + 1] if idx + 1 < len(rewritten) else None
        if next_value is None or next_value.startswith("-"):
            rewritten[idx] = f"{value}=hardcode"
    return rewritten


def main(args: Args) -> None:
    steer_mode = _normalize_steer_mode(args.steer)
    prompt_ui = str(args.prompt_ui).strip().lower()
    if prompt_ui not in {"local", "web"}:
        raise ValueError(f"Unsupported --prompt-ui {args.prompt_ui!r}; expected 'local' or 'web'.")
    web_prompt_client = (
        WebSteerPromptClient(args.web_steer_url, publish_hz=args.web_steer_publish_hz)
        if steer_mode and prompt_ui == "web"
        else None
    )
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    server_has_steerer = _server_has_any_steerer(metadata)
    policy_inference_steps = policy_inference_steps_from_metadata(metadata)
    supports_steerer = bool(steer_mode and _server_supports_steerer(metadata, steer_mode))
    if steer_mode == "evo" and server_has_steerer and not supports_steerer:
        server_mode = (metadata.get("steerer") or {}).get("mode")
        raise ValueError(f"Requested --steer {steer_mode}, but policy server steerer mode is {server_mode!r}.")

    task_prompt = (
        args.task
        or metadata.get("default_prompt")
        or metadata.get("prompt")
        or metadata.get("task_description")
    )
    run_dir = _allocate_run_dir(REPO_ROOT / args.output_dir)
    logging.info("Recording rollouts to %s", run_dir)
    action_chunk_logs: list[dict] = []

    recorder = _recorder.LeRobotRolloutRecorder(
        output_dir=run_dir,
        fps=args.fps,
        server_metadata=metadata,
        default_task=task_prompt,
        render_height=args.render_height,
        render_width=args.render_width,
        steer=bool(steer_mode),
    )

    env = _flexiv_env.FlexivRealEnv(
        reset_pose=metadata.get("reset_pose"),
        render_height=args.render_height,
        render_width=args.render_width,
        fps=args.fps,
        prompt=task_prompt,
    )
    policy = (
        ObservationTargetDeltaPolicy(ws_client_policy)
        if args.execute_observation_target_delta_actions
        else ws_client_policy
    )
    if args.execute_observation_target_delta_actions:
        logging.info("Executing policy actions as observation-to-target deltas.")

    broker = action_chunk_broker.ActionChunkBroker(
        policy=policy,
        action_horizon=args.action_horizon,
        enable_rtc=args.enable_rtc,
        execution_fps=args.fps,
        rtc_default_delay_steps=args.rtc_default_delay_steps,
        rtc_latency_percentile=args.rtc_latency_percentile,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
    )
    prompt_state = InteractivePromptState()
    pending_web_prompt_sequence: int | None = None
    max_episode_steps = args.max_episode_steps if args.max_episode_steps > 0 else float("inf")
    robot_is_home = False

    try:
        for episode_idx in range(args.num_episodes):
            logging.info("Starting episode %d/%d", episode_idx + 1, args.num_episodes)
            if not robot_is_home:
                env.reset()
            _drain_rollout_window_keys()
            robot_is_home = False
            broker.reset()
            prompt_state.clear()
            pending_web_prompt_sequence = None
            recorder.on_episode_start()
            raw_policy_overlay_cache = []
            raw_phase1_overlay_cache = []
            video_policy_overlay_cache = []
            video_phase1_overlay_cache = []
            pending_phase1_update = False
            reused_phase1_actions = None
            prompt_effect_mode = "long_term"
            active_denoise_step = None
            episode_end: EpisodeEnd | None = None
            pat_subtract_count = 0
            action_chunk_idx = 0

            step_idx = 0
            while not env.is_episode_complete() and step_idx < max_episode_steps:
                observation = env.get_observation()
                base_image_rgb = _as_uint8_rgb_image(observation["observation/image"])
                wrist_image_rgb = _as_uint8_rgb_image(observation["observation/wrist_image"])
                raw_overlay_cache = raw_policy_overlay_cache + raw_phase1_overlay_cache
                video_overlay_cache = video_policy_overlay_cache + video_phase1_overlay_cache
                key = _render_env_camera_window(
                    env=env,
                    overlay_cache=raw_overlay_cache,
                    args=args,
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    enable_prompt_hotkey=bool(steer_mode and prompt_ui == "local"),
                    denoise_step=active_denoise_step,
                )
                if _is_pat_subtract_key(key):
                    pat_subtract_count += 1
                if web_prompt_client is not None:
                    web_prompt_client.publish_observation(
                        base_image_rgb=base_image_rgb,
                        wrist_image_rgb=wrist_image_rgb,
                        observation_state=observation["observation/state"],
                        episode=episode_idx,
                        step=step_idx,
                    )
                    web_update = web_prompt_client.poll_prompt()
                    if web_update is not None:
                        prompt_state.update(web_update.payload)
                        pending_web_prompt_sequence = web_update.sequence
                        prompt_effect_mode = str(web_update.payload.get("prompt_effect_mode", "long_term"))
                        logging.info("Received web prompt #%d for episode=%d step=%d.", web_update.sequence, episode_idx, step_idx)
                        broker.reset()
                        raw_policy_overlay_cache = []
                        video_policy_overlay_cache = []
                        raw_phase1_overlay_cache = []
                        video_phase1_overlay_cache = []
                        pending_phase1_update = True
                        reused_phase1_actions = None
                        active_denoise_step = None
                        continue
                if steer_mode and prompt_ui == "local" and key in (ord("p"), ord("P")):
                    prompt_payload = collect_prompt_payload_from_images(
                        base_image_rgb=base_image_rgb,
                        wrist_image_rgb=wrist_image_rgb,
                        observation_state=observation["observation/state"],
                        source_image_shape=env.get_latest_env_image(raw=True).shape,
                        max_phase2_steps=policy_inference_steps,
                    )
                    prompt_state.update(prompt_payload)
                    prompt_effect_mode = str(prompt_payload.get("prompt_effect_mode", "long_term"))
                    render_prompt_window(
                        {
                            **prompt_state.build_display_inputs(),
                            "sample_kwargs": _build_steer_sample_kwargs(
                                prompt_state,
                                observation_state=observation["observation/state"],
                                metadata=metadata,
                                action_horizon=args.action_horizon,
                                enable_hardcode_phase1=_use_hardcode_phase1(
                                    steer_mode=steer_mode,
                                    prompt_state=prompt_state,
                                ),
                            ),
                        },
                        "flexiv_real",
                        episode_idx,
                        step_idx,
                    )
                    broker.reset()
                    raw_policy_overlay_cache = []
                    video_policy_overlay_cache = []
                    raw_phase1_overlay_cache = []
                    video_phase1_overlay_cache = []
                    pending_phase1_update = True
                    reused_phase1_actions = None
                    active_denoise_step = None
                    continue
                episode_end = _episode_end_from_key(key)
                if episode_end is not None:
                    logging.info(
                        "Ending current episode from rollout window. outcome=%s reason=%s save=%s",
                        episode_end.outcome,
                        episode_end.reason,
                        episode_end.save,
                    )
                    break

                needs_new_inference = broker.needs_new_inference()
                use_hardcode_phase1 = _use_hardcode_phase1(steer_mode=steer_mode, prompt_state=prompt_state)
                run_phase1 = bool(
                    steer_mode
                    and supports_steerer
                    and pending_phase1_update
                    and not use_hardcode_phase1
                )
                run_hardcode_phase1 = bool(
                    steer_mode
                    and use_hardcode_phase1
                    and (prompt_effect_mode == "long_term" or pending_phase1_update)
                )
                run_hardcode_evo_2d_phase1 = bool(
                    run_hardcode_phase1
                    and _use_evo_2d_phase1_with_hardcode(metadata=metadata, prompt_state=prompt_state)
                )
                use_prompt_inputs = not (
                    prompt_effect_mode == "short_term"
                    and not run_phase1
                    and not run_hardcode_phase1
                )
                model_observation = {
                    **observation,
                    **(prompt_state.build_model_inputs() if use_prompt_inputs else {}),
                }
                steerer_prompt = (
                    _build_steerer_text_prompt(task_prompt, prompt_state)
                    if use_prompt_inputs
                    else task_prompt
                )
                if steerer_prompt is not None:
                    model_observation["prompt"] = steerer_prompt
                sample_kwargs = (
                    _build_steer_sample_kwargs(
                        prompt_state,
                        observation_state=observation["observation/state"],
                        metadata=metadata,
                        action_horizon=args.action_horizon,
                        enable_hardcode_phase1=run_hardcode_phase1,
                    )
                    if steer_mode
                    else prompt_state.build_sample_kwargs(
                        policy_inference_steps=policy_inference_steps,
                        emit_num_steps=_phase2_policy_type(metadata) == "openpi",
                        phase2_policy_type=_phase2_policy_type(metadata),
                        random_noise_ratio_step=_random_noise_ratio_step(metadata),
                    )
                )
                sample_kwargs = _set_action_horizon(
                    sample_kwargs,
                    metadata=metadata,
                    action_horizon=args.action_horizon,
                )
                if steer_mode and not run_phase1:
                    if (
                        prompt_effect_mode == "short_term"
                        and prompt_state.prompt_mem > 0
                        and not run_hardcode_phase1
                    ):
                        sample_kwargs = _set_short_term_direct_denoise(
                            sample_kwargs,
                            max_phase2_steps=policy_inference_steps,
                        )
                    elif not run_hardcode_phase1:
                        sample_kwargs = _set_reused_phase1_actions(sample_kwargs, reused_phase1_actions)
                if server_has_steerer:
                    sample_kwargs = _set_steerer_enabled(
                        sample_kwargs,
                        mode="evo" if run_hardcode_evo_2d_phase1 else steer_mode,
                        enable=run_phase1 or run_hardcode_evo_2d_phase1,
                    )
                if needs_new_inference:
                    active_denoise_step = _current_denoise_step_from_sample_kwargs(
                        sample_kwargs,
                        policy_inference_steps=policy_inference_steps,
                        phase2_policy_type=_phase2_policy_type(metadata),
                    )
                action = broker.infer(model_observation, sample_kwargs=sample_kwargs or None)
                if needs_new_inference and pending_web_prompt_sequence is not None and web_prompt_client is not None:
                    web_prompt_client.acknowledge_inference(pending_web_prompt_sequence)
                    logging.info("Web prompt #%d was sent to the policy server.", pending_web_prompt_sequence)
                    pending_web_prompt_sequence = None
                if needs_new_inference:
                    env_image_rgb = _as_uint8_rgb_image(env.get_latest_env_image(raw=True))
                    last_results = broker.last_results()
                    inference_timing = _inference_timing_for_log(last_results)
                    openpi_rtc_info = last_results.get("openpi_rtc") if isinstance(last_results, dict) else None
                    _log_inference_timing(
                        episode_idx=episode_idx,
                        chunk_idx=action_chunk_idx,
                        step_idx=step_idx,
                        timing=inference_timing,
                        openpi_rtc=openpi_rtc_info,
                    )
                    policy_action_chunk = _action_chunk_for_overlay(broker, action)
                    _append_action_chunk_log(
                        action_chunk_logs,
                        output_dir=run_dir,
                        episode_idx=episode_idx,
                        chunk_idx=action_chunk_idx,
                        start_step=step_idx,
                        executed_steps=args.action_horizon,
                        action_chunk=policy_action_chunk,
                        denoise_step=active_denoise_step,
                        timing=inference_timing,
                        openpi_rtc=openpi_rtc_info,
                    )
                    action_chunk_idx += 1
                    raw_policy_overlay_cache = _build_overlay_cache(
                        image_shape=env_image_rgb.shape,
                        observation=observation,
                        policy_action_chunk=policy_action_chunk,
                        phase1_action_chunk=None,
                    )
                    video_policy_overlay_cache = _build_overlay_cache(
                        image_shape=base_image_rgb.shape,
                        observation=observation,
                        policy_action_chunk=policy_action_chunk,
                        phase1_action_chunk=None,
                        source_image_shape=env_image_rgb.shape,
                    )
                    if run_phase1 or run_hardcode_phase1:
                        reused_phase1_actions = (
                            _phase1_actions_for_reuse(broker)
                            if prompt_effect_mode == "long_term"
                            else None
                        )
                        phase1_action_chunk = _phase1_action_chunk_for_overlay(broker)
                        # _log_steerer_action_debug(
                        #     broker,
                        #     mode=steer_mode,
                        #     episode_idx=episode_idx,
                        #     step_idx=step_idx,
                        # )
                        if phase1_action_chunk is not None:
                            raw_phase1_overlay_cache = _build_overlay_cache(
                                image_shape=env_image_rgb.shape,
                                observation=observation,
                                policy_action_chunk=None,
                                phase1_action_chunk=phase1_action_chunk,
                            )
                            video_phase1_overlay_cache = _build_overlay_cache(
                                image_shape=base_image_rgb.shape,
                                observation=observation,
                                policy_action_chunk=None,
                                phase1_action_chunk=phase1_action_chunk,
                                source_image_shape=env_image_rgb.shape,
                            )
                        else:
                            raw_phase1_overlay_cache = []
                            video_phase1_overlay_cache = []
                        pending_phase1_update = False
                    elif pending_phase1_update and not supports_steerer:
                        pending_phase1_update = False
                    elif steer_mode and reused_phase1_actions is None:
                        raw_phase1_overlay_cache = []
                        video_phase1_overlay_cache = []
                    raw_overlay_cache = raw_policy_overlay_cache + raw_phase1_overlay_cache
                    video_overlay_cache = video_policy_overlay_cache + video_phase1_overlay_cache
                    key = _render_env_camera_window(
                        env=env,
                        overlay_cache=raw_overlay_cache,
                        args=args,
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        enable_prompt_hotkey=bool(steer_mode),
                        denoise_step=active_denoise_step,
                    )
                    if _is_pat_subtract_key(key):
                        pat_subtract_count += 1
                    episode_end = _episode_end_from_key(key)
                    if episode_end is not None:
                        # logging.info(
                        #     "Ending current episode from rollout window. outcome=%s reason=%s save=%s",
                        #     episode_end.outcome,
                        #     episode_end.reason,
                        #     episode_end.save,
                        # )
                        break
                env.apply_action(action)
                recorder.on_step(
                    observation,
                    action,
                    prompt_overlay_payload=prompt_state.build_display_inputs() if steer_mode else None,
                    prompt_overlay_denoise_step=active_denoise_step,
                    denoise_step=active_denoise_step,
                    prompt_overlay_frame=_build_prompt_overlay_frame(
                        base_image_rgb,
                        video_overlay_cache,
                        prompt_state.build_display_inputs() if steer_mode else None,
                        denoise_step=active_denoise_step,
                    )
                    if steer_mode
                    else None,
                )
                if needs_new_inference:
                    prompt_state.add_prompt_mem()
                step_idx += 1

            if episode_end is None:
                if env.is_episode_complete():
                    episode_end = EpisodeEnd(outcome="unmarked", reason="env_complete")
                elif step_idx >= max_episode_steps:
                    episode_end = EpisodeEnd(outcome="unmarked", reason="max_episode_steps")
                else:
                    episode_end = EpisodeEnd(outcome="unmarked", reason="loop_exit")

            logging.info("Returning robot to home pose after episode.")
            env.reset()
            robot_is_home = True

            prompt_counts = prompt_state.prompt_count_summary()
            if episode_end.save:
                task_progress = _read_task_progress(episode_idx) if step_idx > 0 else None
                pat = int(prompt_counts.get("total", 0)) - pat_subtract_count
                saved = recorder.on_episode_end(
                    prompt_counts=prompt_counts,
                    outcome=episode_end.outcome,
                    end_reason=episode_end.reason,
                    task_progress=task_progress,
                    pat=pat,
                )
                if saved:
                    _print_episode_prompt_count(episode_idx=episode_idx, prompt_counts=prompt_counts)
            else:
                recorder.discard_episode(reason=episode_end.reason)

        if not robot_is_home:
            env.reset()
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            logging.debug("OpenCV GUI backend unavailable; skipping destroyAllWindows().")
        recorder.finalize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    sys.argv = _rewrite_legacy_steer_flag(sys.argv)
    tyro.cli(main)
