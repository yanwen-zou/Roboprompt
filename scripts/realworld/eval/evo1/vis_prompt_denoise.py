#!/usr/bin/env python3
# This script visualizes prompt steering and denoising effect,
# Compares original model output with phase 1 action/ phase 2 action
# Usage: python scripts/realworld/eval/evo1/vis_prompt_denoise.py \
# dataset/real_robot_data/bread/lerobot-pot
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "Evo-1").is_dir() and (path / "openpi").is_dir():
            return path
    raise FileNotFoundError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
OPENPI_ROOT = REPO_ROOT / "openpi"
OPENPI_SRC_ROOT = OPENPI_ROOT / "src"
OPENPI_CLIENT_ROOT = OPENPI_ROOT / "packages" / "openpi-client" / "src"
HARDWARE_ROOT = REPO_ROOT / "hardware"
PREFERRED_PATHS = [str(REPO_ROOT), str(HARDWARE_ROOT), str(OPENPI_ROOT), str(OPENPI_SRC_ROOT), str(OPENPI_CLIENT_ROOT)]
sys.path[:] = [path for path in sys.path if path not in PREFERRED_PATHS]
sys.path[:0] = PREFERRED_PATHS

from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402
from scripts.realworld.eval.eval_ui.interactive_labeling import collect_prompt_payload_from_images  # noqa: E402
from scripts.utils.draw_overlay import draw_future_trajectory  # noqa: E402
from scripts.utils.draw_overlay import make_trajectory_gradient  # noqa: E402
from scripts.utils.flexiv_action_projection import project_flexiv_action_chunk  # noqa: E402
from scripts.utils.interactive_prompt import InteractivePromptState  # noqa: E402
from scripts.utils.interactive_prompt import PROMPT_RANDOM_NOISE_RATIO_STEP  # noqa: E402
from scripts.utils.interactive_prompt import policy_inference_steps_from_metadata  # noqa: E402
from scripts.utils.interactive_prompt import render_prompt_window  # noqa: E402


LOGGER = logging.getLogger("vis_prompt_denoise")
WINDOW_NAME = "Evo Prompt Denoise"
DIRECT_COLORS = (
    (230, 57, 70),
    (29, 147, 236),
    (42, 157, 143),
    (255, 183, 3),
    (157, 78, 221),
    (244, 162, 97),
    (38, 70, 83),
    (128, 185, 24),
)
PHASE1_COLOR = (255, 80, 0)
PHASE2_COLOR = (0, 180, 255)
DEFAULT_RAW_CAMERA_WIDTH = 640
DEFAULT_RAW_CAMERA_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random LeRobot frames, compare direct phase-2 action chunks against "
            "Evo-steered phase1/phase2 chunks, and visualize all 10 trajectories."
        )
    )
    parser.add_argument("dataset_path", type=Path, help="LeRobot dataset root containing meta/, data/, and videos/.")
    parser.add_argument("--host", default="0.0.0.0", help="Policy server host started by a steering server script.")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port.")
    parser.add_argument("--task", default=None, help="Override text task prompt. Defaults to dataset task, then server metadata.")
    parser.add_argument("--num-direct", type=int, default=8, help="Number of direct phase-2 chunks to overlay.")
    parser.add_argument("--direct-num-steps", type=int, default=None, help="Override direct phase-2 denoise steps for the left-panel samples.")
    parser.add_argument("--render-height", type=int, default=224)
    parser.add_argument("--render-width", type=int, default=224)
    parser.add_argument("--phase2-steps", type=float, default=0.6, help="Initial phase2 denoise steps shown in the prompt UI.")
    parser.add_argument("--max-phase2-steps", type=float, default=10.0)
    parser.add_argument(
        "--source-width",
        type=int,
        default=None,
        help="Original camera width for projection remapping. Defaults to 640 when drawing on 224 policy videos.",
    )
    parser.add_argument(
        "--source-height",
        type=int,
        default=None,
        help="Original camera height for projection remapping. Defaults to 480 when drawing on 224 policy videos.",
    )
    parser.add_argument("--processed-padding", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory to save final overlay frames.")
    parser.add_argument(
        "--print-prompt-loss",
        action="store_true",
        help="Log prompt-vs-trajectory consistency losses for phase1 and phase2 chunks.",
    )
    return parser.parse_args()


def phase2_policy_type(metadata: dict[str, Any]) -> str:
    return str(metadata.get("phase2_policy_type") or metadata.get("policy_type") or "openpi").strip().lower()


def random_noise_ratio_step(metadata: dict[str, Any]) -> float:
    try:
        value = float(metadata.get("random_noise_ratio_step", PROMPT_RANDOM_NOISE_RATIO_STEP))
    except (TypeError, ValueError):
        return PROMPT_RANDOM_NOISE_RATIO_STEP
    if not np.isfinite(value) or value < 0.0:
        return PROMPT_RANDOM_NOISE_RATIO_STEP
    return value


def normalize_phase2_steps_for_policy(phase2_steps: float, metadata: dict[str, Any]) -> float | int:
    if phase2_policy_type(metadata) == "diffusion_policy":
        return max(1, int(round(float(phase2_steps))))
    return float(phase2_steps)


def load_tasks(dataset_path: Path) -> dict[int, str]:
    tasks_jsonl = dataset_path / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        tasks: dict[int, str] = {}
        with tasks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "task_index" in obj and "task" in obj:
                    tasks[int(obj["task_index"])] = str(obj["task"])
        return tasks

    tasks_parquet = dataset_path / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        df = pd.read_parquet(tasks_parquet)
        tasks = {}
        for idx, row in df.reset_index().iterrows():
            task_index = int(row["task_index"]) if "task_index" in row else int(idx)
            if "task" in row:
                tasks[task_index] = str(row["task"])
        return tasks

    return {}


def discover_episode_parquets(dataset_path: Path) -> list[Path]:
    parquets = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No LeRobot episode parquet files found under {dataset_path / 'data'}")
    return parquets


def episode_id_from_path(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def find_episode_videos(dataset_path: Path, parquet_path: Path) -> dict[str, Path]:
    chunk = parquet_path.parent.name
    episode = parquet_path.stem
    chunk_videos = dataset_path / "videos" / chunk
    video_paths: dict[str, Path] = {}
    for view_dir in sorted(chunk_videos.iterdir()) if chunk_videos.exists() else []:
        candidate = view_dir / episode / f"{episode}.mp4"
        if candidate.exists():
            video_paths[view_dir.name] = candidate
    if not video_paths:
        matches = sorted((dataset_path / "videos").glob(f"**/{episode}/{episode}.mp4"))
        video_paths = {path.parent.parent.name: path for path in matches}
    if not video_paths:
        raise FileNotFoundError(f"No videos found for {episode} under {dataset_path / 'videos'}")
    return video_paths


def choose_view(video_paths: dict[str, Path], kind: str) -> str:
    candidates = []
    for index, (key, path) in enumerate(video_paths.items()):
        text = f"{key} {path}".lower()
        candidates.append((index, key, text))

    if kind == "base":
        for _, key, text in candidates:
            if any(token in text for token in ("env", "base", "left")):
                return key
        return candidates[0][1]

    for _, key, text in candidates:
        if any(token in text for token in ("wrist", "hand")):
            return key
    base_key = choose_view(video_paths, "base")
    for _, key, _ in candidates:
        if key != base_key:
            return key
    return base_key


def load_video_frame(path: Path, *, row_index: int, timestamp: float | None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        frame_index = int(row_index)
        if timestamp is not None and np.isfinite(timestamp):
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and np.isfinite(fps) and fps > 0:
                frame_index = int(round(float(timestamp) * float(fps)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            frame_index = int(np.clip(frame_index, 0, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def resize_center_crop_rgb(image_rgb: np.ndarray, *, height: int, width: int, padding: int = 8) -> np.ndarray:
    if image_rgb.shape[:2] == (height, width):
        return np.ascontiguousarray(image_rgb)
    resized = cv2.resize(
        image_rgb,
        (int(width + padding), int(height + padding)),
        interpolation=cv2.INTER_CUBIC,
    )
    y0 = max(0, (resized.shape[0] - height) // 2)
    x0 = max(0, (resized.shape[1] - width) // 2)
    return np.ascontiguousarray(resized[y0 : y0 + height, x0 : x0 + width])


def row_value(row: pd.Series, key: str, default: Any = None) -> Any:
    if key not in row:
        return default
    value = row[key]
    if value is None:
        return default
    if np.isscalar(value):
        try:
            if pd.isna(value):
                return default
        except TypeError:
            pass
    return value


def sample_episode_timestep(parquets: list[Path], rng: random.Random) -> tuple[Path, int, pd.Series]:
    by_episode: dict[int, list[Path]] = defaultdict(list)
    for path in parquets:
        by_episode[episode_id_from_path(path)].append(path)
    episode_key = rng.choice(sorted(by_episode))
    parquet_path = rng.choice(by_episode[episode_key])
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        raise ValueError(f"Episode parquet is empty: {parquet_path}")
    row_index = rng.randrange(len(df))
    return parquet_path, row_index, df.iloc[row_index]


def build_observation(
    *,
    dataset_path: Path,
    parquet_path: Path,
    row_index: int,
    row: pd.Series,
    tasks: dict[int, str],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    video_paths = find_episode_videos(dataset_path, parquet_path)
    timestamp = row_value(row, "timestamp")
    timestamp_float = None if timestamp is None else float(timestamp)
    base_key = choose_view(video_paths, "base")
    wrist_key = choose_view(video_paths, "wrist")
    base_raw = load_video_frame(video_paths[base_key], row_index=row_index, timestamp=timestamp_float)
    wrist_raw = load_video_frame(video_paths[wrist_key], row_index=row_index, timestamp=timestamp_float)
    base_policy = resize_center_crop_rgb(base_raw, height=args.render_height, width=args.render_width)
    wrist_policy = resize_center_crop_rgb(wrist_raw, height=args.render_height, width=args.render_width)

    state = np.asarray(row_value(row, "observation.state"), dtype=np.float32)
    if state.ndim != 1:
        state = state.reshape(-1)
    task = args.task
    if task is None:
        task_index = row_value(row, "task_index")
        if task_index is not None:
            task = tasks.get(int(task_index))
    if task is None:
        task = metadata.get("default_prompt") or metadata.get("prompt") or metadata.get("task_description")

    obs = {
        "observation/state": state,
        "observation/image": base_policy,
        "observation/wrist_image": wrist_policy,
    }
    if task:
        obs["prompt"] = str(task)

    context = {
        "dataset_path": dataset_path,
        "episode_id": episode_id_from_path(parquet_path),
        "row_index": int(row_index),
        "parquet_path": parquet_path,
        "base_view": base_key,
        "wrist_view": wrist_key,
        "base_raw": base_raw,
        "base_policy": base_policy,
        "wrist_policy": wrist_policy,
        "state": state,
        "task": task,
        "timestamp": timestamp_float,
    }
    return obs, context


def extract_action_chunk(result: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key not in result:
            continue
        actions = np.asarray(result[key], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim == 2 and actions.shape[-1] >= 3:
            return actions
    return None


def infer_direct_chunks(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    obs: dict[str, Any],
    count: int,
    *,
    metadata: dict[str, Any],
    direct_num_steps: int | None,
) -> list[np.ndarray]:
    chunks = []
    sample_kwargs = {
        "enable_steerer": False,
        "enable_evo1_steerer": False,
    }
    if direct_num_steps is not None:
        if phase2_policy_type(metadata) == "diffusion_policy":
            sample_kwargs["num_inference_steps"] = int(direct_num_steps)
        else:
            sample_kwargs["num_steps"] = int(direct_num_steps)
    for index in range(count):
        result = policy.infer(obs, sample_kwargs=sample_kwargs)
        chunk = extract_action_chunk(result, ("actions",))
        if chunk is None:
            raise RuntimeError(f"Direct inference {index} did not return an action chunk. Keys: {sorted(result)}")
        chunks.append(chunk)
    return chunks


def format_motion_vector(vector: np.ndarray) -> str:
    vector = np.asarray(vector, dtype=np.float32)[:3]
    return "[" + ",".join(f"{float(value):.3f}" for value in vector) + "]"


def format_motion_axis_values(vector: np.ndarray) -> str:
    vector = np.asarray(vector, dtype=np.float32)[:3]
    return ", ".join(
        f"{axis}:{float(value):.3f}"
        for axis, value in zip(("x", "y", "z"), vector, strict=True)
        if np.isfinite(value) and not np.isclose(value, 0.0, atol=1e-6)
    )


def build_steerer_text_prompt(task: str | None, prompt_state: InteractivePromptState) -> str | None:
    parts = []
    if task:
        parts.append(str(task).strip())
    payload = prompt_state.payload
    if payload is None:
        return ", ".join(part for part in parts if part) or None

    global_motion = payload.get("prompt_global_motion")
    global_mask = bool(np.any(np.asarray(payload.get("prompt_global_motion_mask", False), dtype=np.bool_)))
    if global_motion is not None and global_mask:
        motion_text = format_motion_axis_values(np.asarray(global_motion, dtype=np.float32)[:3])
        if motion_text:
            parts.append(f"move {motion_text} in global frame")

    local_motion = payload.get("prompt_local_motion")
    local_mask = bool(np.any(np.asarray(payload.get("prompt_local_motion_mask", False), dtype=np.bool_)))
    if local_motion is not None and local_mask:
        motion_text = format_motion_axis_values(np.asarray(local_motion, dtype=np.float32)[:3])
        if motion_text:
            parts.append(f"move {motion_text} in wrist local frame")

    return ", ".join(part for part in parts if part) or None


def infer_steered_chunk(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    obs: dict[str, Any],
    context: dict[str, Any],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    prompt_payload = collect_prompt_payload_from_images(
        base_image_rgb=context["base_policy"],
        wrist_image_rgb=context["wrist_policy"],
        phase2_steps=args.phase2_steps,
        max_phase2_steps=args.max_phase2_steps,
    )
    prompt_state = InteractivePromptState()
    prompt_state.update(prompt_payload)
    display_payload = prompt_state.build_display_inputs()
    sample_kwargs = prompt_state.build_sample_kwargs(
        policy_inference_steps=args.max_phase2_steps,
        emit_num_steps=phase2_policy_type(metadata) == "openpi",
        phase2_policy_type=phase2_policy_type(metadata),
        random_noise_ratio_step=random_noise_ratio_step(metadata),
    )
    if "phase2_steps" in sample_kwargs:
        sample_kwargs["phase2_steps"] = normalize_phase2_steps_for_policy(sample_kwargs["phase2_steps"], metadata)
    sample_kwargs["enable_steerer"] = True
    sample_kwargs["enable_evo1_steerer"] = True

    model_obs = {
        **obs,
        **prompt_state.build_model_inputs(),
    }
    prompt = build_steerer_text_prompt(context.get("task"), prompt_state)
    if prompt is not None:
        model_obs["prompt"] = prompt

    render_prompt_window(
        {
            **display_payload,
            "sample_kwargs": sample_kwargs,
        },
        "vis_prompt_denoise",
        int(context["episode_id"]),
        int(context["row_index"]),
    )
    result = policy.infer(model_obs, sample_kwargs=sample_kwargs)
    phase1 = extract_action_chunk(
        result,
        (
            "phase1_actions_raw",
            "steerer_phase1_actions_raw",
            "evo1_phase1_actions_raw",
            "steerer_phase1_actions",
            "evo1_phase1_actions",
        ),
    )
    phase2 = extract_action_chunk(result, ("actions",))
    return phase1, phase2, dict(prompt_state.payload or display_payload)


def _active_prompt_vector(payload: dict[str, Any], value_key: str, mask_key: str, dim: int) -> np.ndarray | None:
    if not bool(np.any(np.asarray(payload.get(mask_key, False), dtype=np.bool_))):
        return None
    value = payload.get(value_key)
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape[0] < dim or not np.isfinite(vector[:dim]).all():
        return None
    return vector[:dim]


def _extract_prompt_point_hw(payload: dict[str, Any]) -> np.ndarray | None:
    point_mask = payload.get("prompt_point_mask")
    if point_mask is not None and not bool(np.any(np.asarray(point_mask, dtype=np.bool_))):
        return None
    point = payload.get("prompt_point")
    if point is not None:
        point_hw = np.asarray(point, dtype=np.float32).reshape(-1)
        if point_hw.shape[0] >= 2 and np.isfinite(point_hw[:2]).all():
            return point_hw[:2]

    prompt_result = payload.get("interactive_prompt_result")
    draw_ops = getattr(prompt_result, "draw_ops", None)
    if not isinstance(draw_ops, list):
        draw_ops = payload.get("draw_ops")
    if not isinstance(draw_ops, list):
        return None
    for op in reversed(draw_ops):
        if not isinstance(op, dict) or op.get("type") != "point":
            continue
        point = op.get("point_hw")
        if point is None:
            continue
        point_hw = np.asarray(point, dtype=np.float32).reshape(-1)
        if point_hw.shape[0] >= 2 and np.isfinite(point_hw[:2]).all():
            return point_hw[:2]
    return None


def _prompt_image_shape(payload: dict[str, Any]) -> tuple[int, int] | None:
    prompt_images = payload.get("prompt_images")
    if not isinstance(prompt_images, dict):
        return None
    image = prompt_images.get("prompt_0")
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim < 2:
        return None
    return int(image.shape[0]), int(image.shape[1])


def compute_prompt_losses(
    *,
    context: dict[str, Any],
    prompt_payload: dict[str, Any],
    action_chunk: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "global_motion_mse": None,
        "global_motion_pred_xyz": None,
        "global_motion_target_xyz": None,
        "global_motion_axis_mask": None,
        "drag_mse": None,
        "drag_pred_xy": None,
        "drag_target_xy": None,
        "point_mse": None,
        "point_pred_hw": None,
        "point_target_hw": None,
    }
    if action_chunk is None:
        return metrics

    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < 3:
        return metrics

    global_target = _active_prompt_vector(prompt_payload, "prompt_global_motion", "prompt_global_motion_mask", 3)
    if global_target is not None:
        pred_xyz = actions[:, :3].sum(axis=0).astype(np.float32)
        axis_mask = np.isfinite(global_target) & (np.abs(global_target) >= 1e-6)
        metrics["global_motion_pred_xyz"] = pred_xyz.astype(float).tolist()
        metrics["global_motion_target_xyz"] = global_target.astype(float).tolist()
        metrics["global_motion_axis_mask"] = axis_mask.astype(bool).tolist()
        if bool(axis_mask.any()):
            metrics["global_motion_mse"] = float(np.square(pred_xyz - global_target)[axis_mask].mean())

    drag_target = _active_prompt_vector(prompt_payload, "prompt_2d_drag", "prompt_2d_drag_mask", 2)
    point_target = _extract_prompt_point_hw(prompt_payload)
    if drag_target is None and point_target is None:
        return metrics

    image = np.asarray(context["base_raw"], dtype=np.uint8)
    source_width = args.source_width
    source_height = args.source_height
    if source_width is None and source_height is None and image.shape[:2] == (args.render_height, args.render_width):
        source_width = DEFAULT_RAW_CAMERA_WIDTH
        source_height = DEFAULT_RAW_CAMERA_HEIGHT
    source_image_shape = None
    if source_width is not None and source_height is not None:
        source_image_shape = (int(source_height), int(source_width), 3)
    overlay = project_flexiv_action_chunk(
        image_shape=image.shape,
        observation={"observation/state": np.asarray(context["state"], dtype=np.float32)},
        action_chunk=actions,
        color=PHASE2_COLOR,
        source_image_shape=source_image_shape,
    )
    points_hw = np.asarray(overlay["points_hw"], dtype=np.float32)
    valid_mask = np.asarray(overlay["valid_mask"], dtype=bool)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < 1:
        return metrics

    start = points_hw[int(valid_indices[0])]
    end = points_hw[int(valid_indices[-1])]
    height, width = image.shape[:2]
    denom_xy = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    if drag_target is not None and len(valid_indices) >= 2:
        pred_drag_xy = np.asarray([end[1] - start[1], end[0] - start[0]], dtype=np.float32) / denom_xy
        metrics["drag_pred_xy"] = pred_drag_xy.astype(float).tolist()
        metrics["drag_target_xy"] = drag_target.astype(float).tolist()
        metrics["drag_mse"] = float(np.square(pred_drag_xy - drag_target).mean())

    if point_target is not None:
        prompt_hw = _prompt_image_shape(prompt_payload)
        if prompt_hw is not None:
            scale_hw = np.asarray([height / max(prompt_hw[0], 1), width / max(prompt_hw[1], 1)], dtype=np.float32)
            point_target = point_target * scale_hw
        pred_point_hw = end.astype(np.float32)
        scale_hw = np.asarray([max(height, 1), max(width, 1)], dtype=np.float32)
        metrics["point_pred_hw"] = pred_point_hw.astype(float).tolist()
        metrics["point_target_hw"] = point_target.astype(float).tolist()
        metrics["point_mse"] = float(np.square((pred_point_hw - point_target) / scale_hw).mean())
    return metrics


def format_prompt_loss_line(name: str, metrics: dict[str, Any]) -> str:
    parts = [name]
    if metrics.get("global_motion_mse") is not None:
        parts.append(f"global_mse={float(metrics['global_motion_mse']):.6f}")
    if metrics.get("drag_mse") is not None:
        parts.append(f"drag_mse={float(metrics['drag_mse']):.6f}")
    if metrics.get("point_mse") is not None:
        parts.append(f"point_mse={float(metrics['point_mse']):.6f}")
    return " ".join(parts) if len(parts) > 1 else f"{name} no_active_loss"


def draw_prompt_loss_text(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return frame_rgb
    image = np.ascontiguousarray(frame_rgb.copy())
    x, y0 = 12, 26
    line_h = 24
    box_h = line_h * len(lines) + 14
    box_w = min(image.shape[1] - 24, max(360, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for line in lines) + 20))
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (255, 255, 255), -1)
    image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y0 + index * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return image


def draw_projected_chunk(
    image_rgb: np.ndarray,
    *,
    state: np.ndarray,
    action_chunk: np.ndarray,
    color: tuple[int, int, int],
    source_width: int | None,
    source_height: int | None,
    processed_padding: int,
    outlined: bool,
) -> np.ndarray:
    del processed_padding
    source_image_shape = None
    if source_width is not None and source_height is not None:
        source_image_shape = (int(source_height), int(source_width), 3)
    overlay = project_flexiv_action_chunk(
        image_shape=image_rgb.shape,
        observation={"observation/state": np.asarray(state, dtype=np.float32)},
        action_chunk=np.asarray(action_chunk, dtype=np.float32),
        color=color,
        source_image_shape=source_image_shape,
    )
    points_hw = np.asarray(overlay["points_hw"], dtype=np.int32)
    valid_mask = np.asarray(overlay["valid_mask"], dtype=bool)
    image = image_rgb
    start = 0
    draw_color: Any = color if outlined else make_trajectory_gradient(color)
    thickness = 1
    while start < len(points_hw):
        while start < len(points_hw) and not valid_mask[start]:
            start += 1
        end = start
        while end < len(points_hw) and valid_mask[end]:
            end += 1
        if end - start >= 2:
            image = draw_future_trajectory(image, points_hw[start:end], color=draw_color, thickness=thickness)
        start = end
    if len(points_hw) > 0 and bool(valid_mask[0]):
        cy, cx = int(points_hw[0][0]), int(points_hw[0][1])
        cv2.circle(image, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 2, color, -1, cv2.LINE_AA)
    return image


def draw_overlay_frame(
    *,
    context: dict[str, Any],
    direct_chunks: list[np.ndarray] | None = None,
    phase1_chunk: np.ndarray | None = None,
    phase2_chunk: np.ndarray | None = None,
    args: argparse.Namespace,
) -> np.ndarray:
    image = np.asarray(context["base_raw"], dtype=np.uint8).copy()
    source_width = args.source_width
    source_height = args.source_height
    if source_width is None and source_height is None and image.shape[:2] == (args.render_height, args.render_width):
        source_width = DEFAULT_RAW_CAMERA_WIDTH
        source_height = DEFAULT_RAW_CAMERA_HEIGHT
    if direct_chunks is not None:
        for index, chunk in enumerate(direct_chunks):
            image = draw_projected_chunk(
                image,
                state=context["state"],
                action_chunk=chunk,
                color=DIRECT_COLORS[index % len(DIRECT_COLORS)],
                source_width=source_width,
                source_height=source_height,
                processed_padding=args.processed_padding,
                outlined=False,
            )
    if phase1_chunk is not None:
        image = draw_projected_chunk(
            image,
            state=context["state"],
            action_chunk=phase1_chunk,
            color=PHASE1_COLOR,
            source_width=source_width,
            source_height=source_height,
            processed_padding=args.processed_padding,
            outlined=True,
        )
    if phase2_chunk is not None:
        image = draw_projected_chunk(
            image,
            state=context["state"],
            action_chunk=phase2_chunk,
            color=PHASE2_COLOR,
            source_width=source_width,
            source_height=source_height,
            processed_padding=args.processed_padding,
            outlined=True,
        )
    return image


def compose_comparison_frame(
    *,
    context: dict[str, Any],
    direct_chunks: list[np.ndarray],
    phase1_chunk: np.ndarray | None,
    phase2_chunk: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    direct_frame = draw_overlay_frame(
        context=context,
        direct_chunks=direct_chunks,
        args=args,
    )
    steered_frame = draw_overlay_frame(
        context=context,
        phase1_chunk=phase1_chunk,
        phase2_chunk=phase2_chunk,
        args=args,
    )
    gap = np.full((direct_frame.shape[0], 8, 3), 245, dtype=np.uint8)
    return np.concatenate([direct_frame, gap, steered_frame], axis=1)


def show_rgb(frame_rgb: np.ndarray) -> int:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    return cv2.waitKey(1) & 0xFF


def wait_next_or_quit(frame_rgb: np.ndarray) -> str:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(WINDOW_NAME, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("p"), ord("P")):
            return "next"
        if key in (ord("q"), ord("Q"), 27):
            return "quit"


def maybe_save_frame(frame_rgb: np.ndarray, context: dict[str, Any], save_dir: Path | None) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"episode_{int(context['episode_id']):06d}_step_{int(context['row_index']):06d}.png"
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    LOGGER.info("Saved overlay frame to %s", path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    rng = random.Random(args.seed)
    dataset_path = args.dataset_path.expanduser().resolve()
    if not (dataset_path / "data").is_dir():
        raise FileNotFoundError(f"Expected a LeRobot dataset root with data/: {dataset_path}")

    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = policy.get_server_metadata()
    LOGGER.info("Server metadata: %s", metadata)
    args.max_phase2_steps = policy_inference_steps_from_metadata(metadata, default=args.max_phase2_steps)
    if not isinstance(metadata.get("steerer"), dict) and not isinstance(metadata.get("evo1_steerer"), dict):
        LOGGER.warning("Connected server metadata does not advertise an Evo steerer; steered inference may fail.")

    tasks = load_tasks(dataset_path)
    parquets = discover_episode_parquets(dataset_path)
    LOGGER.info("Loaded %d episode parquet files from %s", len(parquets), dataset_path)

    try:
        while True:
            parquet_path, row_index, row = sample_episode_timestep(parquets, rng)
            obs, context = build_observation(
                dataset_path=dataset_path,
                parquet_path=parquet_path,
                row_index=row_index,
                row=row,
                tasks=tasks,
                metadata=metadata,
                args=args,
            )
            LOGGER.info("Sampled episode=%06d timestep=%d", context["episode_id"], context["row_index"])
            direct_chunks = infer_direct_chunks(
                policy,
                obs,
                max(1, int(args.num_direct)),
                metadata=metadata,
                direct_num_steps=args.direct_num_steps,
            )
            baseline_frame = draw_overlay_frame(context=context, direct_chunks=direct_chunks, args=args)
            show_rgb(baseline_frame)

            phase1_chunk, phase2_chunk, prompt_payload = infer_steered_chunk(policy, obs, context, metadata, args)
            final_frame = compose_comparison_frame(
                context=context,
                direct_chunks=direct_chunks,
                phase1_chunk=phase1_chunk,
                phase2_chunk=phase2_chunk,
                args=args,
            )
            if args.print_prompt_loss:
                phase1_losses = compute_prompt_losses(
                    context=context,
                    prompt_payload=prompt_payload,
                    action_chunk=phase1_chunk,
                    args=args,
                )
                phase2_losses = compute_prompt_losses(
                    context=context,
                    prompt_payload=prompt_payload,
                    action_chunk=phase2_chunk,
                    args=args,
                )
                loss_lines = [
                    format_prompt_loss_line("phase1", phase1_losses),
                    format_prompt_loss_line("phase2", phase2_losses),
                ]
                LOGGER.info("Prompt loss %s | %s", loss_lines[0], loss_lines[1])
                final_frame = draw_prompt_loss_text(final_frame, loss_lines)
            maybe_save_frame(final_frame, context, args.save_dir)
            if wait_next_or_quit(final_frame) == "quit":
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
