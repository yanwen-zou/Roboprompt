#!/usr/bin/env python3
# usage:
#   python scripts/realworld/train/evo1/vis_train.py \
#   --config scripts/realworld/train/evo1/config/stage_1.yaml \
#   --max-steps 100   --batch-size 2   --num-workers 0
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from config.config import DEFAULT_CONFIG_NAME, build_dataset_config, build_train_argv, load_launcher_config
from scripts.utils.draw_overlay import draw_prompt_point
from scripts.utils.realworld_projection import draw_projected_action_horizon, project_action_horizon_to_pixels
from scripts.utils.vis import format_vector, load_mono_font, tensor_image_to_uint8, wrap_text


LOGGER = logging.getLogger("evo1_vis_train")


INT_KEYS = {
    "batch_size",
    "action_smoothness_dims",
    "horizon",
    "image_size",
    "max_steps",
    "num_layers",
    "num_workers",
    "per_action_dim",
    "state_dim",
}
FLOAT_KEYS = {
    "action_smoothness_min_norm",
    "action_smoothness_weight",
    "dropout",
    "grad_clip_norm",
    "lr",
    "primitive_loss_weight",
    "point_loss_weight",
    "traj_loss_weight",
    "weight_decay",
    "wrist_loss_weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Evo-1 training dataloader inputs.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME, help="Config name or YAML path.")
    parser.add_argument("--output-dir", default=None, help="Directory for JSONL summaries and rendered input frames.")
    parser.add_argument("--max-steps", type=int, default=None, help="Number of dataloader steps to inspect. Default: load entire dataset.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size for visualization.")
    parser.add_argument("--num-workers", type=int, default=0, help="Override dataloader workers for easier debugging.")
    parser.add_argument("--dataset-config-path", default=None, help="Override Evo-1 dataset config path.")
    parser.add_argument("--cache-dir", default=None, help="Override Evo-1 processed sample cache directory.")
    parser.add_argument("--max-samples-per-file", type=int, default=100, help="Limit cached samples per parquet for quick inspection.")
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional Evo-1 checkpoint dir. Omit to run a fresh train-from-scratch model from --config.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use bfloat16 autocast for checkpoint inference on CUDA.")
    parser.add_argument("--num-inference-timesteps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_scalar(key: str, value: str) -> Any:
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    return value


def config_from_train_argv(argv: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    idx = 1
    while idx < len(argv):
        token = argv[idx]
        if not token.startswith("--"):
            idx += 1
            continue

        key = token[2:].replace("-", "_")
        next_idx = idx + 1
        if next_idx >= len(argv) or argv[next_idx].startswith("--"):
            config[key] = True
            idx += 1
        else:
            config[key] = parse_scalar(key, argv[next_idx])
            idx += 2
    return config


def import_evo1_dataset(evo1_root: Path):
    sys.path.insert(0, str(evo1_root))
    from dataset.lerobot_dataset_rp import LeRobotDatasetRP

    return LeRobotDatasetRP


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state_dict(ckpt_dir: Path) -> dict[str, Any]:
    checkpoint_file = ckpt_dir / "mp_rank_00_model_states.pt"
    meta_path = ckpt_dir / "checkpoint.json"
    if meta_path.is_file():
        meta = load_json(meta_path)
        checkpoint_file = ckpt_dir / str(meta.get("checkpoints", checkpoint_file.name))
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    return checkpoint.get("module", checkpoint)


def load_ckpt_norm_stats(ckpt_dir: Path) -> dict[str, Any] | None:
    stats_path = ckpt_dir / "norm_stats.json"
    if not stats_path.is_file():
        return None
    stats = load_json(stats_path)
    if len(stats) != 1:
        raise ValueError(f"Evo-1 norm_stats.json should contain one robot key, got {list(stats.keys())}.")
    robot_stats = next(iter(stats.values()))
    for key in ("observation.state", "action"):
        if key not in robot_stats or "min" not in robot_stats[key] or "max" not in robot_stats[key]:
            raise ValueError(f"Invalid Evo-1 norm_stats.json: missing {key}.min/max")
    return robot_stats


def apply_ckpt_norm_stats(dataset: Any, ckpt_norm_stats: dict[str, Any]) -> None:
    for arm_name in dataset.arm2stats_dict:
        dataset.arm2stats_dict[arm_name] = ckpt_norm_stats


def import_evo1_model(evo1_root: Path):
    scripts_root = evo1_root / "scripts"
    if str(evo1_root) not in sys.path:
        sys.path.insert(0, str(evo1_root))
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from Evo1 import EVO1

    return EVO1


def build_model_config(config: dict[str, Any], device: torch.device, *, num_inference_timesteps: int) -> dict[str, Any]:
    model_config = dict(config)
    model_config["device"] = str(device)
    model_config["finetune_vlm"] = False
    model_config["finetune_action_head"] = False
    model_config["num_inference_timesteps"] = int(num_inference_timesteps)
    return model_config


def load_evo1_model(
    config: dict[str, Any],
    device: torch.device,
    *,
    num_inference_timesteps: int,
    evo1_root: Path,
    ckpt_dir: Path | None = None,
):
    EVO1 = import_evo1_model(evo1_root)
    model = EVO1(build_model_config(config, device, num_inference_timesteps=num_inference_timesteps)).eval()
    if ckpt_dir is not None:
        model.load_state_dict(load_state_dict(ckpt_dir), strict=True)
    return model.to(device)


def predict_batch(model: Any, batch: dict[str, Any], device: torch.device, *, amp: bool) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for sample_idx, prompt in enumerate(batch["prompts"]):
        images = [img.to(device=device, dtype=torch.float32) for img in batch["images"][sample_idx]]
        image_mask = batch["image_mask"][sample_idx].to(device=device, dtype=torch.int32)
        state = batch["state"][sample_idx].to(device=device, dtype=torch.float32).unsqueeze(0)
        action_mask = batch["action_mask"][sample_idx, 0].to(device=device, dtype=torch.int32).unsqueeze(0)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                action = model.run_inference(
                    images=images,
                    image_mask=image_mask,
                    prompt=str(prompt),
                    state_input=state,
                    action_mask=action_mask,
                )
        predictions.append(action.reshape(int(model.horizon), int(model.per_action_dim)).float().cpu())
    return torch.stack(predictions, dim=0)


def make_dataset(config: dict[str, Any], evo1_root: Path, dataset_config: dict[str, Any] | None = None):
    LeRobotDatasetRP = import_evo1_dataset(evo1_root)
    if dataset_config is None:
        import yaml
        dataset_config_path = Path(str(config["dataset_config_path"]))
        if not dataset_config_path.is_absolute():
            cwd_path = Path.cwd() / dataset_config_path
            dataset_config_path = cwd_path if cwd_path.exists() else evo1_root / dataset_config_path
        with dataset_config_path.open("r", encoding="utf-8") as f:
            dataset_config = yaml.safe_load(f)

    image_size = config.get("image_size")
    if image_size is None:
        raise KeyError("Missing required config key: 'image_size'")
    action_horizon = config.get("horizon")
    if action_horizon is None:
        raise KeyError("Missing required config key: 'horizon'")

    return LeRobotDatasetRP(
        config=dataset_config,
        image_size=int(image_size),
        max_samples_per_file=config.get("max_samples_per_file"),
        action_horizon=int(action_horizon),
        binarize_gripper=bool(config.get("binarize_gripper", False)),
        use_augmentation=bool(config.get("use_augmentation", False)),
        cache_dir=config.get("cache_dir"),
    )


def vis_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "images",
        "image_mask",
        "state",
        "state_mask",
        "action",
        "action_mask",
        "embodiment_id",
        "prompt_global_motion",
        "prompt_global_motion_axis_mask",
        "prompt_global_motion_mask",
        "prompt_local_motion",
        "prompt_local_motion_mask",
        "visual_prompt_applied",
        "video_dropped",
        "visual_prompt_frame_start",
        "visual_prompt_frame_end",
        "prompt_traj_pixels",
        "prompt_traj_mask",
        "prompt_point",
        "prompt_point_mask",
        "prompt_tcp_world_pos",
        "prompt_base_t_camera",
        "prompt_camera_intrinsic",
        "action_min",
        "action_max",
        "global_motion_min",
        "global_motion_max",
    )
    collated = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    collated["prompts"] = [item["prompt"] for item in batch]
    collated["visual_prompt_type"] = [item["visual_prompt_type"] for item in batch]
    return collated


def denormalize_actions(actions: torch.Tensor, action_min: torch.Tensor, action_max: torch.Tensor) -> torch.Tensor:
    actions = actions.detach().float()
    action_min = action_min.detach().float()
    action_max = action_max.detach().float()
    raw = torch.zeros_like(actions)
    dim = min(actions.shape[-1], action_min.shape[-1], action_max.shape[-1])
    raw[..., :dim] = (actions[..., :dim] + 1.0) / 2.0 * (action_max[:dim] - action_min[:dim] + 1e-8) + action_min[:dim]
    return raw


def compute_action_smoothness_debug(
    batch: dict[str, Any],
    sample_idx: int,
    *,
    pred_action: torch.Tensor | None,
    weight: float,
    num_action_dims: int,
    min_action_norm: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": float(weight) != 0.0,
        "valid": False,
        "weight": float(weight),
        "num_action_dims": int(num_action_dims),
        "min_action_norm": float(min_action_norm),
        "loss": None,
        "aux_loss": None,
        "valid_count": 0,
        "mean_relative_change": None,
        "max_relative_change": None,
        "space": "raw_xyz_rel_l4",
    }
    if pred_action is None:
        result["reason"] = "missing_ckpt_prediction"
        return result
    if not result["enabled"]:
        return result

    action = pred_action.detach().float().cpu()
    action_mask = batch["action_mask"][sample_idx].detach().bool().cpu()
    if action.ndim != 2 or action_mask.shape != action.shape:
        result["reason"] = "action_mask_shape_mismatch"
        return result

    action_min = batch["action_min"][sample_idx].detach().float().cpu()
    action_max = batch["action_max"][sample_idx].detach().float().cpu()
    dims = min(
        max(int(num_action_dims), 0),
        action.shape[-1],
        action_min.shape[-1],
        action_max.shape[-1],
    )
    result["num_action_dims"] = dims
    if action.shape[0] < 2 or dims == 0:
        result["reason"] = "insufficient_horizon_or_dims"
        return result

    raw_action = denormalize_actions(action, action_min, action_max)[:, :dims]
    step_mask = action_mask[:, :dims].all(dim=-1)
    action_norm = torch.linalg.vector_norm(raw_action, dim=-1)
    pair_scale = 0.5 * (action_norm[1:] + action_norm[:-1])
    min_norm = max(float(min_action_norm), 0.0)
    pair_mask = step_mask[1:] & step_mask[:-1] & (pair_scale > min_norm)
    valid_count = int(pair_mask.sum().item())
    result["valid_count"] = valid_count
    if valid_count == 0:
        result["reason"] = "no_valid_adjacent_action_changes"
        return result

    difference_norm = torch.linalg.vector_norm(raw_action[1:] - raw_action[:-1], dim=-1)
    relative_change = difference_norm / pair_scale.clamp_min(max(min_norm, 1e-12))
    valid_relative_change = relative_change[pair_mask]
    loss = float(valid_relative_change.pow(4).mean().item())
    result["valid"] = True
    result["mean_relative_change"] = float(valid_relative_change.mean().item())
    result["max_relative_change"] = float(valid_relative_change.max().item())
    result["loss"] = loss
    result["aux_loss"] = loss * float(weight)
    return result

def compute_primitive_debug(
    batch: dict[str, Any],
    sample_idx: int,
    *,
    pred_action: torch.Tensor | None,
    primitive_loss_weight: float,
) -> dict[str, Any]:
    prompt_global_motion = batch["prompt_global_motion"][sample_idx].detach().float().cpu().numpy()
    prompt_global_motion_axis_mask = batch["prompt_global_motion_axis_mask"][sample_idx].detach().cpu().bool().numpy()
    prompt_global_motion_mask = bool(batch["prompt_global_motion_mask"][sample_idx].item())
    result: dict[str, Any] = {
        "enabled": float(primitive_loss_weight) != 0.0,
        "valid": False,
        "weight": float(primitive_loss_weight),
        "loss": None,
        "aux_loss": None,
        "valid_count": 0,
        "target_mask": prompt_global_motion_mask,
        "target_normalized_sum_xyz": prompt_global_motion.astype(float).tolist(),
        "target_cmd_xyz": None,
        "target_axis_mask": prompt_global_motion_axis_mask.astype(bool).tolist(),
        "pred_normalized_sum_xyz": None,
        "pred_cmd_xyz": None,
        "action_mask_valid": False,
        "horizon": 0,
        "horizon_scale": None,
        "space": "dir_norm_xyz_sum",
    }
    if pred_action is None:
        result["reason"] = "missing_ckpt_prediction"
        return result
    if not result["enabled"] or not prompt_global_motion_mask:
        return result

    action = pred_action.detach().float()
    action_mask = batch["action_mask"][sample_idx].detach().cpu().bool().numpy()
    if action.shape[-1] < 3 or action_mask.shape[-1] < 3:
        result["reason"] = "missing_xyz_action_dims"
        return result

    horizon = min(action.shape[0], action_mask.shape[0])
    result["horizon"] = int(horizon)
    if horizon <= 0:
        return result

    action_mask_valid = bool(action_mask[:horizon, :3].all())
    result["action_mask_valid"] = action_mask_valid
    if not action_mask_valid:
        return result

    axis_mask = prompt_global_motion_axis_mask.astype(bool)
    if not bool(axis_mask.any()):
        result["reason"] = "no_active_motion_axes"
        return result

    action_min = batch["action_min"][sample_idx].detach().float()
    action_max = batch["action_max"][sample_idx].detach().float()
    raw_action = denormalize_actions(action, action_min, action_max)
    pred_raw_sum = raw_action[:horizon, :3].sum(dim=0)
    global_motion_min = batch["global_motion_min"][sample_idx].detach().float()
    global_motion_max = batch["global_motion_max"][sample_idx].detach().float()
    negative_scale = global_motion_min[:3].abs().clamp_min(1e-6)
    positive_scale = global_motion_max[:3].abs().clamp_min(1e-6)
    directional_scale = torch.where(pred_raw_sum >= 0.0, positive_scale, negative_scale)
    pred_normalized_sum = (pred_raw_sum / directional_scale).detach().cpu().numpy()
    target_normalized_sum = prompt_global_motion.astype(np.float32)
    result["horizon_scale"] = 1.0
    result["global_motion_min_xyz"] = global_motion_min[:3].cpu().numpy().astype(float).tolist()
    result["global_motion_max_xyz"] = global_motion_max[:3].cpu().numpy().astype(float).tolist()
    result["pred_normalized_sum_xyz"] = pred_normalized_sum.astype(float).tolist()
    result["target_cmd_xyz"] = target_normalized_sum.astype(float).tolist()
    result["pred_cmd_xyz"] = pred_normalized_sum.astype(float).tolist()
    result["valid"] = True
    result["valid_count"] = 1

    loss = float(np.square(pred_normalized_sum - target_normalized_sum)[axis_mask].mean())
    result["loss"] = loss
    result["aux_loss"] = loss * float(primitive_loss_weight)
    return result


def compute_traj_debug(
    batch: dict[str, Any],
    sample_idx: int,
    *,
    pred_action: torch.Tensor | None,
    traj_loss_weight: float,
    image_shape: tuple[int, int, int],
    source_width: int = 640,
    source_height: int = 480,
    processed_padding: int = 8,
) -> dict[str, Any]:
    prompt_traj_mask = batch["prompt_traj_mask"][sample_idx].detach().cpu().bool().numpy()
    prompt_traj_pixels = batch["prompt_traj_pixels"][sample_idx].detach().float().cpu().numpy()
    result: dict[str, Any] = {
        "enabled": float(traj_loss_weight) != 0.0,
        "valid": False,
        "weight": float(traj_loss_weight),
        "loss": None,
        "aux_loss": None,
        "valid_count": 0,
        "target_points": int(prompt_traj_mask.sum()),
        "pred_valid_points": 0,
        "pred_inside_points": 0,
    }
    if pred_action is None:
        result["reason"] = "missing_ckpt_prediction"
        return result
    if not result["enabled"] or not prompt_traj_mask.any():
        return result

    action = pred_action.detach().float()
    action_mask = batch["action_mask"][sample_idx].detach().cpu().bool().numpy()
    raw_action = denormalize_actions(action, batch["action_min"][sample_idx], batch["action_max"][sample_idx])

    horizon = min(len(prompt_traj_pixels), raw_action.shape[0])
    if horizon <= 0:
        return result

    action_chunk = raw_action[: max(horizon - 1, 0)].detach().float().cpu().numpy()
    points_hw, pred_valid = project_action_horizon_to_pixels(
        image_shape=image_shape,
        current_tcp_position=batch["prompt_tcp_world_pos"][sample_idx].detach().float().cpu().numpy(),
        action_chunk=action_chunk,
        base_t_camera=batch["prompt_base_t_camera"][sample_idx].detach().float().cpu().numpy(),
        intrinsic=batch["prompt_camera_intrinsic"][sample_idx].detach().float().cpu().numpy(),
        source_width=source_width,
        source_height=source_height,
        processed_padding=processed_padding,
    )
    points_hw = points_hw[:horizon]
    pred_valid = pred_valid[:horizon]
    if horizon > 1:
        step_valid = action_mask[: horizon - 1, :3].all(axis=-1)
        action_valid = np.concatenate([[True], np.cumprod(step_valid.astype(np.int64)).astype(bool)])
    else:
        action_valid = np.ones((horizon,), dtype=bool)
    pred_valid = pred_valid & action_valid

    height, width = int(image_shape[0]), int(image_shape[1])
    pred_inside = (
        pred_valid
        & (points_hw[:, 0] >= 0)
        & (points_hw[:, 0] < height)
        & (points_hw[:, 1] >= 0)
        & (points_hw[:, 1] < width)
    )
    target_inside = (
        prompt_traj_mask[:horizon]
        & np.isfinite(prompt_traj_pixels[:horizon]).all(axis=-1)
        & (prompt_traj_pixels[:horizon, 0] >= 0.0)
        & (prompt_traj_pixels[:horizon, 0] < height)
        & (prompt_traj_pixels[:horizon, 1] >= 0.0)
        & (prompt_traj_pixels[:horizon, 1] < width)
    )

    result["pred_valid_points"] = int(pred_valid.sum())
    result["pred_inside_points"] = int(pred_inside.sum())
    sample_valid = int(pred_inside.sum()) >= 1 and int(target_inside.sum()) >= 1
    result["valid"] = sample_valid
    result["valid_count"] = int(sample_valid)
    if not sample_valid:
        return result

    pred_points = points_hw[pred_inside].astype(np.float32)
    target_points = prompt_traj_pixels[:horizon][target_inside].astype(np.float32)
    scale = np.asarray([float(height), float(width)], dtype=np.float32)
    if len(pred_points) >= 2 and len(target_points) >= 2:
        pred_norm = (pred_points - pred_points[:1]) / scale
        target_norm = (target_points - target_points[:1]) / scale
    else:
        pred_norm = pred_points / scale
        target_norm = target_points / scale
    diff = pred_norm[:, None, :] - target_norm[None, :, :]
    dists = np.square(diff).sum(axis=-1)
    loss = 0.5 * (float(dists.min(axis=1).mean()) + float(dists.min(axis=0).mean()))
    result["loss"] = loss
    result["aux_loss"] = loss * float(traj_loss_weight)
    return result


def compute_point_debug(
    batch: dict[str, Any],
    sample_idx: int,
    *,
    pred_action: torch.Tensor | None,
    point_loss_weight: float,
    image_shape: tuple[int, int, int],
    source_width: int = 640,
    source_height: int = 480,
    processed_padding: int = 8,
) -> dict[str, Any]:
    prompt_point = batch["prompt_point"][sample_idx].detach().float().cpu().numpy()
    prompt_point_mask = bool(batch["prompt_point_mask"][sample_idx].item())
    result: dict[str, Any] = {
        "enabled": float(point_loss_weight) != 0.0,
        "valid": False,
        "weight": float(point_loss_weight),
        "loss": None,
        "aux_loss": None,
        "valid_count": 0,
        "target_mask": prompt_point_mask,
        "target_point_hw": prompt_point.astype(float).tolist(),
        "pred_point_hw": None,
        "pred_valid": False,
        "pred_inside": False,
        "target_inside": False,
    }
    if pred_action is None:
        result["reason"] = "missing_ckpt_prediction"
        return result
    if not result["enabled"] or not prompt_point_mask:
        return result

    action = pred_action.detach().float()
    action_mask = batch["action_mask"][sample_idx].detach().cpu().bool().numpy()
    raw_action = denormalize_actions(action, batch["action_min"][sample_idx], batch["action_max"][sample_idx])
    horizon = raw_action.shape[0]
    if horizon <= 0:
        return result

    action_chunk = raw_action[: max(horizon - 1, 0)].detach().float().cpu().numpy()
    points_hw, pred_valid = project_action_horizon_to_pixels(
        image_shape=image_shape,
        current_tcp_position=batch["prompt_tcp_world_pos"][sample_idx].detach().float().cpu().numpy(),
        action_chunk=action_chunk,
        base_t_camera=batch["prompt_base_t_camera"][sample_idx].detach().float().cpu().numpy(),
        intrinsic=batch["prompt_camera_intrinsic"][sample_idx].detach().float().cpu().numpy(),
        source_width=source_width,
        source_height=source_height,
        processed_padding=processed_padding,
    )
    points_hw = points_hw[:horizon]
    pred_valid = pred_valid[:horizon]
    if horizon > 1:
        step_valid = action_mask[: horizon - 1, :3].all(axis=-1)
        action_valid = np.concatenate([[True], np.cumprod(step_valid.astype(np.int64)).astype(bool)])
    else:
        action_valid = np.ones((horizon,), dtype=bool)
    pred_valid = pred_valid & action_valid

    pred_point = points_hw[-1].astype(np.float32)
    height, width = int(image_shape[0]), int(image_shape[1])
    pred_inside = bool(
        pred_valid[-1]
        and np.isfinite(pred_point).all()
        and 0.0 <= pred_point[0] < height
        and 0.0 <= pred_point[1] < width
    )
    target_inside = bool(
        np.isfinite(prompt_point).all()
        and 0.0 <= prompt_point[0] < height
        and 0.0 <= prompt_point[1] < width
    )
    sample_valid = pred_inside and target_inside
    result.update(
        {
            "valid": sample_valid,
            "valid_count": int(sample_valid),
            "pred_point_hw": pred_point.astype(float).tolist(),
            "pred_valid": bool(pred_valid[-1]),
            "pred_inside": pred_inside,
            "target_inside": target_inside,
        }
    )
    if not sample_valid:
        return result

    scale = np.asarray([float(height), float(width)], dtype=np.float32)
    loss = float(np.square((pred_point - prompt_point.astype(np.float32)) / scale).mean())
    result["loss"] = loss
    result["aux_loss"] = loss * float(point_loss_weight)
    return result


def summarize_sample(
    batch: dict[str, Any],
    batch_idx: int,
    sample_idx: int,
    primitive_debug: dict[str, Any] | None = None,
    traj_debug: dict[str, Any] | None = None,
    point_debug: dict[str, Any] | None = None,
    smoothness_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_mask = batch["action_mask"][sample_idx]
    state_mask = batch["state_mask"][sample_idx]
    image_mask = batch["image_mask"][sample_idx]
    action = batch["action"][sample_idx].detach().float()
    state = batch["state"][sample_idx].detach().float()
    active_action = action[action_mask]
    active_state = state[state_mask]

    return {
        "step": batch_idx,
        "sample_in_batch": sample_idx,
        "prompt": batch["prompts"][sample_idx],
        "embodiment_id": int(batch["embodiment_id"][sample_idx].item()),
        "images_shape": list(batch["images"][sample_idx].shape),
        "active_views": int(image_mask.sum().item()),
        "state_shape": list(state.shape),
        "state_active_dims": int(state_mask.sum().item()),
        "state_min": float(active_state.min().item()) if active_state.numel() else None,
        "state_max": float(active_state.max().item()) if active_state.numel() else None,
        "action_shape": list(action.shape),
        "action_active_values": int(action_mask.sum().item()),
        "action_min": float(active_action.min().item()) if active_action.numel() else None,
        "action_max": float(active_action.max().item()) if active_action.numel() else None,
        "prompt_global_motion": format_vector(batch["prompt_global_motion"][sample_idx]),
        "prompt_global_motion_mask": bool(batch["prompt_global_motion_mask"][sample_idx].item()),
        "prompt_local_motion": format_vector(batch["prompt_local_motion"][sample_idx]),
        "prompt_local_motion_mask": bool(batch["prompt_local_motion_mask"][sample_idx].item()),
        "visual_prompt_applied": bool(batch["visual_prompt_applied"][sample_idx].item()),
        "video_dropped": bool(batch["video_dropped"][sample_idx].item()),
        "visual_prompt_type": batch["visual_prompt_type"][sample_idx],
        "visual_prompt_frame_start": int(batch["visual_prompt_frame_start"][sample_idx].item()),
        "visual_prompt_frame_end": int(batch["visual_prompt_frame_end"][sample_idx].item()),
        "prompt_traj_points": int(batch["prompt_traj_mask"][sample_idx].sum().item()),
        "prompt_point": format_vector(batch["prompt_point"][sample_idx]),
        "prompt_point_mask": bool(batch["prompt_point_mask"][sample_idx].item()),
        "primitive_debug": primitive_debug,
        "traj_debug": traj_debug,
        "point_debug": point_debug,
        "smoothness_debug": smoothness_debug,
    }


def save_visualization_frame(
    batch: dict[str, Any],
    output_dir: Path,
    batch_idx: int,
    sample_idx: int,
    primitive_debug: dict[str, Any] | None = None,
    traj_debug: dict[str, Any] | None = None,
    point_debug: dict[str, Any] | None = None,
    smoothness_debug: dict[str, Any] | None = None,
) -> str | None:
    """Save active Evo-1 image inputs plus a right-side info panel."""
    step_dir = output_dir / "frames" / f"step_{batch_idx:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    image_mask = batch["image_mask"][sample_idx]

    image_inputs = batch["images"][sample_idx]
    base_img = Image.fromarray(tensor_image_to_uint8(image_inputs[0]))
    base_w, base_h = base_img.size
    if not image_mask[0].item():
        base_img = Image.new("RGB", (base_w, base_h), color=(0, 0, 0))
    if image_inputs.shape[0] > 1:
        prompt_img = Image.fromarray(tensor_image_to_uint8(image_inputs[1]))
        if image_mask.numel() <= 1 or not image_mask[1].item():
            prompt_img = Image.new("RGB", (base_w, base_h), color=(0, 0, 0))
    else:
        prompt_img = Image.new("RGB", (base_w, base_h), color=(0, 0, 0))

    traj_projection_error = None
    if traj_debug is not None and traj_debug.get("enabled") and traj_debug.get("valid"):
        try:
            pred_action = traj_debug.get("pred_action")
            if not torch.is_tensor(pred_action):
                raise ValueError("Missing predicted action tensor for traj overlay")
            raw_action = denormalize_actions(pred_action, batch["action_min"][sample_idx], batch["action_max"][sample_idx])
            horizon = min(int(batch["prompt_traj_mask"][sample_idx].numel()), raw_action.shape[0])
            action_chunk = raw_action[: max(horizon - 1, 0)].detach().float().cpu().numpy()
            prompt_img = Image.fromarray(
                draw_projected_action_horizon(
                    np.asarray(prompt_img),
                    current_tcp_position=batch["prompt_tcp_world_pos"][sample_idx].detach().float().cpu().numpy(),
                    action_chunk=action_chunk,
                    base_t_camera=batch["prompt_base_t_camera"][sample_idx].detach().float().cpu().numpy(),
                    intrinsic=batch["prompt_camera_intrinsic"][sample_idx].detach().float().cpu().numpy(),
                    source_width=640,
                    source_height=480,
                    processed_padding=8,
                    color=(0, 180, 255),
                )
            )
        except Exception as exc:
            traj_projection_error = str(exc)

    if point_debug is not None and point_debug.get("enabled") and point_debug.get("pred_inside"):
        pred_point_hw = point_debug.get("pred_point_hw")
        if pred_point_hw is not None:
            prompt_img = Image.fromarray(
                draw_prompt_point(
                    np.asarray(prompt_img),
                    np.asarray(pred_point_hw, dtype=np.float32),
                    color=(0, 180, 255),
                )
            )

    view_tiles = [
        ("base image input", base_img, bool(image_mask[0].item())),
        ("prompt img input", prompt_img, bool(image_mask[1].item()) if image_mask.numel() > 1 else False),
    ]
    header_h = 18
    tile_w = base_w
    tile_h = base_h + header_h
    img = Image.new("RGB", (tile_w * len(view_tiles), tile_h), color=(18, 18, 18))
    label_font = load_mono_font(9)
    label_draw = ImageDraw.Draw(img)
    for view_idx, (label, view_img, active) in enumerate(view_tiles):
        x0 = view_idx * tile_w
        label_draw.text(
            (x0 + 6, 4),
            f"{label}  mask={int(active)}",
            fill=(230, 230, 230) if active else (140, 140, 140),
            font=label_font,
        )
        img.paste(view_img, (x0, header_h))
    img_w, img_h = img.size

    # Build info panel
    panel_w = 340
    panel = Image.new("RGB", (panel_w, img_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(panel)
    font = load_mono_font(9)

    # Gather info
    prompt = batch["prompts"][sample_idx]
    overlay = bool(batch["visual_prompt_applied"][sample_idx].item())
    overlay_type = batch["visual_prompt_type"][sample_idx]
    frame_start = int(batch["visual_prompt_frame_start"][sample_idx].item())
    frame_end = int(batch["visual_prompt_frame_end"][sample_idx].item())

    global_motion = format_vector(batch["prompt_global_motion"][sample_idx])
    global_mask = bool(batch["prompt_global_motion_mask"][sample_idx].item())
    local_motion = format_vector(batch["prompt_local_motion"][sample_idx])
    local_mask = bool(batch["prompt_local_motion_mask"][sample_idx].item())
    point_mask = bool(batch["prompt_point_mask"][sample_idx].item())
    point_hw = format_vector(batch["prompt_point"][sample_idx])

    imask = [bool(v.item()) for v in image_mask]

    lines = [
        # "=== MASK ===",
        f"image_mask: {imask}",
        f"video_dropped: {bool(batch['video_dropped'][sample_idx].item())}",
        # "=== OVERLAY ===",
        # f"applied: {overlay}  type: {overlay_type}",
        # f"frames: [{frame_start}, {frame_end}]",
        # "=== GLOBAL ===",
        # f"mask: {global_mask}  motion: {global_motion}",
        # "=== LOCAL ===",
        # f"mask: {local_mask}  motion: {local_motion}",
        "=== POINT ===",
        f"mask: {point_mask}  target_hw: {point_hw}",
    ]
    if smoothness_debug is not None and smoothness_debug.get("enabled"):
        lines.extend(
            [
                "=== ACTION SMOOTH ===",
                "n: {}".format(smoothness_debug.get("valid_count")),
                "r_avg: {}".format(smoothness_debug.get("mean_relative_change")),
                "r_max: {}".format(smoothness_debug.get("max_relative_change")),
                "loss: {}".format(smoothness_debug.get("loss")),
                "aux: {}".format(smoothness_debug.get("aux_loss")),
            ]
        )
    if primitive_debug is not None and primitive_debug.get("enabled"):
        lines.extend(
            [
                "=== PRIMITIVE LOSS ===",
                f"space: {primitive_debug.get('space')}",
                f"mask: {global_mask}  target_normalized_sum_xyz: {primitive_debug.get('target_normalized_sum_xyz')}",
                f"pred_normalized_sum_xyz: {primitive_debug.get('pred_normalized_sum_xyz')}",
                f"primitive_loss: {primitive_debug.get('loss')}",
                f"aux_primitive_loss: {primitive_debug.get('aux_loss')}",
            ]
        )
    if traj_debug is not None and traj_debug.get("enabled"):
        lines.extend(
            [
                "=== TRAJ LOSS ===",
                f"traj_loss: {traj_debug.get('loss')}",
                f"aux_traj_loss: {traj_debug.get('aux_loss')}",
            ]
        )
    if point_debug is not None and point_debug.get("enabled"):
        lines.extend(
            [
                "=== POINT LOSS ===",
                f"point_loss: {point_debug.get('loss')}",
                f"aux_point_loss: {point_debug.get('aux_loss')}",
            ]
        )
    if traj_projection_error is not None:
        lines.extend(["=== TRAJ PROJ ERROR ===", traj_projection_error[:180]])
    lines.append("=== PROMPT ===")

    # Wrap prompt so it fits inside the panel
    max_text_w = panel_w - 16  # 8px margin on each side
    prompt_lines = wrap_text(draw, prompt, font, max_text_w)
    lines.extend(prompt_lines)

    y = 4
    line_h = 11
    for line in lines:
        if y + line_h > img_h:
            break  # stop if we run out of vertical space
        draw.text((8, y), line, fill=(220, 220, 220), font=font)
        y += line_h

    # Concatenate image and panel
    total = Image.new("RGB", (img_w + panel_w, img_h))
    total.paste(img, (0, 0))
    total.paste(panel, (img_w, 0))

    path = step_dir / f"sample_{sample_idx:02d}.png"
    total.save(path)
    return str(path)


def main() -> None:
    args = parse_args()
    init_logging()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_launcher_config(args.config)
    train_argv = build_train_argv(cfg)
    train_config = config_from_train_argv(train_argv)
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    train_config["num_workers"] = args.num_workers
    if args.dataset_config_path is not None:
        train_config["dataset_config_path"] = args.dataset_config_path
    if args.cache_dir is not None:
        train_config["cache_dir"] = args.cache_dir
    train_config["max_samples_per_file"] = args.max_samples_per_file

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else cfg.repo_root / "vis_train_output" / f"evo1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading Evo-1 dataset from config: %s", cfg.config_path)
    dataset_config = build_dataset_config(cfg)
    dataset_config["max_views"] = max(2, int(dataset_config.get("max_views", 1)))
    LOGGER.info("Visualization dataset max_views=%d", dataset_config["max_views"])
    dataset = make_dataset(train_config, cfg.evo1_root, dataset_config=dataset_config)

    device = torch.device(args.device)
    ckpt_dir = args.ckpt.expanduser().resolve() if args.ckpt is not None else None
    if ckpt_dir is not None:
        LOGGER.info("Loading Evo-1 checkpoint for pred action overlay: %s", ckpt_dir)
        ckpt_norm_stats = load_ckpt_norm_stats(ckpt_dir)
        if ckpt_norm_stats is not None:
            apply_ckpt_norm_stats(dataset, ckpt_norm_stats)
            LOGGER.info("Using checkpoint norm stats: %s", ckpt_dir / "norm_stats.json")
    else:
        LOGGER.info("No --ckpt provided; running fresh train-from-scratch Evo-1 model from config.")
    model = load_evo1_model(
        train_config if ckpt_dir is None else dict(load_json(ckpt_dir / "config.json")),
        device,
        num_inference_timesteps=args.num_inference_timesteps,
        evo1_root=cfg.evo1_root,
        ckpt_dir=ckpt_dir,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 0)),
        drop_last=False,
        collate_fn=vis_collate_fn,
    )
    LOGGER.info("Dataset samples=%d, batch_size=%s", len(dataset), train_config.get("batch_size"))

    jsonl_path = output_dir / "steps.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for step, batch in enumerate(dataloader):
            if args.max_steps is not None and step >= args.max_steps:
                break

            LOGGER.info(
                "[Step %d] batch images=%s image_mask=%s visual_prompt=%s actions=%s states=%s",
                step,
                tuple(batch["images"].shape),
                batch["image_mask"].int().tolist(),
                batch["visual_prompt_applied"].int().tolist(),
                tuple(batch["action"].shape),
                tuple(batch["state"].shape),
            )
            traj_loss_weight = float(train_config.get("traj_loss_weight", 0.0))
            primitive_loss_weight = float(train_config.get("primitive_loss_weight", 0.0))
            point_loss_weight = train_config.get("point_loss_weight", traj_loss_weight)
            point_loss_weight = traj_loss_weight if point_loss_weight is None else float(point_loss_weight)
            smoothness_weight = float(train_config.get("action_smoothness_weight", 0.0))
            smoothness_dims = int(train_config.get("action_smoothness_dims", 3))
            smoothness_min_norm = float(train_config.get("action_smoothness_min_norm", 1e-4))
            pred_actions = predict_batch(model, batch, device, amp=args.amp)
            for sample_idx in range(len(batch["prompts"])):
                image_np = tensor_image_to_uint8(batch["images"][sample_idx, 0])
                try:
                    primitive_debug = compute_primitive_debug(
                        batch,
                        sample_idx,
                        pred_action=pred_actions[sample_idx],
                        primitive_loss_weight=primitive_loss_weight,
                    )
                except Exception as exc:
                    primitive_debug = {
                        "enabled": primitive_loss_weight != 0.0,
                        "valid": False,
                        "weight": primitive_loss_weight,
                        "loss": None,
                        "aux_loss": None,
                        "valid_count": 0,
                        "error": str(exc),
                    }
                try:
                    traj_debug = compute_traj_debug(
                        batch,
                        sample_idx,
                        pred_action=pred_actions[sample_idx],
                        traj_loss_weight=traj_loss_weight,
                        image_shape=image_np.shape,
                    )
                except Exception as exc:
                    traj_debug = {
                        "enabled": traj_loss_weight != 0.0,
                        "valid": False,
                        "weight": traj_loss_weight,
                        "loss": None,
                        "aux_loss": None,
                        "valid_count": 0,
                        "error": str(exc),
                    }
                try:
                    point_debug = compute_point_debug(
                        batch,
                        sample_idx,
                        pred_action=pred_actions[sample_idx],
                        point_loss_weight=point_loss_weight,
                        image_shape=image_np.shape,
                    )
                except Exception as exc:
                    point_debug = {
                        "enabled": point_loss_weight != 0.0,
                        "valid": False,
                        "weight": point_loss_weight,
                        "loss": None,
                        "aux_loss": None,
                        "valid_count": 0,
                        "error": str(exc),
                    }
                try:
                    smoothness_debug = compute_action_smoothness_debug(
                        batch,
                        sample_idx,
                        pred_action=pred_actions[sample_idx],
                        weight=smoothness_weight,
                        num_action_dims=smoothness_dims,
                        min_action_norm=smoothness_min_norm,
                    )
                except Exception as exc:
                    smoothness_debug = {
                        "enabled": smoothness_weight != 0.0,
                        "valid": False,
                        "weight": smoothness_weight,
                        "num_action_dims": smoothness_dims,
                        "min_action_norm": smoothness_min_norm,
                        "loss": None,
                        "aux_loss": None,
                        "valid_count": 0,
                        "error": str(exc),
                    }
                traj_debug["pred_action"] = pred_actions[sample_idx]
                summary = summarize_sample(
                    batch,
                    step,
                    sample_idx,
                    primitive_debug=primitive_debug,
                    traj_debug=traj_debug,
                    point_debug=point_debug,
                    smoothness_debug=smoothness_debug,
                )
                vis_path = save_visualization_frame(
                    batch,
                    output_dir,
                    step,
                    sample_idx,
                    primitive_debug=primitive_debug,
                    traj_debug=traj_debug,
                    point_debug=point_debug,
                    smoothness_debug=smoothness_debug,
                )
                summary["saved_frame"] = vis_path
                if isinstance(summary.get("traj_debug"), dict):
                    summary["traj_debug"] = {k: v for k, v in summary["traj_debug"].items() if k != "pred_action"}
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
                LOGGER.info(
                    "[Step %d sample %d] overlay=%s/%s global_mask=%s local_mask=%s primitive_valid=%s primitive_loss=%s traj_valid=%s traj_loss=%s point_valid=%s point_loss=%s smooth_valid=%s smooth_loss=%s aux_smooth_loss=%s prompt=%r",
                    step,
                    sample_idx,
                    summary["visual_prompt_applied"],
                    summary["visual_prompt_type"],
                    summary["prompt_global_motion_mask"],
                    summary["prompt_local_motion_mask"],
                    primitive_debug.get("valid"),
                    primitive_debug.get("loss"),
                    traj_debug.get("valid"),
                    traj_debug.get("loss"),
                    point_debug.get("valid"),
                    point_debug.get("loss"),
                    smoothness_debug.get("valid"),
                    smoothness_debug.get("loss"),
                    smoothness_debug.get("aux_loss"),
                    summary["prompt"],
                )

    LOGGER.info("Wrote dataloader summaries to %s", jsonl_path)
    LOGGER.info("Saved input images under %s", output_dir / "frames")


if __name__ == "__main__":
    main()
