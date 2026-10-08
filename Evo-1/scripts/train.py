from __future__ import annotations

import sys
import os
import math
from torch import amp

# Ensure CUDA toolchain is discoverable for DeepSpeed / flash-attn compilation
if not os.environ.get("CUDA_HOME"):
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.9"
    os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + ":" + os.environ.get("PATH", "")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from Evo1 import EVO1
from accelerate import Accelerator 
import logging
from datetime import datetime
import argparse
from accelerate import Accelerator, DistributedType
import json
import shutil
from torch.optim import AdamW

import warnings

accelerator = Accelerator()

def get_with_warning(config: dict, key: str, default):
    if key in config:
        return config[key]
    else:
        warnings.warn(f"'{key}' not found in config, using default: {default!r}")
        return default


def inspect_named_submodules(module_dict: dict, verbose: bool = True):

    total_all, trainable_all = 0, 0
    logging.info("\n Parameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info(f"\n Module: {module_name}")
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(f"Trainable {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
            elif verbose:
                logging.info(f"Frozen {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
        logging.info("-" * 70)
        logging.info(f"Total     : {total / 1e6:.2f}M")
        logging.info(f"Trainable : {trainable / 1e6:.2f}M")
        logging.info(f"Frozen    : {(total - trainable) / 1e6:.2f}M")
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info(f"ALL TOTAL     : {total_all / 1e6:.2f}M")
    logging.info(f"ALL TRAINABLE : {trainable_all / 1e6:.2f}M")
    logging.info(f"ALL FROZEN    : {(total_all - trainable_all) / 1e6:.2f}M")
    logging.info("=" * 70)


def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    images = [item["images"] for item in batch]
    states = torch.stack([item["state"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    action_mask = torch.stack([item["action_mask"] for item in batch], dim=0)
    prompt_global_motion = torch.stack([item["prompt_global_motion"] for item in batch], dim=0)
    prompt_global_motion_axis_mask = torch.stack([item["prompt_global_motion_axis_mask"] for item in batch], dim=0)
    prompt_global_motion_mask = torch.stack([item["prompt_global_motion_mask"] for item in batch], dim=0)
    prompt_local_motion = torch.stack([item["prompt_local_motion"] for item in batch], dim=0)
    prompt_local_motion_mask = torch.stack([item["prompt_local_motion_mask"] for item in batch], dim=0)
    prompt_traj_pixels = torch.stack([item["prompt_traj_pixels"] for item in batch], dim=0)
    prompt_traj_mask = torch.stack([item["prompt_traj_mask"] for item in batch], dim=0)
    prompt_point = torch.stack([item["prompt_point"] for item in batch], dim=0)
    prompt_point_mask = torch.stack([item["prompt_point_mask"] for item in batch], dim=0)
    prompt_tcp_world_pos = torch.stack([item["prompt_tcp_world_pos"] for item in batch], dim=0)
    prompt_base_t_camera = torch.stack([item["prompt_base_t_camera"] for item in batch], dim=0)
    prompt_camera_intrinsic = torch.stack([item["prompt_camera_intrinsic"] for item in batch], dim=0)
    image_masks = torch.stack([item["image_mask"] for item in batch], dim=0)
    state_mask = torch.stack([item["state_mask"] for item in batch], dim=0)
    embodiment_ids = torch.stack([item["embodiment_id"] for item in batch], dim=0)
    action_min = torch.stack([item["action_min"] for item in batch], dim=0)
    action_max = torch.stack([item["action_max"] for item in batch], dim=0)
    global_motion_min = torch.stack([item["global_motion_min"] for item in batch], dim=0)
    global_motion_max = torch.stack([item["global_motion_max"] for item in batch], dim=0)

    return {
        "prompts": prompts,
        "images": images,
        "states": states,
        "actions": actions,
        "action_mask": action_mask,
        "prompt_global_motion": prompt_global_motion,
        "prompt_global_motion_axis_mask": prompt_global_motion_axis_mask,
        "prompt_global_motion_mask": prompt_global_motion_mask,
        "prompt_local_motion": prompt_local_motion,
        "prompt_local_motion_mask": prompt_local_motion_mask,
        "prompt_traj_pixels": prompt_traj_pixels,
        "prompt_traj_mask": prompt_traj_mask,
        "prompt_point": prompt_point,
        "prompt_point_mask": prompt_point_mask,
        "prompt_tcp_world_pos": prompt_tcp_world_pos,
        "prompt_base_t_camera": prompt_base_t_camera,
        "prompt_camera_intrinsic": prompt_camera_intrinsic,
        "state_mask": state_mask,
        "image_masks": image_masks,
        "embodiment_ids": embodiment_ids,
        "action_min": action_min,
        "action_max": action_max,
        "global_motion_min": global_motion_min,
        "global_motion_max": global_motion_max,
    }

def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step  
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda
    
def setup_logging(log_dir: str) -> str:
    from datetime import datetime
    import logging, os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_path}")
    return log_path

def init_wandb(config: dict, accelerator: Accelerator):

    if accelerator.is_main_process:
        if get_with_warning(config, "disable_wandb", False):
            os.environ["WANDB_MODE"] = "disabled"

        wandb.init(
            project=get_with_warning(config, "wandb_project", "default_run"),
            name=get_with_warning(config, "run_name", "default_run"),
            config=config,
            dir=get_with_warning(config, "save_dir", "checkpoints"),
            mode="online",
        )

        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

def prepare_dataset(config: dict) -> torch.utils.data.Dataset:
    dataset_type = get_with_warning(config, "dataset_type", "lerobot")
    image_size = get_with_warning(config, "image_size", 448)
    max_samples = get_with_warning(config, "max_samples_per_file", None)
    horizon = get_with_warning(config, "horizon", 50)
    binarize_gripper = get_with_warning(config, "binarize_gripper", False)
    use_augmentation = get_with_warning(config, "use_augmentation", False)
    cache_dir = get_with_warning(config, "cache_dir", None)
    cmd_type = get_with_warning(config, "cmd_type", "sigma")
    if dataset_type == "lerobot":
        from dataset.lerobot_dataset_rp import LeRobotDatasetRP
        import yaml
        with open(config.get("dataset_config_path"), 'r') as f:
            dataset_config = yaml.safe_load(f)
        dataset_config["cmd_type"] = cmd_type

        dataset = LeRobotDatasetRP(
            config=dataset_config,
            action_horizon=horizon,
            image_size=image_size,
            max_samples_per_file=max_samples,
            binarize_gripper=binarize_gripper,
            use_augmentation=use_augmentation,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Loaded {len(dataset)} samples from {config['data_paths']} ({dataset_type})")
    return dataset


def prepare_dataloader(dataset, config: dict) -> DataLoader:
    batch_size = get_with_warning(config, "batch_size", 8)
    num_workers = get_with_warning(config, "num_workers", 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Initialized dataloader with batch size {batch_size}")
    return dataloader


def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info(f"[Step {step}] Non-finite detected in {name}")
            return False
    return True

def compute_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_flow_matching_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_velocity = pred_velocity.float()
    target_velocity = target_velocity.to(device=pred_velocity.device, dtype=pred_velocity.dtype)
    action_mask_flat = action_mask.view(action_mask.shape[0], -1).to(
        device=pred_velocity.device,
        dtype=pred_velocity.dtype,
    )
    sq_error = torch.square(pred_velocity - target_velocity) * action_mask_flat
    valid_count = action_mask_flat.sum()
    loss = sq_error.sum() / valid_count.clamp_min(1.0)
    return loss, valid_count


def compute_action_smoothness_loss(
    pred_actions: torch.Tensor,
    action_mask: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    num_action_dims: int,
    min_action_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize sharp relative changes in predicted raw XYZ delta actions.

    The difference between adjacent delta-action vectors is normalized by their
    mean magnitude, making the loss invariant to a common movement scale while
    retaining both direction and relative-magnitude changes. Raising the relative
    change to the fourth power suppresses smooth turns and emphasizes isolated
    spikes. Pairs where both actions are near zero are ignored.
    """
    if pred_actions.ndim != 3 or action_mask.shape != pred_actions.shape:
        raise ValueError(
            "pred_actions and action_mask must have matching [batch, horizon, action_dim] shapes, "
            f"got {tuple(pred_actions.shape)} and {tuple(action_mask.shape)}"
        )
    if action_min.ndim != 2 or action_max.shape != action_min.shape or action_min.shape[0] != pred_actions.shape[0]:
        raise ValueError(
            "action_min and action_max must have matching [batch, action_dim] shapes, "
            f"got {tuple(action_min.shape)} and {tuple(action_max.shape)}"
        )

    dims = min(
        max(int(num_action_dims), 0),
        pred_actions.shape[-1],
        action_min.shape[-1],
    )
    if pred_actions.shape[1] < 2 or dims == 0:
        zero = pred_actions.new_zeros((), dtype=torch.float32)
        return zero, zero

    pred_normalized = pred_actions[:, :, :dims].float()
    amin = action_min[:, :dims].to(device=pred_normalized.device, dtype=pred_normalized.dtype)
    amax = action_max[:, :dims].to(device=pred_normalized.device, dtype=pred_normalized.dtype)
    pred_raw = (pred_normalized + 1.0) / 2.0 * (
        amax[:, None, :] - amin[:, None, :] + 1e-8
    ) + amin[:, None, :]

    step_mask = action_mask[:, :, :dims].to(device=pred_raw.device).bool().all(dim=-1)
    action_norm = torch.linalg.vector_norm(pred_raw, dim=-1)
    pair_scale = 0.5 * (action_norm[:, 1:] + action_norm[:, :-1])
    min_norm = max(float(min_action_norm), 0.0)
    pair_mask = step_mask[:, 1:] & step_mask[:, :-1] & (pair_scale > min_norm)

    difference_norm = torch.linalg.vector_norm(pred_raw[:, 1:] - pred_raw[:, :-1], dim=-1)
    relative_change = difference_norm / pair_scale.clamp_min(max(min_norm, 1e-12))
    per_pair_loss = relative_change.pow(2)
    valid_count = pair_mask.sum()
    loss = (per_pair_loss * pair_mask.to(dtype=per_pair_loss.dtype)).sum() / valid_count.clamp_min(1)
    return loss, valid_count

def compute_primitive_aux_loss(
    pred_actions: torch.Tensor,
    prompt_global_motion: torch.Tensor,
    prompt_global_motion_mask: torch.Tensor,
    prompt_global_motion_axis_mask: torch.Tensor,
    action_mask: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    global_motion_min: torch.Tensor,
    global_motion_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Primitive-direction auxiliary loss (continuous regression).

    ``pred_actions`` are in the action head's min/max-normalised action scale.
    ``prompt_global_motion`` is the raw XYZ horizon sum normalized with the
    dataset-level positive/negative horizon-sum bounds.
    The prediction is converted from the action head's min/max-normalized space
    back to raw units and then mapped into the same sign-preserving space.
    """
    if (
        pred_actions.shape[-1] < 3
        or action_min.shape[-1] < 3
        or action_max.shape[-1] < 3
        or global_motion_min.shape[-1] < 3
        or global_motion_max.shape[-1] < 3
    ):
        zero = pred_actions.new_zeros(())
        return zero, zero

    horizon = pred_actions.shape[1]
    if horizon <= 0:
        zero = pred_actions.new_zeros(())
        return zero, zero
    device = pred_actions.device
    pred_norm_xyz = pred_actions[:, :horizon, :3].float()
    action_min_xyz = action_min[:, :3].to(device=device, dtype=pred_norm_xyz.dtype)
    action_max_xyz = action_max[:, :3].to(device=device, dtype=pred_norm_xyz.dtype)
    pred_raw_xyz = (pred_norm_xyz + 1.0) / 2.0 * (
        action_max_xyz[:, None, :] - action_min_xyz[:, None, :] + 1e-8
    ) + action_min_xyz[:, None, :]
    pred_raw_sum = pred_raw_xyz.sum(dim=1)
    global_motion_min_xyz = global_motion_min[:, :3].to(device=device, dtype=pred_raw_sum.dtype)
    global_motion_max_xyz = global_motion_max[:, :3].to(device=device, dtype=pred_raw_sum.dtype)
    negative_scale = global_motion_min_xyz.abs().clamp_min(1e-6)
    positive_scale = global_motion_max_xyz.abs().clamp_min(1e-6)
    directional_scale = torch.where(pred_raw_sum >= 0.0, positive_scale, negative_scale)
    pred_directional_sum = pred_raw_sum / directional_scale

    prompt_global_motion = prompt_global_motion.to(device=device, dtype=pred_directional_sum.dtype)
    axis_mask = prompt_global_motion_axis_mask.to(device=device).bool()
    valid_action_mask = action_mask[:, :horizon, :3].to(device=device).bool().all(dim=(1, 2))
    valid_mask = prompt_global_motion_mask.to(device=device).bool() & valid_action_mask & axis_mask.any(dim=-1)
    sq_error = (pred_directional_sum - prompt_global_motion) ** 2
    axis_mask_f = axis_mask.to(dtype=sq_error.dtype)
    per_sample_loss = (sq_error * axis_mask_f).sum(dim=-1) / axis_mask_f.sum(dim=-1).clamp_min(1.0)
    loss = compute_masked_mean(per_sample_loss, valid_mask)
    return loss, valid_mask.sum()


def compute_wrist_aux_loss(
    pred_actions: torch.Tensor,
    prompt_local_motion: torch.Tensor,
    prompt_local_motion_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_actions.shape[-1] < 3:
        zero = pred_actions.new_zeros(())
        return zero, zero

    valid_action_mask = action_mask[:, 0, :3].all(dim=-1)
    valid_mask = prompt_local_motion_mask.bool() & valid_action_mask
    sq_error = (pred_actions[:, 0, :3].float() - prompt_local_motion.float()) ** 2
    per_sample_loss = sq_error.mean(dim=-1)
    loss = compute_masked_mean(per_sample_loss, valid_mask)
    return loss, valid_mask.sum()


def _project_pred_actions_to_prompt_pixels(
    pred_actions: torch.Tensor,
    prompt_tcp_world_pos: torch.Tensor,
    prompt_base_t_camera: torch.Tensor,
    prompt_camera_intrinsic: torch.Tensor,
    action_mask: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    image_size: int = 224,
    projection_source_width: int = 640,
    projection_source_height: int = 480,
    projection_processed_padding: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pred_actions.device
    dtype = torch.float32
    horizon = pred_actions.shape[1]
    pred_xyz = pred_actions[:, :, :3].to(dtype=dtype)
    amin = action_min[:, :3].to(device=device, dtype=dtype).unsqueeze(1)
    amax = action_max[:, :3].to(device=device, dtype=dtype).unsqueeze(1)
    pred_xyz_raw = (pred_xyz + 1.0) / 2.0 * (amax - amin) + amin

    batch_size = pred_actions.shape[0]
    current = prompt_tcp_world_pos.to(device=device, dtype=dtype).view(batch_size, 1, 3)
    if horizon == 1:
        points_base = current
        action_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
    else:
        future_points = current + torch.cumsum(pred_xyz_raw[:, : horizon - 1], dim=1)
        points_base = torch.cat([current, future_points], dim=1)
        step_valid = action_mask[:, : horizon - 1, :3].to(device=device).bool().all(dim=-1)
        action_valid = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool, device=device), torch.cumprod(step_valid.long(), dim=1).bool()],
            dim=1,
        )

    ones = torch.ones((*points_base.shape[:2], 1), dtype=dtype, device=device)
    points_h = torch.cat([points_base, ones], dim=-1)
    camera_t_base = torch.linalg.inv(prompt_base_t_camera.to(device=device, dtype=dtype))
    points_camera = torch.einsum("bij,bhj->bhi", camera_t_base, points_h)[..., :3]
    z = points_camera[..., 2]
    uvw = torch.einsum("bij,bhj->bhi", prompt_camera_intrinsic.to(device=device, dtype=dtype), points_camera)
    pixels_xy = uvw[..., :2] / z.clamp_min(1e-6).unsqueeze(-1)

    # Match scripts/labeling/traj_label_rw.py: raw camera pixels are mapped through
    # Resize(output + padding) followed by CenterCrop(output) before being saved.
    resized_width = float(image_size + projection_processed_padding)
    resized_height = float(image_size + projection_processed_padding)
    crop_x = (resized_width - float(image_size)) / 2.0
    crop_y = (resized_height - float(image_size)) / 2.0
    pixels_xy = pixels_xy.clone()
    pixels_xy[..., 0] = pixels_xy[..., 0] * resized_width / float(projection_source_width) - crop_x
    pixels_xy[..., 1] = pixels_xy[..., 1] * resized_height / float(projection_source_height) - crop_y
    pred_pixels_hw = torch.stack([pixels_xy[..., 1], pixels_xy[..., 0]], dim=-1)
    pred_valid = action_valid & torch.isfinite(pred_pixels_hw).all(dim=-1) & (z > 1e-6)
    return pred_pixels_hw, pred_valid, z


def compute_traj_aux_loss(
    pred_actions: torch.Tensor,
    prompt_traj_pixels: torch.Tensor,
    prompt_traj_mask: torch.Tensor,
    prompt_tcp_world_pos: torch.Tensor,
    prompt_base_t_camera: torch.Tensor,
    prompt_camera_intrinsic: torch.Tensor,
    action_mask: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    image_size: int = 224,
    projection_source_width: int = 640,
    projection_source_height: int = 480,
    projection_processed_padding: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_actions.shape[-1] < 3:
        zero = pred_actions.new_zeros(())
        return zero, zero

    device = pred_actions.device
    dtype = torch.float32
    horizon = min(pred_actions.shape[1], prompt_traj_pixels.shape[1])
    if horizon <= 0:
        zero = pred_actions.new_zeros(())
        return zero, zero

    pred_pixels_hw, pred_valid, _ = _project_pred_actions_to_prompt_pixels(
        pred_actions=pred_actions[:, :horizon],
        prompt_tcp_world_pos=prompt_tcp_world_pos,
        prompt_base_t_camera=prompt_base_t_camera,
        prompt_camera_intrinsic=prompt_camera_intrinsic,
        action_mask=action_mask[:, :horizon],
        action_min=action_min,
        action_max=action_max,
        image_size=image_size,
        projection_source_width=projection_source_width,
        projection_source_height=projection_source_height,
        projection_processed_padding=projection_processed_padding,
    )

    target_pixels_hw = prompt_traj_pixels[:, :horizon].to(device=device, dtype=dtype)
    target_valid = (
        prompt_traj_mask[:, :horizon].to(device=device).bool()
        & torch.isfinite(target_pixels_hw).all(dim=-1)
    )
    image_limit = float(image_size)
    pred_inside = (
        pred_valid
        & (pred_pixels_hw[..., 0] >= 0.0)
        & (pred_pixels_hw[..., 0] < image_limit)
        & (pred_pixels_hw[..., 1] >= 0.0)
        & (pred_pixels_hw[..., 1] < image_limit)
    )
    target_inside = (
        target_valid
        & (target_pixels_hw[..., 0] >= 0.0)
        & (target_pixels_hw[..., 0] < image_limit)
        & (target_pixels_hw[..., 1] >= 0.0)
        & (target_pixels_hw[..., 1] < image_limit)
    )
    sample_valid = (pred_inside.sum(dim=1) >= 1) & (target_inside.sum(dim=1) >= 1)
    if not sample_valid.any():
        zero = pred_actions.new_zeros(())
        return zero, sample_valid.sum()

    scale = pred_pixels_hw.new_tensor([float(image_size), float(image_size)]).clamp_min(1.0)
    per_sample_losses = []
    for i in torch.where(sample_valid)[0].tolist():
        pred_points = pred_pixels_hw[i, pred_inside[i]]
        target_points = target_pixels_hw[i, target_inside[i]]

        if pred_points.shape[0] >= 2 and target_points.shape[0] >= 2:
            pred_norm = (pred_points - pred_points[:1]) / scale
            target_norm = (target_points - target_points[:1]) / scale
        else:
            pred_norm = pred_points / scale
            target_norm = target_points / scale
        dists = torch.cdist(pred_norm, target_norm, p=2).square()
        per_sample_losses.append(0.5 * (dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()))

    loss = torch.stack(per_sample_losses).mean()
    return loss, sample_valid.sum()


def compute_point_aux_loss(
    pred_actions: torch.Tensor,
    prompt_point: torch.Tensor,
    prompt_point_mask: torch.Tensor,
    prompt_tcp_world_pos: torch.Tensor,
    prompt_base_t_camera: torch.Tensor,
    prompt_camera_intrinsic: torch.Tensor,
    action_mask: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    image_size: int = 224,
    projection_source_width: int = 640,
    projection_source_height: int = 480,
    projection_processed_padding: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_actions.shape[-1] < 3 or pred_actions.shape[1] <= 0:
        zero = pred_actions.new_zeros(())
        return zero, zero

    device = pred_actions.device
    dtype = torch.float32
    pred_pixels_hw, pred_valid, _ = _project_pred_actions_to_prompt_pixels(
        pred_actions=pred_actions,
        prompt_tcp_world_pos=prompt_tcp_world_pos,
        prompt_base_t_camera=prompt_base_t_camera,
        prompt_camera_intrinsic=prompt_camera_intrinsic,
        action_mask=action_mask,
        action_min=action_min,
        action_max=action_max,
        image_size=image_size,
        projection_source_width=projection_source_width,
        projection_source_height=projection_source_height,
        projection_processed_padding=projection_processed_padding,
    )
    pred_point_hw = pred_pixels_hw[:, -1]
    target_point_hw = prompt_point.to(device=device, dtype=dtype)
    image_limit = float(image_size)
    valid = (
        prompt_point_mask.to(device=device).bool()
        & pred_valid[:, -1]
        & torch.isfinite(target_point_hw).all(dim=-1)
        & (pred_point_hw[:, 0] >= 0.0)
        & (pred_point_hw[:, 0] < image_limit)
        & (pred_point_hw[:, 1] >= 0.0)
        & (pred_point_hw[:, 1] < image_limit)
        & (target_point_hw[:, 0] >= 0.0)
        & (target_point_hw[:, 0] < image_limit)
        & (target_point_hw[:, 1] >= 0.0)
        & (target_point_hw[:, 1] < image_limit)
    )
    if not valid.any():
        zero = pred_actions.new_zeros(())
        return zero, valid.sum()

    scale = pred_point_hw.new_tensor([float(image_size), float(image_size)]).clamp_min(1.0)
    per_sample_loss = ((pred_point_hw - target_point_hw) / scale).square().mean(dim=-1)
    loss = per_sample_loss[valid].mean()
    return loss, valid.sum()


def metric_to_float(value):
    if torch.is_tensor(value):
        return value.detach().float().item()
    return float(value)


def log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator, metrics=None):
    metrics = metrics or {}
    current_epoch = step / len(dataloader)
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        aux_parts = ""
        if "aux_wrist_loss" in metrics and metric_to_float(metrics.get("wrist_valid_count", 0.0)) > 0:
            wrist = metric_to_float(metrics["aux_wrist_loss"])
            aux_parts += f" | wrist: {wrist:.4f}"
        if "aux_primitive_loss" in metrics and metric_to_float(metrics.get("primitive_valid_count", 0.0)) > 0:
            prim = metric_to_float(metrics["aux_primitive_loss"])
            aux_parts += f" | primitive: {prim:.4f}"
        if "aux_traj_loss" in metrics and metric_to_float(metrics.get("traj_valid_count", 0.0)) > 0:
            traj = metric_to_float(metrics["aux_traj_loss"])
            aux_parts += f" | traj: {traj:.4f}"
        if "aux_point_loss" in metrics and metric_to_float(metrics.get("point_valid_count", 0.0)) > 0:
            point = metric_to_float(metrics["aux_point_loss"])
            aux_parts += f" | point: {point:.4f}"
        if (
            "aux_action_smoothness_loss" in metrics
            and metric_to_float(metrics.get("action_smoothness_valid_count", 0.0)) > 0
        ):
            smoothness = metric_to_float(metrics["aux_action_smoothness_loss"])
            aux_parts += f" | action_smoothness: {smoothness:.4f}"
        logging.info(f"[Step {step}] Loss: {loss.item():.4f}{aux_parts}")
        log_payload = {
            "step": step,
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        valid_count_by_prefix = {
            "wrist": metric_to_float(metrics.get("wrist_valid_count", 0.0)),
            "aux_wrist": metric_to_float(metrics.get("wrist_valid_count", 0.0)),
            "primitive": metric_to_float(metrics.get("primitive_valid_count", 0.0)),
            "aux_primitive": metric_to_float(metrics.get("primitive_valid_count", 0.0)),
            "traj": metric_to_float(metrics.get("traj_valid_count", 0.0)),
            "aux_traj": metric_to_float(metrics.get("traj_valid_count", 0.0)),
            "point": metric_to_float(metrics.get("point_valid_count", 0.0)),
            "aux_point": metric_to_float(metrics.get("point_valid_count", 0.0)),
            "action_smoothness": metric_to_float(
                metrics.get("action_smoothness_valid_count", 0.0)
            ),
            "aux_action_smoothness": metric_to_float(
                metrics.get("action_smoothness_valid_count", 0.0)
            ),
        }
        for key, value in metrics.items():
            prefix = key.removesuffix("_loss")
            if key.endswith("_loss") and prefix in valid_count_by_prefix and valid_count_by_prefix[prefix] <= 0:
                continue
            if torch.is_tensor(value):
                value = metric_to_float(value)
            log_payload[key] = value
        wandb.log(log_payload)

def save_checkpoint(save_dir, step, model_engine, loss, accelerator, config=None, norm_stats=None):
    tag = f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, tag)

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} exists. Removing before overwrite.")
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()

    if hasattr(model_engine, 'save_checkpoint'):
        # DeepSpeed path
        client_state = {
            "step": step,
            "best_loss": loss if isinstance(loss, float) else loss.item(),
            "config": config,
        } if accelerator.is_main_process else {}
        model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
        checkpoint_meta = {"type": "ds_model", "version": 0.0, "checkpoints": "mp_rank_00_model_states.pt"}
        if accelerator.is_main_process:
            if config is not None:
                with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)
            if norm_stats is not None:
                with open(os.path.join(checkpoint_dir, "norm_stats.json"), "w") as f:
                    json.dump(norm_stats, f, indent=2)
    else:
        # Standard / DDP path
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(accelerator.unwrap_model(model_engine).state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
            if config is not None:
                with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)
            if norm_stats is not None:
                with open(os.path.join(checkpoint_dir, "norm_stats.json"), "w") as f:
                    json.dump(norm_stats, f, indent=2)
            logging.info(f"Saved checkpoint to {checkpoint_dir}")
        checkpoint_meta = {"type": "pytorch_model", "version": 0.0, "checkpoints": "pytorch_model.bin"}

    if accelerator.is_main_process:
        if config is not None:
            with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
        if norm_stats is not None:
            with open(os.path.join(checkpoint_dir, "norm_stats.json"), "w") as f:
                json.dump(norm_stats, f, indent=2)
        with open(os.path.join(checkpoint_dir, "checkpoint.json"), "w") as f:
            json.dump(checkpoint_meta, f, indent=2)

def load_checkpoint_with_deepspeed(model_engine, load_dir, accelerator, tag="step_best", load_optimizer_states=True, resume_pretrain=False):

    try:
        load_path, client_state = model_engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=load_optimizer_states and not resume_pretrain,
            load_lr_scheduler_states=load_optimizer_states and not resume_pretrain
        )
        if accelerator.is_main_process:
            logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (including optimizer states)")
        return client_state.get("step", 0), client_state
        
    except Exception as e:
        if accelerator.is_main_process:
            logging.warning(f"World size mismatch detected: {str(e)}")
            logging.warning("Attempting to load only model weights (skipping optimizer states)...")
        try:
            load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=False,
                load_lr_scheduler_states=False
            )
            if accelerator.is_main_process:
                logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (model weights only)")
            return client_state.get("step", 0), client_state
            
        except Exception as e2:
            if accelerator.is_main_process:
                logging.error(f"Failed to load checkpoint even without optimizer states: {str(e2)}")
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {load_dir} with tag {tag}: {str(e2)}")

    

def get_and_clip_grad_norm(accelerator, model, loss, max_norm: float = 1.0):

    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):
       
        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:
 
        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(grad_norms) == 0:
            total_norm = torch.tensor(0.0, device=loss.device)
        else:
            total_norm = torch.norm(torch.stack(grad_norms), 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        clipped_grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(clipped_grad_norms) == 0:
            clipped_norm = torch.tensor(0.0, device=loss.device)
        else:
            clipped_norm = torch.norm(torch.stack(clipped_grad_norms), 2)

    return total_norm, clipped_norm

def build_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith("bias") or ".bias" in n
        is_norm = (p.dim() == 1) or ("norm" in n.lower())
        (no_decay if is_bias or is_norm else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def module_param_norm(module: nn.Module, trainable_only: bool = False) -> float:
    total = None
    for p in module.parameters():
        if trainable_only and not p.requires_grad:
            continue
        value = p.detach().float().norm(2).pow(2)
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return torch.sqrt(total).item()


@torch.no_grad()
def module_grad_norm(module: nn.Module) -> float:
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        value = p.grad.detach().float().norm(2).pow(2)
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return torch.sqrt(total).item()

def train(config):


    # === Set logging ===
    save_dir = get_with_warning(config, "save_dir", "checkpoints")
    log_path = setup_logging(save_dir)
    
    # === WandB ===
    init_wandb(config, accelerator)

    # === Debug mode ===
    if get_with_warning(config, "debug", False):
        torch.autograd.set_detect_anomaly(True)

    # === Dataset ===
    # Only rank 0 builds the cache; others wait then reuse it
    if accelerator.is_main_process:
        dataset = prepare_dataset(config)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = prepare_dataset(config)

    # === DataLoader ===
    dataloader = prepare_dataloader(dataset, config)

    # === Model ===
    model = EVO1(config)
    model.train()
    model.set_finetune_flags()

    lr = get_with_warning(config, "lr", 1e-5)
    wd = get_with_warning(config, "weight_decay", 1e-5)
    optimizer = AdamW(build_param_groups(model, wd), lr=lr)
    if accelerator.is_main_process:
        logging.info(f"Optimizer=AdamW, lr={lr}, weight_decay={wd}")


    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model
  
    if accelerator.is_main_process:
        logging.info("Initialized with Accelerate")
    
    
    # === Warmup + Cosine Scheduler ===
    max_steps = get_with_warning(config, "max_steps", 1000)
    warmup_steps = get_with_warning(config, "warmup_steps", 300)
    
    # === Checkpoint and save path setup ===
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_checkpoint.pt")
    best_loss = float("inf")
    
    # === Logging and interval settings ===
    log_interval = get_with_warning(config, "log_interval", 100)
    ckpt_interval = get_with_warning(config, "ckpt_interval", 1000)
    max_norm = get_with_warning(config, "grad_clip_norm", 1.0)
    wrist_loss_weight = get_with_warning(config, "wrist_loss_weight", 0.0)
    primitive_loss_weight = get_with_warning(config, "primitive_loss_weight", 0.0)
    traj_loss_weight = get_with_warning(config, "traj_loss_weight", 0.0)
    point_loss_weight = get_with_warning(config, "point_loss_weight", 0.0)
    action_smoothness_weight = get_with_warning(config, "action_smoothness_weight", 0.0)
    action_smoothness_dims = get_with_warning(config, "action_smoothness_dims", 3)
    action_smoothness_min_norm = get_with_warning(config, "action_smoothness_min_norm", 1e-4)
    # if point_loss_weight is None:
    #     point_loss_weight = traj_loss_weight

    # === Resume training from checkpoint ===
    resume = get_with_warning(config, "resume", False)
    resume_path = get_with_warning(config, "resume_path", None)
    resume_pretrain = get_with_warning(config, "resume_pretrain", False)

    if resume != bool(resume_path):
        raise ValueError("Inconsistent resume configuration: --resume and --resume_path must be set together.")
    
    if resume:
        resume_path = resume_path.rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)

        step, client_state = load_checkpoint_with_deepspeed(
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            load_optimizer_states=True,  
            resume_pretrain=resume_pretrain
        )
        best_loss = client_state.get("best_loss", float("inf"))
        if accelerator.is_main_process:
            logging.info(f"Resuming from {resume_dir}/{resume_tag}, step {step}")
    else:
        step = 0
        if accelerator.is_main_process:
            logging.info("Starting fresh training")

    if resume_pretrain:
        step = 0
        logging.info("Resuming pretraining from scratch, resetting step to 0")

    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=step))


    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        inspect_named_submodules({
            "vision_model": unwrapped.embedder.model.vision_model,
            "language_model": unwrapped.embedder.model.language_model,
            "action_head": unwrapped.action_head
        })

    # === Training Loop ===
    with tqdm(total=max_steps, initial=step, desc="Training", disable=not accelerator.is_main_process) as pbar:
        while step < max_steps:
            for batch in dataloader:
                if step >= max_steps:
                    break
                prompts = batch["prompts"]
                images_batch = batch["images"]
                image_masks = batch["image_masks"]
                states = batch["states"].to(dtype=torch.bfloat16)
                actions_gt = batch["actions"].to(dtype=torch.bfloat16)
                action_mask = batch["action_mask"]
                prompt_global_motion = batch["prompt_global_motion"]
                prompt_global_motion_axis_mask = batch["prompt_global_motion_axis_mask"]
                prompt_global_motion_mask = batch["prompt_global_motion_mask"]
                prompt_local_motion = batch["prompt_local_motion"]
                prompt_local_motion_mask = batch["prompt_local_motion_mask"]
                prompt_traj_pixels = batch["prompt_traj_pixels"]
                prompt_traj_mask = batch["prompt_traj_mask"]
                prompt_point = batch["prompt_point"]
                prompt_point_mask = batch["prompt_point_mask"]
                prompt_tcp_world_pos = batch["prompt_tcp_world_pos"]
                prompt_base_t_camera = batch["prompt_base_t_camera"]
                prompt_camera_intrinsic = batch["prompt_camera_intrinsic"]
                state_mask = batch["state_mask"]
                embodiment_ids = batch["embodiment_ids"]
                action_min = batch["action_min"].to(dtype=torch.float32)
                action_max = batch["action_max"].to(dtype=torch.float32)
                global_motion_min = batch["global_motion_min"].to(dtype=torch.float32)
                global_motion_max = batch["global_motion_max"].to(dtype=torch.float32)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_velocity, noise, noisy_actions, timesteps, fused_tokens = model(
                        images_batch=images_batch,
                        image_masks=image_masks,
                        prompts=prompts,
                        state=states,
                        actions_gt=actions_gt,
                        action_mask=action_mask,
                        return_fused_tokens=True,
                    )
                    
                target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
                
                assert pred_velocity.shape == target_velocity.shape
    
                if action_mask.sum() == 0:
                    raise ValueError(f"[Step {step}] action_mask.sum() is 0! All actions are masked. "
                                f"This indicates a problem with the data or mask generation. "
                                f"action_mask shape: {action_mask.shape}, "
                                f"action_mask: {action_mask}")
                
    
                action_mask_3d = action_mask
                loss, flow_matching_valid = compute_flow_matching_loss(
                    pred_velocity=pred_velocity,
                    target_velocity=target_velocity,
                    action_mask=action_mask_3d,
                )
                flow_matching_loss = loss
    
                timestep_scale = (1.0 - timesteps.float()).view(actions_gt.shape[0], *([1] * (actions_gt.ndim - 1)))
                pred_actions = noisy_actions.float() + timestep_scale * pred_velocity.view_as(actions_gt).float()

                if float(action_smoothness_weight) > 0.0:
                    action_smoothness_loss, action_smoothness_valid = compute_action_smoothness_loss(
                        pred_actions=pred_actions,
                        action_mask=action_mask_3d,
                        action_min=action_min,
                        action_max=action_max,
                        num_action_dims=action_smoothness_dims,
                        min_action_norm=action_smoothness_min_norm,
                    )
                    loss = loss + float(action_smoothness_weight) * action_smoothness_loss
                else:
                    action_smoothness_loss = pred_actions.new_zeros(())
                    action_smoothness_valid = 0
    
                # Only compute wrist aux when local mask is True for at least one sample
                # if prompt_local_motion_mask.any():
                #     wrist_loss, wrist_valid = compute_wrist_aux_loss(
                #         pred_actions=pred_actions,
                #         prompt_local_motion=prompt_local_motion,
                #         prompt_local_motion_mask=prompt_local_motion_mask,
                #         action_mask=action_mask_3d,
                #     )
                #     loss = loss + float(wrist_loss_weight) * wrist_loss
                # else:
                #     wrist_loss = pred_actions.new_zeros(())
                #     wrist_valid = 0
    
                # # Only compute primitive aux when global mask is True for at least one sample
                if prompt_global_motion_mask.any():
                    primitive_loss, primitive_valid = compute_primitive_aux_loss(
                        pred_actions=pred_actions,
                        prompt_global_motion=prompt_global_motion,
                        prompt_global_motion_mask=prompt_global_motion_mask,
                        prompt_global_motion_axis_mask=prompt_global_motion_axis_mask,
                        action_mask=action_mask_3d,
                        action_min=action_min,
                        action_max=action_max,
                        global_motion_min=global_motion_min,
                        global_motion_max=global_motion_max,
                    )
                    loss = loss + float(primitive_loss_weight) * primitive_loss
                else:
                    primitive_loss = pred_actions.new_zeros(())
                    primitive_valid = 0

                if prompt_traj_mask.any():
                    traj_loss, traj_valid = compute_traj_aux_loss(
                        pred_actions=pred_actions,
                        prompt_traj_pixels=prompt_traj_pixels,
                        prompt_traj_mask=prompt_traj_mask,
                        prompt_tcp_world_pos=prompt_tcp_world_pos,
                        prompt_base_t_camera=prompt_base_t_camera,
                        prompt_camera_intrinsic=prompt_camera_intrinsic,
                        action_mask=action_mask_3d,
                        action_min=action_min,
                        action_max=action_max,
                        image_size=get_with_warning(config, "image_size", 224),
                    )
                    loss = loss + float(traj_loss_weight) * traj_loss
                else:
                    traj_loss = pred_actions.new_zeros(())
                    traj_valid = 0

                if prompt_point_mask.any():
                    point_loss, point_valid = compute_point_aux_loss(
                        pred_actions=pred_actions,
                        prompt_point=prompt_point,
                        prompt_point_mask=prompt_point_mask,
                        prompt_tcp_world_pos=prompt_tcp_world_pos,
                        prompt_base_t_camera=prompt_base_t_camera,
                        prompt_camera_intrinsic=prompt_camera_intrinsic,
                        action_mask=action_mask_3d,
                        action_min=action_min,
                        action_max=action_max,
                        image_size=get_with_warning(config, "image_size", 224),
                    )
                    loss = loss + float(point_loss_weight) * point_loss
                else:
                    point_loss = pred_actions.new_zeros(())
                    point_valid = 0
    
                aux_metrics = {
                    "flow_matching_loss": flow_matching_loss,
                    # "wrist_loss": wrist_loss,
                    "primitive_loss": primitive_loss,
                    # "aux_wrist_loss": wrist_loss * float(wrist_loss_weight),
                    "aux_primitive_loss": primitive_loss * float(primitive_loss_weight),
                    "primitive_valid_count": primitive_valid,
                    "traj_loss": traj_loss,
                    "aux_traj_loss": traj_loss * float(traj_loss_weight),
                    "traj_valid_count": traj_valid,
                    "point_loss": point_loss,
                    "aux_point_loss": point_loss * float(point_loss_weight),
                    "point_valid_count": point_valid,
                    "action_smoothness_loss": action_smoothness_loss,
                    "aux_action_smoothness_loss": (
                        action_smoothness_loss * float(action_smoothness_weight)
                    ),
                    "action_smoothness_valid_count": action_smoothness_valid,
                    "flow_matching_valid": flow_matching_valid,
                }
                
                # === NaN/Inf check ===
                if not check_numerical_stability(
                    step,
                    states=states,
                    actions_gt=actions_gt,
                    fused_tokens=fused_tokens,
                    pred_velocity=pred_velocity,
                    pred_actions=pred_actions,
                    # wrist_loss=wrist_loss,
                    primitive_loss=primitive_loss,
                    traj_loss=traj_loss,
                    point_loss=point_loss,
                    action_smoothness_loss=action_smoothness_loss,
                    loss=loss
                ):
                    continue
    
                # === Backward and optimizer step ===
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
    
                # === Clip grad norm ===
                total_norm, clipped_norm = get_and_clip_grad_norm(accelerator, model, loss, max_norm)
    
                optimizer.step()
                scheduler.step()
    
                # === Logging ===
                if step % log_interval == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    aux_metrics["total_grad_norm"] = total_norm
                    aux_metrics["clipped_grad_norm"] = clipped_norm
                    aux_metrics["vlm_param_norm"] = module_param_norm(unwrapped.embedder, trainable_only=False)
                    aux_metrics["action_param_norm"] = module_param_norm(unwrapped.action_head, trainable_only=False)
                    aux_metrics["vlm_trainable_param_norm"] = module_param_norm(unwrapped.embedder, trainable_only=True)
                    aux_metrics["action_trainable_param_norm"] = module_param_norm(unwrapped.action_head, trainable_only=True)
                    aux_metrics["vlm_grad_norm"] = module_grad_norm(unwrapped.embedder)
                    aux_metrics["action_grad_norm"] = module_grad_norm(unwrapped.action_head)
                    log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator, aux_metrics)
       
                # === Save best checkpoint ===
                loss_value = loss.item()
                if accelerator.is_main_process:
                    is_best = loss_value < best_loss
                    if is_best:
                        best_loss = loss_value
                    is_best_tensor = torch.tensor(int(is_best), device=accelerator.device)
                else:
                    is_best_tensor = torch.tensor(0, device=accelerator.device)
                
                if accelerator.distributed_type != DistributedType.NO:
                    torch.distributed.broadcast(is_best_tensor, src=0)
                
                if is_best_tensor.item() == 1 and step > 1000:
                    accelerator.print("start to save best checkpoint")
                    save_checkpoint(
                        save_dir,
                        step="best",
                        model_engine=model_engine,
                        loss=loss,
                        accelerator=accelerator,
                        config=config,
                        norm_stats=dataset.arm2stats_dict 
                    )
                    accelerator.print("end to save best checkpoint")
                    if accelerator.is_main_process:
                        logging.info(f"Saved best checkpoint at step {step} with loss {loss_value:.6f}")
    
                step += 1
    
                pbar.update(1)
                # === Save periodic checkpoint ===
                if step % ckpt_interval == 0 and step > 0:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
                    save_checkpoint(save_dir, step=step, model_engine=model_engine, loss=loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict)
         
    # === Save final model ===
    save_checkpoint(save_dir, step="final", model_engine=model_engine, loss=loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict)
    logging.info(f"Final model saved to step_final/")
    logging.info(f"Best checkpoint saved to step_best/ with loss {best_loss:.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Evo-1")

    # Basic config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--vlm_name", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--action_head", type=str, default="flowmatching", choices=["flowmatching"])
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="lerobot")
    parser.add_argument("--data_paths", type=str, required=False)
    parser.add_argument("--dataset_config_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--cmd_type", type=str, default="sigma", choices=["sigma", "state"])
    parser.add_argument("--binarize_gripper", action="store_true", default=False, help="Whether to binarize gripper state/action (default: False).")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation on images")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--wrist_loss_weight", type=float, default=0.0)
    parser.add_argument("--primitive_loss_weight", type=float, default=0.0)
    parser.add_argument("--traj_loss_weight", type=float, default=0.0)
    parser.add_argument("--point_loss_weight", type=float, default=0.0)
    parser.add_argument("--action_smoothness_weight", type=float, default=0.0)
    parser.add_argument("--action_smoothness_dims", type=int, default=3)
    parser.add_argument("--action_smoothness_min_norm", type=float, default=1e-4)


    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # Resume
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")
   

    # Finetuning
    parser.add_argument("--finetune_vlm", action="store_true")
    parser.add_argument("--finetune_action_head", action="store_true")

    # Misc
    parser.add_argument("--per_action_dim", type=int, default=7)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    # dropout
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    config = vars(args)

    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)
