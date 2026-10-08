#!/usr/bin/env python3
# usage: 
# python scripts/realworld/eval/evo1/eval_dataset.py --ckpt realworld_ckpt/evo1_stage1_30000/ 
# for single episode eval: 
# python scripts/realworld/eval/evo1/eval_dataset.py --ckpt realworld_ckpt/evo1_stage1_30000/ \
# --dataset-path output/evo1_stage1/20260602_153930
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "Evo-1").is_dir() and (path / "openpi").is_dir():
            return path
    raise FileNotFoundError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVO1_ROOT = REPO_ROOT / "Evo-1"
if str(EVO1_ROOT) not in sys.path:
    sys.path.insert(0, str(EVO1_ROOT))
EVO1_SCRIPTS_ROOT = EVO1_ROOT / "scripts"
if str(EVO1_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EVO1_SCRIPTS_ROOT))

from scripts.utils.realworld_paths import normalize_dataset_config_paths, resolve_repo_local_path  # noqa: E402
from scripts.utils.realworld_projection import draw_projected_action_horizon, project_action_horizon_to_pixels  # noqa: E402
from scripts.utils.vis import load_mono_font, tensor_image_to_uint8, wrap_text  # noqa: E402
from dataset.lerobot_dataset_rp import LeRobotDatasetRP  # noqa: E402
from Evo1 import EVO1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Evo-1 action jitter on a LeRobot training dataset.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Evo-1 checkpoint dir, e.g. outputs/stage2_YYYYMMDD/step_final.")
    parser.add_argument("--dataset-config-path", type=Path, default=None, help="Override dataset YAML. Defaults to ckpt config.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Evaluate one LeRobot dataset root while reusing the ckpt dataset config template.")
    parser.add_argument("--dataset-arm", default=None, help="Arm group name for --dataset-path. Defaults to the first arm group in the template config.")
    parser.add_argument("--dataset-name", default=None, help="Dataset name for --dataset-path. Defaults to the dataset directory name.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override dataset root. Defaults to $RP_DATA_ROOT.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override processed sample cache dir.")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples-per-file", type=int, default=None)
    parser.add_argument("--num-inference-timesteps", type=int, default=32)
    parser.add_argument("--num-predictions-per-sample", type=int, default=1, help="Repeat stochastic FM inference per observation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use bfloat16 autocast on CUDA.")
    parser.add_argument("--overlay-output-dir", type=Path, default=None, help="Directory for GT/pred action overlay frames.")
    parser.add_argument("--overlay-source-width", type=int, default=640, help="Original camera image width for projection mapping.")
    parser.add_argument("--overlay-source-height", type=int, default=480, help="Original camera image height for projection mapping.")
    parser.add_argument("--overlay-processed-padding", type=int, default=8, help="Padding used before center-crop/resize in projection mapping.")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_state_dict(ckpt_dir: Path) -> dict[str, Any]:
    checkpoint_file = ckpt_dir / "mp_rank_00_model_states.pt"
    meta_path = ckpt_dir / "checkpoint.json"
    if meta_path.is_file():
        meta = load_json(meta_path)
        checkpoint_file = ckpt_dir / str(meta.get("checkpoints", checkpoint_file.name))
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    return checkpoint.get("module", checkpoint)


def load_ckpt_norm_stats(ckpt_dir: Path) -> dict[str, Any]:
    stats_path = ckpt_dir / "norm_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing Evo-1 normalization stats: {stats_path}")
    stats = load_json(stats_path)
    if len(stats) != 1:
        raise ValueError(f"Evo-1 norm_stats.json should contain one robot key, got {list(stats.keys())}.")
    robot_stats = next(iter(stats.values()))
    for key in ("observation.state", "action"):
        if key not in robot_stats or "min" not in robot_stats[key] or "max" not in robot_stats[key]:
            raise ValueError(f"Invalid Evo-1 norm_stats.json: missing {key}.min/max")
    return robot_stats


def apply_ckpt_norm_stats(dataset: LeRobotDatasetRP, ckpt_norm_stats: dict[str, Any]) -> None:
    # Keep rollout/dataset raw samples, but match the online Evo-1 server's state/action scaling.
    for arm_name in dataset.arm2stats_dict:
        dataset.arm2stats_dict[arm_name] = ckpt_norm_stats


def select_dataset_template(dataset_config: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    data_groups = dataset_config.get("data_groups")
    if not isinstance(data_groups, dict) or not data_groups:
        raise ValueError("Dataset config must contain at least one data_groups entry for --dataset-path.")

    for arm_name, arm_config in data_groups.items():
        if not isinstance(arm_config, dict):
            continue
        for dataset_name, single_dataset_config in arm_config.items():
            if isinstance(single_dataset_config, dict):
                return str(arm_name), str(dataset_name), dict(single_dataset_config)
    raise ValueError("Dataset config has no valid dataset template under data_groups.")


def apply_dataset_path_override(dataset_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_path is None:
        return dataset_config

    template_arm, template_name, template = select_dataset_template(dataset_config)
    dataset_path = resolve_repo_local_path(
        args.dataset_path,
        repo_root=REPO_ROOT,
        fallback_roots=(EVO1_ROOT,),
        must_exist=True,
    )
    arm_name = args.dataset_arm or template_arm
    dataset_name = args.dataset_name or dataset_path.name or template_name
    template["path"] = str(dataset_path)
    dataset_config = dict(dataset_config)
    dataset_config["data_groups"] = {arm_name: {dataset_name: template}}
    return dataset_config


def make_dataset(config: dict[str, Any], args: argparse.Namespace) -> LeRobotDatasetRP:
    dataset_config_path = resolve_repo_local_path(
        args.dataset_config_path or config["dataset_config_path"],
        repo_root=REPO_ROOT,
        fallback_roots=(EVO1_ROOT,),
        must_exist=True,
    )
    cache_dir = args.cache_dir or config.get("cache_dir")
    if cache_dir is not None:
        cache_dir = resolve_repo_local_path(
            cache_dir,
            repo_root=REPO_ROOT,
            fallback_roots=(EVO1_ROOT,),
            must_exist=False,
        )
    dataset_config = load_yaml(dataset_config_path)
    dataset_config = apply_dataset_path_override(dataset_config, args)
    normalize_dataset_config_paths(dataset_config, args.data_root)
    return LeRobotDatasetRP(
        config=dataset_config,
        action_horizon=int(config["horizon"]),
        image_size=int(config.get("image_size", 224)),
        max_samples_per_file=args.max_samples_per_file,
        binarize_gripper=bool(config.get("binarize_gripper", False)),
        use_augmentation=False,
        cache_dir=cache_dir,
    )


def denormalize_actions(actions: torch.Tensor, action_min: torch.Tensor, action_max: torch.Tensor) -> torch.Tensor:
    action_min = action_min.to(device=actions.device, dtype=actions.dtype)
    action_max = action_max.to(device=actions.device, dtype=actions.dtype)
    return (actions + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


def select_valid_action_dims(actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    valid_dims = action_mask.bool()
    if valid_dims.ndim > 1:
        valid_dims = valid_dims[0]
    if actions.shape[-1] == valid_dims.numel():
        return actions[..., valid_dims]
    return actions


def trim_action_chunk_to_visual_prompt_points(action_chunk: torch.Tensor) -> torch.Tensor:
    if action_chunk.shape[0] <= 1:
        return action_chunk
    return action_chunk[:-1]


def save_action_overlay_frame(
    *,
    item: dict[str, Any],
    record: dict[str, Any],
    gt_raw: torch.Tensor,
    pred_raw: torch.Tensor,
    output_dir: Path,
    source_width: int | None,
    source_height: int | None,
    processed_padding: int,
) -> str | None:
    image_mask = item["image_mask"]
    if image_mask.numel() == 0 or not bool(image_mask[0].item()):
        return None

    image = tensor_image_to_uint8(item["images"][0])
    projection_error = None
    projection_lines = []
    try:
        gt_overlay_raw = trim_action_chunk_to_visual_prompt_points(gt_raw)
        pred_overlay_raw = trim_action_chunk_to_visual_prompt_points(pred_raw)
        projection_kwargs = {
            "image_shape": image.shape,
            "current_tcp_position": item["prompt_tcp_world_pos"].detach().float().cpu().numpy(),
            "base_t_camera": item["prompt_base_t_camera"].detach().float().cpu().numpy(),
            "intrinsic": item["prompt_camera_intrinsic"].detach().float().cpu().numpy(),
            "source_width": source_width,
            "source_height": source_height,
            "processed_padding": processed_padding,
        }
        projection_lines.append(
            f"overlay deltas/points: {gt_overlay_raw.shape[0]}/{gt_overlay_raw.shape[0] + 1}"
        )
        for name, action_chunk in (("gt", gt_overlay_raw), ("pred", pred_overlay_raw)):
            points_hw, valid_mask = project_action_horizon_to_pixels(
                action_chunk=action_chunk.detach().float().cpu().numpy(),
                **projection_kwargs,
            )
            inside_mask = (
                valid_mask
                & (points_hw[:, 0] >= 0)
                & (points_hw[:, 0] < image.shape[0])
                & (points_hw[:, 1] >= 0)
                & (points_hw[:, 1] < image.shape[1])
            )
            projection_lines.append(
                f"{name}_proj valid/inside: {int(valid_mask.sum())}/{int(inside_mask.sum())}"
            )
        image = draw_projected_action_horizon(
            image,
            current_tcp_position=projection_kwargs["current_tcp_position"],
            action_chunk=gt_overlay_raw.detach().float().cpu().numpy(),
            base_t_camera=projection_kwargs["base_t_camera"],
            intrinsic=projection_kwargs["intrinsic"],
            source_width=source_width,
            source_height=source_height,
            processed_padding=processed_padding,
            color=(40, 220, 80),
        )
        image = draw_projected_action_horizon(
            image,
            current_tcp_position=projection_kwargs["current_tcp_position"],
            action_chunk=pred_overlay_raw.detach().float().cpu().numpy(),
            base_t_camera=projection_kwargs["base_t_camera"],
            intrinsic=projection_kwargs["intrinsic"],
            source_width=source_width,
            source_height=source_height,
            processed_padding=processed_padding,
            color=(0, 180, 255),
        )
    except Exception as exc:
        projection_error = str(exc)

    img = Image.fromarray(image)
    img_w, img_h = img.size
    panel_w = 380
    panel = Image.new("RGB", (panel_w, img_h), color=(28, 28, 28))
    draw = ImageDraw.Draw(panel)
    font = load_mono_font(10)
    line_h = 13
    lines = [
        "=== ACTION OVERLAY ===",
        "GT: green   PRED: cyan",
        f"idx: {record['index']}",
        f"mse_norm: {record['mse_norm']:.6f}",
        f"mse_raw_xyz: {record['mse_raw_xyz']:.6f}",
        f"raw_adj_ratio: {record['raw_adj_ratio']:.3f}",
        f"raw_second_ratio: {record['raw_second_ratio']:.3f}",
        f"pred/gt_adj: {record['pred_raw']['adj_abs_mean']:.6f}/{record['gt_raw']['adj_abs_mean']:.6f}",
        f"visual_prompt: {record['visual_prompt_applied']} {record['visual_prompt_type']}",
    ]
    lines.extend(projection_lines)
    if projection_error is not None:
        lines.extend(["=== PROJECTION ERROR ===", projection_error[:180]])
    lines.append("=== PROMPT ===")
    prompt_lines = wrap_text(draw, str(record["prompt"]), font, panel_w - 16)
    lines.extend(prompt_lines)

    y = 6
    for line in lines:
        if y + line_h > img_h:
            break
        draw.text((8, y), line, fill=(225, 225, 225), font=font)
        y += line_h

    total = Image.new("RGB", (img_w + panel_w, img_h))
    total.paste(img, (0, 0))
    total.paste(panel, (img_w, 0))
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    path = frame_dir / f"step_{int(record['index']):06d}.png"
    total.save(path)
    return str(path)


def diff_stats(actions: np.ndarray, dims: int = 3) -> dict[str, float]:
    arr = np.asarray(actions, dtype=np.float32)
    use = arr[:, : min(dims, arr.shape[-1])]
    adj = np.diff(use, axis=0)
    second = np.diff(use, n=2, axis=0) if len(use) >= 3 else np.zeros((0, use.shape[-1]), dtype=np.float32)
    return {
        "adj_abs_mean": float(np.mean(np.abs(adj))) if adj.size else 0.0,
        "adj_abs_max": float(np.max(np.abs(adj))) if adj.size else 0.0,
        "second_abs_mean": float(np.mean(np.abs(second))) if second.size else 0.0,
        "second_abs_max": float(np.max(np.abs(second))) if second.size else 0.0,
    }


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-8 else float("inf")


def compute_action_eval_tensors(
    item: dict[str, Any],
    predictions_norm: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action_min = item["action_min"].float()
    action_max = item["action_max"].float()
    gt_norm = select_valid_action_dims(item["action"].float(), item["action_mask"])
    pred_norm = select_valid_action_dims(torch.stack(predictions_norm, dim=0), item["action_mask"])
    gt_raw = denormalize_actions(gt_norm, action_min, action_max)
    pred_raw = denormalize_actions(pred_norm, action_min, action_max)
    # print(f"first 5 pred xyz raw per prediction: {pred_raw[:, :5, :3].numpy()}")
    return gt_norm, pred_norm, gt_raw, pred_raw


def predict_one(model: EVO1, item: dict[str, Any], device: torch.device, *, amp: bool) -> torch.Tensor:
    images = [img.to(device=device, dtype=torch.float32) for img in item["images"]]
    image_mask = item["image_mask"].to(device=device, dtype=torch.int32)
    state = item["state"].to(device=device, dtype=torch.float32).unsqueeze(0)
    action_mask = item["action_mask"][0].to(device=device, dtype=torch.int32).unsqueeze(0)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            action = model.run_inference(
                images=images,
                image_mask=image_mask,
                prompt=str(item["prompt"]),
                state_input=state,
                action_mask=action_mask,
            )
    return action.reshape(1, int(model.horizon), int(model.per_action_dim))[0].float().cpu()


def summarize_sample(
    *,
    index: int,
    item: dict[str, Any],
    predictions_norm: list[torch.Tensor],
) -> dict[str, Any]:
    gt_norm, pred_norm, gt_raw, pred_raw = compute_action_eval_tensors(item, predictions_norm)
    pred_mean_raw = pred_raw.mean(dim=0)
    pred_mean_norm = pred_norm.mean(dim=0)

    gt_raw_stats = diff_stats(gt_raw.numpy())
    pred_raw_stats = diff_stats(pred_mean_raw.numpy())
    gt_norm_stats = diff_stats(gt_norm.numpy())
    pred_norm_stats = diff_stats(pred_mean_norm.numpy())
    mse_norm = torch.mean((pred_mean_norm - gt_norm) ** 2).item()
    mse_raw_xyz = torch.mean((pred_mean_raw[:, :3] - gt_raw[:, :3]) ** 2).item()
    stochastic_std_raw = pred_raw[:, :, :3].std(dim=0).mean().item() if pred_raw.shape[0] > 1 else 0.0

    return {
        "index": index,
        "prompt": item["prompt"],
        "visual_prompt_applied": bool(item["visual_prompt_applied"].item()),
        "visual_prompt_type": str(item["visual_prompt_type"]),
        "mse_norm": float(mse_norm),
        "mse_raw_xyz": float(mse_raw_xyz),
        "stochastic_std_raw_xyz": float(stochastic_std_raw),
        "gt_raw": gt_raw_stats,
        "pred_raw": pred_raw_stats,
        "gt_norm": gt_norm_stats,
        "pred_norm": pred_norm_stats,
        "raw_adj_ratio": safe_ratio(pred_raw_stats["adj_abs_mean"], gt_raw_stats["adj_abs_mean"]),
        "raw_second_ratio": safe_ratio(pred_raw_stats["second_abs_mean"], gt_raw_stats["second_abs_mean"]),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("mse_norm", "mse_raw_xyz", "stochastic_std_raw_xyz", "raw_adj_ratio", "raw_second_ratio")
    out = {f"{key}_mean": float(np.mean([record[key] for record in records])) for key in keys}
    out.update({f"{key}_median": float(np.median([record[key] for record in records])) for key in keys})
    out["pred_raw_adj_abs_mean"] = float(np.mean([record["pred_raw"]["adj_abs_mean"] for record in records]))
    out["gt_raw_adj_abs_mean"] = float(np.mean([record["gt_raw"]["adj_abs_mean"] for record in records]))
    out["pred_raw_second_abs_mean"] = float(np.mean([record["pred_raw"]["second_abs_mean"] for record in records]))
    out["gt_raw_second_abs_mean"] = float(np.mean([record["gt_raw"]["second_abs_mean"] for record in records]))
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = args.ckpt.expanduser().resolve()
    overlay_output_dir = (
        args.overlay_output_dir.expanduser().resolve()
        if args.overlay_output_dir is not None
        else REPO_ROOT / "output" / "eval_dataset" / "evo" / ckpt_dir.name
    )
    print(f"overlay_output_dir {overlay_output_dir}")
    config = load_json(ckpt_dir / "config.json")
    config = dict(config)
    config["device"] = args.device
    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = args.num_inference_timesteps

    device = torch.device(args.device)
    model = EVO1(config).eval()
    model.load_state_dict(load_state_dict(ckpt_dir), strict=True)
    model = model.to(device)
    dataset = make_dataset(config, args)
    ckpt_norm_stats = load_ckpt_norm_stats(ckpt_dir)
    apply_ckpt_norm_stats(dataset, ckpt_norm_stats)
    print(f"Using Evo-1 checkpoint norm stats: {ckpt_dir / 'norm_stats.json'}")

    records = []
    output_f = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_f = args.output_jsonl.open("w", encoding="utf-8")
    try:
        end = min(len(dataset), args.start_index + args.num_samples)
        for index in range(args.start_index, end):
            item = dataset[index]
            predictions = [
                predict_one(model, item, device, amp=args.amp)
                for _ in range(max(1, args.num_predictions_per_sample))
            ]
            record = summarize_sample(index=index, item=item, predictions_norm=predictions)
            _, _, gt_raw, pred_raw = compute_action_eval_tensors(item, predictions)
            record["saved_overlay_frame"] = save_action_overlay_frame(
                item=item,
                record=record,
                gt_raw=gt_raw,
                pred_raw=pred_raw.mean(dim=0),
                output_dir=overlay_output_dir,
                source_width=args.overlay_source_width,
                source_height=args.overlay_source_height,
                processed_padding=args.overlay_processed_padding,
            )
            records.append(record)
            print(
                f"idx={index} mse_norm={record['mse_norm']:.6f} "
                f"raw_adj_ratio={record['raw_adj_ratio']:.3f} "
                f"raw_second_ratio={record['raw_second_ratio']:.3f} "
                f"pred/gt_adj={record['pred_raw']['adj_abs_mean']:.6f}/{record['gt_raw']['adj_abs_mean']:.6f}"
            )
            if output_f is not None:
                output_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_f.flush()
    finally:
        if output_f is not None:
            output_f.close()

    if not records:
        raise RuntimeError("No samples evaluated.")
    summary = aggregate(records)
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
