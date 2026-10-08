import dataclasses
import json
import logging
import os
import pathlib
import re
import sys
from datetime import datetime

import cv2
import numpy as np
import tqdm
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# Must set before importing robosuite / robocasa.
os.environ["MUJOCO_GL"] = "egl"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PACKAGE_ROOTS = [
    REPO_ROOT / "robosuite",
    REPO_ROOT / "robocasa",
    REPO_ROOT,
]
for path in reversed(PACKAGE_ROOTS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import robocasa  # noqa: E402
import robocasa.wrappers.gym_wrapper  # noqa: F401,E402
import robocasa.utils.lerobot_utils as LU  # noqa: E402
from openpi.groot_utils.groot_openpi_dataset import (  # noqa: E402
    _build_rp_prompt_fields,
    _load_video_frame,
)
from robocasa.scripts.dataset_scripts.playback_utils import (  # noqa: E402
    resolve_instruction_from_ep_meta,
)

EEF_POSITION_SLICE = slice(0, 3)
EEF_ROTATION_SLICE = slice(3, 6)


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized or "unknown_policy"


def _get_policy_run_name(
    checkpoint_info: dict[str, str] | None,
    dataset_name: str,
    now: datetime,
) -> str:
    if checkpoint_info is None:
        policy_label = "unknown_policy"
    else:
        policy_label = (
            checkpoint_info.get("config")
            or checkpoint_info.get("label")
            or pathlib.Path(checkpoint_info.get("dir", "")).name
            or "unknown_policy"
        )
    dataset_label = _sanitize_path_component(dataset_name)
    return f"{_sanitize_path_component(policy_label)}_{dataset_label}_{now.strftime('%Y-%m-%d-%H-%M')}"


def _make_eval_run_root(
    log_dir: str,
    checkpoint_info: dict[str, str] | None,
    dataset_path: pathlib.Path,
    now: datetime | None = None,
) -> pathlib.Path:
    now = now or datetime.now()
    return pathlib.Path(log_dir) / "evals_dataset" / _get_policy_run_name(
        checkpoint_info,
        dataset_path.name,
        now,
    )


def _write_summary(
    run_root: pathlib.Path,
    dataset_path: pathlib.Path,
    episode_summaries: list[dict],
    checkpoint_info: dict[str, str] | None,
    eval_config: dict,
) -> dict:
    run_root.mkdir(parents=True, exist_ok=True)
    total_episodes = len(episode_summaries)
    compact_episodes = [
        {
            key: episode_summary[key]
            for key in (
                "episode_index",
                "task",
                "env_name",
                "recorded_steps",
                "evaluated_steps",
            )
            if key in episode_summary
        }
        | (
            {
                "variants": {
                    variant_name: {
                        "mean_chunk_mse": float(variant_summary["mean_chunk_mse"]),
                        "mean_pos_mse": float(variant_summary["mean_pos_mse"]),
                        "mean_rot_mse": float(variant_summary["mean_rot_mse"]),
                    }
                    for variant_name, variant_summary in episode_summary["variants"].items()
                }
            }
            if "variants" in episode_summary
            else {
                "mean_chunk_mse": float(episode_summary["mean_chunk_mse"]),
                "mean_pos_mse": float(episode_summary["mean_pos_mse"]),
                "mean_rot_mse": float(episode_summary["mean_rot_mse"]),
            }
        )
        | (
            {"improvement": episode_summary["improvement"]}
            if "improvement" in episode_summary
            else {}
        )
        for episode_summary in episode_summaries
    ]
    summary = {
        "dataset_path": str(dataset_path),
        "checkpoint": checkpoint_info,
        "run_root": str(run_root),
        "num_episodes": total_episodes,
        "eval_config": eval_config,
        "episodes": compact_episodes,
    }

    if episode_summaries and "variants" in episode_summaries[0]:
        variant_names = list(episode_summaries[0]["variants"].keys())
        variant_summary: dict[str, dict] = {}
        for variant_name in variant_names:
            entries = [episode_summary["variants"][variant_name] for episode_summary in episode_summaries]
            total_chunks = sum(int(entry.get("num_chunks", 0)) for entry in entries)
            total_compared_actions = sum(int(entry.get("num_compared_actions", 0)) for entry in entries)
            total_compared_pos_dims = sum(int(entry.get("num_compared_pos_dims", 0)) for entry in entries)
            total_compared_rot_dims = sum(int(entry.get("num_compared_rot_dims", 0)) for entry in entries)
            pos_squared_error_sum = sum(float(entry.get("pos_squared_error_sum", 0.0)) for entry in entries)
            rot_squared_error_sum = sum(float(entry.get("rot_squared_error_sum", 0.0)) for entry in entries)
            chunk_mse_values = [float(entry["mean_chunk_mse"]) for entry in entries if entry.get("num_chunks", 0) > 0]
            pos_mse_values = [
                float(entry["mean_pos_mse"]) for entry in entries if entry.get("num_compared_pos_dims", 0) > 0
            ]
            rot_mse_values = [
                float(entry["mean_rot_mse"]) for entry in entries if entry.get("num_compared_rot_dims", 0) > 0
            ]
            variant_summary[variant_name] = {
                "num_chunks": total_chunks,
                "num_compared_actions": total_compared_actions,
                "num_compared_pos_dims": total_compared_pos_dims,
                "num_compared_rot_dims": total_compared_rot_dims,
                "mean_episode_chunk_mse": float(np.mean(chunk_mse_values)) if chunk_mse_values else 0.0,
                "mean_episode_pos_mse": float(np.mean(pos_mse_values)) if pos_mse_values else 0.0,
                "mean_episode_rot_mse": float(np.mean(rot_mse_values)) if rot_mse_values else 0.0,
                "global_pos_mse": float(pos_squared_error_sum / float(total_compared_pos_dims))
                if total_compared_pos_dims > 0
                else 0.0,
                "global_rot_mse": float(rot_squared_error_sum / float(total_compared_rot_dims))
                if total_compared_rot_dims > 0
                else 0.0,
            }

        abs_pos_improvements = [
            float(episode_summary["improvement"]["abs_mean_pos_mse"])
            for episode_summary in episode_summaries
            if "improvement" in episode_summary
        ]
        rel_pos_improvements = [
            float(episode_summary["improvement"]["rel_mean_pos_mse"])
            for episode_summary in episode_summaries
            if "improvement" in episode_summary
        ]
        abs_rot_improvements = [
            float(episode_summary["improvement"]["abs_mean_rot_mse"])
            for episode_summary in episode_summaries
            if "improvement" in episode_summary
        ]
        rel_rot_improvements = [
            float(episode_summary["improvement"]["rel_mean_rot_mse"])
            for episode_summary in episode_summaries
            if "improvement" in episode_summary
        ]
        summary["variants"] = variant_summary
        summary["average_abs_mean_pos_mse_improvement"] = (
            float(np.mean(abs_pos_improvements)) if abs_pos_improvements else 0.0
        )
        summary["average_rel_mean_pos_mse_improvement"] = (
            float(np.mean(rel_pos_improvements)) if rel_pos_improvements else 0.0
        )
        summary["average_abs_mean_rot_mse_improvement"] = (
            float(np.mean(abs_rot_improvements)) if abs_rot_improvements else 0.0
        )
        summary["average_rel_mean_rot_mse_improvement"] = (
            float(np.mean(rel_rot_improvements)) if rel_rot_improvements else 0.0
        )
    else:
        total_chunks = sum(int(summary_item.get("num_chunks", 0)) for summary_item in episode_summaries)
        total_compared_actions = sum(int(summary_item.get("num_compared_actions", 0)) for summary_item in episode_summaries)
        total_compared_pos_dims = sum(int(summary_item.get("num_compared_pos_dims", 0)) for summary_item in episode_summaries)
        total_compared_rot_dims = sum(int(summary_item.get("num_compared_rot_dims", 0)) for summary_item in episode_summaries)
        chunk_mse_values = [
            float(summary_item["mean_chunk_mse"]) for summary_item in episode_summaries if summary_item.get("num_chunks", 0) > 0
        ]
        pos_mse_values = [
            float(summary_item["mean_pos_mse"])
            for summary_item in episode_summaries
            if summary_item.get("num_compared_pos_dims", 0) > 0
        ]
        rot_mse_values = [
            float(summary_item["mean_rot_mse"])
            for summary_item in episode_summaries
            if summary_item.get("num_compared_rot_dims", 0) > 0
        ]
        pos_squared_error_sum = sum(
            float(summary_item.get("pos_squared_error_sum", 0.0)) for summary_item in episode_summaries
        )
        rot_squared_error_sum = sum(
            float(summary_item.get("rot_squared_error_sum", 0.0)) for summary_item in episode_summaries
        )
        summary.update(
            {
                "num_chunks": total_chunks,
                "num_compared_actions": total_compared_actions,
                "num_compared_pos_dims": total_compared_pos_dims,
                "num_compared_rot_dims": total_compared_rot_dims,
                "mean_episode_chunk_mse": float(np.mean(chunk_mse_values)) if chunk_mse_values else 0.0,
                "mean_episode_pos_mse": float(np.mean(pos_mse_values)) if pos_mse_values else 0.0,
                "mean_episode_rot_mse": float(np.mean(rot_mse_values)) if rot_mse_values else 0.0,
                "global_pos_mse": float(pos_squared_error_sum / float(total_compared_pos_dims))
                if total_compared_pos_dims > 0
                else 0.0,
                "global_rot_mse": float(rot_squared_error_sum / float(total_compared_rot_dims))
                if total_compared_rot_dims > 0
                else 0.0,
            }
        )

    summary_path = run_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    logging.info("[summary] Wrote %s", summary_path)
    return summary


def _get_checkpoint_info(client, checkpoint_override: str | None) -> dict[str, str] | None:
    checkpoint_info: dict[str, str] = {}
    if checkpoint_override:
        checkpoint_info["label"] = checkpoint_override

    try:
        server_metadata = client.get_server_metadata()
    except Exception as exc:
        logging.warning("Could not fetch server metadata: %s", exc)
        server_metadata = None

    if isinstance(server_metadata, dict):
        checkpoint_dir = server_metadata.get("checkpoint_dir")
        checkpoint_config = server_metadata.get("checkpoint_config")
        if checkpoint_dir:
            checkpoint_info["dir"] = checkpoint_dir
        if checkpoint_config:
            checkpoint_info["config"] = checkpoint_config

    return checkpoint_info or None

def _resolve_episode_indices(
    dataset_path: pathlib.Path,
    n: int | None,
    episode_indices: list[int] | None,
    *,
    random_sample: bool,
    seed: int,
) -> list[int]:
    demos = LU.get_episodes(dataset_path)
    available = list(range(len(demos)))
    if episode_indices:
        wanted = set(int(index) for index in episode_indices)
        available = [index for index in available if index in wanted]
    elif n is not None and random_sample:
        rng = np.random.default_rng(seed)
        sample_size = min(int(n), len(available))
        available = sorted(int(x) for x in rng.choice(available, size=sample_size, replace=False))
    elif n is not None:
        available = available[:n]
    return available


def _load_episode_dataframe(dataset_path: pathlib.Path, episode_index: int):
    data_files = list(dataset_path.glob(f"data/*/episode_{episode_index:06d}.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet file found for episode {episode_index} under {dataset_path}")
    return LU.pd.read_parquet(data_files[0])


def _build_episode_states_from_dataframe(df) -> np.ndarray:
    if "observation.state" not in df.columns:
        raise KeyError(
            f"Expected column 'observation.state' in parquet, got columns: {list(df.columns)}"
        )

    raw_states = np.stack(df["observation.state"].to_list()).astype(np.float32)
    if raw_states.ndim != 2 or raw_states.shape[1] < 16:
        raise ValueError(f"Expected observation.state to have shape [T, 16+], got {raw_states.shape}")

    # Match the same state ordering used in groot_openpi_dataset.py during training:
    # eef_pos_rel, eef_rot_rel, base_pos, base_rot, gripper_qpos
    state_indices = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15]
    return raw_states[:, state_indices]


def _load_lerobot_meta_json(dataset_path: pathlib.Path, filename: str) -> dict:
    meta_path = dataset_path / "meta" / filename
    with open(meta_path, "r") as f:
        return json.load(f)


def _get_dataset_fps(dataset_path: pathlib.Path) -> int:
    info = _load_lerobot_meta_json(dataset_path, "info.json")
    fps = int(info.get("fps", 1))
    return max(1, fps)


def _get_eval_frame_indices(recorded_steps: int, fps: int, max_steps: int | None) -> list[int]:
    capped_steps = recorded_steps if max_steps is None else min(recorded_steps, max_steps)
    stride = max(1, int(fps))
    return list(range(0, capped_steps, stride))


def _keep_with_probability(keep_prob: float) -> bool:
    keep_prob = float(np.clip(keep_prob, 0.0, 1.0))
    return bool(np.random.rand() < keep_prob)


@dataclasses.dataclass
class Args:
    dataset: str
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    enable_prompt: bool = True
    sample_every_first_frame_per_second: bool = True
    compare_prompt_modes: bool = True
    random_sample_episodes: bool = True
    primitive_cmd_apply_prob: float = 0
    wrist_relative_action_apply_prob: float = 1
    drag_2d_apply_prob: float = 0

    
    log_dir: str = "outputs"
    checkpoint: str | None = None
    force: bool = False
    seed: int = 7

    num_episodes: int | None = 25
    episode_indices: list[int] | None = None
    max_steps: int | None = None
    prompt_frame_offset: int = 0


def eval_main(args: Args) -> None:
    np.random.seed(args.seed)
    dataset_path = pathlib.Path(args.dataset).resolve()
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    checkpoint_info = _get_checkpoint_info(client, args.checkpoint)
    run_root = _make_eval_run_root(args.log_dir, checkpoint_info, dataset_path)
    summary_path = run_root / "summary.json"
    if summary_path.exists() and not args.force:
        logging.info("Existing summary found at %s. Pass --force true to rerun.", summary_path)
        return

    episode_indices = _resolve_episode_indices(
        dataset_path,
        args.num_episodes,
        args.episode_indices,
        random_sample=args.random_sample_episodes,
        seed=args.seed,
    )
    if not episode_indices:
        raise ValueError(f"No episodes selected from dataset {dataset_path}")

    env_meta = LU.get_env_metadata(dataset_path)
    env_name = env_meta["env_name"]
    dataset_fps = _get_dataset_fps(dataset_path)
    eval_config = {
        "host": args.host,
        "port": args.port,
        "resize_size": int(args.resize_size),
        "replan_steps": int(args.replan_steps),
        "enable_prompt": bool(args.enable_prompt),
        "sample_every_first_frame_per_second": bool(args.sample_every_first_frame_per_second),
        "compare_prompt_modes": bool(args.compare_prompt_modes),
        "random_sample_episodes": bool(args.random_sample_episodes),
        "primitive_cmd_apply_prob": float(args.primitive_cmd_apply_prob),
        "wrist_relative_action_apply_prob": float(args.wrist_relative_action_apply_prob),
        "drag_2d_apply_prob": float(args.drag_2d_apply_prob),
        "seed": int(args.seed),
        "num_episodes": None if args.num_episodes is None else int(args.num_episodes),
        "episode_indices": None if args.episode_indices is None else [int(x) for x in args.episode_indices],
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "prompt_frame_offset": int(args.prompt_frame_offset),
        "dataset_fps": int(dataset_fps),
        "selected_episode_indices": [int(x) for x in episode_indices],
    }

    episode_summaries: list[dict] = []
    for rollout_episode_idx, episode_index in enumerate(tqdm.tqdm(episode_indices)):
        episode_df = _load_episode_dataframe(dataset_path, episode_index)
        episode_states = _build_episode_states_from_dataframe(episode_df)
        gt_actions = LU.get_episode_actions(dataset_path, episode_index)
        if gt_actions.ndim != 2:
            raise ValueError(f"Expected 2D action array for episode {episode_index}, got shape {gt_actions.shape}")

        ep_meta = LU.get_episode_meta(dataset_path, episode_index)
        task_lang = resolve_instruction_from_ep_meta(ep_meta) or env_name
        if len(episode_states) != len(gt_actions):
            raise ValueError(
                f"State/action length mismatch for episode {episode_index}: "
                f"{len(episode_states)} states vs {len(gt_actions)} actions"
            )

        recorded_steps = int(gt_actions.shape[0])
        eval_frame_indices = (
            _get_eval_frame_indices(recorded_steps, dataset_fps, args.max_steps)
            if args.sample_every_first_frame_per_second
            else list(range(0, recorded_steps if args.max_steps is None else min(recorded_steps, args.max_steps)))
        )
        variant_names = (
            ["no_prompt", "prompt_dropout"]
            if args.compare_prompt_modes
            else ["prompt_dropout" if args.enable_prompt else "no_prompt"]
        )
        variant_trackers = {
            variant_name: {
                "chunk_mse_values": [],
                "pos_chunk_mse_values": [],
                "rot_chunk_mse_values": [],
                "pos_squared_error_sum": 0.0,
                "rot_squared_error_sum": 0.0,
                "compared_action_count": 0,
                "compared_pos_dim_count": 0,
                "compared_rot_dim_count": 0,
                "prompt_enabled": variant_name != "no_prompt",
            }
            for variant_name in variant_names
        }

        for eval_step_idx, t in enumerate(eval_frame_indices):
            state = episode_states[t]
            img = np.ascontiguousarray(
                _load_video_frame(
                    dataset_path=dataset_path,
                    trajectory_id=episode_index,
                    frame_index=t,
                    video_key="robot0_agentview_left",
                )
            )
            wrist_img = np.ascontiguousarray(
                _load_video_frame(
                    dataset_path=dataset_path,
                    trajectory_id=episode_index,
                    frame_index=t,
                    video_key="robot0_eye_in_hand",
                )
            )
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            prompt_frame_index = min(max(t + args.prompt_frame_offset, 0), recorded_steps - 1)

            for variant_name in variant_names:
                use_prompt = variant_name != "no_prompt"
                task_prompt = task_lang 
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": task_prompt,
                }
                prompt_inputs: dict = {}
                if use_prompt:
                    prompt_inputs = _build_rp_prompt_fields(
                        actions=gt_actions[prompt_frame_index:],
                        dataset_path=dataset_path,
                        trajectory_id=episode_index,
                        frame_index=prompt_frame_index,
                        enable_overlay_prompt=True,
                        primitive_cmd_apply_prob=args.primitive_cmd_apply_prob,
                        wrist_relative_action_apply_prob=args.wrist_relative_action_apply_prob,
                        drag_2d_apply_prob=args.drag_2d_apply_prob,
                    )
                    element.update(prompt_inputs)

                predicted_actions = np.asarray(client.infer(element)["actions"], dtype=np.float32)
                if predicted_actions.ndim != 2:
                    raise ValueError(
                        f"Expected predicted action chunk to have shape [chunk, dim], got {predicted_actions.shape}"
                    )

                if len(predicted_actions) < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(predicted_actions)} actions, but replan_steps={args.replan_steps}"
                    )

                predicted_chunk = predicted_actions[: args.replan_steps]
                gt_chunk = gt_actions[t : t + len(predicted_chunk)]
                if len(gt_chunk) == 0:
                    print(f"Warning: No ground truth actions available for episode {episode_index} frame {t}")
                    continue

                if gt_chunk.shape[1] != predicted_chunk.shape[1]:
                    raise ValueError(
                        f"Action dim mismatch for episode {episode_index} frame {t}: "
                        f"predicted {predicted_chunk.shape}, gt {gt_chunk.shape}"
                    )

                if len(gt_chunk) < len(predicted_chunk):
                    predicted_chunk = predicted_chunk[: len(gt_chunk)]

                pos_squared_error = np.square(
                    predicted_chunk[:, EEF_POSITION_SLICE] - gt_chunk[:, EEF_POSITION_SLICE],
                    dtype=np.float32,
                )
                rot_squared_error = np.square(
                    predicted_chunk[:, EEF_ROTATION_SLICE] - gt_chunk[:, EEF_ROTATION_SLICE],
                    dtype=np.float32,
                )
                combined_squared_error = np.concatenate([pos_squared_error, rot_squared_error], axis=1)
                chunk_mse = float(np.mean(combined_squared_error))
                pos_chunk_mse = float(np.mean(pos_squared_error))
                rot_chunk_mse = float(np.mean(rot_squared_error))

                tracker = variant_trackers[variant_name]
                tracker["pos_squared_error_sum"] += float(np.sum(pos_squared_error))
                tracker["rot_squared_error_sum"] += float(np.sum(rot_squared_error))
                tracker["compared_action_count"] += int(predicted_chunk.shape[0])
                tracker["compared_pos_dim_count"] += int(np.prod(pos_squared_error.shape))
                tracker["compared_rot_dim_count"] += int(np.prod(rot_squared_error.shape))
                tracker["chunk_mse_values"].append(chunk_mse)
                tracker["pos_chunk_mse_values"].append(pos_chunk_mse)
                tracker["rot_chunk_mse_values"].append(rot_chunk_mse)

        variant_summaries: dict[str, dict] = {}
        for variant_name, tracker in variant_trackers.items():
            chunk_mse_values = tracker["chunk_mse_values"]
            pos_chunk_mse_values = tracker["pos_chunk_mse_values"]
            rot_chunk_mse_values = tracker["rot_chunk_mse_values"]
            compared_pos_dim_count = int(tracker["compared_pos_dim_count"])
            compared_rot_dim_count = int(tracker["compared_rot_dim_count"])
            variant_summaries[variant_name] = {
                "num_chunks": len(chunk_mse_values),
                "num_compared_actions": int(tracker["compared_action_count"]),
                "num_compared_pos_dims": compared_pos_dim_count,
                "num_compared_rot_dims": compared_rot_dim_count,
                "pos_squared_error_sum": float(tracker["pos_squared_error_sum"]),
                "rot_squared_error_sum": float(tracker["rot_squared_error_sum"]),
                "mean_chunk_mse": float(np.mean(chunk_mse_values)) if chunk_mse_values else 0.0,
                "mean_pos_chunk_mse": float(np.mean(pos_chunk_mse_values)) if pos_chunk_mse_values else 0.0,
                "mean_rot_chunk_mse": float(np.mean(rot_chunk_mse_values)) if rot_chunk_mse_values else 0.0,
                "mean_pos_mse": (
                    float(tracker["pos_squared_error_sum"] / compared_pos_dim_count) if compared_pos_dim_count > 0 else 0.0
                ),
                "mean_rot_mse": (
                    float(tracker["rot_squared_error_sum"] / compared_rot_dim_count) if compared_rot_dim_count > 0 else 0.0
                ),
                "prompt_enabled": bool(tracker["prompt_enabled"]),
            }

        episode_summary = {
            "episode_index": int(episode_index),
            "task": task_lang,
            "env_name": env_name,
            "dataset_fps": int(dataset_fps),
            "recorded_steps": recorded_steps,
            "evaluated_steps": int(len(eval_frame_indices)),
            "evaluated_frame_indices": [int(x) for x in eval_frame_indices],
            "compare_prompt_modes": bool(args.compare_prompt_modes),
            "primitive_cmd_apply_prob": float(args.primitive_cmd_apply_prob),
            "wrist_relative_action_apply_prob": float(args.wrist_relative_action_apply_prob),
            "drag_2d_apply_prob": float(args.drag_2d_apply_prob),
            "checkpoint": checkpoint_info,
            "variants": variant_summaries,
        }

        if "no_prompt" in variant_summaries and "prompt_dropout" in variant_summaries:
            no_prompt_pos_mse = float(variant_summaries["no_prompt"]["mean_pos_mse"])
            no_prompt_rot_mse = float(variant_summaries["no_prompt"]["mean_rot_mse"])
            prompt_dropout_pos_mse = float(variant_summaries["prompt_dropout"]["mean_pos_mse"])
            prompt_dropout_rot_mse = float(variant_summaries["prompt_dropout"]["mean_rot_mse"])
            episode_summary["improvement"] = {
                "abs_mean_pos_mse": float(no_prompt_pos_mse - prompt_dropout_pos_mse),
                "rel_mean_pos_mse": float((no_prompt_pos_mse - prompt_dropout_pos_mse) / no_prompt_pos_mse)
                if no_prompt_pos_mse > 0
                else 0.0,
                "abs_mean_rot_mse": float(no_prompt_rot_mse - prompt_dropout_rot_mse),
                "rel_mean_rot_mse": float((no_prompt_rot_mse - prompt_dropout_rot_mse) / no_prompt_rot_mse)
                if no_prompt_rot_mse > 0
                else 0.0,
            }
        elif len(variant_summaries) == 1:
            episode_summary.update(next(iter(variant_summaries.values())))
        episode_summaries.append(episode_summary)

    _write_summary(run_root, dataset_path, episode_summaries, checkpoint_info, eval_config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_main(tyro.cli(Args))
