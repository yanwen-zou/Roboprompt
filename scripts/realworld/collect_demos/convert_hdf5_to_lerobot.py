#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOTS = [
    REPO_ROOT,
]
for path in reversed(PACKAGE_ROOTS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lerobot.common.datasets.utils import write_info
import openpi.shared.normalize as _normalize
from scripts.utils.lerobot_utils import LerobotDatasetWrapper


OLD_VIDEO_PATH_PATTERN = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
NEW_VIDEO_PATH_PATTERN = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}/episode_{episode_index:06d}.mp4"
)


def _normalize_fps_value(fps: float) -> int | float:
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if fps.is_integer():
        return int(fps)
    return fps


def _video_info(fps: int | float) -> dict[str, Any]:
    return {
        "video.fps": fps,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }


def _find_hdf5_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix not in {".hdf5", ".h5"}:
            raise ValueError(f"Expected an HDF5 file, got: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {".hdf5", ".h5"})
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under: {path}")
    return files


def _collect_hdf5_paths(paths: list[str]) -> list[Path]:
    hdf5_paths: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        resolved = Path(raw_path).resolve()
        for hdf5_path in _find_hdf5_paths(resolved):
            if hdf5_path not in seen:
                seen.add(hdf5_path)
                hdf5_paths.append(hdf5_path)
    return hdf5_paths


def _read_hdf5_fps(hdf5_path: Path) -> int | float:
    with h5py.File(hdf5_path, "r") as raw_file:
        if "data" not in raw_file:
            raise ValueError(f"Missing 'data' group in {hdf5_path}")
        env_info_raw = raw_file["data"].attrs.get("env_info")
        if env_info_raw is None:
            raise ValueError(f"Missing data.attrs['env_info'] in {hdf5_path}; pass --fps explicitly.")
        env_info = json.loads(env_info_raw)
        if "fps" not in env_info:
            raise ValueError(f"Missing env_info['fps'] in {hdf5_path}; pass --fps explicitly.")
        return _normalize_fps_value(env_info["fps"])


def _resolve_fps(hdf5_paths: list[Path], explicit_fps: float | None) -> int | float:
    if explicit_fps is not None:
        return _normalize_fps_value(explicit_fps)

    fps_by_path = {path: _read_hdf5_fps(path) for path in hdf5_paths}
    unique_fps = sorted({float(fps) for fps in fps_by_path.values()})
    if len(unique_fps) > 1:
        details = ", ".join(f"{path.name}: {fps}" for path, fps in fps_by_path.items())
        raise ValueError(f"HDF5 files have different fps values ({details}); pass --fps explicitly or convert separately.")
    return _normalize_fps_value(unique_fps[0])


def _default_output_dir(input_paths: list[str], hdf5_paths: list[Path]) -> Path:
    resolved_inputs = [Path(raw_path).resolve() for raw_path in input_paths]
    input_dirs = [path for path in resolved_inputs if path.is_dir()]
    if len(resolved_inputs) == 1 and input_dirs:
        input_dir = input_dirs[0]
        return input_dir / f"lerobot-{input_dir.name}"

    common_parent = Path(os.path.commonpath([str(path.parent) for path in hdf5_paths]))
    if len(hdf5_paths) > 1:
        return common_parent / f"lerobot-{common_parent.name}"

    last_hdf5_path = hdf5_paths[-1]
    match = re.search(r"(\d{8}_\d{6})$", last_hdf5_path.stem)
    if len(hdf5_paths) == 1:
        if match:
            return hdf5_paths[0].parent / f"lerobot-{match.group(1)}"
        return hdf5_paths[0].parent / f"lerobot-{hdf5_paths[0].stem}"
    raise ValueError("No HDF5 paths provided.")


def _build_task_to_id_map_for_files(hdf5_paths: list[Path]) -> dict[str, int]:
    task_to_id: dict[str, int] = {}
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as raw_file:
            for demo_name in raw_file["data"].keys():
                ep_meta = json.loads(raw_file["data"][demo_name].attrs.get("ep_meta", "{}"))
                lang = ep_meta.get("lang", "") or ""
                if lang not in task_to_id:
                    task_to_id[lang] = len(task_to_id)
    return task_to_id


def _prepare_dataset_spec(first_demo: h5py.Group, fps: int | float) -> tuple[dict[str, dict], bool]:
    first_obs = first_demo["obs"]
    env_images = np.asarray(first_obs["robot0_agentview_left_image"])
    wrist_images = np.asarray(first_obs["robot0_eye_in_hand_image"])
    state_shape = tuple(np.asarray(first_demo["states"]).shape[1:])
    action_shape = tuple(np.asarray(first_demo["actions"]).shape[1:])
    include_right = "robot0_agentview_right_image" in first_obs
    right_shape = None
    if include_right:
        right_shape = tuple(np.asarray(first_obs["robot0_agentview_right_image"]).shape[1:])

    features = {
        "observation.images.robot0_eye_in_hand": {
            "dtype": "video",
            "shape": tuple(wrist_images.shape[1:]),
            "names": ["height", "width", "channel"],
            "video_info": _video_info(fps),
        },
        "observation.images.robot0_agentview_left": {
            "dtype": "video",
            "shape": tuple(env_images.shape[1:]),
            "names": ["height", "width", "channel"],
            "video_info": _video_info(fps),
        },
        "observation.state": {"dtype": "float64", "shape": state_shape},
        "action": {"dtype": "float64", "shape": action_shape},
        "annotation.human.task_description": {"dtype": "int64", "shape": (1,)},
        "annotation.human.task_name": {"dtype": "int64", "shape": (1,)},
    }
    if include_right and right_shape is not None:
        features["observation.images.robot0_agentview_right"] = {
            "dtype": "video",
            "shape": right_shape,
            "names": ["height", "width", "channel"],
            "video_info": _video_info(fps),
        }
    return features, include_right


def _validate_demo_schema(
    demo: h5py.Group,
    *,
    include_right: bool,
    state_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    left_shape: tuple[int, ...],
    wrist_shape: tuple[int, ...],
    right_shape: tuple[int, ...] | None,
    hdf5_path: Path,
) -> None:
    obs = demo["obs"]
    actual_include_right = "robot0_agentview_right_image" in obs
    if actual_include_right != include_right:
        raise ValueError(f"Inconsistent right-camera presence in {hdf5_path}.")
    if tuple(np.asarray(demo["states"]).shape[1:]) != state_shape:
        raise ValueError(f"Inconsistent state shape in {hdf5_path}.")
    if tuple(np.asarray(demo["actions"]).shape[1:]) != action_shape:
        raise ValueError(f"Inconsistent action shape in {hdf5_path}.")
    if tuple(np.asarray(obs["robot0_agentview_left_image"]).shape[1:]) != left_shape:
        raise ValueError(f"Inconsistent left-camera shape in {hdf5_path}.")
    if tuple(np.asarray(obs["robot0_eye_in_hand_image"]).shape[1:]) != wrist_shape:
        raise ValueError(f"Inconsistent wrist-camera shape in {hdf5_path}.")
    if include_right and right_shape is not None:
        actual_right_shape = tuple(np.asarray(obs["robot0_agentview_right_image"]).shape[1:])
        if actual_right_shape != right_shape:
            raise ValueError(f"Inconsistent right-camera shape in {hdf5_path}.")


def _save_extras(raw_file: h5py.File, lerobot_root: Path, *, episode_offset: int, source_name: str) -> int:
    extras_dir = lerobot_root / "extras"
    extras_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta = {key: raw_file["data"].attrs[key] for key in raw_file["data"].attrs}
    for key, value in list(dataset_meta.items()):
        if isinstance(value, np.generic):
            dataset_meta[key] = value.item()
        if isinstance(dataset_meta[key], bytes):
            dataset_meta[key] = dataset_meta[key].decode("utf-8")
    dataset_meta_path = extras_dir / f"dataset_meta_{source_name}.json"
    with open(dataset_meta_path, "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=4)
        f.write("\n")

    episode_count = 0
    for episode_index, demo_name in enumerate(raw_file["data"].keys(), start=episode_offset):
        demo = raw_file["data"][demo_name]
        ep_dir = extras_dir / f"episode_{episode_index:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ep_dir / "states.npz", states=np.asarray(demo["states"]))

        ep_meta = demo.attrs.get("ep_meta", "{}")
        if isinstance(ep_meta, bytes):
            ep_meta = ep_meta.decode("utf-8")
        with open(ep_dir / "ep_meta.json", "w", encoding="utf-8") as f:
            if isinstance(ep_meta, str):
                try:
                    parsed = json.loads(ep_meta)
                except json.JSONDecodeError:
                    parsed = {"raw_ep_meta": ep_meta}
            else:
                parsed = {"ep_meta": ep_meta}
            json.dump(parsed, f, indent=4)
            f.write("\n")

        model_file = demo.attrs.get("model_file", "")
        if isinstance(model_file, bytes):
            model_file = model_file.decode("utf-8")
        with gzip.open(ep_dir / "model.xml.gz", "wb") as f:
            f.write(str(model_file).encode("utf-8"))
        episode_count += 1

    return episode_count


def _normalize_video_layout(lerobot_root: Path, video_keys: list[str]) -> None:
    videos_dir = lerobot_root / "videos"
    if not videos_dir.is_dir():
        return

    for video_key in video_keys:
        for key_dir in sorted(videos_dir.glob(f"chunk-*/{video_key}")):
            if not key_dir.is_dir():
                continue
            for mp4_path in sorted(key_dir.glob("*.mp4")):
                episode_name = mp4_path.stem
                target_dir = key_dir / episode_name
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / mp4_path.name
                if target_path.exists():
                    continue
                mp4_path.rename(target_path)

    info_json = lerobot_root / "meta" / "info.json"
    if info_json.is_file():
        with open(info_json, "r", encoding="utf-8") as f:
            info = json.load(f)
        if info.get("video_path") == OLD_VIDEO_PATH_PATTERN:
            info["video_path"] = NEW_VIDEO_PATH_PATTERN
            write_info(info, lerobot_root)


def _compute_and_save_norm_stats(lerobot_root: Path) -> Path:
    parquet_paths = sorted((lerobot_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet episodes found under {lerobot_root / 'data'}")

    state_stats = _normalize.RunningStats()
    action_stats = _normalize.RunningStats()

    for parquet_path in tqdm(parquet_paths, desc="norm_stats"):
        df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
        state_batch = np.stack(df["observation.state"].to_list()).astype(np.float32)
        action_batch = np.stack(df["action"].to_list()).astype(np.float32)
        state_stats.update(state_batch)
        action_stats.update(action_batch)

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    _normalize.save(lerobot_root, norm_stats)
    return lerobot_root / "norm_stats.json"


def convert_hdf5s(hdf5_paths: list[Path], output_dir: Path, fps: int | float) -> Path:
    fps = _normalize_fps_value(fps)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    first_hdf5_path = hdf5_paths[0]
    with h5py.File(first_hdf5_path, "r") as raw_file:
        demo_names = list(raw_file["data"].keys())
        if not demo_names:
            raise ValueError(f"No demos found in {first_hdf5_path}")
        first_demo = raw_file["data"][demo_names[0]]
        features, include_right = _prepare_dataset_spec(first_demo, fps)

    dataset = LerobotDatasetWrapper.create(
        repo_id="realworld/flexiv",
        root=output_dir,
        robot_type="flexiv",
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    task_to_id = _build_task_to_id_map_for_files(hdf5_paths)
    task_name_idx = len(task_to_id)
    state_shape = tuple(features["observation.state"]["shape"])
    action_shape = tuple(features["action"]["shape"])
    left_shape = tuple(features["observation.images.robot0_agentview_left"]["shape"])
    wrist_shape = tuple(features["observation.images.robot0_eye_in_hand"]["shape"])
    right_shape = tuple(features["observation.images.robot0_agentview_right"]["shape"]) if include_right else None

    episode_offset = 0
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as raw_file:
            data_group = raw_file["data"]
            demo_names = list(data_group.keys())
            if not demo_names:
                raise ValueError(f"No demos found in {hdf5_path}")

            for demo_name in tqdm(demo_names, desc=hdf5_path.name):
                demo = data_group[demo_name]
                _validate_demo_schema(
                    demo,
                    include_right=include_right,
                    state_shape=state_shape,
                    action_shape=action_shape,
                    left_shape=left_shape,
                    wrist_shape=wrist_shape,
                    right_shape=right_shape,
                    hdf5_path=hdf5_path,
                )
                obs = demo["obs"]
                states = np.asarray(demo["states"], dtype=np.float64)
                actions = np.asarray(demo["actions"], dtype=np.float64)
                demo_length = len(actions)
                ep_meta = json.loads(demo.attrs.get("ep_meta", "{}"))
                lang = ep_meta.get("lang", "") or ""
                task_id = task_to_id[lang]

                env_images = np.asarray(obs["robot0_agentview_left_image"], dtype=np.uint8)
                wrist_images = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
                right_images = None
                if include_right:
                    right_images = np.asarray(obs["robot0_agentview_right_image"], dtype=np.uint8)

                for i in range(demo_length):
                    frame = {
                        "observation.images.robot0_eye_in_hand": wrist_images[i],
                        "observation.images.robot0_agentview_left": env_images[i],
                        "observation.state": states[i],
                        "action": actions[i],
                        "annotation.human.task_description": np.array([task_id], dtype=np.int64),
                        "annotation.human.task_name": np.array([task_name_idx], dtype=np.int64),
                        "task": lang,
                    }
                    if include_right and right_images is not None:
                        frame["observation.images.robot0_agentview_right"] = right_images[i]
                    dataset.add_frame(frame)

                dataset.save_episode()

            episode_offset += _save_extras(
                raw_file,
                output_dir,
                episode_offset=episode_offset,
                source_name=hdf5_path.stem,
            )

    video_keys = [
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_left",
    ]
    if include_right:
        video_keys.append("observation.images.robot0_agentview_right")
    _normalize_video_layout(output_dir, video_keys)
    _compute_and_save_norm_stats(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert realworld demo HDF5 files to LeRobot datasets and normalize video layout."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="One or more HDF5 files or directories containing HDF5 files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output dir for the merged LeRobot dataset.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS to write into the LeRobot dataset metadata and encoded videos. Defaults to env_info['fps'] in the HDF5 data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_paths = _collect_hdf5_paths(args.paths)
    output_dir = Path(args.output).resolve() if args.output else _default_output_dir(args.paths, hdf5_paths)
    fps = _resolve_fps(hdf5_paths, args.fps)
    print(f"Converting {len(hdf5_paths)} HDF5 file(s) -> {output_dir} at {fps} fps")
    convert_hdf5s(hdf5_paths, output_dir, fps=fps)
    print(f"Finished {output_dir}")


if __name__ == "__main__":
    main()
