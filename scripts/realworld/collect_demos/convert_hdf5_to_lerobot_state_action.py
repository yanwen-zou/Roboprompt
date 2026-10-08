#!/usr/bin/env python3
"""Convert realworld HDF5 demos to LeRobot using recorded command actions.

This entrypoint is kept for backward compatibility with existing commands. It
does not derive actions from states; LeRobot "action" is copied from
demo["actions"] in the source HDF5.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

import convert_hdf5_to_lerobot as base
from scripts.utils.lerobot_utils import LerobotDatasetWrapper


def _video_info(fps: int | float) -> dict[str, Any]:
    return {
        "video.fps": fps,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }


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


def convert_hdf5s(hdf5_paths: list[Path], output_dir: Path, fps: int | float) -> Path:
    fps = base._normalize_fps_value(fps)
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

    task_to_id = base._build_task_to_id_map_for_files(hdf5_paths)
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

                if len(states) != demo_length:
                    raise ValueError(f"State/action length mismatch in {hdf5_path}:{demo_name}.")
                if len(env_images) != demo_length or len(wrist_images) != demo_length:
                    raise ValueError(f"Image/action length mismatch in {hdf5_path}:{demo_name}.")
                if include_right and right_images is not None and len(right_images) != demo_length:
                    raise ValueError(f"Right image/state length mismatch in {hdf5_path}:{demo_name}.")

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

            episode_offset += base._save_extras(
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
    base._normalize_video_layout(output_dir, video_keys)
    base._compute_and_save_norm_stats(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="FPS to write into metadata/videos. Defaults to env_info['fps'] in the HDF5 data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_paths = base._collect_hdf5_paths(args.paths)
    output_dir = Path(args.output).resolve() if args.output else base._default_output_dir(args.paths, hdf5_paths)
    fps = base._resolve_fps(hdf5_paths, args.fps)
    print(f"Converting {len(hdf5_paths)} HDF5 file(s) -> {output_dir} at {fps} fps")
    print('Copying LeRobot actions from HDF5 demo["actions"].')
    convert_hdf5s(hdf5_paths, output_dir, fps=fps)
    print(f"Finished {output_dir}")


if __name__ == "__main__":
    main()
