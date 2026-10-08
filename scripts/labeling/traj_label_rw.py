#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOTS = [
    REPO_ROOT / "hardware",
    REPO_ROOT,
]
for path in reversed(PACKAGE_ROOTS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from my_device.macros import E2H_CAM_T, E2H_INTRINSIC, Gripper_TCP_T
from scripts.labeling.traj_label_utils import save_combined_overlay_video
from scripts.utils.realworld_projection import (
    compute_gripper_tip_points_base,
    map_camera_pixels_to_video_pixels,
    project_base_points_to_camera_pixels,
)


DEFAULT_DATASET_DIR = REPO_ROOT / "output_noise" / "20260522"
ENV_CAMERA_KEY = "observation.images.robot0_agentview_left"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project a real-world Flexiv gripper-tip trajectory into the environment camera. "
            "Expects a standard LeRobot dataset layout."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to a standard LeRobot dataset root (contains data/, videos/, meta/, extras/).",
    )
    parser.add_argument("--episode-index", type=int, required=True, help="Episode index to process.")
    parser.add_argument("--future-len", type=int, default=40)
    parser.add_argument("--source-width", type=int, default=640)
    parser.add_argument("--source-height", type=int, default=480)
    parser.add_argument(
        "--processed-padding",
        type=int,
        default=8,
        help="Matches Flexiv image preprocessing: Resize(output+padding), then CenterCrop(output).",
    )
    parser.add_argument(
        "--no-transcode",
        action="store_true",
        help="Read the source video directly with OpenCV instead of first transcoding it to H.264.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_episode_states(dataset_dir: Path, episode_index: int) -> np.ndarray:
    ep_name = f"episode_{episode_index:06d}"
    # Try extras/episode_XXXXXX/states.npz first
    states_npz = dataset_dir / "extras" / ep_name / "states.npz"
    if states_npz.exists():
        data = np.load(states_npz)
        if "states" not in data:
            raise KeyError(f"{states_npz} does not contain a 'states' array.")
        return np.asarray(data["states"], dtype=np.float64)

    # Fallback: read from root data/
    import pandas as pd
    parquet_paths = sorted((dataset_dir / "data").glob(f"chunk-*/{ep_name}.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No states.npz or parquet data found for {ep_name} under {dataset_dir}.")
    df = pd.read_parquet(parquet_paths[0], columns=["observation.state"])
    return np.stack(df["observation.state"].to_list()).astype(np.float64)


def get_episode_video_path(dataset_dir: Path, episode_index: int, video_key: str = ENV_CAMERA_KEY) -> Path:
    ep_name = f"episode_{episode_index:06d}"
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        video_template = info.get("video_path", "videos/chunk-{chunk_index:03d}/{video_key}/{episode_name}/{episode_name}.mp4")
        return dataset_dir / video_template.format(
            episode_chunk=0,
            episode_index=episode_index,
            chunk_index=0,
            file_index=episode_index,
            video_key=video_key,
            episode_name=ep_name,
        )
    # Fallback hard-coded standard path
    return dataset_dir / "videos" / "chunk-000" / video_key / ep_name / f"{ep_name}.mp4"


def get_video_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source episode video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not determine video size for {video_path}.")
    return width, height


def can_read_first_frame(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        ret, _ = cap.read()
        return bool(ret)
    finally:
        cap.release()


def make_cv2_readable_video(source_video_path: Path, output_dir: Path, *, transcode_first: bool = True) -> Path:
    if not transcode_first and can_read_first_frame(source_video_path):
        return source_video_path

    converted_path = output_dir / f"{source_video_path.stem}_h264_for_overlay.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(converted_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not can_read_first_frame(converted_path):
        raise RuntimeError(f"Converted video is still not readable by OpenCV: {converted_path}")
    return converted_path


def process_one_episode(
    dataset_dir: Path,
    episode_index: int,
    *,
    future_len: int,
    source_width: int,
    source_height: int,
    processed_padding: int,
    no_transcode: bool,
    overwrite: bool,
) -> None:
    ep_name = f"episode_{episode_index:06d}"
    output_dir = dataset_dir / "extras" / ep_name
    output_dir.mkdir(parents=True, exist_ok=True)

    states = get_episode_states(dataset_dir, episode_index)
    source_video_path = get_episode_video_path(dataset_dir, episode_index)
    overlay_source_video_path = make_cv2_readable_video(
        source_video_path,
        output_dir,
        transcode_first=not no_transcode,
    )
    video_width, video_height = get_video_size(overlay_source_video_path)

    points_base = compute_gripper_tip_points_base(states, Gripper_TCP_T)
    raw_pixels_xy, points_camera = project_base_points_to_camera_pixels(
        points_base,
        base_t_camera=E2H_CAM_T,
        intrinsic=E2H_INTRINSIC,
    )
    pixels_xy = map_camera_pixels_to_video_pixels(
        raw_pixels_xy,
        source_width=source_width,
        source_height=source_height,
        video_width=video_width,
        video_height=video_height,
        processed_padding=processed_padding,
    )

    # Keep the saved pixel convention consistent with existing target/tcp pixels:
    # [row, col] == [y, x].
    pixels_hw = pixels_xy[:, [1, 0]]

    target_pixels_path = output_dir / "target_pixels.npy"
    np.save(target_pixels_path, pixels_hw.astype(np.int32))

    overlay_path = source_video_path.with_name(f"{source_video_path.stem}_rw_gripper_tip_overlay.mp4")
    if overlay_path.exists() and not overwrite:
        raise FileExistsError(f"Overlay already exists: {overlay_path}. Use --overwrite to replace it.")

    save_combined_overlay_video(
        source_video_path=overlay_source_video_path,
        object_pixels=pixels_hw,
        tcp_pixels=pixels_hw,
        output_video_path=overlay_path,
        future_len=future_len,
    )
    if overlay_source_video_path != source_video_path:
        overlay_source_video_path.unlink(missing_ok=True)

    print(f"states: {states.shape}")
    print(f"target pixels [y, x]: {target_pixels_path}")
    print(f"overlay: {overlay_path}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    process_one_episode(
        dataset_dir=dataset_dir,
        episode_index=args.episode_index,
        future_len=args.future_len,
        source_width=args.source_width,
        source_height=args.source_height,
        processed_padding=args.processed_padding,
        no_transcode=args.no_transcode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
