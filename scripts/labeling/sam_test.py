#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SAM2_REPO = ROOT / "sam2"
if str(SAM2_REPO) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2_video_predictor
from sam_click_prompt import (
    get_click_point,
    overlay_mask,
    sam2_autocast_context,
    to_numpy_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 tracking on a local mp4 after one click on the first frame."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "01_dog.mp4",
        help="Input mp4 path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM2 config path relative to sam2 package root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_small.pt",
        help="SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "sam",
        help="Directory for rendered outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
        help="Mask logit threshold.",
    )
    parser.add_argument(
        "--point-x",
        type=int,
        default=None,
        help="Optional x coordinate for the initial positive click.",
    )
    parser.add_argument(
        "--point-y",
        type=int,
        default=None,
        help="Optional y coordinate for the initial positive click.",
    )
    return parser.parse_args()


def load_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")
    return frames, fps


def main() -> None:
    args = parse_args()
    video_path = args.video.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    frames, fps = load_video_frames(video_path)
    if (args.point_x is None) != (args.point_y is None):
        raise ValueError("Pass both --point-x and --point-y together.")
    if args.point_x is not None and args.point_y is not None:
        click_x, click_y = args.point_x, args.point_y
    else:
        click_x, click_y = get_click_point(frames[0], output_dir)

    device = torch.device(args.device)
    autocast_ctx = sam2_autocast_context(str(device))

    predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=str(checkpoint_path),
        device=str(device),
    )

    with torch.inference_mode(), autocast_ctx:
        inference_state = predictor.init_state(
            video_path=str(video_path),
            offload_video_to_cpu=True,
            async_loading_frames=False,
        )
        _, _, init_masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=np.array([[click_x, click_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

        mask_by_frame: dict[int, np.ndarray] = {
            0: to_numpy_mask(init_masks[0], args.score_thresh)
        }
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            if not obj_ids:
                continue
            obj_index = obj_ids.index(1)
            mask_by_frame[frame_idx] = to_numpy_mask(
                mask_logits[obj_index], args.score_thresh
            )

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_dir / "tracked.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create output video writer.")

    for frame_idx, frame in enumerate(frames):
        mask = mask_by_frame.get(frame_idx)
        rendered = frame.copy()
        if mask is not None:
            mask_uint8 = (mask.astype(np.uint8) * 255)
            cv2.imwrite(str(masks_dir / f"{frame_idx:05d}.png"), mask_uint8)
            rendered = overlay_mask(rendered, mask)
        if frame_idx == 0:
            cv2.circle(rendered, (click_x, click_y), 8, (0, 255, 0), -1)
            cv2.circle(rendered, (click_x, click_y), 16, (255, 255, 255), 2)
        cv2.putText(
            rendered,
            f"frame={frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(rendered)
    writer.release()

    with open(output_dir / "click.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": str(video_path),
                "config": args.config,
                "checkpoint": str(checkpoint_path),
                "device": str(device),
                "click": {"x": click_x, "y": click_y, "frame": 0, "label": 1},
            },
            f,
            indent=2,
        )

    print(f"Saved video to: {output_dir / 'tracked.mp4'}")
    print(f"Saved masks to: {masks_dir}")
    print(f"Saved click metadata to: {output_dir / 'click.json'}")


if __name__ == "__main__":
    main()
