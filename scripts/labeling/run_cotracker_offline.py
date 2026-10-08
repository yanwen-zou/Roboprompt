from __future__ import annotations

import gc
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch

from sam_click_prompt import (
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    get_prompt_points,
    load_sam2_image_predictor,
    overlay_mask,
    predict_mask_from_points,
    sam2_autocast_context,
)


def normalize_visibility(pred_visibility: torch.Tensor) -> np.ndarray:
    visibility = pred_visibility[0].detach().cpu().numpy()
    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility[..., 0]
    elif visibility.ndim != 2:
        raise RuntimeError(
            f"Unexpected visibility shape after removing batch dim: {visibility.shape}"
        )
    return visibility > 0.5


def draw_prompt_points(
    frame: np.ndarray, prompt_points: np.ndarray, prompt_labels: np.ndarray
) -> np.ndarray:
    rendered = frame.copy()
    for (x, y), label in zip(prompt_points.astype(int), prompt_labels):
        if label == 1:
            cv2.circle(rendered, (x, y), 8, (0, 255, 0), -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(
                rendered,
                (x, y),
                (0, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=18,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        cv2.circle(rendered, (x, y), 16, (255, 255, 255), 2, cv2.LINE_AA)
    return rendered


def save_segment_mask_outputs(
    segment_dir: Path, frame_bgr: np.ndarray, segm_mask_np: np.ndarray
) -> tuple[Path, Path]:
    mask_overlay_path = segment_dir / "segment_mask_overlay.png"
    mask_binary_path = segment_dir / "segment_mask_binary.png"
    cv2.imwrite(str(mask_overlay_path), overlay_mask(frame_bgr, segm_mask_np))
    cv2.imwrite(str(mask_binary_path), (segm_mask_np.astype(np.uint8) * 255))
    return mask_overlay_path, mask_binary_path


def save_tracks_json(
    *,
    segment_dir: Path,
    segment_idx: int,
    start_frame: int,
    end_frame: int,
    prompt_points: np.ndarray,
    prompt_labels: np.ndarray,
    mask_score: float,
    tracks: np.ndarray,
    visibility: np.ndarray,
) -> Path:
    tracks_payload = {
        "segment_index": segment_idx,
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "num_frames": int(end_frame - start_frame),
        "num_points": int(tracks.shape[1]),
        "mask_score": float(mask_score),
        "prompt_points": prompt_points.astype(float).tolist(),
        "prompt_labels": prompt_labels.astype(int).tolist(),
        "tracks": [],
    }

    for point_idx in range(tracks.shape[1]):
        point_track = []
        for local_frame_idx in range(tracks.shape[0]):
            point_track.append(
                {
                    "frame_index": int(start_frame + local_frame_idx),
                    "x": float(tracks[local_frame_idx, point_idx, 0]),
                    "y": float(tracks[local_frame_idx, point_idx, 1]),
                    "visible": bool(visibility[local_frame_idx, point_idx]),
                }
            )
        tracks_payload["tracks"].append(
            {
                "point_index": int(point_idx),
                "frames": point_track,
            }
        )

    tracks_json_path = segment_dir / "tracks.json"
    with open(tracks_json_path, "w", encoding="utf-8") as f:
        json.dump(tracks_payload, f, indent=2)
    return tracks_json_path


def build_segment_ranges(num_frames: int, num_segments: int) -> list[tuple[int, int]]:
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")
    boundaries = np.linspace(0, num_frames, num_segments + 1, dtype=int)
    ranges: list[tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            ranges.append((int(start), int(end)))
    return ranges


def load_cotracker(device: str) -> torch.nn.Module:
    return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)


def main() -> None:
    video_path = Path("outputs/rollout_2_success.mp4")
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    frames = iio.imread(video_path, plugin="FFMPEG")
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {video_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_size = 30
    num_segments = 3
    backward_tracking = False
    sam_config = DEFAULT_SAM2_CONFIG
    sam_checkpoint = DEFAULT_SAM2_CHECKPOINT

    output_dir = Path("outputs/cotracker")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / "tracked.mp4"

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video writer at {output_video_path}.")

    segment_ranges = build_segment_ranges(len(frames), num_segments)
    segment_summaries: list[str] = []

    for segment_idx, (start_frame, end_frame) in enumerate(segment_ranges):
        segment_frames = frames[start_frame:end_frame]
        first_frame_bgr = cv2.cvtColor(segment_frames[0], cv2.COLOR_RGB2BGR)
        segment_dir = output_dir / f"segment_{segment_idx:02d}_frame_{start_frame:06d}"
        segment_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[segment {segment_idx + 1}/{len(segment_ranges)}] "
            f"frames {start_frame}..{end_frame - 1}: waiting for prompt"
        )
        prompt_points, prompt_labels = get_prompt_points(first_frame_bgr, segment_dir)

        sam_predictor = load_sam2_image_predictor(
            device=device,
            config=sam_config,
            checkpoint=sam_checkpoint,
        )
        with torch.inference_mode(), sam2_autocast_context(device):
            segm_mask_np, mask_score = predict_mask_from_points(
                first_frame_bgr,
                point_coords=prompt_points,
                point_labels=prompt_labels,
                predictor=sam_predictor,
            )
        if not np.any(segm_mask_np):
            raise RuntimeError(
                f"SAM returned an empty mask for segment {segment_idx} at frame {start_frame}."
            )

        mask_overlay_path, mask_binary_path = save_segment_mask_outputs(
            segment_dir, first_frame_bgr, segm_mask_np
        )

        del sam_predictor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        video = (
            torch.from_numpy(segment_frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T C H W
        segm_mask = torch.from_numpy(segm_mask_np.astype(np.float32))[None, None].to(device)

        cotracker = load_cotracker(device)
        try:
            with torch.inference_mode(), sam2_autocast_context(device):
                pred_tracks, pred_visibility = cotracker(
                    video,
                    segm_mask=segm_mask,
                    grid_size=grid_size,
                    grid_query_frame=0,
                    backward_tracking=backward_tracking,
                )
        except torch.AcceleratorError as exc:
            if device == "cuda" and "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CoTracker ran out of CUDA memory. "
                    "Try lowering grid_size, reducing num_segments, or keeping backward_tracking=False."
                ) from exc
            raise
        finally:
            del cotracker
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        tracks = pred_tracks[0].detach().cpu().numpy()
        visibility = normalize_visibility(pred_visibility)
        tracks_json_path = save_tracks_json(
            segment_dir=segment_dir,
            segment_idx=segment_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            prompt_points=prompt_points,
            prompt_labels=prompt_labels,
            mask_score=mask_score,
            tracks=tracks,
            visibility=visibility,
        )

        rng = np.random.default_rng(segment_idx)
        colors = rng.integers(64, 256, size=(tracks.shape[1], 3), dtype=np.uint8)
        trail_length = 10

        for local_frame_idx, frame in enumerate(segment_frames):
            global_frame_idx = start_frame + local_frame_idx
            rendered = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if local_frame_idx == 0:
                rendered = overlay_mask(rendered, segm_mask_np)
            for point_idx in range(tracks.shape[1]):
                if not visibility[local_frame_idx, point_idx]:
                    continue
                color = tuple(int(x) for x in colors[point_idx].tolist())
                trail_start = max(0, local_frame_idx - trail_length + 1)
                visible_steps = (
                    np.where(visibility[trail_start : local_frame_idx + 1, point_idx])[0]
                    + trail_start
                )

                for prev_idx, curr_idx in zip(visible_steps[:-1], visible_steps[1:]):
                    pt1 = tuple(np.round(tracks[prev_idx, point_idx]).astype(int))
                    pt2 = tuple(np.round(tracks[curr_idx, point_idx]).astype(int))
                    cv2.line(rendered, pt1, pt2, color, 1, cv2.LINE_AA)

                center = tuple(np.round(tracks[local_frame_idx, point_idx]).astype(int))
                cv2.circle(rendered, center, 2, color, -1, cv2.LINE_AA)

            cv2.putText(
                rendered,
                f"frame={global_frame_idx} segment={segment_idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(rendered)

        segment_summaries.append(
            "segment="
            f"{segment_idx} frames={start_frame}-{end_frame - 1} "
            f"mask_score={mask_score:.4f} "
            f"pos={(prompt_labels == 1).sum()} neg={(prompt_labels == 0).sum()} "
            f"points={tracks.shape[1]} overlay={mask_overlay_path} "
            f"binary={mask_binary_path} tracks={tracks_json_path}"
        )

        del pred_tracks, pred_visibility, tracks, visibility, video, segm_mask
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    writer.release()

    print(f"Saved video to: {output_video_path}")
    print(f"Num segments: {len(segment_ranges)}")
    print(f"CoTracker grid_size: {grid_size}")
    print(f"Backward tracking: {backward_tracking}")
    for summary in segment_summaries:
        print(summary)


if __name__ == "__main__":
    main()
