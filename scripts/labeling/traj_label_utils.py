from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scripts.utils.draw_overlay import (
    OBJECT_KEY,
    TCP_KEY,
    draw_disk,
    draw_future_trajectory,
    draw_line,
    draw_tcp_point,
    overlay_future_trajectory_for_frame,
)

try:
    import imageio
except ImportError:
    imageio = None


def save_combined_overlay_video(
    source_video_path,
    object_pixels,
    tcp_pixels,
    target_pixels=None,
    point_draw_fn=None,
    output_video_path=None,
    future_len=20,
):
    source_video_path = Path(source_video_path)
    if output_video_path is None:
        output_video_path = source_video_path.with_name(f"{source_video_path.stem}_overlay.mp4")

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source episode video: {source_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20

    object_pixels = np.asarray(object_pixels, dtype=np.int32)
    tcp_pixels = np.asarray(tcp_pixels, dtype=np.int32)
    if target_pixels is not None:
        target_pixels = np.asarray(target_pixels, dtype=np.int32)
        if point_draw_fn is None:
            raise ValueError("point_draw_fn is required when target_pixels is provided.")

    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    cv2_writer = None

    frame_index = 0
    try:
        frame_limit = len(target_pixels) if target_pixels is not None else min(len(object_pixels), len(tcp_pixels))
        while frame_index < frame_limit:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if writer is None and cv2_writer is None:
                if imageio is not None:
                    writer = imageio.get_writer(str(output_video_path), fps=fps)
                else:
                    height, width = frame_rgb.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    cv2_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
                    if not cv2_writer.isOpened():
                        raise RuntimeError(f"Could not open output video writer: {output_video_path}")
            rendered = overlay_future_trajectory_for_frame(
                frame_rgb,
                {
                    OBJECT_KEY: object_pixels,
                    TCP_KEY: tcp_pixels,
                },
                current_frame=frame_index,
                future_len=future_len,
            )
            if target_pixels is not None:
                rendered = point_draw_fn(rendered, target_pixels[frame_index])
            if writer is not None:
                writer.append_data(rendered)
            else:
                cv2_writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()
        if cv2_writer is not None:
            cv2_writer.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames were read from source episode video: {source_video_path}")

    return output_video_path
