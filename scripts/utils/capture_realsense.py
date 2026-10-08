#!/usr/bin/env python3
"""Preview a RealSense color camera: p saves a 1280x960 PNG, q quits."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


WIDTH, HEIGHT = 1280, 960
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "output" / "realsense"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="Serial number of the RealSense camera to capture")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as error:
        parser.exit(1, f"Missing dependency: {error}\nInstall pyrealsense2, numpy and GUI-enabled opencv-python.\n")

    context = rs.context()
    device = next(
        (device for device in context.query_devices()
         if device.get_info(rs.camera_info.serial_number) == args.serial),
        None,
    )
    if device is None:
        parser.exit(1, f"RealSense camera {args.serial} was not found.\n")

    profiles = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8:
                video = profile.as_video_stream_profile()
                profiles.append((video.width(), video.height(), video.fps()))
    if not profiles:
        parser.exit(1, "Camera has no BGR8 color stream profiles.\n")

    def profile_rank(profile):
        width, height, fps = profile
        usable_scale = min(width / WIDTH, height / HEIGHT)
        return (
            (width, height) != (WIDTH, HEIGHT),
            usable_scale < 1,
            abs(usable_scale - 1),
            abs(fps - 30),
        )

    width, height, fps = min(profiles, key=profile_rank)
    print(f"Camera {args.serial}: color stream {width}x{height} @ {fps} FPS")
    if (width, height) != (WIDTH, HEIGHT):
        print("Native 1280x960 unavailable: center-crop to 4:3 and resize to 1280x960.")

    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    window = "RealSense | p: save | q: quit"
    started = False
    try:
        pipeline.start(config)
        started = True
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 720)
        print(f"Save directory: {args.output.resolve()}")
        print("Focus the preview window. Press p to save, q to quit.")
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color = frames.get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())
            h, w = frame.shape[:2]
            if w * HEIGHT > h * WIDTH:
                crop_width = h * WIDTH // HEIGHT
                left = (w - crop_width) // 2
                frame = frame[:, left:left + crop_width]
            elif w * HEIGHT < h * WIDTH:
                crop_height = w * HEIGHT // WIDTH
                top = (h - crop_height) // 2
                frame = frame[top:top + crop_height, :]
            if frame.shape[:2] != (HEIGHT, WIDTH):
                interpolation = cv2.INTER_AREA if frame.shape[0] > HEIGHT else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=interpolation)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("p"), ord("P")):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = args.output / f"{args.serial}_{stamp}.png"
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"Failed to save {path}")
                print(f"Saved: {path} (1280x960)")
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
