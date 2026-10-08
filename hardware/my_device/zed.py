import time
import sys
import argparse

import cv2
import numpy as np


class ZEDCamera:
    """ZED SDK wrapper returning left frames as BGR numpy arrays."""

    def __init__(self, resolution: str = "720P", fps: int = 60):
        import pyzed.sl as sl

        self._sl = sl
        self._zed = sl.Camera()
        params = sl.InitParameters()
        res_map = {
            "2K": sl.RESOLUTION.HD2K,
            "1080P": sl.RESOLUTION.HD1080,
            "720P": sl.RESOLUTION.HD720,
            "WVGA": sl.RESOLUTION.VGA,
        }
        params.camera_resolution = res_map.get(resolution.upper(), sl.RESOLUTION.HD720)
        params.camera_fps = int(fps)
        params.depth_mode = sl.DEPTH_MODE.NONE

        status = self._zed.open(params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {repr(status)}")

        self._runtime = sl.RuntimeParameters()
        self._mat_left = sl.Mat()
        self._mat_right = sl.Mat()
        self._w = None
        self._h = None

    def read(self):
        if self._zed.grab(self._runtime) != self._sl.ERROR_CODE.SUCCESS:
            return None
        self._zed.retrieve_image(self._mat_left, self._sl.VIEW.LEFT)
        img = self._mat_left.get_data()
        if img is not None and self._w is None:
            self._h, self._w = img.shape[:2]
        return img

    def read_stereo(self):
        """Grab and return (left_bgr, right_bgr) frames, or None if failed."""
        if self._zed.grab(self._runtime) != self._sl.ERROR_CODE.SUCCESS:
            return None
        self._zed.retrieve_image(self._mat_left, self._sl.VIEW.LEFT)
        self._zed.retrieve_image(self._mat_right, self._sl.VIEW.RIGHT)
        left = self._mat_left.get_data()
        right = self._mat_right.get_data()

        # Drop alpha channel if present (BGRA -> BGR)
        if left is not None and left.ndim == 3 and left.shape[2] == 4:
            left = left[:, :, :3]
        if right is not None and right.ndim == 3 and right.shape[2] == 4:
            right = right[:, :, :3]

        if left is not None and self._w is None:
            self._h, self._w = left.shape[:2]
        return left, right

    @property
    def size(self):
        return (self._w, self._h)

    def close(self):
        self._zed.close()


def _run_with_pyzed(resolution: str = "WVGA", fps: int = 30, show_right: bool = False):
    try:
        import pyzed.sl as sl
    except Exception as e:
        raise ImportError("pyzed.sl not available: " + str(e))

    zed = sl.Camera()
    init_params = sl.InitParameters()
    # resolution
    res_map = {
        "2K": sl.RESOLUTION.HD2K,
        "1080P": sl.RESOLUTION.HD1080,
        "720P": sl.RESOLUTION.HD720,
        "WVGA": sl.RESOLUTION.VGA,
    }
    init_params.camera_resolution = res_map.get(resolution.upper(), sl.RESOLUTION.HD720)
    init_params.camera_fps = int(fps)
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # color stream only for now

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {repr(status)}")

    runtime_params = sl.RuntimeParameters()
    mat_left = sl.Mat()
    mat_right = sl.Mat()

    win_name = "ZED Left"
    win_name_r = "ZED Right"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if show_right:
        cv2.namedWindow(win_name_r, cv2.WINDOW_NORMAL)

    prev = time.time()
    frames = 0
    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(mat_left, sl.VIEW.LEFT)
                if show_right:
                    zed.retrieve_image(mat_right, sl.VIEW.RIGHT)

                left = mat_left.get_data()
                if show_right:
                    right = mat_right.get_data()

                cv2.imshow(win_name, left)
                if show_right:
                    cv2.imshow(win_name_r, right)

                frames += 1
                now = time.time()
                if now - prev >= 1.0:
                    fps_txt = f"FPS: {frames/(now-prev):.1f}"
                    cv2.setWindowTitle(win_name, f"{win_name} - {fps_txt}")
                    if show_right:
                        cv2.setWindowTitle(win_name_r, f"{win_name_r} - {fps_txt}")
                    prev = now
                    frames = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cv2.destroyAllWindows()
        zed.close()

def main():
    parser = argparse.ArgumentParser(description="ZED live viewer (SDK only)")
    parser.add_argument("--resolution", default="WVGA", help="ZED resolution: 2K/1080P/720P/WVGA")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--show_right", action="store_true", help="Display right camera as well")

    args = parser.parse_args()

    _run_with_pyzed(args.resolution, args.fps, args.show_right)


if __name__ == "__main__":
    main()
