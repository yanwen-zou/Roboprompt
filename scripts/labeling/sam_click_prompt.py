from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SAM2_REPO = ROOT / "sam2"
if str(SAM2_REPO) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_SAM2_CHECKPOINT = ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_small.pt"


def _can_open_cv_window() -> bool:
    # On headless / remote shells, attempting to create a Qt window can abort the
    # whole process before OpenCV raises a Python exception. Guard it proactively.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


def save_click_reference(frame: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = output_dir / "frame0_for_click.png"
    cv2.imwrite(str(ref_path), frame)
    return ref_path


def prompt_point_from_terminal(frame: np.ndarray, output_dir: Path) -> tuple[int, int]:
    ref_path = save_click_reference(frame, output_dir)
    height, width = frame.shape[:2]
    print(f"Open this image and choose a point: {ref_path}")
    print(f"Frame size: width={width}, height={height}")
    while True:
        raw = input("Enter click coordinates as 'x y': ").strip()
        try:
            x_str, y_str = raw.split()
            x = int(x_str)
            y = int(y_str)
        except ValueError:
            print("Invalid input. Expected two integers like: 320 180")
            continue
        if 0 <= x < width and 0 <= y < height:
            return x, y
        print("Coordinates out of bounds. Try again.")


def prompt_points_from_terminal(
    frame: np.ndarray, output_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    ref_path = save_click_reference(frame, output_dir)
    height, width = frame.shape[:2]
    print(f"Open this image and choose prompt points: {ref_path}")
    print(f"Frame size: width={width}, height={height}")
    print("Enter one point per line as: <p|n> x y")
    print("Example: p 320 180")
    print("Enter an empty line to finish.")

    points: list[tuple[int, int]] = []
    labels: list[int] = []
    while True:
        raw = input("prompt> ").strip()
        if not raw:
            break
        try:
            kind, x_str, y_str = raw.split()
            x = int(x_str)
            y = int(y_str)
        except ValueError:
            print("Invalid input. Expected: p 320 180 or n 320 180")
            continue
        if kind not in {"p", "n"}:
            print("Point type must be 'p' for positive or 'n' for negative.")
            continue
        if not (0 <= x < width and 0 <= y < height):
            print("Coordinates out of bounds. Try again.")
            continue
        points.append((x, y))
        labels.append(1 if kind == "p" else 0)

    if not points or 1 not in labels:
        raise SystemExit("Need at least one positive point.")

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def get_click_point(frame: np.ndarray, output_dir: Path) -> tuple[int, int]:
    window_name = "SAM2 First Frame"
    selected: list[tuple[int, int]] = []
    preview = frame.copy()

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.clear()
            selected.append((x, y))

    if not _can_open_cv_window():
        return prompt_point_from_terminal(frame, output_dir)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
    except cv2.error:
        return prompt_point_from_terminal(frame, output_dir)

    while True:
        canvas = preview.copy()
        if selected:
            cv2.circle(canvas, selected[0], 8, (0, 255, 0), -1)
            cv2.circle(canvas, selected[0], 16, (255, 255, 255), 2)
        cv2.putText(
            canvas,
            "Left click a point on frame 0, then press Enter/Space to continue. Esc to quit.",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and selected:
            cv2.destroyWindow(window_name)
            return selected[0]
        if key == 27:
            cv2.destroyWindow(window_name)
            raise SystemExit("Canceled by user.")


def get_prompt_points(frame: np.ndarray, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    window_name = "SAM2 First Frame"
    selected_points: list[tuple[int, int]] = []
    selected_labels: list[int] = []
    preview = frame.copy()

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x, y))
            selected_labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            selected_points.append((x, y))
            selected_labels.append(0)

    if not _can_open_cv_window():
        return prompt_points_from_terminal(frame, output_dir)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
    except cv2.error:
        return prompt_points_from_terminal(frame, output_dir)

    while True:
        canvas = preview.copy()
        for (x, y), label in zip(selected_points, selected_labels):
            if label == 1:
                color = (0, 255, 0)
                cv2.circle(canvas, (x, y), 8, color, -1)
            else:
                color = (0, 0, 255)
                cv2.drawMarker(
                    canvas,
                    (x, y),
                    color,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=18,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
            cv2.circle(canvas, (x, y), 16, (255, 255, 255), 2)
        cv2.putText(
            canvas,
            "Left=positive Right=negative Backspace=undo Enter/Space=confirm Esc=quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (8, 127) and selected_points:
            selected_points.pop()
            selected_labels.pop()
        elif key in (13, 32):
            if not selected_points or 1 not in selected_labels:
                print("Need at least one positive point before confirming.")
                continue
            cv2.destroyWindow(window_name)
            return (
                np.array(selected_points, dtype=np.float32),
                np.array(selected_labels, dtype=np.int32),
            )
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise SystemExit("Canceled by user.")


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.array([0, 200, 80], dtype=np.uint8)
    overlay = frame.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * color).astype(np.uint8)
    return overlay


def to_numpy_mask(mask_tensor: torch.Tensor, score_thresh: float) -> np.ndarray:
    return (mask_tensor > score_thresh).squeeze().detach().cpu().numpy().astype(bool)


def load_sam2_image_predictor(
    *,
    device: str,
    config: str = DEFAULT_SAM2_CONFIG,
    checkpoint: Path = DEFAULT_SAM2_CHECKPOINT,
) -> SAM2ImagePredictor:
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    predictor = SAM2ImagePredictor(
        build_sam2(
            config_file=config,
            ckpt_path=str(checkpoint),
            device=device,
        )
    )
    return predictor


def sam2_autocast_context(device: str):
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def predict_mask_from_points(
    frame: np.ndarray,
    *,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    predictor: SAM2ImagePredictor,
) -> tuple[np.ndarray, float]:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords.astype(np.float32),
        point_labels=point_labels.astype(np.int32),
        multimask_output=True,
        normalize_coords=True,
    )
    best_index = int(np.argmax(scores))
    return masks[best_index].astype(bool), float(scores[best_index])


def predict_mask_from_click(
    frame: np.ndarray,
    *,
    point_xy: tuple[int, int],
    predictor: SAM2ImagePredictor,
) -> tuple[np.ndarray, float]:
    return predict_mask_from_points(
        frame,
        point_coords=np.array([point_xy], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        predictor=predictor,
    )
