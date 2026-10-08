from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

OBJECT_KEY = "object"
TCP_KEY = "tcp"
TARGET_KEY = "target"
DEFAULT_TRAJECTORY_COLOR = (255, 80, 0)
DEFAULT_TRAJECTORY_START_COLOR = (255, 255, 255)
DEFAULT_PROMPT_POINT_COLOR = DEFAULT_TRAJECTORY_COLOR


def make_trajectory_gradient(color=DEFAULT_TRAJECTORY_COLOR, start_color=DEFAULT_TRAJECTORY_START_COLOR):
    return (tuple(int(v) for v in start_color), tuple(int(v) for v in color))


def draw_disk(image, center_hw, color, radius=6):
    """Draw a filled disk directly into an RGB numpy image."""
    h, w = image.shape[:2]
    cy, cx = int(center_hw[0]), int(center_hw[1])
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius + 1)
    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius + 1)
    if y_min >= y_max or x_min >= x_max:
        return image

    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    image[y_min:y_max, x_min:x_max][mask] = np.asarray(color, dtype=np.uint8)
    return image


def draw_line(image, start_hw, end_hw, color, thickness=2):
    """Draw a simple thick line between two pixels."""
    if start_hw is None or end_hw is None:
        return image

    y0, x0 = np.asarray(start_hw, dtype=float)
    y1, x1 = np.asarray(end_hw, dtype=float)
    steps = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    if steps <= 1:
        return draw_disk(image, start_hw, color, radius=thickness)

    ys = np.linspace(y0, y1, steps)
    xs = np.linspace(x0, x1, steps)
    for y, x in zip(ys, xs):
        draw_disk(image, (int(round(y)), int(round(x))), color, radius=thickness)
    return image


def draw_future_trajectory(image, points_hw, color=DEFAULT_TRAJECTORY_COLOR, thickness=1):
    """Draw future trajectory.

    If color is a single color, draws with white border and that color line.
    If color is a tuple/list of two colors, draws a gradient line from
    start_color to end_color without border.
    """
    if len(points_hw) < 2:
        return image

    base_thickness = max(1, int(thickness * 1.2))

    if isinstance(color, (tuple, list)) and len(color) == 2 and isinstance(color[0], (tuple, list, np.ndarray)):
        start_color = np.asarray(color[0], dtype=np.float32)
        end_color = np.asarray(color[1], dtype=np.float32)

        for idx in range(1, len(points_hw)):
            t = (idx - 1) / max(len(points_hw) - 2, 1)
            c = (start_color * (1 - t) + end_color * t).astype(np.uint8)
            draw_line(image, points_hw[idx - 1], points_hw[idx], c.tolist(), thickness=base_thickness)
        return image

    white = np.asarray([255, 255, 255], dtype=np.uint8)
    color_arr = np.asarray(color, dtype=np.uint8)

    for idx in range(1, len(points_hw)):
        draw_line(image, points_hw[idx - 1], points_hw[idx], white, thickness=base_thickness + 2)

    for idx in range(1, len(points_hw)):
        draw_line(image, points_hw[idx - 1], points_hw[idx], color_arr, thickness=base_thickness)

    return image


def draw_outlined_circle(image, center_hw, radius, thickness=1):
    """Draw a black circle with a slightly thicker white outline."""
    if center_hw is None or radius <= 0:
        return image

    white_radius = max(1, int(round(radius + thickness + 1)))
    black_radius = max(1, int(round(radius)))
    draw_disk(image, center_hw, [255, 255, 255], radius=white_radius)
    draw_disk(image, center_hw, [0, 0, 0], radius=black_radius)
    return image


def draw_outlined_ellipse(image, center_hw, axes_hw, angle_deg, thickness=1):
    """Draw a black ellipse with a slightly thicker white outline."""
    import cv2

    if center_hw is None:
        return image

    cy, cx = int(round(center_hw[0])), int(round(center_hw[1]))
    axis_h = max(1, int(round(axes_hw[0])))
    axis_w = max(1, int(round(axes_hw[1])))
    outline_gap = max(2, thickness + 2)
    white_axes = (axis_w + outline_gap, axis_h + outline_gap)
    black_axes = (axis_w, axis_h)
    white_thickness = max(4, thickness + 4)
    black_thickness = max(2, thickness + 2)

    cv2.ellipse(
        image,
        (cx, cy),
        white_axes,
        angle_deg,
        0,
        360,
        (255, 255, 255),
        thickness=white_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (cx, cy),
        black_axes,
        angle_deg,
        0,
        360,
        (0, 0, 0),
        thickness=black_thickness,
        lineType=cv2.LINE_AA,
    )
    return image


def draw_tcp_point(image_rgb, point_hw, radius=6):
    return draw_prompt_point(image_rgb, point_hw, radius=radius)


def draw_prompt_point_inplace(image_rgb, point_hw, radius=6, color=DEFAULT_PROMPT_POINT_COLOR):
    point = np.asarray(point_hw, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return image_rgb
    draw_disk(image_rgb, point, [255, 255, 255], radius=radius + 2)
    draw_disk(image_rgb, point, color, radius=radius)
    return image_rgb


def draw_prompt_point(image_rgb, point_hw, radius=6, color=DEFAULT_PROMPT_POINT_COLOR):
    rendered = image_rgb.copy()
    draw_prompt_point_inplace(rendered, point_hw, radius=radius, color=color)
    return rendered


def draw_prompt_overlays(
    image_rgb,
    *,
    trajectory_points_hw=None,
    point_hw=None,
    draw_trajectory=False,
    draw_point=False,
    trajectory_color=DEFAULT_TRAJECTORY_COLOR,
    point_color=DEFAULT_PROMPT_POINT_COLOR,
    trajectory_thickness=1,
    point_radius=6,
):
    rendered = image_rgb.copy()
    if draw_trajectory and trajectory_points_hw is not None and len(trajectory_points_hw) >= 2:
        rendered = draw_future_trajectory(
            rendered,
            trajectory_points_hw,
            color=make_trajectory_gradient(trajectory_color),
            thickness=trajectory_thickness,
        )
    if draw_point and point_hw is not None:
        draw_prompt_point_inplace(rendered, point_hw, radius=point_radius, color=point_color)
    return rendered


def randomize_prompt_point(point_hw, image_shape, noise_std=3.0):
    point = np.asarray(point_hw, dtype=np.float32).reshape(2)
    if noise_std > 0:
        point = point + np.random.normal(0.0, float(noise_std), size=2).astype(np.float32)
    height, width = int(image_shape[0]), int(image_shape[1])
    point[0] = np.clip(point[0], 0.0, max(height - 1, 0))
    point[1] = np.clip(point[1], 0.0, max(width - 1, 0))
    return point.astype(np.float32)


@functools.lru_cache(maxsize=1024)
def _load_npy_cached(path_str):
    return np.load(path_str)


def _coerce_overlay_points(points_or_path):
    if isinstance(points_or_path, np.ndarray):
        return points_or_path
    if isinstance(points_or_path, (list, tuple)):
        return np.asarray(points_or_path)
    return np.asarray(_load_npy_cached(str(Path(points_or_path).resolve())))


def load_overlay_points(overlay_sources):
    if OBJECT_KEY not in overlay_sources or TCP_KEY not in overlay_sources:
        raise KeyError(f"overlay_sources must contain both '{OBJECT_KEY}' and '{TCP_KEY}' keys")
    return {
        OBJECT_KEY: _coerce_overlay_points(overlay_sources[OBJECT_KEY]),
        TCP_KEY: _coerce_overlay_points(overlay_sources[TCP_KEY]),
    }


def overlay_future_trajectory_for_frame(
    image,
    overlay_sources,
    current_frame,
    future_len=20,
):
    overlay_points = load_overlay_points(overlay_sources)
    object_pixels = overlay_points[OBJECT_KEY]
    tcp_pixels = overlay_points[TCP_KEY]

    frame = image.copy()
    future_end = min(len(object_pixels), current_frame + future_len + 1)
    future_object_pixels = object_pixels[current_frame:future_end]
    future_tcp_pixels = tcp_pixels[current_frame:future_end]

    draw_future_trajectory(frame, future_object_pixels, color=[255, 180, 0], thickness=1)
    draw_future_trajectory(frame, future_tcp_pixels, color=[0, 180, 255], thickness=1)
    return frame


def render_annotation_overlay(
    image,
    object_pixels,
    tcp_pixels,
    frame_index,
    future_len=20,
):
    return overlay_future_trajectory_for_frame(
        image,
        {
            OBJECT_KEY: object_pixels,
            TCP_KEY: tcp_pixels,
        },
        current_frame=frame_index,
        future_len=future_len,
    )
