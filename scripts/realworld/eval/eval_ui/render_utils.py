from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from scripts.utils.draw_overlay import draw_future_trajectory
from scripts.utils.draw_overlay import draw_prompt_point
from scripts.utils.draw_overlay import draw_tcp_point
from scripts.utils.draw_overlay import make_trajectory_gradient

PANEL_HEADER_HEIGHT = 32
PREVIEW_GAP = 16
CONTROL_PANEL_HEIGHT = 420
PREVIEW_IMAGE_TARGET_HEIGHT = 520


def compose_preview(
    left_image: np.ndarray,
    wrist_image: np.ndarray,
    control_panel: np.ndarray,
    *,
    gap: int = PREVIEW_GAP,
) -> np.ndarray:
    left_panel = draw_panel_header(left_image, "Image Prompt")
    wrist_panel = draw_panel_header(wrist_image, "Wrist Image")
    max_height = max(left_panel.shape[0], wrist_panel.shape[0])
    left_panel = pad_to_height(left_panel, max_height)
    wrist_panel = pad_to_height(wrist_panel, max_height)
    spacer = np.full((max_height, gap, 3), 235, dtype=np.uint8)
    top_row = np.concatenate([left_panel, spacer, wrist_panel], axis=1)
    final_width = max(top_row.shape[1], control_panel.shape[1])
    top_row = pad_panel_width(top_row, final_width, fill_value=235)
    control_panel = pad_panel_width(control_panel, final_width, fill_value=246)
    row_gap = np.full((gap, final_width, 3), 235, dtype=np.uint8)
    return np.concatenate([top_row, row_gap, control_panel], axis=0)


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if target_height <= 0 or image.shape[0] <= 0:
        return image
    scale = target_height / float(image.shape[0])
    target_width = max(1, int(round(image.shape[1] * scale)))
    interpolation = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def draw_panel_header(image: np.ndarray, title: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(image, PANEL_HEADER_HEIGHT, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), 245, dtype=np.uint8)
    return np.concatenate([image, pad], axis=0)


def pad_panel_width(image: np.ndarray, width: int, *, fill_value: int) -> np.ndarray:
    if image.shape[1] >= width:
        return image
    pad = np.full((image.shape[0], width - image.shape[1], 3), fill_value, dtype=np.uint8)
    return np.concatenate([image, pad], axis=1)


def render_prompt_controls(
    *,
    wrist_values: np.ndarray,
    primitive_values: np.ndarray,
    phase2_steps: float | None = None,
    max_phase2_steps: float | None = None,
    prompt_effect_mode: str = "long_term",
    prompt_phase1_source: str = "evo",
    width: int = 600,
) -> np.ndarray:
    canvas = np.full((CONTROL_PANEL_HEIGHT, width, 3), 246, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Prompt Controls",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Enter confirms. Backspace removes the last image stroke. [/] adjust phase2 steps by 0.2.",
        (18, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Image: left-drag trajectory, right-click target point. Local/Global: drag sliders.",
        (18, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Global action:",
        (18, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    global_regions = get_global_bar_regions(width)
    for axis, value, region in zip("xyz", primitive_values, global_regions, strict=False):
        y_mid = (region["y0"] + region["y1"]) // 2
        x_mid = (region["x0"] + region["x1"]) // 2
        cv2.line(canvas, (region["x0"], y_mid), (region["x1"], y_mid), (255, 255, 255), 10, cv2.LINE_AA)
        cv2.line(canvas, (region["x0"], y_mid), (region["x1"], y_mid), (55, 110, 210), 3, cv2.LINE_AA)
        cv2.line(canvas, (x_mid, region["y0"] - 5), (x_mid, region["y1"] + 5), (175, 190, 215), 2, cv2.LINE_AA)
        handle_x = slider_x_from_value(region, float(value))
        cv2.circle(canvas, (handle_x, y_mid), 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (handle_x, y_mid), 10, (55, 110, 210), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{axis}:{float(value):+.2f}",
            (18, y_mid + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )

    effect_y0 = 224
    effect_button_width = 164
    effect_button_height = 42
    effect_gap = 12
    cv2.putText(
        canvas,
        "Phase1 scope:",
        (18, effect_y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    effect_start_x = 180
    for index, (mode, label) in enumerate(
        (("short_term", "Short-term"), ("long_term", "Long-term"))
    ):
        x0 = effect_start_x + index * (effect_button_width + effect_gap)
        x1 = x0 + effect_button_width
        y1 = effect_y0 + effect_button_height
        active = prompt_effect_mode == mode
        fill = (225, 238, 255) if active else (238, 238, 238)
        border = (55, 110, 210) if active else (150, 150, 150)
        cv2.rectangle(canvas, (x0, effect_y0), (x1, y1), fill, -1)
        cv2.rectangle(canvas, (x0, effect_y0), (x1, y1), border, 2)
        cv2.putText(
            canvas,
            label,
            (x0 + 18, effect_y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )

    source_y0 = 276
    source_button_width = 164
    source_button_height = 42
    source_gap = 12
    cv2.putText(
        canvas,
        "Phase1 source:",
        (18, source_y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    source_start_x = 180
    for index, (source, label) in enumerate((("evo", "Evo-1"), ("hardcode", "Hardcode"))):
        x0 = source_start_x + index * (source_button_width + source_gap)
        x1 = x0 + source_button_width
        y1 = source_y0 + source_button_height
        active = prompt_phase1_source == source
        fill = (225, 238, 255) if active else (238, 238, 238)
        border = (55, 110, 210) if active else (150, 150, 150)
        cv2.rectangle(canvas, (x0, source_y0), (x1, y1), fill, -1)
        cv2.rectangle(canvas, (x0, source_y0), (x1, y1), border, 2)
        cv2.putText(
            canvas,
            label,
            (x0 + 18, source_y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )

    wrist_text = "Local motion: " + " ".join(
        f"{axis}={value:+.2f}" for axis, value in zip("xyz", wrist_values, strict=False)
    )
    primitive_text = "Global motion: " + " ".join(
        f"{axis}={value:+.2f}" for axis, value in zip("xyz", primitive_values, strict=False)
    )
    cv2.putText(
        canvas,
        wrist_text,
        (18, 348),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        primitive_text,
        (18, 372),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    if phase2_steps is not None:
        phase2_text = f"Phase2 steps: {float(phase2_steps):.1f}"
        if max_phase2_steps is not None:
            phase2_text += f" / {float(max_phase2_steps):.1f}"
        cv2.putText(
            canvas,
            phase2_text,
            (18, 396),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
    return canvas


def get_global_bar_regions(width: int) -> list[dict[str, int]]:
    bar_width = min(440, max(220, int(width * 0.46)))
    bar_height = 18
    gap = 18
    left = 150
    top = 118
    regions = []
    for index in range(3):
        y0 = top + index * (bar_height + gap)
        regions.append({"x0": left, "x1": left + bar_width, "y0": y0, "y1": y0 + bar_height})
    return regions


def get_wrist_bar_regions(height: int, width: int, wrist_action_dim: int) -> list[dict[str, int]]:
    bar_width = min(220, max(140, int(width * 0.65)))
    bar_height = 24
    gap = 18
    left = 24
    bottom_margin = 28
    total_height = wrist_action_dim * bar_height + (wrist_action_dim - 1) * gap
    top = max(24, height - bottom_margin - total_height)
    regions = []
    for index in range(wrist_action_dim):
        y0 = top + index * (bar_height + gap)
        regions.append({"x0": left, "x1": left + bar_width, "y0": y0, "y1": y0 + bar_height})
    return regions


def find_bar_at_point(bar_regions: list[dict[str, int]], x: int, y: int) -> int | None:
    for index, region in enumerate(bar_regions):
        if region["x0"] <= x <= region["x1"] and region["y0"] <= y <= region["y1"]:
            return index
    return None


def slider_value_from_x(region: dict[str, int], x: int) -> float:
    clipped_x = min(max(x, region["x0"]), region["x1"])
    ratio = (clipped_x - region["x0"]) / max(1, region["x1"] - region["x0"])
    return float(np.clip(ratio * 2.0 - 1.0, -1.0, 1.0))


def slider_x_from_value(region: dict[str, int], value: float) -> int:
    ratio = (float(np.clip(value, -1.0, 1.0)) + 1.0) / 2.0
    return int(round(region["x0"] + ratio * (region["x1"] - region["x0"])))


def render_wrist_controls(wrist_image: np.ndarray, wrist_values: np.ndarray, active_index: int | None) -> np.ndarray:
    canvas = wrist_image.copy()
    axis_labels = "xyz"
    bar_regions = get_wrist_bar_regions(canvas.shape[0], canvas.shape[1], len(wrist_values))
    for index, (axis, value, region) in enumerate(zip(axis_labels, wrist_values, bar_regions, strict=False)):
        y_mid = (region["y0"] + region["y1"]) // 2
        x_mid = (region["x0"] + region["x1"]) // 2
        cv2.line(canvas, (region["x0"], y_mid), (region["x1"], y_mid), (255, 255, 255), 10, cv2.LINE_AA)
        cv2.line(canvas, (region["x0"], y_mid), (region["x1"], y_mid), (40, 40, 40), 3, cv2.LINE_AA)
        cv2.line(canvas, (x_mid, region["y0"] - 6), (x_mid, region["y1"] + 6), (200, 200, 200), 2, cv2.LINE_AA)
        handle_x = slider_x_from_value(region, float(value))
        handle_color = (230, 230, 230) if active_index == index else (255, 255, 255)
        cv2.circle(canvas, (handle_x, y_mid), 12, handle_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (handle_x, y_mid), 12, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{axis}:{value:+.2f}",
            (region["x0"], region["y0"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{axis}:{value:+.2f}",
            (region["x0"], region["y0"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
    return canvas


def shape_from_stroke(points_xy: list[tuple[int, int]]) -> dict[str, Any]:
    if not points_xy:
        return {"type": "stroke", "points_hw": [], "thickness": 1}
    return {
        "type": "stroke",
        "points_hw": [[float(y), float(x)] for x, y in points_xy],
        "thickness": 1,
    }


def shape_from_point(point_xy: tuple[float, float]) -> dict[str, Any]:
    return {
        "type": "point",
        "point_hw": [float(point_xy[1]), float(point_xy[0])],
        "source": "target_point",
    }


def render_draw_ops(image: np.ndarray, draw_ops: list[dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    for op in draw_ops:
        if op["type"] == "stroke":
            draw_outlined_stroke(
                canvas,
                points_hw=np.asarray(op["points_hw"], dtype=float),
                thickness=int(op.get("thickness", 2)),
            )
        elif op["type"] == "point":
            draw_point(canvas, point_hw=np.asarray(op["point_hw"], dtype=float))
    return canvas


def draw_point(image: np.ndarray, point_hw: np.ndarray, radius: int = 6) -> None:
    if point_hw.shape != (2,) or not np.all(np.isfinite(point_hw)):
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rendered_rgb = draw_prompt_point(image_rgb, point_hw, radius=radius)
    image[...] = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)


def draw_outlined_stroke(image: np.ndarray, points_hw: np.ndarray, thickness: int = 1) -> None:
    if len(points_hw) == 0:
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if len(points_hw) == 1:
        rendered_rgb = draw_tcp_point(image_rgb, points_hw[0])
        image[...] = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        return
    rendered_rgb = draw_future_trajectory(
        image_rgb,
        points_hw,
        color=make_trajectory_gradient(),
        thickness=thickness,
    )
    image[...] = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
