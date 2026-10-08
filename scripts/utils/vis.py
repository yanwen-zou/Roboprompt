from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import ImageDraw, ImageFont


DEFAULT_MONO_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().float().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {arr.shape}")
    arr = np.transpose(arr, (1, 2, 0))
    if arr.max(initial=0.0) <= 1.01:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_mono_font(size: int = 10) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(DEFAULT_MONO_FONT, size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    if draw.textlength(text, font=font) <= max_width:
        return [text]

    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines if lines else [text]


def format_vector(value: Any, *, precision: int = 5) -> list[float]:
    if isinstance(value, torch.Tensor):
        values = value.detach().float().cpu().reshape(-1).tolist()
    else:
        values = np.asarray(value, dtype=np.float32).reshape(-1).tolist()
    return [round(float(x), precision) for x in values]
