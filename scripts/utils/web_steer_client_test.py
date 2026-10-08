from __future__ import annotations

import base64

import cv2
import numpy as np

from scripts.utils.web_steer_client import _decode_prompt_payload
from scripts.utils.web_steer_client import _encode_jpeg_data_url


def test_camera_encode_and_prompt_decode() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[..., 0] = 220
    data_url = _encode_jpeg_data_url(image)
    assert data_url.startswith("data:image/jpeg;base64,")

    png_ok, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert png_ok
    prompt = _decode_prompt_payload(
        {
            "sequence": 5,
            "mode": "combined",
            "prompt_images": {"prompt_0": "data:image/png;base64," + base64.b64encode(png).decode()},
            "prompt_image_masks": {"prompt_0": True},
            "prompt_global_motion": [0.1, 0.2, -0.3],
            "prompt_global_motion_mask": True,
            "prompt_local_motion": [0, 0, 0],
            "prompt_local_motion_mask": False,
            "prompt_2d_drag": [0.4, -0.2],
            "prompt_2d_drag_mask": True,
            "sample_kwargs": {"phase2_steps": 1.2},
            "prompt_effect_mode": "long_term",
            "prompt_phase1_source": "evo",
            "draw_ops": [{"type": "point"}],
        }
    )
    assert prompt["prompt_images"]["prompt_0"].shape == image.shape
    assert prompt["prompt_global_motion"].dtype == np.float32
    assert bool(prompt["prompt_2d_drag_mask"])
    assert "sequence" not in prompt
    assert "draw_ops" not in prompt
