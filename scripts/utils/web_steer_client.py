from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
_MODEL_PROMPT_KEYS = {
    "prompt_arms",
    "prompt_images",
    "prompt_image_masks",
    "prompt_global_motion",
    "prompt_global_motion_mask",
    "prompt_local_motion",
    "prompt_local_motion_mask",
    "prompt_2d_drag",
    "prompt_2d_drag_mask",
    "sample_kwargs",
    "prompt_effect_mode",
    "prompt_phase1_source",
}


@dataclass(frozen=True)
class WebPromptUpdate:
    sequence: int
    payload: dict[str, Any]


class WebSteerPromptClient:
    """Bridge camera observations and interactive prompts through web_steer."""

    def __init__(self, base_url: str, *, publish_hz: float = 5.0, timeout: float = 1.0) -> None:
        if publish_hz <= 0:
            raise ValueError(f"publish_hz must be positive, got {publish_hz}.")
        self._base_url = base_url.rstrip("/")
        self._publish_interval = 1.0 / publish_hz
        self._timeout = timeout
        self._last_publish_time = float("-inf")
        self._last_prompt_sequence = self._initial_prompt_sequence()
        self._connection_warning_logged = False

    def publish_observation(
        self,
        *,
        base_image_rgb: np.ndarray,
        wrist_image_rgb: np.ndarray | None,
        observation_state: np.ndarray | None,
        episode: int,
        step: int,
        force: bool = False,
    ) -> bool:
        now = time.monotonic()
        if not force and now - self._last_publish_time < self._publish_interval:
            return False
        payload: dict[str, Any] = {
            "base_image": _encode_jpeg_data_url(base_image_rgb),
            "frame_id": f"E{episode:03d}-S{step:06d}",
            "episode": int(episode),
            "step": int(step),
        }
        if wrist_image_rgb is not None:
            payload["wrist_image"] = _encode_jpeg_data_url(wrist_image_rgb)
        if observation_state is not None:
            payload["state"] = np.asarray(observation_state, dtype=np.float32).reshape(-1).tolist()
        try:
            self._request_json("/api/observation", method="POST", payload=payload)
        except (OSError, ValueError) as exc:
            self._warn_connection(exc)
            return False
        self._last_publish_time = now
        self._connection_warning_logged = False
        return True

    def poll_prompt(self) -> WebPromptUpdate | None:
        try:
            response = self._request_json(f"/api/prompt?after={self._last_prompt_sequence}", allow_empty=True)
        except (OSError, ValueError) as exc:
            self._warn_connection(exc)
            return None
        if response is None:
            return None
        sequence = int(response.get("sequence", 0))
        if sequence <= self._last_prompt_sequence:
            return None
        payload = _decode_prompt_payload(response)
        self._last_prompt_sequence = sequence
        self._connection_warning_logged = False
        return WebPromptUpdate(sequence=sequence, payload=payload)

    def acknowledge_inference(self, sequence: int) -> None:
        try:
            self._request_json(f"/api/prompt/{int(sequence)}/ack", method="POST", payload={})
        except (OSError, ValueError) as exc:
            self._warn_connection(exc)

    def _initial_prompt_sequence(self) -> int:
        try:
            state = self._request_json("/api/state")
        except (OSError, ValueError):
            return 0
        return int((state or {}).get("prompt_sequence", 0))

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> dict[str, Any] | None:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = url_request.Request(self._base_url + path, data=body, headers=headers, method=method)
        try:
            with url_request.urlopen(request, timeout=self._timeout) as response:
                if response.status == 204 and allow_empty:
                    return None
                data = response.read()
        except url_error.HTTPError as exc:
            if exc.code == 204 and allow_empty:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            raise OSError(f"web_steer returned HTTP {exc.code}: {detail}") from exc
        if not data:
            return None
        decoded = json.loads(data)
        if not isinstance(decoded, dict):
            raise ValueError("web_steer response must be a JSON object.")
        return decoded

    def _warn_connection(self, exc: Exception) -> None:
        if self._connection_warning_logged:
            return
        LOGGER.warning("web_steer is unavailable at %s: %s", self._base_url, exc)
        self._connection_warning_logged = True


def _encode_jpeg_data_url(image_rgb: np.ndarray) -> str:
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise ValueError("Failed to encode camera observation as JPEG.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _decode_prompt_payload(response: dict[str, Any]) -> dict[str, Any]:
    payload = {key: response[key] for key in _MODEL_PROMPT_KEYS if key in response}
    prompt_images = payload.get("prompt_images")
    if isinstance(prompt_images, dict):
        payload["prompt_images"] = {
            key: _decode_image_data_url(value) for key, value in prompt_images.items() if isinstance(value, str)
        }
    masks = payload.get("prompt_image_masks")
    if isinstance(masks, dict):
        payload["prompt_image_masks"] = {key: np.bool_(value) for key, value in masks.items()}
    for key, size in (
        ("prompt_global_motion", 3),
        ("prompt_local_motion", 3),
        ("prompt_2d_drag", 2),
    ):
        value = payload.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (size,):
            raise ValueError(f"web_steer {key} has shape {array.shape}, expected ({size},).")
        payload[key] = array
    for key in ("prompt_global_motion_mask", "prompt_local_motion_mask", "prompt_2d_drag_mask"):
        if key in payload:
            payload[key] = np.bool_(payload[key])
    return payload


def _decode_image_data_url(value: str) -> np.ndarray:
    encoded = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("web_steer prompt image is not valid base64.") from exc
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("web_steer prompt image could not be decoded.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
