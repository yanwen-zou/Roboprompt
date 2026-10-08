from __future__ import annotations

import argparse
import base64
import binascii
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 16 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@dataclass
class SteerState:
    condition: threading.Condition = field(default_factory=threading.Condition)
    observation_sequence: int = 0
    prompt_sequence: int = 0
    observation: dict[str, Any] | None = None
    images: dict[str, tuple[bytes, str]] = field(default_factory=dict)
    prompt: dict[str, Any] | None = None
    acknowledged_prompt_sequence: int = 0
    acknowledged_at: float | None = None


def create_app(*, allowed_origins: list[str] | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    origins = {origin.rstrip("/") for origin in (allowed_origins or [])}
    state = SteerState()
    app.config["STEER_STATE"] = state

    @app.before_request
    def check_origin() -> Response | None:
        origin = request.headers.get("Origin")
        if request.path.startswith("/api/") and origin:
            if origin != request.host_url.rstrip("/") and origin not in origins:
                return jsonify(error="Origin is not allowed. Configure --allowed-origin on the server."), 403
        return None

    @app.after_request
    def prevent_stale_api(response: Response) -> Response:
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.vary.add("Origin")
            origin = request.headers.get("Origin")
            if origin in origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type"
                if request.headers.get("Access-Control-Request-Private-Network") == "true":
                    response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    @app.get("/")
    def index() -> Response:
        return send_from_directory(ROOT, "index.html")

    @app.get("/<path:asset>")
    def assets(asset: str) -> Response:
        if asset not in {"app.js", "styles.css"}:
            return jsonify(error="Not found"), 404
        return send_from_directory(ROOT, asset)

    @app.post("/api/observation")
    def receive_observation() -> Response:
        try:
            images, metadata = _parse_observation_request()
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if "base" not in images:
            return jsonify(error="A base camera image is required."), 400

        with state.condition:
            state.observation_sequence += 1
            sequence = state.observation_sequence
            metadata["sequence"] = sequence
            metadata["received_at"] = time.time()
            state.observation = metadata
            state.images = images
            state.condition.notify_all()
        return jsonify(ok=True, sequence=sequence), 202

    @app.get("/api/state")
    def current_state() -> Response:
        with state.condition:
            observation = dict(state.observation or {})
            image_keys = sorted(state.images)
            prompt_sequence = state.prompt_sequence
            has_prompt = state.prompt is not None
            acknowledged_prompt_sequence = state.acknowledged_prompt_sequence
            acknowledged_at = state.acknowledged_at
        last_seen = observation.get("received_at")
        connected = isinstance(last_seen, (int, float)) and time.time() - last_seen < 3.0
        return jsonify(
            connected=connected,
            observation=observation or None,
            image_keys=image_keys,
            prompt_sequence=prompt_sequence,
            has_prompt=has_prompt,
            acknowledged_prompt_sequence=acknowledged_prompt_sequence,
            acknowledged_at=acknowledged_at,
        )

    @app.get("/api/observation/<camera>")
    def camera_image(camera: str) -> Response:
        with state.condition:
            image = state.images.get(camera)
        if image is None:
            return jsonify(error=f"Camera {camera!r} is unavailable."), 404
        data, mime = image
        return Response(data, mimetype=mime, headers={"Content-Length": str(len(data))})

    @app.get("/api/events")
    def events() -> Response:
        def stream():
            last_observation = -1
            last_prompt = -1
            while True:
                with state.condition:
                    state.condition.wait_for(
                        lambda: state.observation_sequence != last_observation
                        or state.prompt_sequence != last_prompt,
                        timeout=15,
                    )
                    current_observation = state.observation_sequence
                    current_prompt = state.prompt_sequence
                if current_observation == last_observation and current_prompt == last_prompt:
                    yield ": keepalive\n\n"
                    continue
                last_observation = current_observation
                last_prompt = current_prompt
                payload = json.dumps(
                    {"observation_sequence": current_observation, "prompt_sequence": current_prompt}
                )
                yield f"data: {payload}\n\n"

        return Response(stream(), mimetype="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.post("/api/prompt")
    def submit_prompt() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a JSON prompt object."), 400
        try:
            normalized = _validate_prompt(payload)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        with state.condition:
            state.prompt_sequence += 1
            sequence = state.prompt_sequence
            normalized["sequence"] = sequence
            normalized["created_at"] = time.time()
            normalized.setdefault("observation_sequence", state.observation_sequence)
            state.prompt = normalized
            state.condition.notify_all()
        return jsonify(ok=True, sequence=sequence), 201

    @app.post("/api/prompt/<int:sequence>/ack")
    def acknowledge_prompt(sequence: int) -> Response:
        with state.condition:
            if sequence <= 0 or sequence > state.prompt_sequence:
                return jsonify(error=f"Unknown prompt sequence {sequence}."), 404
            state.acknowledged_prompt_sequence = max(state.acknowledged_prompt_sequence, sequence)
            state.acknowledged_at = time.time()
            state.condition.notify_all()
        return jsonify(ok=True, sequence=sequence, status="inferred")

    @app.get("/api/prompt")
    def get_prompt() -> Response:
        after = request.args.get("after", default=0, type=int)
        wait = min(max(request.args.get("wait", default=0.0, type=float), 0.0), 30.0)
        with state.condition:
            if state.prompt_sequence <= after and wait:
                state.condition.wait_for(lambda: state.prompt_sequence > after, timeout=wait)
            if state.prompt is None or state.prompt_sequence <= after:
                return Response(status=204)
            prompt = dict(state.prompt)
        return jsonify(prompt)

    @app.delete("/api/prompt")
    def clear_prompt() -> Response:
        with state.condition:
            state.prompt = None
            state.condition.notify_all()
        return jsonify(ok=True)

    @app.get("/healthz")
    def health() -> Response:
        return jsonify(ok=True)

    return app


def _parse_observation_request() -> tuple[dict[str, tuple[bytes, str]], dict[str, Any]]:
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON observation object.")
        images: dict[str, tuple[bytes, str]] = {}
        aliases = {
            "base": ("base_image", "image", "observation/image"),
            "wrist": ("wrist_image", "observation/wrist_image"),
        }
        for camera, keys in aliases.items():
            encoded = next((payload[key] for key in keys if payload.get(key)), None)
            if encoded is not None:
                images[camera] = _decode_data_image(encoded)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object.")
        for key in ("frame_id", "timestamp", "state", "episode", "step"):
            if key in payload:
                metadata[key] = payload[key]
        return images, metadata

    images = {}
    for camera, names in {"base": ("base_image", "image"), "wrist": ("wrist_image",)}.items():
        upload = next((request.files[name] for name in names if name in request.files), None)
        if upload is None:
            continue
        data = upload.read(MAX_IMAGE_BYTES + 1)
        mime = upload.mimetype or "application/octet-stream"
        _validate_image(data, mime)
        images[camera] = (data, mime)
    raw_metadata = request.form.get("metadata", "{}")
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata is not valid JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object.")
    return images, metadata


def _decode_data_image(value: Any) -> tuple[bytes, str]:
    if not isinstance(value, str):
        raise ValueError("JSON images must be base64 strings or data URLs.")
    mime = "image/jpeg"
    encoded = value
    if value.startswith("data:"):
        try:
            header, encoded = value.split(",", 1)
            mime = header[5:].split(";", 1)[0]
        except ValueError as exc:
            raise ValueError("Malformed image data URL.") from exc
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image is not valid base64.") from exc
    _validate_image(data, mime)
    return data, mime


def _validate_image(data: bytes, mime: str) -> None:
    if not data:
        raise ValueError("Uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Uploaded image exceeds 16 MiB.")
    if mime not in ALLOWED_IMAGE_TYPES:
        raise ValueError(f"Unsupported image type {mime!r}.")


def _vector(payload: dict[str, Any], key: str, size: int) -> list[float]:
    value = payload.get(key, [0.0] * size)
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{key} must contain exactly {size} values.")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must contain numbers.") from exc
    if any(not -1.0 <= item <= 1.0 for item in result):
        raise ValueError(f"{key} values must be within [-1, 1].")
    return result


def _validate_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode", "trajectory"))
    if mode not in {"trajectory", "point", "global", "combined"}:
        raise ValueError("mode must be trajectory, point, global, or combined.")
    effect = str(payload.get("prompt_effect_mode", "long_term"))
    if effect not in {"short_term", "long_term"}:
        raise ValueError("prompt_effect_mode must be short_term or long_term.")
    source = str(payload.get("prompt_phase1_source", "evo"))
    if source not in {"evo", "hardcode"}:
        raise ValueError("prompt_phase1_source must be evo or hardcode.")

    global_motion = _vector(payload, "prompt_global_motion", 3)
    drag = _vector(payload, "prompt_2d_drag", 2)
    draw_ops = payload.get("draw_ops", [])
    if not isinstance(draw_ops, list):
        raise ValueError("draw_ops must be a list.")
    prompt_image = payload.get("prompt_image")
    if prompt_image is not None:
        _decode_data_image(prompt_image)

    phase2_steps = payload.get("phase2_steps", 0.0)
    try:
        phase2_steps = max(0.0, float(phase2_steps))
    except (TypeError, ValueError) as exc:
        raise ValueError("phase2_steps must be a non-negative number.") from exc
    image_active = bool(draw_ops and prompt_image)
    global_active = any(abs(value) > 1e-6 for value in global_motion)
    drag_active = mode in {"trajectory", "combined"} and any(abs(value) > 1e-6 for value in drag)
    return {
        "mode": mode,
        "prompt_images": {"prompt_0": prompt_image} if prompt_image else {},
        "prompt_image_masks": {"prompt_0": image_active},
        "prompt_global_motion": global_motion,
        "prompt_global_motion_mask": global_active,
        "prompt_local_motion": [0.0, 0.0, 0.0],
        "prompt_local_motion_mask": False,
        "prompt_2d_drag": drag,
        "prompt_2d_drag_mask": drag_active,
        "sample_kwargs": {"phase2_steps": phase2_steps},
        "prompt_effect_mode": effect,
        "prompt_phase1_source": source,
        "draw_ops": draw_ops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboPrompt browser steering UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allowed-origin", action="append", default=[],
                        help="Allowed browser origin, e.g. https://USER.github.io (repeatable; no repository path)")
    args = parser.parse_args()
    create_app(allowed_origins=args.allowed_origin).run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
