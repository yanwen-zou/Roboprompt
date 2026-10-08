#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "openpi").is_dir() and (path / "steering").is_dir():
            return path
    raise FileNotFoundError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
for path in (
    REPO_ROOT,
    REPO_ROOT / "openpi" / "src",
    REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep FastWAM FRS denoise settings on dumped Evo1 observations.")
    parser.add_argument("--input-dir", type=Path, default=Path("output/evo1/raw_fastwam"))
    parser.add_argument("--indices", default="0", help="Comma-separated dump indices, e.g. 0,1,2.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--phase2-steps", type=float, default=0.6)
    parser.add_argument("--sigmas", default="0,0.2,0.4,0.6,0.8,1")
    parser.add_argument("--plain-steps", default="0,0.6,1,2,4,8,12,20")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("/tmp/fastwam_frs_step_sweep.jsonl"))
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_dump_observation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_paths = data["raw_image_paths"]
    debug = data.get("debug", {})
    raw_debug = debug.get("raw_input_debug", {})

    raw_state = debug.get("raw_state_first8")
    if raw_state is None:
        raise KeyError(f"{path} does not contain debug.raw_state_first8")

    obs: dict[str, Any] = {
        "observation/state": np.asarray(raw_state, dtype=np.float32),
        "observation/image": load_rgb(raw_paths["observation_image"]),
        "observation/wrist_image": load_rgb(raw_paths["observation_wrist_image"]),
        "prompt": str(data.get("prompt", "")),
    }

    if "prompt_0" in raw_paths:
        obs["prompt_images"] = {"prompt_0": load_rgb(raw_paths["prompt_0"])}
        if "prompt_overlay_0" in raw_paths:
            obs["prompt_images"]["prompt_overlay_0"] = load_rgb(raw_paths["prompt_overlay_0"])
        prompt_mask = bool(raw_debug.get("prompt_image_mask_prompt_0", False))
        obs["prompt_image_masks"] = {"prompt_0": prompt_mask}

    return obs, data


def action_metrics(actions: Any) -> dict[str, Any]:
    action = np.asarray(actions, dtype=np.float32)
    if action.ndim != 2 or action.shape[-1] < 3:
        raise ValueError(f"Expected action chunk [T,D>=3], got {action.shape}")
    xyz = action[:, :3]
    delta = np.diff(xyz, axis=0)
    accel = np.diff(delta, axis=0)
    return {
        "delta_mean": float(np.linalg.norm(delta, axis=-1).mean()) if len(delta) else 0.0,
        "delta_max": float(np.linalg.norm(delta, axis=-1).max()) if len(delta) else 0.0,
        "accel_mean": float(np.linalg.norm(accel, axis=-1).mean()) if len(accel) else 0.0,
        "accel_max": float(np.linalg.norm(accel, axis=-1).max()) if len(accel) else 0.0,
        "sum_xyz": xyz.sum(axis=0).astype(float).tolist(),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    return value


def run_one(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    obs: dict[str, Any],
    *,
    label: str,
    sample_kwargs: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    result = policy.infer(obs, sample_kwargs=sample_kwargs)
    elapsed = time.time() - started
    row = {
        "label": label,
        "elapsed_s": elapsed,
        "sample_kwargs": sample_kwargs,
        "steerer": result.get("steerer"),
        "phase2_debug": result.get("phase2_debug"),
        "metrics": action_metrics(result["actions"]),
    }
    return json_ready(row)


def main() -> None:
    args = parse_args()
    indices = parse_int_list(args.indices)
    sigmas = parse_float_list(args.sigmas)
    plain_steps = parse_float_list(args.plain_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = policy.get_server_metadata()
    print("metadata", json.dumps(json_ready(metadata), ensure_ascii=False))

    base_kwargs = {
        "enable_steerer": True,
        "enable_evo1_steerer": True,
        "seed": int(args.seed),
    }
    with args.output.open("w", encoding="utf-8") as f:
        for index in indices:
            dump_path = args.input_dir / f"input_{index:06d}.json"
            obs, dump = load_dump_observation(dump_path)
            print(f"input={dump_path} prompt={dump.get('prompt')!r}")

            for sigma in sigmas:
                sample_kwargs = {
                    **base_kwargs,
                    "phase2_steps": float(args.phase2_steps),
                    "random_noise_ratio": float(sigma),
                }
                row = run_one(policy, obs, label=f"frs_sigma_{sigma:g}", sample_kwargs=sample_kwargs)
                row["input_index"] = index
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(row["label"], row["phase2_debug"], row["metrics"])

            for step in plain_steps:
                sample_kwargs = {
                    **base_kwargs,
                    "phase2_steps": float(step),
                    "random_noise_ratio": 0.0,
                }
                row = run_one(policy, obs, label=f"plain_step_{step:g}", sample_kwargs=sample_kwargs)
                row["input_index"] = index
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(row["label"], row["phase2_debug"], row["metrics"])

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
