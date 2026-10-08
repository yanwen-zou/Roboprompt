#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


ACTION_KEYS = (
    "phase1_actions_raw",
    "steerer_phase1_actions_raw",
    "evo1_phase1_actions_raw",
    "phase1_actions",
    "steerer_phase1_actions",
    "evo1_phase1_actions",
    "actions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot DP/OpenPI/FastWAM denoise-step sensitivity on the same dumped observation, "
            "state, and fixed phase-1 delta-action chunk."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("output/evo1/raw_fastwam"))
    parser.add_argument("--index", type=int, default=0, help="Dump index such as input_000000.json.")
    parser.add_argument(
        "--aggregate-jsonl",
        action="append",
        default=None,
        help="Aggregate one or more per-input jsonl files and plot the mean relative action curves.",
    )
    parser.add_argument(
        "--normalization-jsonl",
        action="append",
        default=None,
        help=(
            "Per-input jsonl files used to provide the normalization denominator. "
            "Use no-FRS jsonl files so all conditions share policy/input sigma=1 denominators."
        ),
    )
    parser.add_argument(
        "--plot-jsonl",
        action="append",
        default=None,
        help="Plot existing jsonl rows without running inference or aggregation.",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        default=None,
        help=(
            "Policy websocket endpoint as name=host:port or name=ws://host:port. "
            "Repeat for dp/openpi/fastwam."
        ),
    )
    parser.add_argument(
        "--phase1-policy",
        default="fastwam",
        help="Endpoint label used to generate phase1 when --phase1-actions is not provided.",
    )
    parser.add_argument(
        "--phase1-actions",
        type=Path,
        default=None,
        help="Optional fixed phase1 action chunk (.npy, .npz, .json, or .jsonl).",
    )
    parser.add_argument(
        "--sigmas",
        default=None,
        help="Comma-separated normalized denoise positions. Default: 0,0.05,...,1.",
    )
    parser.add_argument(
        "--all-policy-steps",
        action="store_true",
        help="Sweep every integer denoise step for each policy, using x=current_step/total_step.",
    )
    parser.add_argument("--num-sigmas", type=int, default=21, help="Number of shared sigma points when --sigmas is omitted.")
    parser.add_argument(
        "--dims",
        default="all",
        help="Action dims for the norm: all, xyz, 0:7, or comma-separated indices.",
    )
    parser.add_argument(
        "--metric",
        default="chunk_l2",
        choices=("chunk_l2", "mean_timestep_l2", "max_timestep_l2"),
        help="Scalar plotted on the y axis.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed for policy backends that support it, e.g. FastWAM.")
    parser.add_argument("--frs", action="store_true", help="Pass frs=True in refinement sample kwargs.")
    parser.add_argument(
        "--random-noise-ratio-mode",
        choices=("none", "sigma", "one"),
        default="none",
        help=(
            "How to set random_noise_ratio during refinement. "
            "'sigma' aligns it to the plotted normalized denoise coordinate."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path without suffix. Defaults under output/evo1/denoise_delta/.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional png path. Useful for writing aggregate pngs into a separate plots directory.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Write jsonl/csv but skip the per-input png.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional shared y-axis maximum for plot output.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing rows in output-prefix.jsonl for other policies and replace only the current endpoint policy rows.",
    )
    parser.add_argument(
        "--allow-label-type-mismatch",
        action="store_true",
        help="Allow standard labels such as dp/openpi/fastwam to connect to a different policy_type.",
    )
    return parser.parse_args()


def parse_endpoint(spec: str) -> tuple[str, str, int | None]:
    if "=" not in spec:
        raise ValueError(f"Endpoint must be name=host:port or name=ws://host:port, got {spec!r}.")
    name, target = spec.split("=", 1)
    name = name.strip().lower()
    target = target.strip()
    if not name or not target:
        raise ValueError(f"Invalid endpoint spec: {spec!r}.")
    if target.startswith("ws://") or target.startswith("wss://"):
        return name, target, None
    if ":" not in target:
        return name, target, None
    host, port_text = target.rsplit(":", 1)
    return name, host, int(port_text)


def default_endpoints() -> list[str]:
    return [
        "dp=127.0.0.1:8001",
        "openpi=127.0.0.1:8002",
        "fastwam=127.0.0.1:8003",
    ]


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one sigma value.")
    return values


def parse_dims(text: str) -> list[int] | None:
    text = str(text).strip().lower()
    if text in {"", "all"}:
        return None
    if text == "xyz":
        return [0, 1, 2]
    if ":" in text:
        start_text, end_text = text.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text)
        return list(range(start, end))
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def resolve_path(path: str | Path, *, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in (base, REPO_ROOT):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return candidate


def load_rgb(path: str | Path, *, base: Path) -> np.ndarray:
    return np.asarray(Image.open(resolve_path(path, base=base)).convert("RGB"))


def load_dump_observation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_paths = data["raw_image_paths"]
    debug = data.get("debug", {})
    raw_debug = debug.get("raw_input_debug", {})

    raw_state = debug.get("raw_state_first8")
    if raw_state is None:
        raw_state = data.get("raw_state")
    if raw_state is None:
        raise KeyError(f"{path} does not contain debug.raw_state_first8 or raw_state.")
    raw_state = np.asarray(raw_state, dtype=np.float32)

    obs: dict[str, Any] = {
        "observation/state": raw_state,
        "state": raw_state,
        "observation/image": load_rgb(raw_paths["observation_image"], base=path.parent),
        "observation/wrist_image": load_rgb(raw_paths["observation_wrist_image"], base=path.parent),
        "prompt": str(data.get("prompt", "")),
    }
    if "observation_right_wrist_image" in raw_paths:
        obs["observation/right_wrist_image"] = load_rgb(raw_paths["observation_right_wrist_image"], base=path.parent)

    if "prompt_0" in raw_paths:
        obs["prompt_images"] = {"prompt_0": load_rgb(raw_paths["prompt_0"], base=path.parent)}
        if "prompt_overlay_0" in raw_paths:
            obs["prompt_images"]["prompt_overlay_0"] = load_rgb(raw_paths["prompt_overlay_0"], base=path.parent)
        prompt_mask = bool(raw_debug.get("prompt_image_mask_prompt_0", False))
        obs["prompt_image_masks"] = {"prompt_0": prompt_mask}

    return obs, data


def extract_action_chunk(result: Any, keys: tuple[str, ...] = ACTION_KEYS) -> np.ndarray:
    if isinstance(result, np.ndarray):
        actions = result.astype(np.float32)
        return normalize_action_rank(actions)
    if isinstance(result, list):
        return normalize_action_rank(np.asarray(result, dtype=np.float32))
    if not isinstance(result, dict):
        raise TypeError(f"Expected action result dict/list/ndarray, got {type(result).__name__}.")
    for key in keys:
        if key not in result:
            continue
        actions = normalize_action_rank(np.asarray(result[key], dtype=np.float32))
        if actions.shape[-1] >= 1:
            return actions
    raise KeyError(f"No action chunk found. Tried keys {keys}; available keys: {sorted(result)}")


def extract_metric_action_chunk(result: dict[str, Any], *, policy_type: str) -> tuple[np.ndarray, str]:
    if policy_type == "diffusion_policy" and "diffusion_policy_action_pred" in result:
        return extract_action_chunk(result, ("diffusion_policy_action_pred",)), "diffusion_policy_action_pred"
    return extract_action_chunk(result, ("actions",)), "actions"


def normalize_action_rank(actions: np.ndarray) -> np.ndarray:
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"Expected action chunk [T,D], got {actions.shape}.")
    return actions.astype(np.float32)


def load_phase1_actions(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return extract_action_chunk(np.load(path, allow_pickle=False))
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        for key in ACTION_KEYS:
            if key in data:
                return extract_action_chunk(data[key])
        if len(data.files) == 1:
            return extract_action_chunk(data[data.files[0]])
        raise KeyError(f"{path} does not contain any known action key. Keys: {data.files}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return extract_action_chunk(json.loads(line))
        raise ValueError(f"{path} is empty.")
    with path.open("r", encoding="utf-8") as f:
        return extract_action_chunk(json.load(f))


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


def load_existing_rows(path: Path, *, skip_policies: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("policy", "")) not in skip_policies:
                rows.append(row)
    return rows


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def policy_type_from_metadata(label: str, metadata: dict[str, Any]) -> str:
    value = metadata.get("phase2_policy_type") or metadata.get("policy_type") or label
    value = str(value).strip().lower()
    if value in {"dp", "diffusion", "diffusion-policy"}:
        return "diffusion_policy"
    if value in {"pi05", "pi0.5"}:
        return "openpi"
    return value


def expected_policy_type_for_label(label: str) -> str | None:
    normalized = str(label).strip().lower()
    if normalized in {"dp", "dp_ddim", "diffusion_policy", "diffusion_policy_ddim"}:
        return "diffusion_policy"
    if normalized in {"openpi", "fastwam"}:
        return normalized
    return None


def validate_policy_label_type(label: str, policy_type: str, metadata: dict[str, Any], *, allow_mismatch: bool) -> None:
    if allow_mismatch:
        return
    expected = expected_policy_type_for_label(label)
    if expected is None or policy_type == expected:
        return
    metadata_hint = {
        key: metadata.get(key)
        for key in ("phase2_policy_type", "policy_type", "phase2_policy_config", "phase2_policy_dir")
        if key in metadata
    }
    raise ValueError(
        f"Endpoint label {label!r} expects policy_type={expected!r}, but connected server reports "
        f"policy_type={policy_type!r}. Metadata hint: {metadata_hint}. If this is intentional, pass "
        "--allow-label-type-mismatch."
    )


def total_steps_from_metadata(label: str, metadata: dict[str, Any]) -> int:
    candidates = [
        metadata.get("policy_inference_steps"),
        metadata.get("max_phase2_steps"),
        metadata.get("phase2_max_steps"),
        metadata.get("num_steps"),
        metadata.get("num_inference_steps"),
    ]
    for nested_key in ("diffusion_policy", "fastwam", "openpi"):
        nested = metadata.get(nested_key)
        if isinstance(nested, dict):
            candidates.extend(
                [
                    nested.get("policy_inference_steps"),
                    nested.get("max_phase2_steps"),
                    nested.get("num_steps"),
                    nested.get("num_inference_steps"),
                ]
            )
    for value in candidates:
        if value is None:
            continue
        try:
            steps = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        if steps > 0:
            return steps
    raise ValueError(f"Could not infer total denoise steps for {label!r} from metadata: {metadata}")


def make_refinement_kwargs(
    *,
    policy_type: str,
    total_steps: int,
    current_step: int,
    sigma: float,
    phase1_actions: np.ndarray,
    seed: int | None,
    frs: bool,
    random_noise_ratio_mode: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "enable_steerer": False,
        "enable_evo1_steerer": False,
        "phase1_actions": phase1_actions,
        "phase2_steps": float(current_step),
        "frs": bool(frs),
    }
    if random_noise_ratio_mode == "sigma":
        kwargs["random_noise_ratio"] = float(np.clip(sigma, 0.0, 1.0))
    elif random_noise_ratio_mode == "one":
        kwargs["random_noise_ratio"] = 1.0
    elif random_noise_ratio_mode != "none":
        raise ValueError(f"Unsupported random noise ratio mode: {random_noise_ratio_mode!r}")
    if seed is not None and policy_type == "fastwam":
        kwargs["seed"] = int(seed)
    if policy_type == "diffusion_policy":
        kwargs["num_inference_steps"] = int(total_steps)
    elif policy_type == "openpi":
        kwargs["num_steps"] = int(total_steps)
    else:
        kwargs["num_inference_steps"] = int(total_steps)
        kwargs["num_steps"] = int(total_steps)
    return kwargs


def metric_value(output_actions: np.ndarray, phase1_actions: np.ndarray, dims: list[int] | None, metric: str) -> dict[str, Any]:
    output_actions = normalize_action_rank(output_actions)
    phase1_actions = normalize_action_rank(phase1_actions)
    horizon = min(output_actions.shape[0], phase1_actions.shape[0])
    dim = min(output_actions.shape[1], phase1_actions.shape[1])
    if horizon <= 0 or dim <= 0:
        raise ValueError(f"Cannot compare empty action chunks: output={output_actions.shape}, phase1={phase1_actions.shape}")

    if dims is None:
        dim_indices = list(range(dim))
    else:
        dim_indices = [idx for idx in dims if 0 <= idx < dim]
        if not dim_indices:
            raise ValueError(f"No requested dims {dims} are valid for aligned dim={dim}.")

    diff = output_actions[:horizon, dim_indices] - phase1_actions[:horizon, dim_indices]
    timestep_l2 = np.linalg.norm(diff, axis=-1)
    values = {
        "chunk_l2": float(np.linalg.norm(diff)),
        "mean_timestep_l2": float(np.mean(timestep_l2)),
        "max_timestep_l2": float(np.max(timestep_l2)),
        "aligned_horizon": int(horizon),
        "aligned_dim": int(dim),
        "used_dims": dim_indices,
    }
    values["plot_y"] = values[metric]
    return values


def infer_phase1(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    obs: dict[str, Any],
    *,
    policy_type: str,
    total_steps: int,
    seed: int | None,
) -> np.ndarray:
    sample_kwargs: dict[str, Any] = {
        "enable_steerer": True,
        "enable_evo1_steerer": True,
        "phase2_steps": 0.0,
    }
    if seed is not None and policy_type == "fastwam":
        sample_kwargs["seed"] = int(seed)
    if policy_type == "diffusion_policy":
        sample_kwargs["num_inference_steps"] = int(total_steps)
    else:
        sample_kwargs["num_steps"] = int(total_steps)
    result = policy.infer(obs, sample_kwargs=sample_kwargs)
    return extract_action_chunk(result, ACTION_KEYS[:-1])


def connect_policies(endpoint_specs: list[str], *, allow_label_type_mismatch: bool = False) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    for spec in endpoint_specs:
        label, host, port = parse_endpoint(spec)
        print(f"connecting {label} -> {host}{'' if port is None else f':{port}'}", flush=True)
        policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        metadata = policy.get_server_metadata()
        policy_type = policy_type_from_metadata(label, metadata)
        validate_policy_label_type(
            label,
            policy_type,
            metadata,
            allow_mismatch=allow_label_type_mismatch,
        )
        total_steps = total_steps_from_metadata(label, metadata)
        policies[label] = {
            "policy": policy,
            "metadata": metadata,
            "policy_type": policy_type,
            "total_steps": total_steps,
        }
        print(f"{label}: type={policy_type} total_steps={total_steps}", flush=True)
    return policies


def sigma_points(total_steps: int, args: argparse.Namespace) -> list[tuple[int, float]]:
    if args.all_policy_steps:
        return [(step, float(step) / float(total_steps)) for step in range(total_steps + 1)]
    if args.sigmas is not None:
        raw_sigmas = parse_float_list(args.sigmas)
    else:
        raw_sigmas = np.linspace(0.0, 1.0, int(args.num_sigmas)).astype(float).tolist()
    points = []
    seen_steps = set()
    for sigma in raw_sigmas:
        clipped = float(np.clip(float(sigma), 0.0, 1.0))
        step = int(round(clipped * total_steps))
        if step in seen_steps:
            continue
        seen_steps.add(step)
        points.append((step, float(step) / float(total_steps)))
    return points


def add_policy_normalized_plot_y(rows: list[dict[str, Any]]) -> None:
    """Normalize each policy curve by its own sigma=1 delta."""
    for policy in sorted({str(row["policy"]) for row in rows}):
        series = [row for row in rows if str(row["policy"]) == policy]
        sigma_one_rows = [row for row in series if abs(float(row.get("sigma", 0.0)) - 1.0) <= 1e-9]
        if not sigma_one_rows:
            raise ValueError(f"Cannot normalize {policy!r}: no row at sigma=1.")
        denominator = float(sigma_one_rows[-1]["plot_y"])
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(f"Cannot normalize {policy!r}: sigma=1 delta is {denominator}.")
        for row in series:
            row["plot_y_sigma1"] = denominator
            row["plot_y_normalized"] = float(row["plot_y"]) / denominator


def normalization_denominators(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    denominators: dict[tuple[str, int], float] = {}
    for row in rows:
        if abs(float(row.get("sigma", 0.0)) - 1.0) > 1e-9:
            continue
        input_index = row.get("input_index")
        if input_index is None:
            continue
        key = (str(row["policy"]), int(input_index))
        denominator = float(row["plot_y"])
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(f"Invalid normalization denominator for {key}: {denominator}.")
        denominators[key] = denominator
    if not denominators:
        raise ValueError("No sigma=1 denominators found in normalization jsonl rows.")
    return denominators


def aggregate_mean_rows(
    rows: list[dict[str, Any]],
    *,
    denominators: dict[tuple[str, int], float] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, str, bool, str], list[dict[str, Any]]] = {}
    for row in rows:
        if "plot_y_normalized" not in row:
            raise ValueError("Aggregate inputs must contain plot_y_normalized; run per-input sweeps first.")
        key = (
            str(row["policy"]),
            str(row["policy_type"]),
            float(row["sigma"]),
            str(row["metric"]),
            bool(row.get("frs", False)),
            str(row.get("random_noise_ratio_mode", "none")),
        )
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (policy, policy_type, sigma, metric, frs, random_noise_ratio_mode), series in groups.items():
        plot_y_values = np.asarray([float(row["plot_y"]) for row in series], dtype=np.float64)
        if denominators is None:
            relative_values = np.asarray([float(row["plot_y_normalized"]) for row in series], dtype=np.float64)
            normalization_source = "condition sigma=1 per-input"
            plot_y_sigma1: str | float = "per-input"
        else:
            relative = []
            for row in series:
                input_index = int(row["input_index"])
                denominator_key = (policy, input_index)
                if denominator_key not in denominators:
                    raise KeyError(f"Missing no-FRS normalization denominator for {denominator_key}.")
                relative.append(float(row["plot_y"]) / denominators[denominator_key])
            relative_values = np.asarray(relative, dtype=np.float64)
            normalization_source = "no_frs sigma=1 per-input"
            plot_y_sigma1 = normalization_source
        input_indices = sorted({int(row["input_index"]) for row in series if row.get("input_index") is not None})
        row0 = series[0]
        out = {
            "input": "aggregate",
            "input_index": "mean",
            "input_indices": input_indices,
            "num_inputs": len(input_indices),
            "prompt": "aggregate",
            "phase1_source": "per-input",
            "policy": policy,
            "policy_type": policy_type,
            "total_steps": row0.get("total_steps"),
            "current_step": row0.get("current_step"),
            "sigma": sigma,
            "metric": metric,
            "metric_action_key": row0.get("metric_action_key"),
            "frs": frs,
            "random_noise_ratio_mode": random_noise_ratio_mode,
            "plot_y": float(np.mean(plot_y_values)),
            "plot_y_std": float(np.std(plot_y_values, ddof=1)) if len(plot_y_values) > 1 else 0.0,
            "plot_y_sigma1": plot_y_sigma1,
            "normalization_source": normalization_source,
            "plot_y_normalized": float(np.mean(relative_values)),
            "plot_y_normalized_std": float(np.std(relative_values, ddof=1)) if len(relative_values) > 1 else 0.0,
            "plot_y_normalized_sem": (
                float(np.std(relative_values, ddof=1) / np.sqrt(len(relative_values)))
                if len(relative_values) > 1
                else 0.0
            ),
            "elapsed_s": float(np.mean([float(row.get("elapsed_s", 0.0) or 0.0) for row in series])),
            "aligned_horizon": row0.get("aligned_horizon"),
            "aligned_dim": row0.get("aligned_dim"),
            "used_dims": row0.get("used_dims"),
        }
        for metric_name in ("chunk_l2", "mean_timestep_l2", "max_timestep_l2"):
            values = [float(row[metric_name]) for row in series if row.get(metric_name) is not None]
            if values:
                out[metric_name] = float(np.mean(values))
        aggregate_rows.append(out)

    aggregate_rows.sort(
        key=lambda row: (str(row.get("policy", "")), float(row.get("sigma", 0.0)), int(row.get("current_step", 0)))
    )
    return aggregate_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "input",
        "input_index",
        "input_indices",
        "num_inputs",
        "prompt",
        "phase1_source",
        "policy",
        "policy_type",
        "total_steps",
        "current_step",
        "sigma",
        "metric",
        "metric_action_key",
        "frs",
        "random_noise_ratio_mode",
        "plot_y",
        "plot_y_std",
        "plot_y_sigma1",
        "normalization_source",
        "plot_y_normalized",
        "plot_y_normalized_std",
        "plot_y_normalized_sem",
        "chunk_l2",
        "mean_timestep_l2",
        "max_timestep_l2",
        "elapsed_s",
        "aligned_horizon",
        "aligned_dim",
        "used_dims",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if key in {"used_dims", "input_indices"} and key in row else row.get(key)
                    for key in fieldnames
                }
            )


def plot_rows(path: Path, rows: list[dict[str, Any]], *, metric: str, y_max: float | None = None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plot_rows_with_pil(path, rows, metric=metric, y_max=y_max)
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=160)
    colors = {
        "dp": "#1f77b4",
        "dp_ddim": "#9467bd",
        "diffusion_policy": "#1f77b4",
        "openpi": "#2ca02c",
        "fastwam": "#d62728",
    }
    normalization_source = str(rows[0].get("normalization_source", "")) if rows else ""
    if normalization_source == "no_frs sigma=1 per-input":
        y_label = f"Delta action / no-FRS delta at sigma=1 ({metric})"
        title = "Policy denoise-step action deviation (no-FRS normalized)"
    else:
        y_label = f"Delta action / delta action at sigma=1 ({metric})"
        title = "Policy denoise-step action deviation (per-policy normalized)"
    for policy in sorted({row["policy"] for row in rows}):
        series = sorted((row for row in rows if row["policy"] == policy), key=lambda row: row["sigma"])
        num_inputs = series[0].get("num_inputs")
        label = "{} ({} steps)".format(policy, series[0]["total_steps"])
        if num_inputs:
            label = "{} (n={}, {} steps)".format(policy, num_inputs, series[0]["total_steps"])
        ax.plot(
            [row["sigma"] for row in series],
            [row["plot_y_normalized"] for row in series],
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors.get(str(policy), colors.get(str(series[0].get("policy_type")), None)),
            label=label,
        )
    ax.set_xlabel("Normalized denoise step sigma = current_step / total_step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if y_max is not None:
        ax.set_ylim(0.0, float(y_max))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rows_with_pil(path: Path, rows: list[dict[str, Any]], *, metric: str, y_max: float | None = None) -> None:
    from PIL import ImageDraw

    width, height = 1400, 820
    left, right, top, bottom = 120, 40, 90, 130
    colors = {
        "dp": (31, 119, 180),
        "dp_ddim": (148, 103, 189),
        "diffusion_policy": (31, 119, 180),
        "openpi": (44, 160, 44),
        "fastwam": (214, 39, 40),
    }
    normalization_source = str(rows[0].get("normalization_source", "")) if rows else ""
    if normalization_source == "no_frs sigma=1 per-input":
        y_label = f"y = delta action / no-FRS delta at sigma=1 ({metric})"
        title = "Policy denoise-step action deviation (no-FRS normalized)"
    else:
        y_label = f"y = delta action / delta action at sigma=1 ({metric})"
        title = "Policy denoise-step action deviation (per-policy normalized)"
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_y = float(y_max) if y_max is not None else max(max(float(row["plot_y_normalized"]) for row in rows), 1e-9)
    max_y = max(max_y, 1e-9)

    def point(x_value: float, y_value: float) -> tuple[int, int]:
        x = left + int(np.clip(x_value, 0.0, 1.0) * plot_width)
        y = top + plot_height - int(np.clip(y_value / max_y, 0.0, 1.0) * plot_height)
        return x, y

    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=(40, 40, 40), width=2)
    for tick in np.linspace(0.0, 1.0, 6):
        x = left + int(tick * plot_width)
        draw.line((x, top, x, top + plot_height), fill=(225, 225, 225), width=1)
        draw.text((x - 18, top + plot_height + 12), f"{tick:.1f}", fill=(30, 30, 30))
    for tick in np.linspace(0.0, max_y, 6):
        y = top + plot_height - int(tick / max_y * plot_height)
        draw.line((left, y, left + plot_width, y), fill=(225, 225, 225), width=1)
        draw.text((12, y - 7), f"{tick:.3g}", fill=(30, 30, 30))

    legend_x = left
    for policy in sorted({row["policy"] for row in rows}):
        series = sorted((row for row in rows if row["policy"] == policy), key=lambda row: row["sigma"])
        color = colors.get(policy, colors.get(series[0]["policy_type"], (148, 103, 189)))
        points = [point(float(row["sigma"]), float(row["plot_y_normalized"])) for row in series]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        num_inputs = series[0].get("num_inputs")
        label = "{} ({} steps)".format(policy, series[0]["total_steps"])
        if num_inputs:
            label = "{} (n={}, {} steps)".format(policy, num_inputs, series[0]["total_steps"])
        draw.rectangle((legend_x, 34, legend_x + 22, 48), fill=color)
        draw.text((legend_x + 30, 30), label, fill=(20, 20, 20))
        legend_x += 220

    draw.text((left, 16), title, fill=(20, 20, 20))
    draw.text((left + int(plot_width / 2) - 170, height - 48), "sigma = current_step / total_step", fill=(20, 20, 20))
    draw.text((left, height - 22), y_label, fill=(20, 20, 20))
    image.save(path)


def main() -> None:
    args = parse_args()
    if args.plot_jsonl:
        output_prefix = args.output_prefix
        if output_prefix is None:
            output_prefix = Path("output/evo1/denoise_delta") / "denoise_delta_plot"
        rows: list[dict[str, Any]] = []
        for path_text in args.plot_jsonl:
            rows.extend(load_jsonl_rows(Path(path_text)))
        png_path = args.plot_path if args.plot_path is not None else output_prefix.with_suffix(".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plot_rows(png_path, rows, metric=str(rows[0]["metric"]) if rows else args.metric, y_max=args.y_max)
        print(f"wrote {png_path}", flush=True)
        return

    if args.aggregate_jsonl:
        output_prefix = args.output_prefix
        if output_prefix is None:
            output_prefix = Path("output/evo1/denoise_delta") / "denoise_delta_mean"
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for path_text in args.aggregate_jsonl:
            rows.extend(load_jsonl_rows(Path(path_text)))
        denominators = None
        if args.normalization_jsonl:
            normalization_rows: list[dict[str, Any]] = []
            for path_text in args.normalization_jsonl:
                normalization_rows.extend(load_jsonl_rows(Path(path_text)))
            denominators = normalization_denominators(normalization_rows)
        aggregate_rows = aggregate_mean_rows(rows, denominators=denominators)
        jsonl_path = output_prefix.with_suffix(".jsonl")
        csv_path = output_prefix.with_suffix(".csv")
        png_path = args.plot_path if args.plot_path is not None else output_prefix.with_suffix(".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in aggregate_rows:
                f.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
        write_csv(csv_path, aggregate_rows)
        if not args.skip_plot:
            plot_rows(
                png_path,
                aggregate_rows,
                metric=str(aggregate_rows[0]["metric"]) if aggregate_rows else args.metric,
                y_max=args.y_max,
            )
        print(f"wrote {jsonl_path}", flush=True)
        print(f"wrote {csv_path}", flush=True)
        if not args.skip_plot:
            print(f"wrote {png_path}", flush=True)
        return

    dump_path = args.input_dir / f"input_{args.index:06d}.json"
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump not found: {dump_path}")

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = Path("output/evo1/denoise_delta") / f"denoise_delta_input{args.index:06d}"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    obs, dump = load_dump_observation(dump_path)
    endpoint_specs = args.endpoint if args.endpoint is not None else default_endpoints()
    policies = connect_policies(endpoint_specs, allow_label_type_mismatch=args.allow_label_type_mismatch)

    if args.phase1_actions is not None:
        phase1_actions = load_phase1_actions(args.phase1_actions)
        phase1_source = str(args.phase1_actions)
    else:
        phase1_label = str(args.phase1_policy).strip().lower()
        if phase1_label not in policies:
            raise KeyError(f"--phase1-policy {phase1_label!r} is not among endpoints: {sorted(policies)}")
        info = policies[phase1_label]
        phase1_actions = infer_phase1(
            info["policy"],
            obs,
            policy_type=info["policy_type"],
            total_steps=info["total_steps"],
            seed=args.seed,
        )
        phase1_source = f"{phase1_label}:steerer"
    phase1_actions = normalize_action_rank(phase1_actions)
    np.save(output_prefix.with_name(output_prefix.name + "_phase1.npy"), phase1_actions)
    print(f"phase1 source={phase1_source} shape={phase1_actions.shape}", flush=True)

    dims = parse_dims(args.dims)
    jsonl_path = output_prefix.with_suffix(".jsonl")
    policy_labels = set(policies)
    rows: list[dict[str, Any]] = load_existing_rows(jsonl_path, skip_policies=policy_labels) if args.append else []
    new_rows: list[dict[str, Any]] = []
    for label, info in policies.items():
        policy = info["policy"]
        policy_type = info["policy_type"]
        total_steps = int(info["total_steps"])
        for current_step, sigma in sigma_points(total_steps, args):
            sample_kwargs = make_refinement_kwargs(
                policy_type=policy_type,
                total_steps=total_steps,
                current_step=current_step,
                sigma=sigma,
                phase1_actions=phase1_actions,
                seed=args.seed,
                frs=args.frs,
                random_noise_ratio_mode=args.random_noise_ratio_mode,
            )
            started = time.time()
            result = policy.infer(obs, sample_kwargs=sample_kwargs)
            elapsed = time.time() - started
            output_actions, metric_action_key = extract_metric_action_chunk(result, policy_type=policy_type)
            metrics = metric_value(output_actions, phase1_actions, dims, args.metric)
            row = {
                "input": str(dump_path),
                "input_index": int(args.index),
                "prompt": dump.get("prompt"),
                "phase1_source": phase1_source,
                "policy": label,
                "policy_type": policy_type,
                "total_steps": total_steps,
                "current_step": int(current_step),
                "sigma": float(sigma),
                "metric": args.metric,
                "frs": bool(args.frs),
                "random_noise_ratio_mode": args.random_noise_ratio_mode,
                "elapsed_s": float(elapsed),
                "sample_kwargs": {
                    key: value
                    for key, value in sample_kwargs.items()
                    if key not in {"phase1_actions"}
                },
                "phase2_debug": result.get("phase2_debug"),
                "metric_action_key": metric_action_key,
                **metrics,
            }
            new_rows.append(row)
            print(
                f"{label:8s} step={current_step:3d}/{total_steps:<3d} "
                f"sigma={sigma:.3f} {args.metric}={row['plot_y']:.6g} elapsed={elapsed:.2f}s",
                flush=True,
            )

    rows.extend(new_rows)
    rows.sort(key=lambda row: (str(row.get("policy", "")), float(row.get("sigma", 0.0)), int(row.get("current_step", 0))))
    add_policy_normalized_plot_y(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")

    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    write_csv(csv_path, rows)
    if not args.skip_plot:
        if args.plot_path is not None:
            png_path = args.plot_path
            png_path.parent.mkdir(parents=True, exist_ok=True)
        plot_rows(png_path, rows, metric=args.metric, y_max=args.y_max)
    print(f"wrote {jsonl_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    if not args.skip_plot:
        print(f"wrote {png_path}", flush=True)


if __name__ == "__main__":
    main()
