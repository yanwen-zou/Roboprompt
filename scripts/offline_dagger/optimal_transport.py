"""Compute rollout/expert optimal-transport scores from policy embeddings.

Usage:
    python scripts/offline_dagger/optimal_transport.py \
        --config scripts/offline_dagger/config/dp.yaml

Useful debug limits:
    python scripts/offline_dagger/optimal_transport.py \
        --config scripts/offline_dagger/config/dp.yaml \
        --max-experts 4 \
        --max-rollouts 2

Outputs:
    The summary CSV contains the best expert match for each rollout episode.
    If ot.length_penalty is enabled, ``min_total_ot_cost_w_time`` additionally
    penalizes rollout episodes longer than the run-level median expert length.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is optional for CPU-only use.
    torch = None

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Missing Python package: PyYAML. Install it in the active environment.") from exc

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is optional for this script.
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.offline_dagger.lerobot_data import (  # noqa: E402
    Trajectory,
    collect_trajectory_specs,
    load_trajectories_from_specs,
    resolve_data_groups,
)
from scripts.offline_dagger.openpi_embedder import OpenPIPolicyEmbedder  # noqa: E402


SUPPORTED_COSTS = {
    "cosine_distance",
    "cosine_similarity",
    "euclidean",
    "squared_euclidean",
}


def create_policy_embedder(feature_cfg: dict[str, Any], *, config_dir: Path) -> Any:
    kind = feature_cfg.get("kind")
    if kind is None:
        kind = "diffusion_policy" if "diffusion_policy" in feature_cfg else "openpi"
    kind = str(kind)
    if kind == "openpi":
        return OpenPIPolicyEmbedder(feature_cfg, config_dir=config_dir)
    if kind in {"diffusion_policy", "dp"}:
        from scripts.offline_dagger.diffusion_policy_embedder import DiffusionPolicyEmbedder

        return DiffusionPolicyEmbedder(feature_cfg, config_dir=config_dir)
    raise ValueError("features.kind must be one of: openpi, diffusion_policy.")


def feature_mode_name(feature_cfg: dict[str, Any]) -> str:
    kind = feature_cfg.get("kind")
    if kind is None:
        kind = "diffusion_policy" if "diffusion_policy" in feature_cfg else "openpi"
    return f"{kind}_policy_embedding"


@dataclass(frozen=True)
class SinkhornResult:
    total_cost: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class LengthPenaltyConfig:
    enabled: bool
    weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ARMADA/FLOAT-style optimal-transport costs between rollout "
            "trajectories and expert trajectories using policy embeddings."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML config containing expert.data_groups, rollout.data_groups, and policy feature settings.",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Override output.csv from the YAML config.")
    parser.add_argument(
        "--pairwise-output-csv",
        type=Path,
        default=None,
        help="Optional CSV containing every rollout/expert OT cost pair.",
    )
    parser.add_argument("--max-experts", type=int, default=None, help="Debug limit for expert trajectories.")
    parser.add_argument("--max-rollouts", type=int, default=None, help="Debug limit for rollout trajectories.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a YAML mapping: {path}")
    return cfg


def section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return value


def expand_path(path_value: str | Path, *, base_dir: Path | None = None) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(path_value)))
    path = Path(text)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def safe_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def resolve_output_dir(output_cfg: dict[str, Any], config_name: str) -> Path:
    output_dir = output_cfg.get("dir", f"output/offline_dagger/{safe_path_component(config_name)}")
    return expand_path(output_dir, base_dir=REPO_ROOT)


def resolve_output_file(
    path_value: str | Path | None,
    default_name: str,
    output_dir: Path,
    *,
    timestamp: str,
) -> Path:
    if path_value is None:
        return output_dir / default_name.format(timestamp=timestamp)

    text = os.path.expandvars(os.path.expanduser(str(path_value).format(timestamp=timestamp)))
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    if path.parent == Path("."):
        return (output_dir / path).resolve()
    return (REPO_ROOT / path).resolve()


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError(f"Block size must be positive, got {size}.")
    return [items[start : start + size] for start in range(0, len(items), size)]


def resolve_length_penalty_config(ot_cfg: dict[str, Any]) -> LengthPenaltyConfig:
    nested_cfg = ot_cfg.get("length_penalty", {})
    if nested_cfg is None:
        nested_cfg = {}
    if not isinstance(nested_cfg, dict):
        raise TypeError("ot.length_penalty must be a mapping when provided.")

    weight = nested_cfg.get("weight", ot_cfg.get("length_penalty_weight", 0.0))
    enabled_value = nested_cfg.get("enabled", ot_cfg.get("length_penalty_enabled"))

    weight = float(weight)
    enabled = bool(enabled_value) if enabled_value is not None else weight > 0.0
    if weight < 0.0:
        raise ValueError(f"Length penalty weight must be non-negative, got {weight}.")
    return LengthPenaltyConfig(enabled=enabled, weight=weight)


def rollout_length_penalty(
    rollout_length: int,
    median_expert_length: float,
    length_cfg: LengthPenaltyConfig,
) -> float:
    if median_expert_length <= 0.0:
        raise ValueError(f"Median expert length must be positive, got {median_expert_length}.")
    length_ratio = float(rollout_length) / median_expert_length
    raw_penalty = max(0.0, length_ratio - 1.0)
    return length_cfg.weight * raw_penalty if length_cfg.enabled else 0.0


def pad_experts_to_max_length(experts: list[Trajectory]) -> list[Trajectory]:
    max_len = max(expert.length for expert in experts)
    padded: list[Trajectory] = []
    for expert in experts:
        if expert.length == max_len:
            padded.append(expert)
            continue
        pad = np.repeat(expert.features[-1:, :], max_len - expert.length, axis=0)
        padded.append(
            Trajectory(
                key=expert.key,
                split=expert.split,
                arm=expert.arm,
                dataset=expert.dataset,
                dataset_path=expert.dataset_path,
                parquet_path=expert.parquet_path,
                episode_id=expert.episode_id,
                features=np.concatenate([expert.features, pad], axis=0),
            )
        )
    return padded


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, eps)


def token_space_cosine_matrix(x: np.ndarray, y: np.ndarray, cost: str, eps: float = 1e-12) -> np.ndarray:
    if cost not in {"cosine_distance", "cosine_similarity"}:
        raise ValueError("Token-space features currently support only cosine_distance or cosine_similarity.")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Token-space features must be 3-D, got expert={x.shape}, rollout={y.shape}.")
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"Token-space feature shapes differ: expert={x.shape[1:]}, rollout={y.shape[1:]}.")
    if x.shape[-1] < 2:
        raise ValueError(f"Token-space feature dim must include hidden dim plus mask, got {x.shape[-1]}.")

    x_tokens = x[:, :, :-1].astype(np.float32, copy=False)
    y_tokens = y[:, :, :-1].astype(np.float32, copy=False)
    x_mask = x[:, :, -1].astype(bool, copy=False)
    y_mask = y[:, :, -1].astype(bool, copy=False)

    x_norm = x_tokens / np.maximum(np.linalg.norm(x_tokens, axis=-1, keepdims=True), eps)
    y_norm = y_tokens / np.maximum(np.linalg.norm(y_tokens, axis=-1, keepdims=True), eps)
    matrix = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)

    # Chunk over expert timesteps so token-space comparison does not materialize
    # a huge expert_len x rollout_len x token_count tensor at once.
    chunk_size = 32
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        sim = np.einsum("asd,bsd->abs", x_norm[start:end], y_norm, optimize=True)
        valid = x_mask[start:end, None, :] & y_mask[None, :, :]
        denom = np.maximum(valid.sum(axis=-1), 1)
        mean_sim = np.sum(sim * valid, axis=-1) / denom
        mean_sim = np.clip(mean_sim, -1.0, 1.0)
        matrix[start:end] = (1.0 - mean_sim) if cost == "cosine_distance" else mean_sim
    return matrix


def cost_matrix(x: np.ndarray, y: np.ndarray, cost: str) -> np.ndarray:
    if cost not in SUPPORTED_COSTS:
        raise ValueError(f"Unsupported ot.cost '{cost}'. Choose one of {sorted(SUPPORTED_COSTS)}.")
    if x.ndim == 3 or y.ndim == 3:
        return token_space_cosine_matrix(x, y, cost)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dims differ: expert dim={x.shape[1]}, rollout dim={y.shape[1]}.")

    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    if cost in {"cosine_distance", "cosine_similarity"}:
        sim = l2_normalize(x64) @ l2_normalize(y64).T
        sim = np.clip(sim, -1.0, 1.0)
        return (1.0 - sim) if cost == "cosine_distance" else sim

    x_sq = np.sum(np.square(x64), axis=1, keepdims=True)
    y_sq = np.sum(np.square(y64), axis=1, keepdims=True).T
    sq_dist = np.maximum(x_sq + y_sq - 2.0 * (x64 @ y64.T), 0.0)
    return np.sqrt(sq_dist) if cost == "euclidean" else sq_dist


def sinkhorn_total_cost(
    cost: np.ndarray,
    *,
    reg: float,
    max_iter: int,
    tol: float,
    eps: float,
) -> SinkhornResult:
    if cost.ndim != 2 or 0 in cost.shape:
        raise ValueError(f"Cost matrix must be non-empty and 2-D, got shape {cost.shape}.")
    if reg <= 0:
        raise ValueError("ot.sinkhorn_reg must be positive.")

    rows, cols = cost.shape
    a = np.full(rows, 1.0 / rows, dtype=np.float64)
    b = np.full(cols, 1.0 / cols, dtype=np.float64)
    shifted_cost = cost - float(np.nanmin(cost))
    kernel = np.exp(-shifted_cost / reg)
    kernel = np.maximum(kernel, eps)

    u = np.ones(rows, dtype=np.float64)
    v = np.ones(cols, dtype=np.float64)
    converged = False
    last_err = math.inf

    for iteration in range(1, max_iter + 1):
        prev_u = u.copy()
        kv = kernel @ v
        u = a / np.maximum(kv, eps)
        ktu = kernel.T @ u
        v = b / np.maximum(ktu, eps)

        if iteration % 10 == 0 or iteration == max_iter:
            row_marginal = u * (kernel @ v)
            col_marginal = v * (kernel.T @ u)
            last_err = float(max(np.max(np.abs(row_marginal - a)), np.max(np.abs(col_marginal - b))))
            if last_err < tol:
                converged = True
                break
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise FloatingPointError(
                "Sinkhorn diverged. Try increasing ot.sinkhorn_reg or using normalized features."
            )
        if np.max(np.abs(u - prev_u)) < eps and last_err < tol:
            converged = True
            break

    plan = (u[:, None] * kernel) * v[None, :]
    return SinkhornResult(
        total_cost=float(np.sum(plan * cost)),
        converged=converged,
        iterations=iteration,
    )


class TorchOTComputer:
    def __init__(self, rollout: Trajectory, ot_cfg: dict[str, Any], device: "torch.device") -> None:
        if torch is None:
            raise RuntimeError("TorchOTComputer requires PyTorch.")
        dtype_name = str(ot_cfg.get("compute_dtype", "float32"))
        if dtype_name == "float32":
            self.dtype = torch.float32
        elif dtype_name == "float64":
            self.dtype = torch.float64
        else:
            raise ValueError("ot.compute_dtype must be one of: float32, float64.")

        self.rollout = torch.as_tensor(rollout.features, dtype=self.dtype, device=device)
        self.device = device
        self.cost = str(ot_cfg.get("cost", "cosine_distance"))
        self.reg = float(ot_cfg.get("sinkhorn_reg", 0.05))
        self.max_iter = int(ot_cfg.get("sinkhorn_max_iter", 500))
        self.tol = float(ot_cfg.get("sinkhorn_tol", 1e-9))
        self.eps = float(ot_cfg.get("sinkhorn_eps", 1e-12))
        self.token_chunk_size = int(ot_cfg.get("token_chunk_size", 32))
        if self.token_chunk_size <= 0:
            raise ValueError(f"ot.token_chunk_size must be positive, got {self.token_chunk_size}.")

    def compute(self, expert: Trajectory) -> SinkhornResult:
        expert_features = torch.as_tensor(expert.features, dtype=self.dtype, device=self.device)
        matrix = self._cost_matrix(expert_features, self.rollout)
        result = self._sinkhorn_total_cost(matrix)
        del expert_features, matrix
        return result

    def _cost_matrix(self, x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        if self.cost not in SUPPORTED_COSTS:
            raise ValueError(f"Unsupported ot.cost '{self.cost}'. Choose one of {sorted(SUPPORTED_COSTS)}.")
        if x.ndim == 3 or y.ndim == 3:
            return self._token_space_cosine_matrix(x, y)
        if x.shape[1] != y.shape[1]:
            raise ValueError(f"Feature dims differ: expert dim={x.shape[1]}, rollout dim={y.shape[1]}.")

        if self.cost in {"cosine_distance", "cosine_similarity"}:
            x_norm = x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(self.eps)
            y_norm = y / torch.linalg.vector_norm(y, dim=1, keepdim=True).clamp_min(self.eps)
            sim = (x_norm @ y_norm.T).clamp(-1.0, 1.0)
            return (1.0 - sim) if self.cost == "cosine_distance" else sim

        x_sq = torch.sum(torch.square(x), dim=1, keepdim=True)
        y_sq = torch.sum(torch.square(y), dim=1, keepdim=True).T
        sq_dist = (x_sq + y_sq - 2.0 * (x @ y.T)).clamp_min(0.0)
        return torch.sqrt(sq_dist) if self.cost == "euclidean" else sq_dist

    def _token_space_cosine_matrix(self, x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        if self.cost not in {"cosine_distance", "cosine_similarity"}:
            raise ValueError("Token-space features currently support only cosine_distance or cosine_similarity.")
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(f"Token-space features must be 3-D, got expert={tuple(x.shape)}, rollout={tuple(y.shape)}.")
        if x.shape[1:] != y.shape[1:]:
            raise ValueError(f"Token-space feature shapes differ: expert={tuple(x.shape[1:])}, rollout={tuple(y.shape[1:])}.")
        if x.shape[-1] < 2:
            raise ValueError(f"Token-space feature dim must include hidden dim plus mask, got {x.shape[-1]}.")

        x_tokens = x[:, :, :-1]
        y_tokens = y[:, :, :-1]
        x_mask = x[:, :, -1] > 0.5
        y_mask = y[:, :, -1] > 0.5

        x_norm = x_tokens / torch.linalg.vector_norm(x_tokens, dim=-1, keepdim=True).clamp_min(self.eps)
        y_norm = y_tokens / torch.linalg.vector_norm(y_tokens, dim=-1, keepdim=True).clamp_min(self.eps)
        matrix = torch.empty((x.shape[0], y.shape[0]), dtype=self.dtype, device=self.device)

        for start in range(0, x.shape[0], self.token_chunk_size):
            end = min(start + self.token_chunk_size, x.shape[0])
            sim = torch.einsum("asd,bsd->abs", x_norm[start:end], y_norm)
            valid = x_mask[start:end, None, :] & y_mask[None, :, :]
            denom = valid.sum(dim=-1).clamp_min(1).to(self.dtype)
            mean_sim = torch.sum(sim * valid.to(self.dtype), dim=-1) / denom
            mean_sim = mean_sim.clamp(-1.0, 1.0)
            matrix[start:end] = (1.0 - mean_sim) if self.cost == "cosine_distance" else mean_sim
            del sim, valid, denom, mean_sim
        return matrix

    def _sinkhorn_total_cost(self, cost: "torch.Tensor") -> SinkhornResult:
        if cost.ndim != 2 or 0 in cost.shape:
            raise ValueError(f"Cost matrix must be non-empty and 2-D, got shape {tuple(cost.shape)}.")
        if self.reg <= 0:
            raise ValueError("ot.sinkhorn_reg must be positive.")

        rows, cols = cost.shape
        a = torch.full((rows,), 1.0 / rows, dtype=self.dtype, device=self.device)
        b = torch.full((cols,), 1.0 / cols, dtype=self.dtype, device=self.device)
        shifted_cost = cost - torch.min(cost)
        kernel = torch.exp(-shifted_cost / self.reg).clamp_min(self.eps)

        u = torch.ones(rows, dtype=self.dtype, device=self.device)
        v = torch.ones(cols, dtype=self.dtype, device=self.device)
        converged = False
        last_err = math.inf

        for iteration in range(1, self.max_iter + 1):
            prev_u = u.clone()
            kv = kernel @ v
            u = a / kv.clamp_min(self.eps)
            ktu = kernel.T @ u
            v = b / ktu.clamp_min(self.eps)

            if iteration % 10 == 0 or iteration == self.max_iter:
                row_marginal = u * (kernel @ v)
                col_marginal = v * (kernel.T @ u)
                row_err = torch.max(torch.abs(row_marginal - a))
                col_err = torch.max(torch.abs(col_marginal - b))
                last_err = float(torch.maximum(row_err, col_err).detach().cpu())
                if last_err < self.tol:
                    converged = True
                    break
            if not torch.isfinite(u).all().item() or not torch.isfinite(v).all().item():
                raise FloatingPointError(
                    "Sinkhorn diverged. Try increasing ot.sinkhorn_reg or using normalized features."
                )
            if float(torch.max(torch.abs(u - prev_u)).detach().cpu()) < self.eps and last_err < self.tol:
                converged = True
                break

        plan = (u[:, None] * kernel) * v[None, :]
        total_cost = float(torch.sum(plan * cost).detach().cpu())
        return SinkhornResult(
            total_cost=total_cost,
            converged=converged,
            iterations=iteration,
        )


def compute_pair_cost(expert: Trajectory, rollout: Trajectory, ot_cfg: dict[str, Any]) -> SinkhornResult:
    matrix = cost_matrix(expert.features, rollout.features, str(ot_cfg.get("cost", "cosine_distance")))
    return sinkhorn_total_cost(
        matrix,
        reg=float(ot_cfg.get("sinkhorn_reg", 0.05)),
        max_iter=int(ot_cfg.get("sinkhorn_max_iter", 500)),
        tol=float(ot_cfg.get("sinkhorn_tol", 1e-9)),
        eps=float(ot_cfg.get("sinkhorn_eps", 1e-12)),
    )


def make_pair_row(rollout: Trajectory, expert: Trajectory, result: SinkhornResult) -> dict[str, Any]:
    return {
        "rollout_dataset": rollout.dataset,
        "rollout_episode_id": rollout.episode_id,
        "rollout_length": rollout.length,
        "expert_key": expert.key,
        "expert_episode_id": expert.episode_id,
        "expert_length": expert.length,
        "total_ot_cost": result.total_cost,
        "sinkhorn_converged": result.converged,
    }


def compute_pair_row(rollout: Trajectory, expert: Trajectory, ot_cfg: dict[str, Any]) -> dict[str, Any]:
    return make_pair_row(rollout, expert, compute_pair_cost(expert, rollout, ot_cfg))


def compute_pair_row_streaming(
    rollout: Trajectory,
    expert: Trajectory,
    ot_cfg: dict[str, Any],
    torch_computer: TorchOTComputer | None,
) -> dict[str, Any]:
    if torch_computer is None:
        return compute_pair_row(rollout, expert, ot_cfg)
    return make_pair_row(rollout, expert, torch_computer.compute(expert))


def resolve_torch_device(ot_cfg: dict[str, Any], feature_cfg: dict[str, Any]) -> "torch.device | None":
    requested = str(ot_cfg.get("compute_device", "auto"))
    if requested == "cpu":
        return None
    if torch is None:
        if requested in {"auto", "none"}:
            return None
        raise ModuleNotFoundError("ot.compute_device requested PyTorch, but PyTorch is not installed.")
    if requested in {"none", "numpy"}:
        return None
    if requested == "auto":
        if not torch.cuda.is_available():
            return None
        device_index = int(section(feature_cfg, "openpi").get("device_index", 0))
        return torch.device(f"cuda:{device_index}")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"ot.compute_device={requested!r} requested CUDA, but torch.cuda.is_available() is false.")
    return device


def clear_torch_cache(device: "torch.device | None") -> None:
    if torch is not None and device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


def progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, file=sys.stdout, **kwargs)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()


def append_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerows(rows)


def maybe_json_costs(costs: list[dict[str, Any]]) -> str:
    compact = [
        {
            "expert_key": item["expert_key"],
            "total_ot_cost": item["total_ot_cost"],
        }
        for item in costs
    ]
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    cfg_path = args.config.expanduser().resolve()
    cfg = load_yaml(cfg_path)
    feature_cfg = section(cfg, "features")
    ot_cfg = section(cfg, "ot")
    output_cfg = section(cfg, "output")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    embedder = create_policy_embedder(feature_cfg, config_dir=cfg_path.parent)
    output_dir = resolve_output_dir(output_cfg, embedder.config_name)
    output_csv = (
        expand_path(args.output_csv, base_dir=REPO_ROOT)
        if args.output_csv is not None
        else resolve_output_file(
            output_cfg.get("csv"),
            "ot_costs_{timestamp}.csv",
            output_dir,
            timestamp=run_timestamp,
        )
    )

    pairwise_output_csv = args.pairwise_output_csv or output_cfg.get("pairwise_csv")
    pairwise_output_path = (
        resolve_output_file(
            pairwise_output_csv,
            "pairwise_ot_costs_{timestamp}.csv",
            output_dir,
            timestamp=run_timestamp,
        )
        if pairwise_output_csv
        else None
    )

    expert_cfg = section(cfg, "expert")
    if "data_groups" in expert_cfg:
        expert_groups = resolve_data_groups(cfg, "expert", config_dir=cfg_path.parent)
    else:
        expert_groups = embedder.expert_data_groups()
        print(f"Using expert data from OpenPI train config '{embedder.config_name}'.")
    rollout_groups = resolve_data_groups(cfg, "rollout", config_dir=cfg_path.parent)

    expert_specs = collect_trajectory_specs(expert_groups, split="expert", limit=args.max_experts)
    rollout_specs = collect_trajectory_specs(rollout_groups, split="rollout", limit=args.max_rollouts)
    expert_block_size = int(ot_cfg.get("expert_block_size", 8))
    compute_device = resolve_torch_device(ot_cfg, feature_cfg)
    length_penalty_cfg = resolve_length_penalty_config(ot_cfg)
    expert_blocks = chunked(expert_specs, expert_block_size)

    if bool(ot_cfg.get("pad_expert_to_max_length", False)):
        print("Warning: ot.pad_expert_to_max_length is applied within each expert block in blockwise mode.")

    print(f"Found {len(expert_specs)} expert trajectories and {len(rollout_specs)} rollout trajectories.")
    print(f"Using expert-block-first OT: {expert_block_size} experts x 1 rollout.")
    if compute_device is None:
        print("Using OT compute device: numpy/cpu")
    else:
        print(f"Using OT compute device: {compute_device}")
    print(f"Using feature mode: {feature_mode_name(feature_cfg)}")
    print(f"Using OT cost: {ot_cfg.get('cost', 'cosine_distance')}")
    if length_penalty_cfg.enabled:
        print(
            "Using rollout length penalty: "
            f"weight={length_penalty_cfg.weight:g}, reference=median_expert_length"
        )

    include_pairwise_json = bool(output_cfg.get("include_pairwise_costs_json", False))
    best_by_rollout: dict[str, dict[str, Any]] = {}
    costs_by_rollout: dict[str, list[dict[str, Any]]] = (
        {spec.key: [] for spec in rollout_specs} if include_pairwise_json else {}
    )
    expert_lengths: list[int] = []

    summary_fields = [
        "rollout_dataset",
        "rollout_episode_id",
        "rollout_length",
        "matched_expert_key",
        "matched_expert_episode_id",
        "matched_expert_length",
        "min_total_ot_cost",
        "min_total_ot_cost_w_time",
        "sinkhorn_converged",
    ]
    if include_pairwise_json:
        summary_fields.append("pairwise_costs_json")

    pairwise_fields = [
        "rollout_dataset",
        "rollout_episode_id",
        "rollout_length",
        "expert_key",
        "expert_episode_id",
        "expert_length",
        "total_ot_cost",
        "sinkhorn_converged",
    ]
    if pairwise_output_path is not None:
        write_csv_header(pairwise_output_path, pairwise_fields)

    for expert_block_index, expert_block_specs in enumerate(
        progress(expert_blocks, desc="Expert blocks"),
        start=1,
    ):
        experts = load_trajectories_from_specs(
            expert_block_specs,
            feature_cfg=feature_cfg,
            embedder=embedder,
            desc=f"Loading expert block {expert_block_index}/{len(expert_blocks)}",
        )
        expert_lengths.extend(expert.length for expert in experts)
        if bool(ot_cfg.get("pad_expert_to_max_length", False)):
            experts = pad_experts_to_max_length(experts)

        for rollout_index, rollout_spec in enumerate(
            progress(rollout_specs, desc=f"Rollouts for expert block {expert_block_index}", unit="traj", leave=False),
            start=1,
        ):
            loaded_rollouts = load_trajectories_from_specs(
                [rollout_spec],
                feature_cfg=feature_cfg,
                embedder=embedder,
                desc=f"Loading rollout {rollout_index}/{len(rollout_specs)}",
            )
            rollout = loaded_rollouts[0]
            torch_computer = TorchOTComputer(rollout, ot_cfg, compute_device) if compute_device is not None else None

            pairwise_rows: list[dict[str, Any]] = []
            for expert in progress(
                experts,
                desc=f"OT experts {expert_block_index}/{len(expert_blocks)}",
                leave=False,
            ):
                pair_row = compute_pair_row_streaming(rollout, expert, ot_cfg, torch_computer)
                if pairwise_output_path is not None:
                    pairwise_rows.append(pair_row)
                if include_pairwise_json:
                    costs_by_rollout[rollout.key].append(pair_row)

                current_best = best_by_rollout.get(rollout.key)
                if current_best is None or float(pair_row["total_ot_cost"]) < float(current_best["total_ot_cost"]):
                    best_by_rollout[rollout.key] = pair_row

            if pairwise_output_path is not None:
                append_csv_rows(pairwise_output_path, pairwise_rows, pairwise_fields)
            del loaded_rollouts, rollout, torch_computer, pairwise_rows
            clear_torch_cache(compute_device)
            gc.collect()

        del experts
        clear_torch_cache(compute_device)
        gc.collect()

    if not expert_lengths:
        raise ValueError("No expert trajectory lengths were observed while computing OT costs.")
    median_expert_length = float(np.median(np.asarray(expert_lengths, dtype=np.float64)))
    if length_penalty_cfg.enabled:
        print(f"Median expert length for rollout length penalty: {median_expert_length:g}")

    summary_rows: list[dict[str, Any]] = []
    for rollout_spec in rollout_specs:
        best = best_by_rollout.get(rollout_spec.key)
        if best is None:
            raise ValueError(f"No expert costs computed for rollout: {rollout_spec.key}")
        length_penalty_cost = rollout_length_penalty(
            int(best["rollout_length"]),
            median_expert_length,
            length_penalty_cfg,
        )
        min_total_ot_cost = float(best["total_ot_cost"])
        summary = {
            "rollout_dataset": best["rollout_dataset"],
            "rollout_episode_id": best["rollout_episode_id"],
            "rollout_length": best["rollout_length"],
            "matched_expert_key": best["expert_key"],
            "matched_expert_episode_id": best["expert_episode_id"],
            "matched_expert_length": best["expert_length"],
            "min_total_ot_cost": min_total_ot_cost,
            "min_total_ot_cost_w_time": min_total_ot_cost + length_penalty_cost,
            "sinkhorn_converged": best["sinkhorn_converged"],
        }
        if include_pairwise_json:
            summary["pairwise_costs_json"] = maybe_json_costs(costs_by_rollout[rollout_spec.key])
        summary_rows.append(summary)

    write_csv(output_csv, summary_rows, summary_fields)

    print(f"Wrote rollout min OT costs: {output_csv}")
    if pairwise_output_path is not None:
        print(f"Wrote pairwise OT costs: {pairwise_output_path}")


if __name__ == "__main__":
    main()
