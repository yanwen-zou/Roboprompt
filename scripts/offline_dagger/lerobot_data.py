from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is optional for this script.
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Trajectory:
    key: str
    split: str
    arm: str
    dataset: str
    dataset_path: Path
    parquet_path: Path
    episode_id: int | None
    features: np.ndarray

    @property
    def length(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class TrajectorySpec:
    key: str
    split: str
    arm: str
    dataset: str
    dataset_path: Path
    parquet_path: Path
    episode_id: int | None


class TrajectoryEmbedder(Protocol):
    def embed_parquet(self, traj_key: str, parquet_path: Path, feature_cfg: dict[str, Any]) -> np.ndarray: ...


def resolve_dataset_path(path_value: str | Path, *, config_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if path.is_absolute():
        return path.resolve()

    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
        config_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_data_groups(cfg: dict[str, Any], split: str, *, config_dir: Path) -> dict[str, Any]:
    split_cfg = _section(cfg, split)
    data_groups = split_cfg.get("data_groups")
    if data_groups is None:
        legacy_key = f"{split}_data_groups"
        data_groups = cfg.get(legacy_key)
    if not isinstance(data_groups, dict) or not data_groups:
        raise ValueError(f"Missing non-empty {split}.data_groups in config.")

    resolved: dict[str, Any] = {}
    for arm_name, arm_config in data_groups.items():
        if not isinstance(arm_config, dict):
            raise TypeError(f"{split}.data_groups.{arm_name} must be a mapping.")
        resolved[str(arm_name)] = {}
        for dataset_name, dataset_config in arm_config.items():
            if not isinstance(dataset_config, dict) or "path" not in dataset_config:
                raise ValueError(f"{split}.data_groups.{arm_name}.{dataset_name} must contain a path.")
            copied = dict(dataset_config)
            copied["path"] = str(resolve_dataset_path(copied["path"], config_dir=config_dir))
            resolved[str(arm_name)][str(dataset_name)] = copied
    return resolved


def iter_le_robot_roots(dataset_path: Path) -> list[Path]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if (dataset_path / "meta").is_dir() and (dataset_path / "data").is_dir():
        return [dataset_path]

    child_roots = [
        child
        for child in sorted(dataset_path.iterdir())
        if child.is_dir() and (child / "meta").is_dir() and (child / "data").is_dir()
    ]
    if child_roots:
        return child_roots

    episode_dirs = sorted(d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("episode_"))
    if episode_dirs:
        return episode_dirs

    raise FileNotFoundError(f"No LeRobot dataset found under {dataset_path}")


def iter_le_robot_parquets(dataset_path: Path) -> list[Path]:
    roots = iter_le_robot_roots(dataset_path)
    parquets: list[Path] = []
    for root in roots:
        parquets.extend(sorted(root.glob("data/*/*.parquet")))
        if not (root / "meta").is_dir():
            parquets.extend(sorted(root.glob("*.parquet")))
    if not parquets:
        raise FileNotFoundError(f"No LeRobot parquet files found under {dataset_path}")
    return parquets


def parse_episode_id(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def make_trajectory_key(split: str, arm: str, dataset: str, parquet_path: Path) -> str:
    episode_id = parse_episode_id(parquet_path)
    episode = f"episode_{episode_id:06d}" if episode_id is not None else parquet_path.stem
    dataset_root = parquet_path.parents[2].name if len(parquet_path.parents) >= 3 else parquet_path.parent.name
    return f"{split}/{arm}/{dataset}/{dataset_root}/{parquet_path.parent.name}/{episode}"


def trim_features(features: np.ndarray, feature_cfg: dict[str, Any]) -> np.ndarray:
    max_steps = feature_cfg.get("max_steps")
    if max_steps is not None:
        features = features[: int(max_steps)]
    max_dim = feature_cfg.get("max_dim")
    if max_dim is not None:
        max_dim = int(max_dim)
        if features.ndim == 2:
            features = features[:, :max_dim]
        elif features.ndim == 3:
            features = np.concatenate([features[:, :, :max_dim], features[:, :, -1:]], axis=-1)
        else:
            raise ValueError(f"Unsupported feature rank for max_dim: {features.ndim}")
    if bool(feature_cfg.get("drop_nan_rows", False)):
        mask = np.isfinite(features).all(axis=tuple(range(1, features.ndim)))
        features = features[mask]
    if bool(feature_cfg.get("require_finite", True)) and not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or Inf values.")
    if features.shape[0] == 0:
        raise ValueError("Feature matrix has no rows after filtering.")
    return np.ascontiguousarray(features, dtype=np.float32)


def load_trajectories(
    data_groups: dict[str, Any],
    *,
    split: str,
    feature_cfg: dict[str, Any],
    embedder: TrajectoryEmbedder,
    limit: int | None = None,
) -> list[Trajectory]:
    jobs = collect_trajectory_specs(data_groups, split=split, limit=limit)
    return load_trajectories_from_specs(
        jobs,
        feature_cfg=feature_cfg,
        embedder=embedder,
        desc=f"Loading {split}",
    )


def collect_trajectory_specs(
    data_groups: dict[str, Any],
    *,
    split: str,
    limit: int | None = None,
) -> list[TrajectorySpec]:
    specs: list[TrajectorySpec] = []
    for arm_name, arm_config in data_groups.items():
        if not isinstance(arm_config, dict):
            raise TypeError(f"{split}.data_groups.{arm_name} must be a mapping.")
        for dataset_name, dataset_config in arm_config.items():
            if not isinstance(dataset_config, dict) or "path" not in dataset_config:
                raise ValueError(f"{split}.data_groups.{arm_name}.{dataset_name} must contain a path.")
            dataset_path = Path(dataset_config["path"])
            for parquet_path in iter_le_robot_parquets(dataset_path):
                key = make_trajectory_key(split, arm_name, dataset_name, parquet_path)
                specs.append(
                    TrajectorySpec(
                        key=key,
                        split=split,
                        arm=str(arm_name),
                        dataset=str(dataset_name),
                        dataset_path=dataset_path,
                        parquet_path=parquet_path,
                        episode_id=parse_episode_id(parquet_path),
                    )
                )
                if limit is not None and len(specs) >= limit:
                    break
            if limit is not None and len(specs) >= limit:
                break
        if limit is not None and len(specs) >= limit:
            break
    if not specs:
        raise ValueError(f"No trajectories found for split '{split}'.")
    return specs


def load_trajectories_from_specs(
    specs: list[TrajectorySpec],
    *,
    feature_cfg: dict[str, Any],
    embedder: TrajectoryEmbedder,
    desc: str,
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for spec in _progress(
        specs,
        desc=desc,
        unit="traj",
    ):
        trajectories.append(
            Trajectory(
                key=spec.key,
                split=spec.split,
                arm=spec.arm,
                dataset=spec.dataset,
                dataset_path=spec.dataset_path,
                parquet_path=spec.parquet_path,
                episode_id=spec.episode_id,
                features=embedder.embed_parquet(spec.key, spec.parquet_path, feature_cfg),
            )
        )
    if not trajectories:
        raise ValueError(f"No trajectories loaded for '{desc}'.")
    return trajectories


def _progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    if tqdm is not None:
        return tqdm(iterable, dynamic_ncols=True, file=sys.stdout, **kwargs)

    desc = str(kwargs.get("desc", "Progress"))
    unit = str(kwargs.get("unit", "item"))
    total = len(iterable) if hasattr(iterable, "__len__") else None

    def _iter() -> Iterable[Any]:
        for index, item in enumerate(iterable, start=1):
            suffix = f"/{total}" if total is not None else ""
            print(f"{desc}: {index}{suffix} {unit}", flush=True)
            yield item

    return _iter()


def _section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return value
