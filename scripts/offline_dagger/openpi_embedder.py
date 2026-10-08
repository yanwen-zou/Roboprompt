from __future__ import annotations

import dataclasses
import gc
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.offline_dagger.lerobot_data import resolve_dataset_path, trim_features

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is optional for this script.
    tqdm = None


class OpenPIPolicyEmbedder:
    def __init__(self, feature_cfg: dict[str, Any], *, config_dir: Path) -> None:
        openpi_cfg = _section(feature_cfg, "openpi")
        self.config_name = str(openpi_cfg.get("config_name", "pi05_flexiv_bread"))
        self.config_dir = config_dir
        checkpoint_dir = openpi_cfg.get("checkpoint_dir")
        if checkpoint_dir is None:
            raise ValueError("features.openpi.checkpoint_dir is required for OpenPI policy embeddings.")

        self.batch_size = int(openpi_cfg.get("batch_size", 16))
        self.device_index = int(openpi_cfg.get("device_index", 0))
        if "xla_preallocate" not in openpi_cfg:
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        else:
            os.environ.setdefault(
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "true" if bool(openpi_cfg["xla_preallocate"]) else "false",
            )
        if "xla_mem_fraction" in openpi_cfg:
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(openpi_cfg["xla_mem_fraction"]))
        self.pooling = str(openpi_cfg.get("pooling", "mask_mean"))
        if self.pooling not in {"mask_mean", "cls", "tokens"}:
            raise ValueError("features.openpi.pooling must be one of: mask_mean, cls, tokens.")
        self.default_prompt = openpi_cfg.get("default_prompt")
        self.state_column = str(openpi_cfg.get("state_column", "observation.state"))
        self.base_image_view = str(openpi_cfg.get("base_image_view", "observation.images.robot0_agentview_left"))
        self.wrist_image_view = str(openpi_cfg.get("wrist_image_view", "observation.images.robot0_eye_in_hand"))
        cache_base_dir = resolve_dataset_path(
            openpi_cfg.get("cache_dir", "outputs/offline_dagger/embedding_cache"),
            config_dir=config_dir,
        )
        self.cache_dir = cache_base_dir / _safe_path_component(self.config_name)

        self.checkpoint_path = resolve_dataset_path(checkpoint_dir, config_dir=config_dir)

        import jax
        import jax.numpy as jnp
        from openpi import transforms
        from openpi.models import model as openpi_model
        from openpi.training import config as training_config

        print(f"Loading OpenPI config '{self.config_name}'...", flush=True)
        train_config = training_config.get_config(self.config_name)
        assets_base_dir = openpi_cfg.get("assets_base_dir")
        if assets_base_dir is not None:
            train_config = dataclasses.replace(
                train_config,
                assets_base_dir=str(resolve_dataset_path(assets_base_dir, config_dir=config_dir)),
            )
        devices = jax.devices()
        if not devices:
            raise RuntimeError("JAX did not report any available devices.")
        if self.device_index >= len(devices):
            raise ValueError(f"features.openpi.device_index={self.device_index} but JAX only sees {len(devices)} devices.")
        self.device = devices[self.device_index]
        param_sharding = jax.sharding.SingleDeviceSharding(self.device)

        print(f"JAX devices: {devices}", flush=True)
        print(f"Using OpenPI embedding device: {self.device}", flush=True)
        print(f"Restoring OpenPI params from {self.checkpoint_path / 'params'}...", flush=True)
        params = openpi_model.restore_params(
            self.checkpoint_path / "params",
            dtype=jnp.bfloat16,
            sharding=param_sharding,
        )
        print("OpenPI params restored; building model...", flush=True)
        self.model = train_config.model.load(params)

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if self.default_prompt is None:
            self.default_prompt = getattr(train_config.data, "default_prompt", None)
        if self.default_prompt is None:
            self.default_prompt = ""
        else:
            self.default_prompt = str(self.default_prompt)

        self.repo_id = data_config.repo_id
        self.asset_id = data_config.asset_id
        self.local_dataset_root = data_config.local_dataset_root
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Embedding cache dir: {self.cache_dir}")

        norm_stats = self._load_norm_stats(openpi_cfg, self.checkpoint_path, config_dir, default_asset_id=self.asset_id)
        input_norm_stats = norm_stats if norm_stats is not None else data_config.norm_stats
        self.input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(self.default_prompt),
                *data_config.data_transforms.inputs,
                transforms.Normalize(input_norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        print("OpenPI embedding model loaded.", flush=True)

        self._jax = jax
        self._jnp = jnp
        self._openpi_model = openpi_model
        self._embed_batch_jit = jax.jit(self._embed_batch, device=self.device)

    def expert_data_groups(self) -> dict[str, Any]:
        if self.local_dataset_root is None:
            raise ValueError(
                f"OpenPI config '{self.config_name}' does not define data.local_dataset_root; "
                "set expert.data_groups explicitly in the OT config."
            )

        roots = (
            [self.local_dataset_root]
            if isinstance(self.local_dataset_root, str)
            else list(self.local_dataset_root)
        )
        arm_name = _safe_path_component(str(self.asset_id or self.config_name))
        datasets: dict[str, dict[str, str]] = {}
        for index, root in enumerate(roots):
            resolved = self._resolve_train_dataset_path(root)
            dataset_name = _safe_path_component(resolved.name or f"dataset_{index}")
            if dataset_name in datasets:
                dataset_name = f"{dataset_name}_{index}"
            datasets[dataset_name] = {"path": str(resolved)}
        return {arm_name: datasets}

    def _resolve_train_dataset_path(self, path_value: str | Path) -> Path:
        path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
        if path.is_absolute():
            if not path.exists():
                raise FileNotFoundError(f"OpenPI train config dataset path not found: {path}")
            return path.resolve()

        candidates = [
            Path.cwd() / path,
            self.config_dir / path,
            self.checkpoint_path.parent / path,
            self.checkpoint_path.parent.parent / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        searched = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError(
            f"OpenPI train config dataset path not found: {path_value}\n"
            f"Searched:\n  - {searched}"
        )

    def _load_norm_stats(
        self,
        openpi_cfg: dict[str, Any],
        checkpoint_path: Path,
        config_dir: Path,
        *,
        default_asset_id: str | None,
    ) -> dict[str, Any] | None:
        norm_stats_path = openpi_cfg.get("norm_stats_path")
        norm_stats_asset_id = openpi_cfg.get("norm_stats_asset_id", default_asset_id)
        if norm_stats_path is not None:
            target = resolve_dataset_path(norm_stats_path, config_dir=config_dir)
        elif norm_stats_asset_id is not None:
            target = checkpoint_path / "assets" / str(norm_stats_asset_id)
        else:
            return None

        from openpi.shared import normalize

        try:
            return normalize.load(target)
        except FileNotFoundError:
            if norm_stats_path is not None or "norm_stats_asset_id" in openpi_cfg:
                raise
            return None

    def _embed_batch(self, batch: dict[str, Any]) -> Any:
        observation = self._openpi_model.Observation.from_dict(batch)
        tokens, mask, _ = self.model.embed_prefix(observation)
        if self.pooling == "tokens":
            return self._jnp.concatenate([tokens, mask.astype(tokens.dtype)[..., None]], axis=-1)
        if self.pooling == "cls":
            return tokens[:, 0, :]
        mask = mask.astype(tokens.dtype)
        denom = self._jnp.maximum(self._jnp.sum(mask, axis=1, keepdims=True), 1.0)
        return self._jnp.sum(tokens * mask[..., None], axis=1) / denom

    def cache_path_for(self, traj_key: str, parquet_path: Path, feature_cfg: dict[str, Any]) -> Path:
        digest = hashlib.sha1(
            "|".join(
                [
                    self.config_name,
                    str(self.repo_id),
                    str(self.asset_id),
                    str(self.local_dataset_root),
                    str(self.checkpoint_path.resolve()),
                    traj_key,
                    str(parquet_path.resolve()),
                    str(parquet_path.stat().st_mtime_ns),
                    self.default_prompt,
                    self.state_column,
                    self.base_image_view,
                    self.wrist_image_view,
                    self.pooling,
                    str(feature_cfg.get("max_steps")),
                    str(feature_cfg.get("max_dim")),
                    str(feature_cfg.get("drop_nan_rows", False)),
                ]
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def embed_parquet(self, traj_key: str, parquet_path: Path, feature_cfg: dict[str, Any]) -> np.ndarray:
        cache_path = self.cache_path_for(traj_key, parquet_path, feature_cfg)
        if cache_path.exists() and bool(feature_cfg.get("use_cache", True)):
            openpi_cfg = _section(feature_cfg, "openpi")
            if bool(openpi_cfg.get("log_cache_hits", False)):
                _log(f"Embedding cache hit [{traj_key}] -> {cache_path.name}")
            mmap_mode = "r" if bool(openpi_cfg.get("mmap_cache", True)) else None
            return np.load(cache_path, mmap_mode=mmap_mode)

        _log(f"Embedding cache miss [{traj_key}]; reading parquet and videos...")
        df = pd.read_parquet(parquet_path)
        max_steps = feature_cfg.get("max_steps")
        if max_steps is not None:
            df = df.iloc[: int(max_steps)]
        if self.state_column not in df.columns:
            raise KeyError(f"Missing OpenPI state column in parquet: {self.state_column}")

        base_video, wrist_video = self._video_paths(parquet_path)
        frame_indices = (
            df["frame_index"].astype(int).tolist() if "frame_index" in df.columns else list(range(len(df)))
        )

        embeddings: list[np.ndarray] = []
        starts = range(0, len(df), self.batch_size)
        total_batches = math.ceil(len(df) / self.batch_size)
        for start in _progress(
            starts,
            desc=f"Embedding {traj_key}",
            unit="batch",
            total=total_batches,
            leave=False,
        ):
            end = min(start + self.batch_size, len(df))
            batch_df = df.iloc[start:end]
            batch_frame_indices = frame_indices[start:end]
            base_frames = read_video_frames(base_video, batch_frame_indices)
            wrist_frames = read_video_frames(wrist_video, batch_frame_indices)
            batch_samples = [
                self.input_transform(dict(sample))
                for sample in self._build_batch_samples(batch_df, base_frames, wrist_frames)
            ]
            actual_batch_size = len(batch_samples)
            if actual_batch_size < self.batch_size:
                batch_samples.extend([batch_samples[-1]] * (self.batch_size - actual_batch_size))
            batch = self._jax.tree.map(lambda *xs: np.stack(xs, axis=0), *batch_samples)
            batch = self._jax.tree.map(lambda x: self._jax.device_put(x, self.device), batch)
            embedded = self._embed_batch_jit(batch)
            embedded.block_until_ready()
            embeddings.append(np.asarray(embedded[:actual_batch_size], dtype=np.float32))
            del batch, batch_samples, base_frames, wrist_frames, embedded
        features = trim_features(np.concatenate(embeddings, axis=0), feature_cfg)
        del embeddings
        if bool(feature_cfg.get("use_cache", True)):
            np.save(cache_path, features)
        gc.collect()
        return features

    def _video_paths(self, parquet_path: Path) -> tuple[Path, Path]:
        dataset_root = parquet_path.parents[2]
        chunk_name = parquet_path.parent.name
        episode_name = parquet_path.stem
        base_video = self._video_path(dataset_root, chunk_name, episode_name, self.base_image_view)
        wrist_video = self._video_path(dataset_root, chunk_name, episode_name, self.wrist_image_view)
        return base_video, wrist_video

    def _build_batch_samples(
        self,
        df: pd.DataFrame,
        base_frames: list[np.ndarray],
        wrist_frames: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        samples = []
        for idx, (_, row) in enumerate(df.iterrows()):
            samples.append(
                {
                    "observation/state": np.asarray(row[self.state_column], dtype=np.float32),
                    "observation/image": base_frames[idx],
                    "observation/wrist_image": wrist_frames[idx],
                    "prompt": self._prompt_from_row(row),
                }
            )
        return samples

    def _prompt_from_row(self, row: pd.Series) -> str:
        for key in ("annotation.human.task_description", "annotation.human.task_name"):
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, str) and value:
                return value
            if isinstance(value, np.ndarray) and value.size:
                item = value.reshape(-1)[0]
                if isinstance(item, str) and item:
                    return item
        return self.default_prompt

    @staticmethod
    def _video_path(dataset_root: Path, chunk_name: str, episode_name: str, view_name: str) -> Path:
        path = dataset_root / "videos" / chunk_name / view_name / episode_name / f"{episode_name}.mp4"
        if not path.exists():
            raise FileNotFoundError(f"Required OpenPI video view not found: {path}")
        return path


def read_video_frames(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    import cv2

    if not frame_indices:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames_by_index: dict[int, np.ndarray] = {}
    try:
        unique_indices = sorted(set(int(idx) for idx in frame_indices))
        cursor = 0
        target_pos = 0
        while target_pos < len(unique_indices):
            target = unique_indices[target_pos]
            if cursor != target:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                cursor = target
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Could not read frame {target} from {video_path}")
            frames_by_index[target] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            cursor = target + 1
            target_pos += 1
    finally:
        cap.release()

    return [frames_by_index[int(idx)] for idx in frame_indices]


def _progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    if tqdm is not None:
        return tqdm(iterable, dynamic_ncols=True, file=sys.stdout, **kwargs)

    desc = str(kwargs.get("desc", "Progress"))
    unit = str(kwargs.get("unit", "item"))
    total = kwargs.get("total")

    def _iter() -> Iterable[Any]:
        for index, item in enumerate(iterable, start=1):
            suffix = f"/{total}" if total is not None else ""
            print(f"{desc}: {index}{suffix} {unit}", flush=True)
            yield item

    return _iter()


def _log(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message, file=sys.stdout)
    else:
        print(message, flush=True)


def _safe_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return value
