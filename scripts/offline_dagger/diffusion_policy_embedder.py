from __future__ import annotations

import dataclasses
import gc
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import dill
import numpy as np
import pandas as pd
import torch
from PIL import Image

from scripts.offline_dagger.lerobot_data import resolve_dataset_path, trim_features

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is optional for this script.
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFUSION_POLICY_ROOT = REPO_ROOT / "diffusion_policy"
if str(DIFFUSION_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))


@dataclasses.dataclass(frozen=True)
class _ObsKey:
    name: str
    source: str
    shape: tuple[int, ...]
    obs_type: str


class DiffusionPolicyEmbedder:
    def __init__(self, feature_cfg: dict[str, Any], *, config_dir: Path) -> None:
        dp_cfg = _section(feature_cfg, "diffusion_policy")
        checkpoint_path = dp_cfg.get("checkpoint_path")
        if checkpoint_path is None:
            checkpoint_path = dp_cfg.get("checkpoint_dir")
        if checkpoint_path is None:
            raise ValueError("features.diffusion_policy.checkpoint_path is required.")

        self.checkpoint_path = self._resolve_checkpoint(
            resolve_dataset_path(checkpoint_path, config_dir=config_dir)
        )
        self.use_ema = bool(dp_cfg.get("use_ema", True))
        self.batch_size = int(dp_cfg.get("batch_size", 32))
        self.device = torch.device(str(dp_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
        self.pooling = str(dp_cfg.get("pooling", "obs_encoder"))
        if self.pooling != "obs_encoder":
            raise ValueError("features.diffusion_policy.pooling currently supports only 'obs_encoder'.")
        self.mmap_cache = bool(dp_cfg.get("mmap_cache", True))
        self.log_cache_hits = bool(dp_cfg.get("log_cache_hits", False))

        cache_base_dir = resolve_dataset_path(
            dp_cfg.get("cache_dir", "outputs/offline_dagger/embedding_cache"),
            config_dir=config_dir,
        )
        self.config_name = str(dp_cfg.get("config_name", self._default_config_name()))
        self.cache_dir = cache_base_dir / self.config_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading Diffusion Policy checkpoint: {self.checkpoint_path}", flush=True)
        self.policy, self.dp_train_cfg = self._load_policy(self.checkpoint_path)
        self.policy.eval().to(self.device)
        self.shape_meta = self.dp_train_cfg.shape_meta
        self.obs_keys = self._parse_obs_keys(self.shape_meta)
        self.rgb_keys = [key for key in self.obs_keys if key.obs_type == "rgb"]
        self.lowdim_keys = [key for key in self.obs_keys if key.obs_type == "low_dim"]
        if not self.rgb_keys and not self.lowdim_keys:
            raise ValueError("Diffusion Policy shape_meta does not define any supported obs keys.")
        print(f"Using Diffusion Policy embedding device: {self.device}", flush=True)
        print(f"Diffusion Policy embedding cache dir: {self.cache_dir}", flush=True)

    def expert_data_groups(self) -> dict[str, Any]:
        dataset_paths = self.dp_train_cfg.task.get("dataset_paths", None)
        if dataset_paths is None:
            raise ValueError(
                "Diffusion Policy checkpoint does not define task.dataset_paths; "
                "set expert.data_groups explicitly in the OT config."
            )
        datasets: dict[str, dict[str, str]] = {}
        roots = dataset_paths if isinstance(dataset_paths, list) else [dataset_paths]
        for index, root in enumerate(roots):
            resolved = self._resolve_train_dataset_path(root)
            dataset_name = _safe_path_component(resolved.name or f"dataset_{index}")
            if dataset_name in datasets:
                dataset_name = f"{dataset_name}_{index}"
            datasets[dataset_name] = {"path": str(resolved)}
        return {_safe_path_component(self.config_name): datasets}

    def cache_path_for(self, traj_key: str, parquet_path: Path, feature_cfg: dict[str, Any]) -> Path:
        digest = hashlib.sha1(
            "|".join(
                [
                    self.config_name,
                    str(self.checkpoint_path.resolve()),
                    str(self.checkpoint_path.stat().st_mtime_ns),
                    traj_key,
                    str(parquet_path.resolve()),
                    str(parquet_path.stat().st_mtime_ns),
                    self.pooling,
                    str(self.batch_size),
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
            if self.log_cache_hits:
                _log(f"Diffusion Policy embedding cache hit [{traj_key}] -> {cache_path.name}")
            mmap_mode = "r" if self.mmap_cache else None
            return np.load(cache_path, mmap_mode=mmap_mode)

        _log(f"Diffusion Policy embedding cache miss [{traj_key}]; reading parquet and videos...")
        df = pd.read_parquet(parquet_path)
        max_steps = feature_cfg.get("max_steps")
        if max_steps is not None:
            df = df.iloc[: int(max_steps)]
        if len(df) == 0:
            raise ValueError(f"Trajectory has no rows: {parquet_path}")

        frame_indices = (
            df["frame_index"].astype(int).tolist() if "frame_index" in df.columns else list(range(len(df)))
        )
        obs_arrays = self._read_observations(parquet_path, df, frame_indices)
        embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for start in _progress(
                range(0, len(df), self.batch_size),
                desc=f"Embedding DP {traj_key}",
                unit="batch",
                total=(len(df) + self.batch_size - 1) // self.batch_size,
                leave=False,
            ):
                end = min(start + self.batch_size, len(df))
                obs_batch = {
                    key: torch.as_tensor(value[start:end], dtype=torch.float32, device=self.device)
                    for key, value in obs_arrays.items()
                }
                nobs = self.policy.normalizer.normalize(obs_batch)
                encoded = self.policy.obs_encoder(nobs)
                embeddings.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
                del obs_batch, nobs, encoded

        features = trim_features(np.concatenate(embeddings, axis=0), feature_cfg)
        del embeddings, obs_arrays
        if bool(feature_cfg.get("use_cache", True)):
            np.save(cache_path, features)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return features

    def _read_observations(
        self,
        parquet_path: Path,
        df: pd.DataFrame,
        frame_indices: list[int],
    ) -> dict[str, np.ndarray]:
        obs_arrays: dict[str, np.ndarray] = {}
        for key in self.lowdim_keys:
            if key.source not in df.columns:
                raise KeyError(f"Missing Diffusion Policy lowdim column in parquet: {key.source}")
            obs_arrays[key.name] = _column_to_array(df[key.source].tolist(), expected_dim=key.shape[0])

        for key in self.rgb_keys:
            video_path = self._video_path(parquet_path, key.source)
            frames = read_video_frames(video_path, frame_indices)
            _, height, width = key.shape
            obs_arrays[key.name] = np.stack(
                [_center_crop_resize(frame, width=width, height=height) for frame in frames],
                axis=0,
            ).astype(np.float32) / 255.0
        return obs_arrays

    def _video_path(self, parquet_path: Path, view_name: str) -> Path:
        dataset_root = parquet_path.parents[2]
        chunk_name = parquet_path.parent.name
        episode_name = parquet_path.stem
        path = dataset_root / "videos" / chunk_name / view_name / episode_name / f"{episode_name}.mp4"
        if not path.exists():
            raise FileNotFoundError(f"Required Diffusion Policy video view not found: {path}")
        return path

    def _load_policy(self, checkpoint_path: Path) -> tuple[torch.nn.Module, Any]:
        import hydra
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
        train_cfg = payload["cfg"]
        policy = hydra.utils.instantiate(train_cfg.policy)
        state_dicts = payload.get("state_dicts", {})
        state_key = "ema_model" if self.use_ema and state_dicts.get("ema_model") is not None else "model"
        if state_key not in state_dicts:
            raise KeyError(f"Checkpoint does not contain state_dicts.{state_key}")
        policy.load_state_dict(state_dicts[state_key])
        return policy, train_cfg

    def _resolve_checkpoint(self, path: Path) -> Path:
        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(f"Diffusion Policy checkpoint path not found: {path}")
        checkpoints_dir = path / "checkpoints"
        search_dir = checkpoints_dir if checkpoints_dir.is_dir() else path
        candidates = sorted(
            p for p in search_dir.glob("*.ckpt")
            if p.is_file() and not any(part.startswith(".") for part in p.name.split("."))
        )
        if not candidates:
            raise FileNotFoundError(f"No .ckpt file found under {search_dir}")
        latest = search_dir / "latest.ckpt"
        if latest.exists():
            return latest.resolve()
        return max(candidates, key=lambda item: item.stat().st_mtime).resolve()

    def _resolve_train_dataset_path(self, path_value: str | Path) -> Path:
        path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
        if path.exists():
            return path.resolve()

        candidates: list[Path] = []
        text = str(path)
        marker = "/real_robot_data/"
        if marker in text:
            suffix = text.split(marker, 1)[1]
            candidates.append(REPO_ROOT / "robocasa_data_ckpt" / "real_robot_data" / suffix)
        candidates.append(REPO_ROOT / "robocasa_data_ckpt" / "real_robot_data" / "bread" / path.name)
        candidates.append(REPO_ROOT / "robocasa_data_ckpt" / "real_robot_data" / path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        searched = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError(
            f"Diffusion Policy train dataset path not found: {path_value}\n"
            f"Searched:\n  - {searched}"
        )

    def _parse_obs_keys(self, shape_meta: Any) -> list[_ObsKey]:
        obs_meta = shape_meta["obs"]
        keys = []
        for name, attr in obs_meta.items():
            obs_type = str(attr.get("type", "low_dim"))
            if obs_type not in {"rgb", "low_dim"}:
                continue
            shape = tuple(int(v) for v in attr["shape"])
            source = str(attr.get("lerobot_key", name))
            keys.append(_ObsKey(name=str(name), source=source, shape=shape, obs_type=obs_type))
        return keys

    def _default_config_name(self) -> str:
        run_dir = self.checkpoint_path.parents[1] if self.checkpoint_path.parent.name == "checkpoints" else self.checkpoint_path.parent
        return f"diffusion_policy_{_safe_path_component(run_dir.name)}"


def read_video_frames(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    if not frame_indices:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames_by_index: dict[int, np.ndarray] = {}
    try:
        unique_indices = sorted(set(int(idx) for idx in frame_indices))
        cursor = 0
        for target in unique_indices:
            if cursor != target:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                cursor = target
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Could not read frame {target} from {video_path}")
            frames_by_index[target] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            cursor = target + 1
    finally:
        cap.release()

    return [frames_by_index[int(idx)] for idx in frame_indices]


def _center_crop_resize(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(image).convert("RGB")
    src_w, src_h = pil.size
    scale = max(width / src_w, height / src_h)
    resized = pil.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    resized_w, resized_h = resized.size
    left = max((resized_w - width) // 2, 0)
    top = max((resized_h - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8).transpose(2, 0, 1)


def _column_to_array(values: list[Any], expected_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[-1] != int(expected_dim):
        raise ValueError(f"Expected dim {expected_dim}, got {array.shape[-1]}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _progress(iterable: Any, **kwargs: Any) -> Any:
    if tqdm is not None:
        return tqdm(iterable, dynamic_ncols=True, file=sys.stdout, **kwargs)
    return iterable


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
