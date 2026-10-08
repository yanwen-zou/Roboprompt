from __future__ import annotations

import bisect
import copy
import json
import os
import pathlib
import warnings
from typing import Any, Dict, Iterable, Mapping

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Sampler
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.sampler import create_indices, downsample_mask, get_val_mask
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class LeRobotV21ImageDataset(BaseImageDataset):
    """Direct reader for LeRobot v2.1 parquet/video datasets.

    This intentionally avoids importing the installed ``lerobot`` package. The
    bundled datasets in this repo are v2.1 layouts, while the Python dependency
    may be a different LeRobot API version.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_paths: str | Iterable[str] | None = None,
        offline_dataset_paths: str | Iterable[str] | None = None,
        online_dataset_paths: str | Iterable[str] | None = None,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int | None = None,
        n_latency_steps: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        normalizer_mode: str = "limits",
        online_filter_denoise_step_lt_max: bool = False,
        online_denoise_step_max: float | None = None,
    ):
        self.shape_meta = shape_meta
        if offline_dataset_paths is None:
            offline_dataset_paths = dataset_paths
        self.offline_dataset_paths = _normalize_paths(offline_dataset_paths)
        self.online_dataset_paths = _normalize_paths(online_dataset_paths)
        if not self.offline_dataset_paths:
            raise ValueError("LeRobotV21ImageDataset requires at least one offline dataset path.")
        self.dataset_paths = self.offline_dataset_paths + self.online_dataset_paths
        self.horizon = int(horizon)
        self.n_latency_steps = int(n_latency_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.n_obs_steps = n_obs_steps
        self.normalizer_mode = normalizer_mode
        self.online_filter_denoise_step_lt_max = bool(online_filter_denoise_step_lt_max)
        self.online_denoise_step_max = online_denoise_step_max

        self.rgb_keys = []
        self.lowdim_keys = []
        self._obs_sources = {}
        self._image_shapes = {}
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            source_key = attr.get("lerobot_key", key)
            self._obs_sources[key] = source_key
            if obs_type == "rgb":
                self.rgb_keys.append(key)
                self._image_shapes[key] = tuple(int(v) for v in attr["shape"])
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported obs type {obs_type!r} for {key!r}.")

        self.action_key = shape_meta.get("action", {}).get("lerobot_key", "action")
        dataset_splits = (["offline"] * len(self.offline_dataset_paths)) + (
            ["online"] * len(self.online_dataset_paths)
        )
        arrays, episode_ends, episode_infos = _load_lerobot_v21_arrays(
            self.dataset_paths,
            dataset_splits=dataset_splits,
            shape_meta=shape_meta,
            lowdim_sources=[self._obs_sources[key] for key in self.lowdim_keys],
            action_key=self.action_key,
            image_sources=[self._obs_sources[key] for key in self.rgb_keys],
            online_filter_denoise_step_lt_max=self.online_filter_denoise_step_lt_max,
            online_denoise_step_max=self.online_denoise_step_max,
        )
        self._arrays = arrays
        self.episode_ends = episode_ends
        self.episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self._episode_infos = episode_infos
        self._video_cache = None

        if self.online_dataset_paths:
            val_mask = _get_split_val_mask(episode_infos, val_ratio=val_ratio, seed=seed)
        else:
            val_mask = get_val_mask(n_episodes=len(episode_ends), val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
        self.indices = create_indices(
            episode_ends,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=train_mask,
        )
        self.val_mask = val_mask
        self._balanced_sampling_enabled = len(self.online_dataset_paths) > 0
        self._refresh_sample_groups()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_cache"] = None
        return state

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = create_indices(
            self.episode_ends,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        val_set._video_cache = None
        val_set._balanced_sampling_enabled = False
        val_set._refresh_sample_groups()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        del kwargs
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self._arrays["action"], mode=self.normalizer_mode
        )
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self._arrays[key], mode=self.normalizer_mode
            )
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._arrays["action"])

    def __len__(self):
        if self._balanced_sampling_enabled:
            return 2 * max(len(self._offline_sample_indices), len(self._online_sample_indices))
        return len(self.indices)

    @property
    def balanced_sampling_enabled(self) -> bool:
        return self._balanced_sampling_enabled

    def get_balanced_sampler(self, seed: int = 0) -> Sampler[int]:
        if not self._balanced_sampling_enabled:
            raise ValueError("Balanced sampler requested but online dataset paths are empty.")
        return BalancedOfflineOnlineSampler(self, seed=seed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        idx = self._resolve_sample_index(idx)
        sequence_indices = self._sequence_indices(idx)
        obs_indices = sequence_indices[: self.n_obs_steps] if self.n_obs_steps is not None else sequence_indices

        obs_dict = {}
        for key in self.lowdim_keys:
            obs_dict[key] = self._arrays[key][obs_indices].astype(np.float32)
        for key in self.rgb_keys:
            frames = [self._read_image_frame(key, int(global_idx)) for global_idx in obs_indices]
            obs_dict[key] = np.stack(frames, axis=0).astype(np.float32) / 255.0

        action = self._arrays["action"][sequence_indices].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps :]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }

    def _sequence_indices(self, idx: int) -> np.ndarray:
        buffer_start, buffer_end, sample_start, sample_end = self.indices[idx]
        sequence = np.empty(self.horizon + self.n_latency_steps, dtype=np.int64)
        if sample_start > 0:
            sequence[:sample_start] = buffer_start
        if sample_end < len(sequence):
            sequence[sample_end:] = buffer_end - 1
        sequence[sample_start:sample_end] = np.arange(buffer_start, buffer_end, dtype=np.int64)
        return sequence

    def _refresh_sample_groups(self):
        offline_indices = []
        online_indices = []
        for sample_idx, (buffer_start, *_rest) in enumerate(self.indices):
            episode_idx = bisect.bisect_right(self.episode_ends, int(buffer_start))
            split = self._episode_infos[episode_idx].get("split", "offline")
            if split == "online":
                if not self._online_sample_allowed(int(buffer_start)):
                    continue
                online_indices.append(sample_idx)
            else:
                offline_indices.append(sample_idx)
        self._offline_sample_indices = np.asarray(offline_indices, dtype=np.int64)
        self._online_sample_indices = np.asarray(online_indices, dtype=np.int64)
        if self._balanced_sampling_enabled:
            if len(self._offline_sample_indices) == 0:
                raise ValueError("Balanced offline/online sampling requires at least one offline sample.")
            if len(self._online_sample_indices) == 0:
                raise ValueError("Balanced offline/online sampling requires at least one online sample.")

    def _resolve_sample_index(self, idx: int) -> int:
        if not self._balanced_sampling_enabled:
            return int(idx)

        idx = int(idx)
        if idx % 2 == 0:
            pool = self._offline_sample_indices
        else:
            pool = self._online_sample_indices
        return int(pool[(idx // 2) % len(pool)])

    def _online_sample_allowed(self, buffer_start: int) -> bool:
        if not self.online_filter_denoise_step_lt_max:
            return True
        denoise_steps = self._arrays.get("_denoise_step")
        denoise_step_max = self._arrays.get("_denoise_step_max")
        if denoise_steps is None or denoise_step_max is None:
            return True
        step = float(denoise_steps[buffer_start, 0])
        threshold = float(denoise_step_max[buffer_start, 0])
        if not np.isfinite(step) or not np.isfinite(threshold):
            return True
        return step < threshold

    def _read_image_frame(self, key: str, global_idx: int) -> np.ndarray:
        episode_idx = bisect.bisect_right(self.episode_ends, global_idx)
        local_idx = global_idx - int(self.episode_starts[episode_idx])
        source_key = self._obs_sources[key]
        video_path = self._episode_infos[episode_idx]["videos"][source_key]
        image = self._video_reader().read(video_path, local_idx)
        _, height, width = self._image_shapes[key]
        return _center_crop_resize(image, width=width, height=height)

    def _video_reader(self) -> "_VideoReader":
        if self._video_cache is None:
            self._video_cache = _VideoReader()
        return self._video_cache


class _VideoReader:
    def __init__(self):
        self._captures: dict[str, cv2.VideoCapture] = {}

    def read(self, path: str, frame_idx: int) -> np.ndarray:
        cap = self._captures.get(path)
        if cap is None:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video {path}")
            self._captures[path] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __del__(self):
        for cap in self._captures.values():
            cap.release()


class BalancedOfflineOnlineSampler(Sampler[int]):
    def __init__(self, dataset: LeRobotV21ImageDataset, seed: int = 0):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        half_len = len(self.dataset) // 2
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        offline_order = torch.randperm(half_len, generator=generator).tolist()
        online_order = torch.randperm(half_len, generator=generator).tolist()
        indices = []
        for offline_idx, online_idx in zip(offline_order, online_order):
            indices.append(2 * offline_idx)
            indices.append(2 * online_idx + 1)
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


def _load_lerobot_v21_arrays(
    dataset_paths: list[str],
    *,
    dataset_splits: list[str] | None = None,
    shape_meta: Mapping[str, Any],
    lowdim_sources: list[str],
    action_key: str,
    image_sources: list[str],
    online_filter_denoise_step_lt_max: bool = False,
    online_denoise_step_max: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    arrays: dict[str, list[np.ndarray]] = {"action": []}
    for obs_key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") == "low_dim":
            arrays[obs_key] = []

    episode_ends = []
    episode_infos = []
    total_steps = 0
    if dataset_splits is None:
        dataset_splits = ["offline"] * len(dataset_paths)
    if len(dataset_splits) != len(dataset_paths):
        raise ValueError("dataset_splits must have the same length as dataset_paths.")

    for dataset_path, dataset_split in zip(dataset_paths, dataset_splits):
        root = pathlib.Path(dataset_path)
        info = _read_json(root / "meta" / "info.json")
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{root} is not a LeRobot v2.1 dataset: {info.get('codebase_version')!r}")
        features = info.get("features", {})
        for source in lowdim_sources + [action_key] + image_sources:
            if source not in features:
                raise KeyError(f"{root} is missing LeRobot feature {source!r}.")

        episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
        chunks_size = int(info.get("chunks_size", 1000))
        data_template = info["data_path"]
        video_template = info["video_path"]
        denoise_threshold = None
        has_denoise_step = "denoise_step" in features
        should_filter_online = online_filter_denoise_step_lt_max and dataset_split == "online"
        if should_filter_online:
            denoise_threshold = _resolve_online_denoise_step_max(root, online_denoise_step_max)
            if denoise_threshold is None and has_denoise_step:
                denoise_threshold = _read_observed_denoise_step_max(root, info, episodes)
                if denoise_threshold is not None:
                    warnings.warn(
                        f"{root} has no configured max denoise step; using observed max "
                        f"denoise_step={denoise_threshold:g} as the online filter threshold.",
                        RuntimeWarning,
                    )
            if denoise_threshold is None:
                warnings.warn(
                    f"{root} is configured as online data but no max denoise step was found; "
                    "online denoise_step filtering is disabled for this dataset.",
                    RuntimeWarning,
                )
            if not has_denoise_step:
                warnings.warn(
                    f"{root} is configured as online data but has no denoise_step feature; "
                    "online denoise_step filtering is disabled for this dataset.",
                    RuntimeWarning,
                )

        for episode in episodes:
            episode_index = int(episode["episode_index"])
            episode_chunk = episode_index // chunks_size
            parquet_path = root / data_template.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            columns = lowdim_sources + [action_key]
            if should_filter_online and denoise_threshold is not None and has_denoise_step:
                columns = columns + ["denoise_step"]
            table = _read_parquet_columns(parquet_path, columns)
            expected_length = int(episode["length"])
            if len(table[action_key]) != expected_length:
                raise ValueError(f"{parquet_path} length mismatch: expected {expected_length}, got {len(table[action_key])}")

            arrays["action"].append(_column_to_array(table[action_key], expected_dim=shape_meta["action"]["shape"][0]))
            for obs_key, attr in shape_meta["obs"].items():
                if attr.get("type", "low_dim") != "low_dim":
                    continue
                source = attr.get("lerobot_key", obs_key)
                arrays[obs_key].append(_column_to_array(table[source], expected_dim=attr["shape"][0]))
            if should_filter_online and denoise_threshold is not None and has_denoise_step:
                denoise_step = _column_to_array(table["denoise_step"], expected_dim=1)
                max_step = np.full((expected_length, 1), denoise_threshold, dtype=np.float32)
            else:
                denoise_step = np.full((expected_length, 1), np.nan, dtype=np.float32)
                max_step = np.full((expected_length, 1), np.inf, dtype=np.float32)
            arrays.setdefault("_denoise_step", []).append(denoise_step)
            arrays.setdefault("_denoise_step_max", []).append(max_step)

            videos = {
                source: str(
                    root
                    / video_template.format(
                        episode_chunk=episode_chunk,
                        episode_index=episode_index,
                        video_key=source,
                    )
                )
                for source in image_sources
            }
            episode_infos.append(
                {"root": str(root), "episode_index": episode_index, "split": dataset_split, "videos": videos}
            )
            total_steps += expected_length
            episode_ends.append(total_steps)

    stacked = {key: np.concatenate(parts, axis=0).astype(np.float32) for key, parts in arrays.items()}
    return stacked, np.asarray(episode_ends, dtype=np.int64), episode_infos


def _resolve_online_denoise_step_max(root: pathlib.Path, override: float | None) -> float | None:
    if override is not None:
        return _as_float(override)
    meta_path = root / "extras" / "dataset_meta_server.json"
    if not meta_path.exists():
        return None
    try:
        meta = _read_json(meta_path)
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(f"Could not read {meta_path}: {exc}", RuntimeWarning)
        return None
    return _find_first_float(
        meta,
        (
            "max_denoise_step",
            "max_denoise_steps",
            "max_phase2_steps",
            "phase2_max_steps",
            "policy_inference_steps",
            "num_inference_steps",
            "num_steps",
        ),
    )


def _read_observed_denoise_step_max(
    root: pathlib.Path,
    info: Mapping[str, Any],
    episodes: list[dict[str, Any]],
) -> float | None:
    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info["data_path"]
    max_values = []
    try:
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            episode_chunk = episode_index // chunks_size
            parquet_path = root / data_template.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            table = _read_parquet_columns(parquet_path, ["denoise_step"])
            denoise_step = _column_to_array(table["denoise_step"], expected_dim=1)
            if len(denoise_step):
                max_values.append(float(np.nanmax(denoise_step[:, 0])))
    except Exception as exc:
        warnings.warn(f"Could not infer observed max denoise_step for {root}: {exc}", RuntimeWarning)
        return None
    if not max_values:
        return None
    parsed = max(max_values)
    if not np.isfinite(parsed):
        return None
    return parsed


def _find_first_float(value: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                parsed = _as_float(value[key])
                if parsed is not None:
                    return parsed
        for child in value.values():
            parsed = _find_first_float(child, keys)
            if parsed is not None:
                return parsed
    elif isinstance(value, list):
        for child in value:
            parsed = _find_first_float(child, keys)
            if parsed is not None:
                return parsed
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _read_parquet_columns(path: pathlib.Path, columns: list[str]) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "LeRobotV21ImageDataset requires pyarrow to read parquet files. "
            "Install the project dependencies before training Diffusion Policy on LeRobot v2.1 data."
        ) from exc
    table = pq.read_table(path, columns=columns)
    return {name: table[name].to_pylist() for name in columns}


def _column_to_array(values: list[Any], expected_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[-1] != int(expected_dim):
        raise ValueError(f"Expected dim {expected_dim}, got {array.shape[-1]}")
    return array


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _as_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_paths(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    return [str(pathlib.Path(p).expanduser()) for p in _as_list(value)]


def _get_split_val_mask(episode_infos: list[dict[str, Any]], *, val_ratio: float, seed: int) -> np.ndarray:
    val_mask = np.zeros(len(episode_infos), dtype=bool)
    if val_ratio <= 0:
        return val_mask

    rng = np.random.default_rng(seed=seed)
    for split in ("offline", "online"):
        split_indices = np.asarray(
            [idx for idx, info in enumerate(episode_infos) if info.get("split", "offline") == split],
            dtype=np.int64,
        )
        if len(split_indices) <= 1:
            continue
        n_val = min(max(1, round(len(split_indices) * val_ratio)), len(split_indices) - 1)
        selected = rng.choice(split_indices, size=n_val, replace=False)
        val_mask[selected] = True
    return val_mask


def _center_crop_resize(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(image).convert("RGB")
    src_w, src_h = pil.size
    scale = max(width / src_w, height / src_h)
    resized = pil.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    resized_w, resized_h = resized.size
    left = max((resized_w - width) // 2, 0)
    top = max((resized_h - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8).transpose(2, 0, 1)
