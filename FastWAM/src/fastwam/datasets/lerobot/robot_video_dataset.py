import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs=None,
        offline_dataset_dirs=None,
        online_dataset_dirs=None,
        shape_meta=None,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        target_fps: Optional[float] = None,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        online_filter_denoise_step_lt_max: bool = False,
        online_denoise_step_max: Optional[float] = None,
    ):
        if offline_dataset_dirs is None:
            offline_dataset_dirs = dataset_dirs
        self.offline_dataset_dirs = _as_list(offline_dataset_dirs)
        self.online_dataset_dirs = _as_list(online_dataset_dirs)
        if not self.offline_dataset_dirs:
            raise ValueError("RobotVideoDataset requires at least one offline dataset dir.")
        if shape_meta is None:
            raise ValueError("RobotVideoDataset requires shape_meta.")
        dataset_dirs = self.offline_dataset_dirs + self.online_dataset_dirs
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            target_fps=target_fps,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.online_filter_denoise_step_lt_max = bool(online_filter_denoise_step_lt_max)
        self.online_denoise_step_max = online_denoise_step_max
        self.balanced_sampling_indices = self._build_balanced_sampling_indices()

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        if self.balanced_sampling_indices is not None:
            return 2 * max(
                len(self.balanced_sampling_indices["offline"]),
                len(self.balanced_sampling_indices["online"]),
            )
        return len(self.lerobot_dataset)

    def _get(self, idx):
        idx = self._resolve_sample_index(idx)
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = self._resolve_sample_index(np.random.randint(len(self)))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data

    def _build_balanced_sampling_indices(self):
        if not self.online_dataset_dirs:
            return None

        offline_count = len(self.offline_dataset_dirs)
        offline_indices = []
        online_indices = []
        start = 0
        for dataset_idx, dataset in enumerate(self.lerobot_dataset.multi_dataset._datasets):
            end = start + len(dataset)
            if dataset_idx < offline_count:
                offline_indices.extend(range(start, end))
            else:
                candidate_indices = np.arange(start, end, dtype=np.int64)
                if self.online_filter_denoise_step_lt_max:
                    selected_episodes = getattr(dataset, "episodes", None)
                    mask = _read_online_denoise_filter_mask(
                        Path(dataset.root),
                        selected_episodes=selected_episodes,
                        expected_length=len(dataset),
                        override=self.online_denoise_step_max,
                    )
                    candidate_indices = candidate_indices[mask]
                online_indices.extend(candidate_indices.tolist())
            start = end

        if not offline_indices:
            raise ValueError("Balanced offline/online sampling requires at least one offline sample.")
        if not online_indices:
            raise ValueError("Balanced offline/online sampling requires at least one online sample.")

        return {
            "offline": np.asarray(offline_indices, dtype=np.int64),
            "online": np.asarray(online_indices, dtype=np.int64),
        }

    def _resolve_sample_index(self, idx):
        if self.balanced_sampling_indices is None:
            return int(idx)

        idx = int(idx)
        if idx % 2 == 0:
            pool = self.balanced_sampling_indices["offline"]
        else:
            pool = self.balanced_sampling_indices["online"]
        return int(pool[(idx // 2) % len(pool)])


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _read_online_denoise_filter_mask(
    root: Path,
    *,
    selected_episodes: list[int] | None,
    expected_length: int,
    override: float | None,
) -> np.ndarray:
    threshold = _resolve_online_denoise_step_max(root, override)
    try:
        info = _read_json(root / "meta" / "info.json")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s metadata for denoise_step filtering: %s", root, exc)
        return np.ones(expected_length, dtype=bool)

    if "denoise_step" not in info.get("features", {}):
        logger.warning(
            "%s is configured as online data but has no denoise_step feature; "
            "online denoise_step filtering is disabled for this dataset.",
            root,
        )
        return np.ones(expected_length, dtype=bool)

    try:
        episodes_by_index = {int(ep["episode_index"]): ep for ep in _read_jsonl(root / "meta" / "episodes.jsonl")}
        episodes = selected_episodes
        if episodes is None:
            episodes = sorted(episodes_by_index)
        parts = []
        chunks_size = int(info.get("chunks_size", 1000))
        data_template = info["data_path"]
        for episode_index in episodes:
            episode_index = int(episode_index)
            episode = episodes_by_index[episode_index]
            episode_chunk = episode_index // chunks_size
            parquet_path = root / data_template.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            values = _read_parquet_column(parquet_path, "denoise_step")
            denoise_step = _denoise_column_to_array(values)
            expected_episode_length = int(episode["length"])
            if len(denoise_step) != expected_episode_length:
                raise ValueError(
                    f"{parquet_path} denoise_step length mismatch: "
                    f"expected {expected_episode_length}, got {len(denoise_step)}"
                )
            parts.append(denoise_step)
        if not parts:
            return np.ones(expected_length, dtype=bool)
        denoise_steps = np.concatenate(parts, axis=0).astype(np.float32)
    except Exception as exc:
        logger.warning("Could not build denoise_step filter for %s: %s", root, exc)
        return np.ones(expected_length, dtype=bool)

    if threshold is None:
        if len(denoise_steps) == 0:
            logger.warning(
                "%s is configured as online data but no max denoise step was found; "
                "online denoise_step filtering is disabled for this dataset.",
                root,
            )
            return np.ones(expected_length, dtype=bool)
        threshold = float(np.nanmax(denoise_steps))
        if not np.isfinite(threshold):
            logger.warning(
                "%s denoise_step values are not finite; online denoise_step filtering is disabled for this dataset.",
                root,
            )
            return np.ones(expected_length, dtype=bool)
        logger.warning(
            "%s has no configured max denoise step; using observed max denoise_step=%s "
            "as the online filter threshold.",
            root,
            threshold,
        )

    mask = (denoise_steps < threshold).astype(bool)

    if len(mask) != expected_length:
        logger.warning(
            "Denoise filter length mismatch for %s: expected %s, got %s; "
            "online denoise_step filtering is disabled for this dataset.",
            root,
            expected_length,
            len(mask),
        )
        return np.ones(expected_length, dtype=bool)
    return mask


def _resolve_online_denoise_step_max(root: Path, override: float | None) -> float | None:
    if override is not None:
        return _as_float(override)
    meta_path = root / "extras" / "dataset_meta_server.json"
    if not meta_path.exists():
        return None
    try:
        meta = _read_json(meta_path)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", meta_path, exc)
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_parquet_column(path: Path, column: str) -> list[Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[column])
    return table[column].to_pylist()


def _denoise_column_to_array(values: list[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim > 1:
        array = array.reshape(array.shape[0], -1)[:, 0]
    return array.reshape(-1)
