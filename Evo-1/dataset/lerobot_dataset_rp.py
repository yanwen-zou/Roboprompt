import os
import torch
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm  
from typing import List, Union, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import multiprocessing as mp
import logging
import pickle
from dataset.utils import (
    build_visual_prompt_inputs,
    build_rp_prompt_fields,
    compute_gripper_tip_world_position,
    compute_lerobot_normalization_stats_from_minmax,
    get_left_image_index,
    get_env_camera_params,
    load_episode_target_pixels,
    merge_lerobot_stats,
    normalize_cmd_type,
    process_parquet_file_worker,
    replace_action_stats_with_state_delta,
)


class LeRobotDatasetRP(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        action_horizon: int,
        image_size: int = 448,
        max_samples_per_file: Union[int, None] = None,
        video_backend: str = "av",
        video_backend_kwargs: Dict[str, Any] = None,
        binarize_gripper: bool = False,
        cache_dir: Union[str, Path] = None,
        use_augmentation: bool = False,
    ):
        self.config = config

        sorted_datasets = sorted(self.config['data_groups'].keys())
        self.arm_to_embodiment_id = {key: i for i, key in enumerate(sorted_datasets)}

        self.max_action_dim = config['max_action_dim']
        self.max_state_dim = config['max_state_dim']
        self.max_views = config['max_views']
        self.cmd_type = normalize_cmd_type(config.get("cmd_type", "sigma"))

        self.image_size = image_size
        self.max_samples_per_file = max_samples_per_file
        self.binarize_gripper = binarize_gripper
        self.use_augmentation = use_augmentation

        if cache_dir is None:
            self.cache_dir = Path("/home/dell/code/lintao/Evo_1/training_data_cache/")
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data = []
        self.arm2stats_dict = {}
        self.arm2global_motion_stats = {}
        self.action_horizon = action_horizon
        self.prompt_task_dropout_prob = config.get("prompt_task_dropout_prob", 0.0)
        self.prompt_motion_noise_std = float(config.get("prompt_motion_noise_std", 0.005))
        self.video_dropout = float(config.get("video_dropout", config.get("prompt_video_dropout_prob", 0.3)))
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs or {}  

        if self.video_backend == "decord" and not self.video_backend_kwargs:
            self.video_backend_kwargs = {"ctx": "cpu"}  

        self._load_metadata()
        self._load_trajectories()

        self.basic_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

        self.aug_transform = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.95, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BICUBIC), 
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            T.ToTensor()
        ])

    def _discover_sub_datasets(self, dataset_path: Path) -> list[Path]:
        """如果 dataset_path 本身是 LeRobot 数据集根目录，返回 [dataset_path]；
        如果 dataset_path 下有 episode_* 子目录，返回这些子目录列表。"""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        if (dataset_path / "meta").is_dir() and (dataset_path / "data").is_dir():
            return [dataset_path]
        episode_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        if episode_dirs:
            return episode_dirs
        raise FileNotFoundError(f"No valid LeRobot dataset found under {dataset_path}")

    def _load_tasks(self, sub_path: Path) -> list[dict]:
        tasks_path = sub_path / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            return pd.read_json(tasks_path, lines=True).to_dict("records")
        tasks_path = sub_path / "meta" / "tasks.parquet"
        if tasks_path.exists():
            df = pd.read_parquet(tasks_path)
            df = df.reset_index().rename(columns={"index": "task"})
            return df.to_dict("records")
        raise FileNotFoundError(f"tasks file not found in {sub_path / 'meta'}")

    def _load_stats(self, sub_path: Path) -> dict:
        stats_path_after_compute = sub_path / "meta" / "stats.json"
        if stats_path_after_compute.exists():
            print(f"already have stats file: {stats_path_after_compute}")
            with open(stats_path_after_compute, "r") as f:
                return json.load(f)
        stats_path = sub_path / "meta" / "episodes_stats.jsonl"
        if stats_path.exists():
            stats = compute_lerobot_normalization_stats_from_minmax(stats_path)
            with open(stats_path_after_compute, "w") as f:
                json.dump(stats, f, indent=4)
            print(f"computed stats and saved to: {stats_path_after_compute}")
            return stats
        raise FileNotFoundError(f"normalization stats file not found in {sub_path / 'meta'}")

    def _load_episodes(self, sub_path: Path) -> None:
        episodes_path = sub_path / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            self.episodes += pd.read_json(episodes_path, lines=True).to_dict("records")

    def _load_metadata(self):
        self.episodes = []
        self.tasks = {}

        for arm_name, arm_config in self.config['data_groups'].items():
            print(f"  -- Processing arm group: '{arm_name}'")
            norm_arm_list = []
            self.tasks[arm_name] = {}
            for dataset_name, dataset_config in arm_config.items():
                print(f"    -- Processing dataset: '{dataset_name}'")
                print(f"    -- Dataset config: {dataset_config}")
                path_str = dataset_config['path']
                dataset_path = Path(path_str)
                sub_paths = self._discover_sub_datasets(dataset_path)

                all_tasks = []
                for sub_path in sub_paths:
                    sub_tasks = self._load_tasks(sub_path)
                    all_tasks.extend(sub_tasks)
                    self._load_episodes(sub_path)
                    stats = self._load_stats(sub_path)
                    if self.cmd_type == "state":
                        stats = replace_action_stats_with_state_delta(stats, sub_path)
                    norm_arm_list.append(stats)

                task_index_to_task = {
                    task_obj["task_index"]: task_obj["task"]
                    for task_obj in all_tasks
                    if "task_index" in task_obj and "task" in task_obj
                }
                self.tasks[arm_name][dataset_name] = task_index_to_task

            self.arm2stats_dict[arm_name] = merge_lerobot_stats(norm_arm_list)

    def _load_trajectories(self):
        parquet_process_units = []
        arm_horizon_mins = {arm_name: [] for arm_name in self.config["data_groups"]}
        arm_horizon_maxs = {arm_name: [] for arm_name in self.config["data_groups"]}
        for arm_name, arm_config in self.config['data_groups'].items():
            for dataset_name, dataset_config in arm_config.items():
                dataset_path = dataset_config.get('path', None)
                if dataset_path is None:
                    raise ValueError(f"Dataset path for '{arm_name}-{dataset_name}' is not configured, please check the config")
                dataset_path = Path(dataset_path)
                sub_paths = self._discover_sub_datasets(dataset_path)
                task_mapping = self.tasks[arm_name][dataset_name]
                for sub_path in sub_paths:
                    parquet_files = list(sub_path.glob("data/*/*.parquet"))
                    for parquet_path in parquet_files:
                        parquet_process_units.append((
                            parquet_path,
                            arm_name,
                            dataset_name,
                            dataset_config,
                            sub_path,
                            task_mapping,
                            self.action_horizon,
                            self.max_samples_per_file,
                            self.cache_dir,
                            self.cmd_type,
                        ))

        print(f"total {len(parquet_process_units)} parquet files to process")
        num_processes = min(16, len(parquet_process_units))
        print(f"Using {num_processes} processes for concurrent processing")

        with mp.Pool(processes=num_processes) as pool:
            total_episodes = 0
            with tqdm(total=len(parquet_process_units), desc="Processing Parquet files to cache") as pbar:
                for episode_files, error, horizon_sum_stats in pool.imap_unordered(process_parquet_file_worker, parquet_process_units):
                    if error:
                        logging.error(error)
                    else:
                        self.data.extend(episode_files)
                        total_episodes += len(episode_files)
                        stats_arm_name = horizon_sum_stats["arm_name"]
                        arm_horizon_mins[stats_arm_name].append(horizon_sum_stats["min"])
                        arm_horizon_maxs[stats_arm_name].append(horizon_sum_stats["max"])
                    pbar.set_postfix({
                        'episodes_this_file': len(episode_files),
                        'total_episodes': total_episodes
                    })
                    pbar.update(1)

        for arm_name in self.config["data_groups"]:
            if not arm_horizon_mins[arm_name]:
                raise ValueError(f"No horizon-sum action stats generated for arm group {arm_name!r}")
            self.arm2global_motion_stats[arm_name] = {
                "min": np.min(np.stack(arm_horizon_mins[arm_name]), axis=0).astype(np.float32),
                "max": np.max(np.stack(arm_horizon_maxs[arm_name]), axis=0).astype(np.float32),
            }
            motion_stats = self.arm2global_motion_stats[arm_name]
            print(
                f"Global motion horizon-sum stats for {arm_name} ({self.cmd_type} cmd): "
                f"min={motion_stats['min']}, max={motion_stats['max']}"
            )

        print(f"Data processing completed, total {len(self.data)} files generated")


    def _pad_tensor(
        self, 
        source_tensor: torch.Tensor, 
        max_dim: int
    ) -> (torch.Tensor, torch.Tensor):

        source_dim = source_tensor.shape[-1]
        
        if source_tensor.dim() > 1:
            padded_shape = (*source_tensor.shape[:-1], max_dim)
        else:
            padded_shape = (max_dim,)

        padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype, device=source_tensor.device)
        mask = torch.zeros(padded_shape, dtype=torch.bool, device=source_tensor.device)

        data_slice = (..., slice(0, source_dim))
        
        padded_tensor[data_slice] = source_tensor
        mask[data_slice] = True
            
        return padded_tensor, mask


    def _load_video_frame(self, video_paths: dict, timestamp: float) -> List[Image.Image]:
    
        frames = []
        for view, path in video_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"video file not found: {path}")
            
            if self.video_backend == "decord":
                import decord

                try:
                    ctx = self.video_backend_kwargs.get("ctx", "cpu")
                    if ctx == "cpu":
                        ctx = decord.cpu(0)
                    elif ctx == "gpu":
                        ctx = decord.gpu(0)
                    logging.info(f"Using video backend {self.video_backend}, context: {ctx}")
                    vr = decord.VideoReader(path, ctx=ctx)
                    logging.info(f"Successfully opened video file: {path}")
                    fps = vr.get_avg_fps()
                    logging.info(f"Video {path} FPS: {fps}")
                    if fps is None or np.isnan(fps):
                        raise ValueError(f"Unable to read FPS, video may be corrupted: {path}")

                    frame_idx = int(timestamp * fps)
                    logging.info(f"Reading video {path} frame index: {frame_idx} (timestamp: {timestamp}, fps: {fps})")
                    if frame_idx >= len(vr):
                        logging.info(f"the requested frame index exceeds video length: frame_idx={frame_idx}, len={len(vr)}. Using last frame instead.")
                        
                        frame_idx = len(vr) - 1

                    frame = vr[frame_idx].asnumpy()
                    frames.append(Image.fromarray(frame))
                    logging.info(f"Successfully read video frame: {path}, frame index: {frame_idx}")

                except Exception as e:
                    logging.info(f"Failed to read video file: {path}")
                    logging.info(f"Error message: {str(e)}")
                    raise

            elif self.video_backend == "av":
                import av
                try:
                    with av.open(path) as container:
                        for frame in container.decode(video=0):
                            if frame.time >= timestamp:
                                frames.append(Image.fromarray(frame.to_ndarray(format='rgb24')))
                                break

                except Exception as e:
                    print(f"Failed to read video file: {path}")
                    print(f"Error message: {str(e)}")
                    raise
            else:
                raise NotImplementedError(f"Video backend {self.video_backend} not implemented")
        
        return frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cache_filepath = self.data[idx]
        
        try:
            with open(cache_filepath, 'rb') as f:
                item = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Cannot load cache file {cache_filepath}: {e}") from e
 
        
        arm_key = item["arm_key"]
        dataset_key = item["dataset_key"]
        embodiment_id = self.arm_to_embodiment_id[arm_key]

 
        frames = self._load_video_frame(item["video_paths"], item["timestamp"])

        images = frames

        if item["state"] is None:
            raise ValueError("missing observation.state, please check data integrity")

        try:
            norm_stats = self.arm2stats_dict[arm_key]
        except KeyError:
            raise KeyError(f"Normalization stats not found for arm_key={arm_key} and dataset_key={dataset_key}")

        state = torch.tensor(item["state"], dtype=torch.float32)
        device = state.device
        state_min = torch.tensor(norm_stats["observation.state"]["min"], dtype=torch.float32, device=device)
        state_max = torch.tensor(norm_stats["observation.state"]["max"], dtype=torch.float32, device=device)
        state = 2 * (state - state_min) / (state_max - state_min + 1e-8) - 1
        state = torch.clamp(state, -1.0, 1.0)
        state_padded, state_mask = self._pad_tensor(state, self.max_state_dim)

        if item["action"] is None:
            raise ValueError("missing action, please check data integrity")

        raw_action = np.stack(item["action"]).astype(np.float32)
        action = torch.from_numpy(raw_action).float()
        device = action.device
        action_min = torch.tensor(norm_stats["action"]["min"], dtype=torch.float32, device=device)
        action_max = torch.tensor(norm_stats["action"]["max"], dtype=torch.float32, device=device)
        global_motion_stats = self.arm2global_motion_stats[arm_key]
        global_motion_min = np.asarray(global_motion_stats["min"], dtype=np.float32)
        global_motion_max = np.asarray(global_motion_stats["max"], dtype=np.float32)
        action = 2 * (action - action_min.unsqueeze(0)) / (action_max.unsqueeze(0) - action_min.unsqueeze(0) + 1e-8) - 1
        action = torch.clamp(action, -1.0, 1.0)
        action_padded, action_mask = self._pad_tensor(action, self.max_action_dim)

        # ------------------------------------------------------------------
        # Decide prompt strategy first (visual / global / local)
        # ------------------------------------------------------------------
        prompt_fields = build_rp_prompt_fields(
            item["prompt"] if item["prompt"] is not None else "",
            item=item,
            actions=raw_action,
            action_mask=action_mask.cpu().numpy(),
            horizon=self.action_horizon,
            global_motion_min=global_motion_min,
            global_motion_max=global_motion_max,
            task_dropout_prob=self.prompt_task_dropout_prob,
            prompt_motion_noise_std=self.prompt_motion_noise_std,
        )

        # ------------------------------------------------------------------
        # Build visual prompt as a separate image input.
        # ------------------------------------------------------------------
        visual_prompt_info = {
            "visual_prompt_applied": np.bool_(False),
            "visual_prompt_type": "none",
            "visual_prompt_traj_applied": np.bool_(False),
            "visual_prompt_point_applied": np.bool_(False),
            "visual_prompt_frame_start": np.int32(-1),
            "visual_prompt_frame_end": np.int32(-1),
            "prompt_point": np.zeros((2,), dtype=np.float32),
        }
        if len(images) > 0:
            left_image_index = min(get_left_image_index(item.get("video_paths", {})), len(images) - 1)
            base_image = images[left_image_index]
            images = [base_image]
            overlay_base_image, visual_prompt_image, visual_prompt_info = build_visual_prompt_inputs(
                base_image,
                item,
                self.action_horizon,
            )
            images[0] = overlay_base_image
            if visual_prompt_image is not None:
                images.append(visual_prompt_image)

        has_conditioning_prompt = (
            len(images) > 1
            or bool(prompt_fields["prompt_global_motion_mask"])
            or bool(prompt_fields["prompt_local_motion_mask"])
        )
        video_dropped = len(images) > 0 and has_conditioning_prompt and random.random() < self.video_dropout
        if video_dropped:
            images[0] = Image.new("RGB", images[0].size, (0, 0, 0))

        prompt_traj_pixels = np.zeros((self.action_horizon, 2), dtype=np.float32)
        prompt_traj_mask = np.zeros((self.action_horizon,), dtype=np.bool_)
        prompt_point = np.zeros((2,), dtype=np.float32)
        prompt_point_mask = np.bool_(False)
        frame_start = int(visual_prompt_info.get("visual_prompt_frame_start", -1))
        if bool(visual_prompt_info.get("visual_prompt_traj_applied", False)):
            target_pixels = load_episode_target_pixels(Path(item["dataset_path"]), int(item["trajectory_id"]))
            frame_end = min(len(target_pixels), frame_start + self.action_horizon)
            if frame_start >= 0 and frame_end > frame_start:
                traj = np.asarray(target_pixels[frame_start:frame_end], dtype=np.float32)
                count = min(len(traj), self.action_horizon)
                if count > 0:
                    prompt_traj_pixels[:count] = traj[:count]
                    prompt_traj_mask[:count] = True
        if bool(visual_prompt_info.get("visual_prompt_point_applied", False)):
            point = np.asarray(visual_prompt_info.get("prompt_point", prompt_point), dtype=np.float32)
            if point.shape == (2,) and np.isfinite(point).all():
                prompt_point = point
                prompt_point_mask = np.bool_(True)

        dataset_config = self.config["data_groups"][arm_key][dataset_key]
        prompt_base_t_camera, prompt_camera_intrinsic, prompt_camera_valid = get_env_camera_params(dataset_config)
        if not prompt_camera_valid:
            prompt_traj_mask[:] = False
            prompt_point_mask = np.bool_(False)
        prompt_tcp_world_pos = compute_gripper_tip_world_position(item["state"])

        if self.use_augmentation:
            images = [
                self.aug_transform(img) if random.random() < 0.5 else self.basic_transform(img)
                for img in images
            ]
        else:
            images = [self.basic_transform(img) for img in images]

        images = images[: self.max_views]
        num_real_views = len(images)
        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:num_real_views] = True
        if video_dropped:
            image_mask[0] = False

        while len(images) < self.max_views:
            if len(images) == 0:
                dummy_image = torch.zeros(3, self.image_size, self.image_size)
                logging.info("Warning: Image list is empty, using zero tensor for padding")
            else:
                dummy_image = torch.zeros_like(images[0])
            images.append(dummy_image)

        images = torch.stack(images)

        return {
            "images": images,
            "image_mask": image_mask,
            "prompt": prompt_fields["prompt"],
            "prompt_global_motion": torch.tensor(prompt_fields["prompt_global_motion"], dtype=torch.float32),
            "prompt_global_motion_axis_mask": torch.tensor(
                prompt_fields["prompt_global_motion_axis_mask"], dtype=torch.bool
            ),
            "prompt_global_motion_mask": torch.tensor(prompt_fields["prompt_global_motion_mask"], dtype=torch.bool),
            "prompt_local_motion": torch.tensor(prompt_fields["prompt_local_motion"], dtype=torch.float32),
            "prompt_local_motion_mask": torch.tensor(prompt_fields["prompt_local_motion_mask"], dtype=torch.bool),
            "visual_prompt_applied": torch.tensor(visual_prompt_info["visual_prompt_applied"], dtype=torch.bool),
            "video_dropped": torch.tensor(video_dropped, dtype=torch.bool),
            "visual_prompt_type": visual_prompt_info["visual_prompt_type"],
            "visual_prompt_frame_start": torch.tensor(visual_prompt_info["visual_prompt_frame_start"], dtype=torch.int32),
            "visual_prompt_frame_end": torch.tensor(visual_prompt_info["visual_prompt_frame_end"], dtype=torch.int32),
            "prompt_traj_pixels": torch.tensor(prompt_traj_pixels, dtype=torch.float32),
            "prompt_traj_mask": torch.tensor(prompt_traj_mask, dtype=torch.bool),
            "prompt_point": torch.tensor(prompt_point, dtype=torch.float32),
            "prompt_point_mask": torch.tensor(prompt_point_mask, dtype=torch.bool),
            "prompt_tcp_world_pos": torch.tensor(prompt_tcp_world_pos, dtype=torch.float32),
            "prompt_base_t_camera": torch.tensor(prompt_base_t_camera, dtype=torch.float32),
            "prompt_camera_intrinsic": torch.tensor(prompt_camera_intrinsic, dtype=torch.float32),
            "state": state_padded.to(dtype=torch.bfloat16),
            "state_mask": state_mask,
            "action": action_padded.to(dtype=torch.bfloat16),
            "action_mask": action_mask,
            "action_min": action_min.to(dtype=torch.bfloat16),
            "action_max": action_max.to(dtype=torch.bfloat16),
            "global_motion_min": torch.tensor(global_motion_min, dtype=torch.float32),
            "global_motion_max": torch.tensor(global_motion_max, dtype=torch.float32),
            "embodiment_id": torch.tensor(embodiment_id, dtype=torch.long)
        }
