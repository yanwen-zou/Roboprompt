from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from openpi_client.runtime import subscriber as _subscriber

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import write_info
    from lerobot.common.datasets.video_utils import encode_video_frames
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    from lerobot.datasets.utils import write_info  # type: ignore
    from lerobot.datasets.video_utils import encode_video_frames  # type: ignore

import openpi.shared.normalize as _normalize
from scripts.utils.interactive_prompt import compose_prompt_overlay_frame


FPS_VIDEO_INFO = {
    "video.codec": "h264",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "has_audio": False,
}
PROMPT_OVERLAY_GAP_WIDTH = 8
DATA_PATH_PATTERN = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_PATTERN = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}/episode_{episode_index:06d}.mp4"
)


class FlexivLeRobotDataset(LeRobotDataset):
    """Small wrapper to keep encoded video settings aligned with existing real-robot datasets."""

    def encode_episode_videos(self, episode_index: int) -> dict:
        import shutil

        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(
                img_dir,
                video_path,
                self.fps,
                overwrite=True,
                vcodec="h264",
                pix_fmt="yuv420p",
                crf=23,
                g=None,
                fast_decode=0,
            )
            shutil.rmtree(img_dir)

        if len(self.meta.video_keys) > 0 and episode_index == 0:
            self.meta.update_video_info()
            write_info(self.meta.info, self.meta.root)
        return video_paths


class LeRobotRolloutRecorder(_subscriber.Subscriber):
    def __init__(
        self,
        *,
        output_dir: Path,
        fps: int,
        server_metadata: dict[str, Any] | None,
        default_task: str | None,
        render_height: int,
        render_width: int,
        steer: bool = False,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._fps = int(fps)
        self._server_metadata = server_metadata or {}
        self._default_task = default_task or "unspecified task"
        self._render_height = int(render_height)
        self._render_width = int(render_width)
        self._steer = bool(steer)

        self._dataset = self._create_dataset()
        self._task_to_id: dict[str, int] = {}
        self._task_name_to_id = {"task": 1}
        self._episode_idx = 0
        self._frame_idx = 0
        self._current_task = self._default_task
        self._episode_states: list[np.ndarray] = []
        self._episode_denoise_steps: list[float] = []
        self._episode_results: list[dict[str, Any]] = []

    def on_episode_start(self) -> None:
        self._frame_idx = 0
        self._current_task = self._default_task
        self._episode_states = []
        self._episode_denoise_steps = []

    def on_step(
        self,
        observation: dict,
        action: dict,
        *,
        prompt_overlay_payload: dict | None = None,
        prompt_overlay_frame: np.ndarray | None = None,
        prompt_overlay_denoise_step: float | None = None,
        denoise_step: float | None = None,
    ) -> None:
        task = str(observation.get("prompt") or self._default_task)
        self._current_task = task
        task_id = self._task_to_id.setdefault(task, len(self._task_to_id))

        state = np.asarray(observation["observation/state"], dtype=np.float64)
        left_image = self._to_hwc_uint8(observation["observation/image"])
        if denoise_step is None:
            denoise_step = prompt_overlay_denoise_step
        denoise_step_value = float(denoise_step) if denoise_step is not None else float("nan")
        frame = {
            "observation.images.robot0_eye_in_hand": self._to_hwc_uint8(observation["observation/wrist_image"]),
            "observation.images.robot0_agentview_left": left_image,
            "observation.state": state,
            "action": np.asarray(action["actions"], dtype=np.float64),
            "denoise_step": np.asarray([denoise_step_value], dtype=np.float32),
            "annotation.human.task_description": np.asarray([task_id], dtype=np.int64),
            "annotation.human.task_name": np.asarray([self._task_name_to_id["task"]], dtype=np.int64),
            "task": task,
        }
        if self._steer:
            if prompt_overlay_frame is None:
                prompt_overlay_frame = compose_prompt_overlay_frame(
                    left_image,
                    prompt_overlay_payload,
                    denoise_step=prompt_overlay_denoise_step,
                )
            frame["observation.images.robot0_agentview_left_prompt_overlay"] = self._to_hwc_uint8(prompt_overlay_frame)
        try:
            frame_without_task = {key: value for key, value in frame.items() if key != "task"}
            self._dataset.add_frame(frame_without_task, task=task)
        except TypeError as exc:
            if "task" not in str(exc):
                raise
            self._dataset.add_frame(frame)
        self._episode_states.append(state.copy())
        self._episode_denoise_steps.append(denoise_step_value)
        self._frame_idx += 1

    def on_episode_end(
        self,
        *,
        prompt_counts: dict[str, int] | None = None,
        outcome: str | None = None,
        end_reason: str | None = None,
        task_progress: str | None = None,
        pat: int | None = None,
    ) -> bool:
        if self._frame_idx == 0:
            logging.warning("Skipping empty episode %d.", self._episode_idx)
            self.discard_episode(reason=end_reason or "empty")
            return False

        self._dataset.save_episode()
        self._save_episode_extras(outcome=outcome, end_reason=end_reason, task_progress=task_progress, pat=pat)
        self._record_episode_result(
            prompt_counts=prompt_counts,
            outcome=outcome,
            end_reason=end_reason,
            task_progress=task_progress,
            pat=pat,
        )
        logging.info("Saved episode %03d to %s", self._episode_idx, self._output_dir)
        self._episode_idx += 1
        return True

    def discard_episode(self, *, reason: str | None = None) -> None:
        if self._frame_idx > 0:
            logging.info("Discarding episode %03d with %d frames. reason=%s", self._episode_idx, self._frame_idx, reason)
        if hasattr(self._dataset, "_wait_image_writer"):
            self._dataset._wait_image_writer()
        self._dataset.clear_episode_buffer()
        self._frame_idx = 0
        self._current_task = self._default_task
        self._episode_states = []
        self._episode_denoise_steps = []

    def finalize(self) -> None:
        self._save_server_metadata()
        self._save_results_json()
        self._compute_and_save_norm_stats()

    def _create_dataset(self) -> FlexivLeRobotDataset:
        features = {
            "observation.images.robot0_eye_in_hand": {
                "dtype": "video",
                "shape": (self._render_height, self._render_width, 3),
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": self._fps, **FPS_VIDEO_INFO},
            },
            "observation.images.robot0_agentview_left": {
                "dtype": "video",
                "shape": (self._render_height, self._render_width, 3),
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": self._fps, **FPS_VIDEO_INFO},
            },
            "observation.state": {"dtype": "float64", "shape": (8,)},
            "action": {"dtype": "float64", "shape": (7,)},
            "denoise_step": {"dtype": "float32", "shape": (1,)},
            "annotation.human.task_description": {"dtype": "int64", "shape": (1,)},
            "annotation.human.task_name": {"dtype": "int64", "shape": (1,)},
        }
        if self._steer:
            prompt_overlay_width = self._render_width + PROMPT_OVERLAY_GAP_WIDTH + self._render_height
            features["observation.images.robot0_agentview_left_prompt_overlay"] = {
                "dtype": "video",
                "shape": (self._render_height, prompt_overlay_width, 3),
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": self._fps, **FPS_VIDEO_INFO},
            }
        dataset = FlexivLeRobotDataset.create(
            repo_id="realworld/flexiv",
            root=self._output_dir,
            robot_type="flexiv",
            fps=self._fps,
            features=features,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        dataset.meta.info["data_path"] = DATA_PATH_PATTERN
        dataset.meta.info["video_path"] = VIDEO_PATH_PATTERN
        write_info(dataset.meta.info, dataset.meta.root)
        return dataset

    def _save_episode_extras(
        self,
        *,
        outcome: str | None,
        end_reason: str | None,
        task_progress: str | None,
        pat: int | None,
    ) -> None:
        extras_dir = self._output_dir / "extras" / f"episode_{self._episode_idx:06d}"
        extras_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            extras_dir / "states.npz",
            states=np.asarray(self._episode_states, dtype=np.float64),
            denoise_steps=np.asarray(self._episode_denoise_steps, dtype=np.float32),
        )
        with open(extras_dir / "ep_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "lang": self._current_task,
                    "num_steps": self._frame_idx,
                    "outcome": outcome,
                    "end_reason": end_reason,
                    "task_progress": task_progress,
                    "PAT": pat,
                    "server_metadata": self._server_metadata,
                },
                f,
                indent=4,
            )
            f.write("\n")

    def _save_server_metadata(self) -> None:
        extras_dir = self._output_dir / "extras"
        extras_dir.mkdir(parents=True, exist_ok=True)
        with open(extras_dir / "dataset_meta_server.json", "w", encoding="utf-8") as f:
            json.dump(self._server_metadata, f, indent=4)
            f.write("\n")

    def _record_episode_result(
        self,
        *,
        prompt_counts: dict[str, int] | None,
        outcome: str | None,
        end_reason: str | None,
        task_progress: str | None,
        pat: int | None,
    ) -> None:
        counts = dict(prompt_counts or {})
        self._episode_results.append(
            {
                "episode_index": int(self._episode_idx),
                "num_steps": int(self._frame_idx),
                "task": self._current_task,
                "outcome": outcome,
                "task_progress": task_progress,
                "end_reason": end_reason,
                "prompt_counts": {
                    "total": int(counts.get("total", 0)),
                    "img_overlay": int(counts.get("img_overlay", 0)),
                    "global_action": int(counts.get("global_action", 0)),
                    "local_action": int(counts.get("local_action", 0)),
                },
                "PAT": pat,
            }
        )
        self._save_results_json()

    def _save_results_json(self) -> None:
        with open(self._output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({"episodes": self._episode_results}, f, indent=2)
            f.write("\n")

    def _compute_and_save_norm_stats(self) -> None:
        parquet_paths = sorted((self._output_dir / "data").glob("chunk-*/episode_*.parquet"))
        if not parquet_paths:
            return

        state_stats = _normalize.RunningStats()
        action_stats = _normalize.RunningStats()
        for parquet_path in parquet_paths:
            try:
                import pandas as pd
            except ImportError:
                logging.warning("pandas is not installed; skip norm_stats.json generation.")
                return
            df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
            state_batch = np.stack(df["observation.state"].to_list()).astype(np.float32)
            action_batch = np.stack(df["action"].to_list()).astype(np.float32)
            state_stats.update(state_batch)
            action_stats.update(action_batch)

        _normalize.save(
            self._output_dir,
            {
                "state": state_stats.get_statistics(),
                "actions": action_stats.get_statistics(),
            },
        )

    @staticmethod
    def _to_hwc_uint8(image: Any) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {image.shape}.")
        if image.shape[0] == 3:
            return np.transpose(image, (1, 2, 0))
        return image
