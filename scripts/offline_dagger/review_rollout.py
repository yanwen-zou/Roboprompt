#!/usr/bin/env python3
"""Interactively review rollout episodes and keep only DAgger-valid ones.

Usage:
    python scripts/offline_dagger/review_rollout.py \
        robocasa_data_ckpt/real_robot_data/rollouts/lerobot-toaster-right-rollout1

Controls:
    p: mark the current episode as DAgger-valid and move to the next one
    q: mark the current episode as invalid and move to the next one
    r: replay the current episode
    Esc: abort without changing the dataset

After review, the script asks for confirmation before writing the output dataset.
Valid episodes are reindexed to contiguous episode_index values, parquet index
values are rebuilt to be frame-contiguous, and meta/*.jsonl plus info.json are
rewritten to stay aligned with the valid episodes. The source dataset is never
modified. By default, the output is written next to the source dataset as
``fastwam_dagger1``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


PREFERRED_VIDEO_KEYS = (
    "observation.images.robot0_agentview_left_prompt_overlay",
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
)
DEFAULT_MIN_TASK_PROGRESS = 97.0
DEFAULT_REPLAY_SPEED = 2.0


@dataclass(frozen=True)
class EpisodeDecision:
    episode_index: int
    keep: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="LeRobot rollout dataset root.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Replay FPS override. By default, uses dataset fps multiplied by --speed.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_REPLAY_SPEED,
        help=f"Replay speed multiplier when --fps is not set. Default: {DEFAULT_REPLAY_SPEED:g}.",
    )
    parser.add_argument(
        "--video-key",
        default=None,
        help="Video feature to replay. Defaults to prompt overlay, agentview, wrist, then first available video.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Write the output dataset without the final confirmation prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output dataset root. A bare directory name is created under <dataset parent>. "
            "Default: <dataset parent>/fastwam_dagger1."
        ),
    )
    parser.add_argument(
        "--min-task-progress",
        type=float,
        default=DEFAULT_MIN_TASK_PROGRESS,
        help=f"Only replay episodes whose task_progress is at least this value. Default: {DEFAULT_MIN_TASK_PROGRESS:g}.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_lerobot_root(dataset: Path) -> None:
    required = [
        dataset / "meta" / "info.json",
        dataset / "meta" / "episodes.jsonl",
        dataset / "meta" / "episodes_stats.jsonl",
        dataset / "data",
        dataset / "videos",
        dataset / "extras",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset root is missing required paths: {missing}")


def format_lerobot_path(pattern: str, episode_index: int, video_key: str | None, chunks_size: int) -> Path:
    values: dict[str, Any] = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return Path(pattern.format(**values))


def video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


def find_episode_video_path(dataset: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path | None:
    chunks_size = int(info.get("chunks_size", 1000))
    path = dataset / format_lerobot_path(str(info["video_path"]), episode_index, video_key, chunks_size)
    if path.is_file():
        return path

    episode_name = f"episode_{episode_index:06d}"
    candidates = sorted((dataset / "videos").glob(f"chunk-*/{video_key}/{episode_name}/{episode_name}.mp4"))
    candidates.extend(sorted((dataset / "videos").glob(f"chunk-*/{video_key}/{episode_name}.mp4")))
    return candidates[0] if candidates else None


def select_video_key(
    dataset: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    requested: str | None,
) -> str:
    keys = video_keys(info)
    if not keys:
        raise ValueError("Dataset info.json does not define any video features.")
    if requested is not None:
        if requested not in keys:
            raise ValueError(f"Requested --video-key {requested!r} is not in info.json video features: {keys}")
        missing = [
            int(row["episode_index"])
            for row in episodes
            if find_episode_video_path(dataset, info, int(row["episode_index"]), requested) is None
        ]
        if missing:
            raise FileNotFoundError(f"Requested --video-key {requested!r} is missing episodes: {missing}")
        return requested

    ordered_keys = [key for key in PREFERRED_VIDEO_KEYS if key in keys] + [
        key for key in keys if key not in PREFERRED_VIDEO_KEYS
    ]
    coverage: dict[str, int] = {}
    for key in ordered_keys:
        coverage[key] = sum(
            find_episode_video_path(dataset, info, int(row["episode_index"]), key) is not None for row in episodes
        )
    for key in ordered_keys:
        if coverage[key] == len(episodes):
            return key
    best_key = max(ordered_keys, key=lambda key: coverage[key])
    print(
        f"Warning: no video key is present for every episode; using {best_key!r} "
        f"with {coverage[best_key]}/{len(episodes)} episodes and falling back per episode."
    )
    return best_key


def episode_video_path(dataset: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    path = find_episode_video_path(dataset, info, episode_index, video_key)
    if path is not None:
        return path

    keys = video_keys(info)
    ordered_keys = [key for key in PREFERRED_VIDEO_KEYS if key in keys] + [
        key for key in keys if key not in PREFERRED_VIDEO_KEYS
    ]
    for fallback_key in ordered_keys:
        fallback = find_episode_video_path(dataset, info, episode_index, fallback_key)
        if fallback is not None:
            print(f"Warning: episode {episode_index} missing {video_key!r}; replaying {fallback_key!r}.")
            return fallback
    raise FileNotFoundError(f"Could not find any replay video for episode {episode_index}.")


def data_parquet_path(dataset: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    path = dataset / format_lerobot_path(str(info["data_path"]), episode_index, None, chunks_size)
    if path.is_file():
        return path
    episode_name = f"episode_{episode_index:06d}.parquet"
    candidates = sorted((dataset / "data").glob(f"chunk-*/{episode_name}"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find parquet for episode {episode_index}.")


def resolve_output_path(dataset: Path, output_arg: Path | None) -> Path:
    if output_arg is None:
        return dataset.parent / "fastwam_dagger1"

    output_arg = output_arg.expanduser()
    if output_arg.is_absolute():
        return output_arg.resolve()

    if len(output_arg.parts) == 1:
        return (dataset.parent / output_arg).resolve()

    return output_arg.resolve()


def parse_task_progress(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        progress = float(text)
    except ValueError:
        return None
    return progress if math.isfinite(progress) else None


def load_task_progress_by_episode(dataset: Path) -> dict[int, float]:
    results_path = dataset / "results.json"
    if results_path.is_file():
        results = read_json(results_path)
        source_episodes = results.get("episodes", [])
        if isinstance(source_episodes, list):
            progress_by_episode: dict[int, float] = {}
            for row in source_episodes:
                if not isinstance(row, dict) or "episode_index" not in row:
                    continue
                progress = parse_task_progress(row.get("task_progress"))
                if progress is not None:
                    progress_by_episode[int(row["episode_index"])] = progress
            return progress_by_episode

    csv_path = dataset / "results_merged.csv"
    if csv_path.is_file():
        rows = pd.read_csv(csv_path)
        if "Task Progress (TR) %" not in rows.columns:
            return {}
        progress_by_episode = {}
        for row_number, row in rows.iterrows():
            if "episode_index" in rows.columns and not pd.isna(row["episode_index"]):
                episode_index = int(row["episode_index"])
            elif "FastWAM Round1" in rows.columns and not pd.isna(row["FastWAM Round1"]):
                episode_index = int(row["FastWAM Round1"]) - 1
            else:
                episode_index = int(row_number)
            progress = parse_task_progress(row["Task Progress (TR) %"])
            if progress is not None:
                progress_by_episode[episode_index] = progress
        return progress_by_episode

    return {}


def draw_hud(frame: np.ndarray, text_lines: list[str]) -> np.ndarray:
    panel_width = 190
    line_height = 22
    rendered = np.full((frame.shape[0], frame.shape[1] + panel_width, 3), 238, dtype=np.uint8)
    rendered[:, : frame.shape[1]] = frame
    panel_x = frame.shape[1]
    cv2.rectangle(rendered, (panel_x, 0), (rendered.shape[1] - 1, rendered.shape[0] - 1), (245, 245, 245), -1)
    cv2.line(rendered, (panel_x, 0), (panel_x, rendered.shape[0] - 1), (190, 190, 190), 1)

    for index, text in enumerate(text_lines):
        cv2.putText(
            rendered,
            text[:28],
            (panel_x + 12, 24 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    return rendered


def replay_episode(
    video_path: Path,
    episode_index: int,
    review_position: int,
    review_count: int,
    length: int,
    fps: float,
    task_progress: float,
) -> str:
    window_name = "review_rollout"
    delay_ms = max(1, int(round(1000.0 / fps)))

    while True:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        final_frame: np.ndarray | None = None
        frame_index = 0
        decision: str | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            final_frame = frame.copy()
            rendered_frame = draw_hud(
                frame,
                [
                    f"episode {episode_index}",
                    f"review {review_position}/{review_count}",
                    f"progress {task_progress:.1f}%",
                    f"frames {length}",
                    "",
                    "p dagger valid",
                    "q invalid",
                    "r replay",
                    "Esc abort",
                ],
            )
            cv2.imshow(window_name, rendered_frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("p"), ord("P")):
                decision = "keep"
                break
            if key in (ord("q"), ord("Q")):
                decision = "delete"
                break
            if key in (ord("r"), ord("R")):
                decision = "replay"
                break
            if key == 27:
                decision = "abort"
                break
            frame_index += 1
        cap.release()

        if decision in {"keep", "delete", "abort"}:
            return decision
        if decision == "replay":
            continue
        if final_frame is None:
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        continue


def deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def update_episode_stats(row: dict[str, Any], new_episode_index: int, index_offset: int, length: int) -> dict[str, Any]:
    row = deep_copy_json(row)
    row["episode_index"] = new_episode_index
    stats = row.get("stats", {})

    if "episode_index" in stats:
        count = stats["episode_index"].get("count", [length])
        stats["episode_index"] = {
            "min": [new_episode_index],
            "max": [new_episode_index],
            "mean": [float(new_episode_index)],
            "std": [0.0],
            "count": count,
        }

    if "index" in stats:
        stats["index"]["min"] = [index_offset]
        stats["index"]["max"] = [index_offset + max(length - 1, 0)]
        stats["index"]["mean"] = [index_offset + (length - 1) / 2.0 if length > 0 else float(index_offset)]
        stats["index"].setdefault("count", [length])

    return row


def rewrite_parquet(src: Path, dst: Path, new_episode_index: int, index_offset: int) -> int:
    df = pd.read_parquet(src)
    if "episode_index" not in df.columns:
        raise ValueError(f"Missing episode_index column in {src}")
    if "index" not in df.columns:
        raise ValueError(f"Missing index column in {src}")
    length = len(df)
    df["episode_index"] = new_episode_index
    df["index"] = np.arange(index_offset, index_offset + length, dtype=np.int64)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return length


def copy_episode_video_dir(src_video: Path, dst_video: Path, old_episode_index: int, new_episode_index: int) -> None:
    src_dir = src_video.parent
    dst_dir = dst_video.parent
    dst_dir.mkdir(parents=True, exist_ok=True)
    old_name = f"episode_{old_episode_index:06d}"
    new_name = f"episode_{new_episode_index:06d}"

    for src_item in src_dir.iterdir():
        dst_name = src_item.name.replace(old_name, new_name)
        dst_item = dst_dir / dst_name
        if src_item.is_dir():
            shutil.copytree(src_item, dst_item)
        elif src_item.is_file():
            shutil.copy2(src_item, dst_item)


def update_extra_metadata(extra_dir: Path, new_episode_index: int) -> None:
    for name in ("ep_stats.json", "ep_meta.json"):
        path = extra_dir / name
        if not path.is_file():
            continue
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        changed = False
        for key in ("episode_idx", "episode_index"):
            if key in payload:
                payload[key] = new_episode_index
                changed = True
        if changed:
            write_json(path, payload)


def calculate_norm_stats(output: Path) -> dict[str, Any] | None:
    parquet_paths = sorted((output / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        return None

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for path in parquet_paths:
        df = pd.read_parquet(path)
        if "observation.state" in df:
            states.append(np.vstack([np.asarray(value, dtype=np.float32) for value in df["observation.state"]]))
        if "action" in df:
            actions.append(np.vstack([np.asarray(value, dtype=np.float32) for value in df["action"]]))

    def block(parts: list[np.ndarray]) -> dict[str, Any]:
        values = np.concatenate(parts, axis=0)
        return {
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    norm_stats: dict[str, Any] = {}
    if states:
        norm_stats["state"] = block(states)
    if actions:
        norm_stats["actions"] = block(actions)
    return {"norm_stats": norm_stats} if norm_stats else None


def write_filtered_results(dataset: Path, output: Path, keep_episode_indices: list[int]) -> None:
    results_path = dataset / "results.json"
    if not results_path.is_file():
        return
    results = read_json(results_path)
    source_episodes = results.get("episodes", [])
    if not isinstance(source_episodes, list):
        return
    by_index = {int(row["episode_index"]): row for row in source_episodes if "episode_index" in row}
    filtered: list[dict[str, Any]] = []
    for new_episode_index, old_episode_index in enumerate(keep_episode_indices):
        if old_episode_index not in by_index:
            continue
        row = deep_copy_json(by_index[old_episode_index])
        row["episode_index"] = new_episode_index
        filtered.append(row)
    write_json(output / "results.json", {"episodes": filtered})
    write_results_csv(output / "results_merged.csv", filtered)


def write_results_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "FastWAM Round1",
            "Task Progress (TR) %",
            "Steering Times (ST)",
            "Traj",
            "Local cmd",
            "global cmd",
            "Prompt Aligned Times(PAT)",
            "Outcome",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, ep in enumerate(episodes, start=1):
            counts = ep.get("prompt_counts") or {}
            progress = str(ep.get("task_progress", "")).strip()
            if progress.endswith("%"):
                progress = progress[:-1]
            writer.writerow(
                {
                    "FastWAM Round1": idx,
                    "Task Progress (TR) %": progress,
                    "Steering Times (ST)": counts.get("total", ""),
                    "Traj": counts.get("img_overlay", ""),
                    "Local cmd": counts.get("local_action", ""),
                    "global cmd": counts.get("global_action", ""),
                    "Prompt Aligned Times(PAT)": ep.get("PAT", ""),
                    "Outcome": ep.get("outcome", ""),
                }
            )


def copy_static_dataset_files(src: Path, dst: Path) -> None:
    skip = {"data", "videos", "extras", "meta"}
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in skip:
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    shutil.copytree(src / "meta", dst / "meta")


def copy_existing_video_if_present(
    dataset: Path,
    output: Path,
    info: dict[str, Any],
    old_episode_index: int,
    new_episode_index: int,
    video_key: str,
) -> bool:
    chunks_size = int(info.get("chunks_size", 1000))
    src_video = dataset / format_lerobot_path(str(info["video_path"]), old_episode_index, video_key, chunks_size)
    if not src_video.is_file():
        src_video = find_episode_video_path(dataset, info, old_episode_index, video_key)
        if src_video is None:
            return False
    dst_video = output / format_lerobot_path(str(info["video_path"]), new_episode_index, video_key, chunks_size)
    copy_episode_video_dir(src_video, dst_video, old_episode_index, new_episode_index)
    return True


def build_filtered_dataset(
    dataset: Path,
    output: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    keep_episode_indices: list[int],
) -> None:
    if output.exists():
        raise FileExistsError(f"Temporary output already exists: {output}")
    copy_static_dataset_files(dataset, output)

    for name in ("data", "videos", "extras"):
        path = output / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    episode_by_index = {int(row["episode_index"]): row for row in episodes}
    stats_by_index = {int(row["episode_index"]): row for row in episode_stats}
    keys = video_keys(info)
    chunks_size = int(info.get("chunks_size", 1000))

    new_episodes: list[dict[str, Any]] = []
    new_episode_stats: list[dict[str, Any]] = []
    total_frames = 0
    total_videos = 0

    for new_episode_index, old_episode_index in enumerate(keep_episode_indices):
        old_row = episode_by_index[old_episode_index]
        new_row = deep_copy_json(old_row)
        new_row["episode_index"] = new_episode_index

        src_parquet = data_parquet_path(dataset, info, old_episode_index)
        dst_parquet = output / format_lerobot_path(str(info["data_path"]), new_episode_index, None, chunks_size)
        length = rewrite_parquet(src_parquet, dst_parquet, new_episode_index, total_frames)
        new_row["length"] = length
        new_episodes.append(new_row)

        old_stats = stats_by_index[old_episode_index]
        new_episode_stats.append(update_episode_stats(old_stats, new_episode_index, total_frames, length))

        for key in keys:
            if copy_existing_video_if_present(dataset, output, info, old_episode_index, new_episode_index, key):
                total_videos += 1

        src_extra = dataset / "extras" / f"episode_{old_episode_index:06d}"
        dst_extra = output / "extras" / f"episode_{new_episode_index:06d}"
        if src_extra.is_dir():
            shutil.copytree(src_extra, dst_extra)
            update_extra_metadata(dst_extra, new_episode_index)

        total_frames += length

    new_info = deep_copy_json(info)
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = total_videos
    new_info["total_chunks"] = math.ceil(len(new_episodes) / chunks_size) if new_episodes else 0
    new_info["splits"] = {"train": f"0:{len(new_episodes)}"}

    write_jsonl(output / "meta" / "episodes.jsonl", new_episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", new_episode_stats)
    write_json(output / "meta" / "info.json", new_info)
    write_filtered_results(dataset, output, keep_episode_indices)
    norm_stats = calculate_norm_stats(output)
    if norm_stats is not None:
        write_json(output / "norm_stats.json", norm_stats)


def validate_metadata(episodes: list[dict[str, Any]], episode_stats: list[dict[str, Any]]) -> None:
    episode_indices = [int(row["episode_index"]) for row in episodes]
    stats_indices = [int(row["episode_index"]) for row in episode_stats]
    if episode_indices != stats_indices:
        raise ValueError("meta/episodes.jsonl and meta/episodes_stats.jsonl episode_index values are not aligned.")
    expected = list(range(len(episodes)))
    if sorted(episode_indices) != expected:
        raise ValueError(f"Episode indices must be contiguous 0..N-1 before review, got {episode_indices}")


def confirm_apply(delete_indices: list[int], keep_indices: list[int], yes: bool, output: Path) -> bool:
    if yes:
        return True
    text = ", ".join(str(index) for index in delete_indices)
    response = input(
        f"Write {len(keep_indices)} DAgger-valid episode(s) to {output} "
        f"and skip {len(delete_indices)} invalid episode(s) [{text}]? [y/N] "
    )
    return response.strip().lower() in {"y", "yes"}


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    require_lerobot_root(dataset)
    if args.fps is not None and args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.speed <= 0:
        raise ValueError(f"--speed must be positive, got {args.speed}")

    info = read_json(dataset / "meta" / "info.json")
    dataset_fps = float(info.get("fps", 30.0))
    if dataset_fps <= 0:
        raise ValueError(f"Dataset fps must be positive, got {dataset_fps}")
    replay_fps = float(args.fps) if args.fps is not None else dataset_fps * float(args.speed)
    replay_speed = replay_fps / dataset_fps
    episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
    episode_stats = read_jsonl(dataset / "meta" / "episodes_stats.jsonl")
    validate_metadata(episodes, episode_stats)

    progress_by_episode = load_task_progress_by_episode(dataset)
    review_episodes: list[dict[str, Any]] = []
    skipped_low_progress: list[int] = []
    skipped_missing_progress: list[int] = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        progress = progress_by_episode.get(episode_index)
        if progress is None:
            skipped_missing_progress.append(episode_index)
        elif progress >= args.min_task_progress:
            review_episodes.append(row)
        else:
            skipped_low_progress.append(episode_index)

    if not review_episodes:
        print(
            f"No episodes have task_progress >= {args.min_task_progress:g}. "
            "Dataset unchanged."
        )
        if skipped_missing_progress:
            print(f"Skipped episodes with missing task_progress: {skipped_missing_progress}")
        return

    chosen_video_key = select_video_key(dataset, info, review_episodes, args.video_key)
    print(f"Reviewing {len(review_episodes)}/{len(episodes)} episode(s) from: {dataset}")
    print(f"Minimum task_progress: {args.min_task_progress:g}")
    print(f"Replay video key: {chosen_video_key}")
    print(f"Replay speed: {replay_speed:g}x ({replay_fps:g} FPS)")
    if skipped_low_progress:
        print(f"Skipping {len(skipped_low_progress)} episode(s) below threshold: {skipped_low_progress}")
    if skipped_missing_progress:
        print(f"Skipping {len(skipped_missing_progress)} episode(s) without task_progress: {skipped_missing_progress}")
    print("Controls: p=dagger valid, q=invalid, r=replay, Esc=abort")

    decisions: list[EpisodeDecision] = []
    try:
        for review_position, row in enumerate(review_episodes, start=1):
            episode_index = int(row["episode_index"])
            video_path = episode_video_path(dataset, info, episode_index, chosen_video_key)
            decision = replay_episode(
                video_path,
                episode_index,
                review_position,
                len(review_episodes),
                int(row.get("length", 0)),
                replay_fps,
                progress_by_episode[episode_index],
            )
            if decision == "abort":
                print("Aborted. Dataset unchanged.")
                return
            keep = decision == "keep"
            decisions.append(EpisodeDecision(episode_index=episode_index, keep=keep))
            print(f"episode {episode_index}: {'dagger valid' if keep else 'invalid'}")
    finally:
        cv2.destroyAllWindows()

    delete_indices = [item.episode_index for item in decisions if not item.keep]
    keep_indices = [item.episode_index for item in decisions if item.keep]
    output = resolve_output_path(dataset, args.output)
    if not keep_indices:
        print("No episodes marked DAgger-valid. Dataset unchanged.")
        return
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not confirm_apply(delete_indices, keep_indices, bool(args.yes), output):
        print("Review not confirmed. Dataset unchanged.")
        return

    build_filtered_dataset(dataset, output, info, episodes, episode_stats, keep_indices)
    print(f"Wrote {len(keep_indices)} DAgger-valid episode(s): {output}")
    if delete_indices:
        print(f"Skipped {len(delete_indices)} invalid episode(s): {delete_indices}")


if __name__ == "__main__":
    main()
