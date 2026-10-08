#!/usr/bin/env python3
"""Merge LeRobot rollout datasets and rebuild contiguous episode/frame indices.

Usage:
    .venv/bin/python scripts/offline_dagger/merge_lerobot_rollouts.py \
        output/evo1/fastwam/20260804_110008 \
        output/evo1/fastwam/20260804_113601 \
        --output output/evo1/fastwam/20260804_110008_113601_merged \
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path, help="Input LeRobot dataset roots, in merge order.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output dataset root. It must not already exist unless --overwrite is set.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def require_lerobot_root(dataset: Path) -> None:
    required = [
        dataset / "meta" / "info.json",
        dataset / "meta" / "episodes.jsonl",
        dataset / "meta" / "episodes_stats.jsonl",
        dataset / "data",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{dataset} is missing required paths: {missing}")


def format_lerobot_path(pattern: str, episode_index: int, video_key: str | None, chunks_size: int) -> Path:
    values: dict[str, Any] = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return Path(pattern.format(**values))


def data_parquet_path(dataset: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    path = dataset / format_lerobot_path(str(info["data_path"]), episode_index, None, chunks_size)
    if path.is_file():
        return path
    episode_name = f"episode_{episode_index:06d}.parquet"
    candidates = sorted((dataset / "data").glob(f"chunk-*/{episode_name}"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find parquet for episode {episode_index} in {dataset}.")


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


def copy_episode_video_dir(src_video: Path, dst_video: Path, old_episode_index: int, new_episode_index: int) -> None:
    src_dir = src_video.parent
    dst_dir = dst_video.parent
    dst_dir.mkdir(parents=True, exist_ok=True)
    old_name = f"episode_{old_episode_index:06d}"
    new_name = f"episode_{new_episode_index:06d}"

    for src_item in src_dir.iterdir():
        dst_item = dst_dir / src_item.name.replace(old_name, new_name)
        if src_item.is_dir():
            shutil.copytree(src_item, dst_item)
        elif src_item.is_file():
            shutil.copy2(src_item, dst_item)


def copy_episode_images(dataset: Path, output: Path, old_episode_index: int, new_episode_index: int) -> None:
    images_root = dataset / "images"
    if not images_root.is_dir():
        return
    old_name = f"episode_{old_episode_index:06d}"
    new_name = f"episode_{new_episode_index:06d}"
    for src in images_root.rglob(f"*{old_name}*"):
        rel = src.relative_to(images_root)
        dst_rel = Path(*[part.replace(old_name, new_name) for part in rel.parts])
        dst = output / "images" / dst_rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


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


def update_episode_stats(row: dict[str, Any], new_episode_index: int, index_offset: int, length: int) -> dict[str, Any]:
    row = deep_copy_json(row)
    row["episode_index"] = new_episode_index
    stats = row.get("stats", {})
    if "episode_index" in stats:
        stats["episode_index"] = {
            "min": [new_episode_index],
            "max": [new_episode_index],
            "mean": [float(new_episode_index)],
            "std": [0.0],
            "count": [length],
        }
    if "index" in stats:
        stats["index"]["min"] = [index_offset]
        stats["index"]["max"] = [index_offset + max(length - 1, 0)]
        stats["index"]["mean"] = [index_offset + (length - 1) / 2.0 if length else float(index_offset)]
        stats["index"]["count"] = [length]
    return row


def copy_static_files(template: Path, output: Path) -> None:
    skip = {"data", "videos", "images", "extras", "meta", "results.json", "results_merged.csv", "norm_stats.json"}
    output.mkdir(parents=True, exist_ok=True)
    for item in template.iterdir():
        if item.name in skip:
            continue
        target = output / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)
    for name in ("data", "videos", "images", "extras", "meta"):
        (output / name).mkdir(parents=True, exist_ok=True)


def merge_tasks(datasets: list[Path], output: Path) -> None:
    seen = set()
    tasks: list[dict[str, Any]] = []
    for dataset in datasets:
        path = dataset / "meta" / "tasks.jsonl"
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            key = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(row)
    if tasks:
        write_jsonl(output / "meta" / "tasks.jsonl", tasks)


def calculate_norm_stats(output: Path) -> dict[str, Any] | None:
    parquet_paths = sorted((output / "data").glob("chunk-*/*.parquet"))
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


def write_results_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
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


def validate_compatible_infos(infos: list[dict[str, Any]]) -> None:
    first = infos[0]
    for idx, info in enumerate(infos[1:], start=2):
        for key in ("data_path", "video_path", "fps"):
            if first.get(key) != info.get(key):
                raise ValueError(f"Dataset {idx} has incompatible info.json {key}: {info.get(key)!r}")
        if first.get("features") != info.get("features"):
            raise ValueError(f"Dataset {idx} has incompatible info.json features.")


def merge_datasets(datasets: list[Path], output: Path, overwrite: bool = False) -> None:
    datasets = [dataset.expanduser().resolve() for dataset in datasets]
    output = output.expanduser().resolve()
    for dataset in datasets:
        require_lerobot_root(dataset)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    infos = [read_json(dataset / "meta" / "info.json") for dataset in datasets]
    validate_compatible_infos(infos)
    template_info = infos[0]
    chunks_size = int(template_info.get("chunks_size", 1000))
    keys = video_keys(template_info)

    copy_static_files(datasets[0], output)

    merged_episodes: list[dict[str, Any]] = []
    merged_stats: list[dict[str, Any]] = []
    merged_results: list[dict[str, Any]] = []
    total_frames = 0
    total_videos = 0

    for dataset, info in zip(datasets, infos):
        episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
        stats_rows = read_jsonl(dataset / "meta" / "episodes_stats.jsonl")
        stats_by_index = {int(row["episode_index"]): row for row in stats_rows}
        result_rows = []
        results_path = dataset / "results.json"
        if results_path.is_file():
            source_results = read_json(results_path).get("episodes", [])
            if isinstance(source_results, list):
                result_rows = source_results
        results_by_index = {int(row["episode_index"]): row for row in result_rows if "episode_index" in row}

        for row in sorted(episodes, key=lambda item: int(item["episode_index"])):
            old_episode_index = int(row["episode_index"])
            new_episode_index = len(merged_episodes)

            src_parquet = data_parquet_path(dataset, info, old_episode_index)
            dst_parquet = output / format_lerobot_path(
                str(template_info["data_path"]), new_episode_index, None, chunks_size
            )
            length = rewrite_parquet(src_parquet, dst_parquet, new_episode_index, total_frames)

            new_row = deep_copy_json(row)
            new_row["episode_index"] = new_episode_index
            new_row["length"] = length
            merged_episodes.append(new_row)
            merged_stats.append(update_episode_stats(stats_by_index[old_episode_index], new_episode_index, total_frames, length))

            for key in keys:
                src_video = find_episode_video_path(dataset, info, old_episode_index, key)
                if src_video is None:
                    continue
                dst_video = output / format_lerobot_path(
                    str(template_info["video_path"]), new_episode_index, key, chunks_size
                )
                copy_episode_video_dir(src_video, dst_video, old_episode_index, new_episode_index)
                total_videos += 1

            copy_episode_images(dataset, output, old_episode_index, new_episode_index)

            src_extra = dataset / "extras" / f"episode_{old_episode_index:06d}"
            if src_extra.is_dir():
                dst_extra = output / "extras" / f"episode_{new_episode_index:06d}"
                shutil.copytree(src_extra, dst_extra)
                update_extra_metadata(dst_extra, new_episode_index)

            if old_episode_index in results_by_index:
                result = deep_copy_json(results_by_index[old_episode_index])
                result["episode_index"] = new_episode_index
                merged_results.append(result)

            total_frames += length

    for item in datasets[0].glob("extras/*"):
        if item.is_file():
            shutil.copy2(item, output / "extras" / item.name)

    info = deep_copy_json(template_info)
    info["total_episodes"] = len(merged_episodes)
    info["total_frames"] = total_frames
    info["total_videos"] = total_videos
    info["total_chunks"] = math.ceil(len(merged_episodes) / chunks_size) if merged_episodes else 0
    info["splits"] = {"train": f"0:{len(merged_episodes)}"}

    write_jsonl(output / "meta" / "episodes.jsonl", merged_episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", merged_stats)
    write_json(output / "meta" / "info.json", info)
    merge_tasks(datasets, output)
    write_json(output / "results.json", {"episodes": merged_results})
    write_results_csv(output / "results_merged.csv", merged_results)

    norm_stats = calculate_norm_stats(output)
    if norm_stats is not None:
        write_json(output / "norm_stats.json", norm_stats)

    print(f"Wrote {output}")
    print(f"episodes={len(merged_episodes)} frames={total_frames} videos={total_videos}")


def main() -> None:
    args = parse_args()
    merge_datasets(args.datasets, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
