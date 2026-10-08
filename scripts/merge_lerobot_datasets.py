#!/usr/bin/env python3
"""Append one local LeRobot dataset after another with episode reindexing."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Base LeRobot dataset root.")
    parser.add_argument("--append", type=Path, required=True, help="LeRobot dataset root to append.")
    parser.add_argument("--output", type=Path, required=True, help="Output LeRobot dataset root. Must not exist.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_lerobot_root(path: Path) -> None:
    required = [
        path / "meta" / "info.json",
        path / "meta" / "episodes.jsonl",
        path / "meta" / "episodes_stats.jsonl",
        path / "meta" / "modality.json",
        path / "meta" / "embodiment.json",
        path / "meta" / "tasks.jsonl",
        path / "data",
        path / "videos",
        path / "extras",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset root is missing required paths: {missing}")


def require_preprocessed_extras(dataset: Path, episodes: list[dict[str, Any]]) -> None:
    missing: list[str] = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        extra_dir = dataset / "extras" / f"episode_{episode_index:06d}"
        for filename in ("states.npz", "ep_stats.json", "ep_meta.json", "model.xml.gz", "wrist_relative_action.npy"):
            path = extra_dir / filename
            if not path.is_file():
                missing.append(str(path))
    if missing:
        preview = "\n".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Dataset has incomplete preprocessed extras:\n{preview}{suffix}")


def assert_same_file(base: Path, append: Path, relative: str) -> None:
    base_payload = (base / relative).read_bytes()
    append_payload = (append / relative).read_bytes()
    if base_payload != append_payload:
        raise ValueError(f"Cannot merge datasets with different {relative}.")


def format_lerobot_path(pattern: str, episode_index: int, video_key: str | None, chunks_size: int) -> Path:
    episode_chunk = episode_index // chunks_size
    kwargs = {
        "episode_chunk": episode_chunk,
        "episode_index": episode_index,
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    return Path(pattern.format(**kwargs))


def video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]


def copy_video_sidecars(src_video: Path, dst_video: Path) -> None:
    """Copy per-episode video sidecar arrays such as trajectory overlay pixels."""
    for src_sidecar in sorted(src_video.parent.glob("*.npy")):
        dst_sidecar = dst_video.parent / src_sidecar.name
        shutil.copy2(src_sidecar, dst_sidecar)


def reindex_episode_stats(row: dict[str, Any], new_episode_index: int, index_offset: int) -> dict[str, Any]:
    row = json.loads(json.dumps(row))
    row["episode_index"] = new_episode_index

    stats = row.get("stats", {})
    if "episode_index" in stats:
        count = stats["episode_index"].get("count", [None])
        stats["episode_index"] = {
            "min": [new_episode_index],
            "max": [new_episode_index],
            "mean": [float(new_episode_index)],
            "std": [0.0],
            "count": count,
        }
    if "index" in stats:
        for key in ("min", "max", "mean"):
            stats["index"][key] = [value + index_offset for value in stats["index"][key]]
    return row


def reindex_parquet(src: Path, dst: Path, new_episode_index: int, index_offset: int) -> int:
    df = pd.read_parquet(src)
    if "episode_index" not in df.columns:
        raise ValueError(f"Missing episode_index column in {src}.")
    if "index" not in df.columns:
        raise ValueError(f"Missing index column in {src}.")
    df["episode_index"] = new_episode_index
    df["index"] = df["index"] + index_offset
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return len(df)


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict[str, Any]:
    frames = [pd.read_parquet(path) for path in tqdm(sorted(parquet_paths), desc="Reading parquet")]
    if not frames:
        raise ValueError("No parquet files found for stats calculation.")
    data = pd.concat(frames, axis=0)

    stats: dict[str, Any] = {}
    for column in data.columns:
        first = data[column].iloc[0]
        if isinstance(first, str):
            continue
        values = np.vstack([np.asarray(value, dtype=np.float32) for value in data[column]])
        stats[column] = {
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return stats


def update_extra_metadata(extra_dir: Path, new_episode_index: int) -> None:
    ep_stats_path = extra_dir / "ep_stats.json"
    if ep_stats_path.exists():
        ep_stats = read_json(ep_stats_path)
        if "episode_idx" in ep_stats:
            ep_stats["episode_idx"] = new_episode_index
        write_json(ep_stats_path, ep_stats)


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    append = args.append.resolve()
    output = args.output.resolve()

    require_lerobot_root(base)
    require_lerobot_root(append)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    for relative in ("meta/modality.json", "meta/embodiment.json", "meta/tasks.jsonl"):
        assert_same_file(base, append, relative)

    base_info = read_json(base / "meta" / "info.json")
    append_info = read_json(append / "meta" / "info.json")
    for key in ("codebase_version", "robot_type", "fps", "features", "data_path", "video_path", "chunks_size"):
        if base_info.get(key) != append_info.get(key):
            raise ValueError(f"Cannot merge datasets with different info.json field: {key}")

    base_episodes = read_jsonl(base / "meta" / "episodes.jsonl")
    append_episodes = read_jsonl(append / "meta" / "episodes.jsonl")
    base_episode_stats = read_jsonl(base / "meta" / "episodes_stats.jsonl")
    append_episode_stats = read_jsonl(append / "meta" / "episodes_stats.jsonl")
    if len(append_episodes) != len(append_episode_stats):
        raise ValueError("Append dataset episodes.jsonl and episodes_stats.jsonl lengths differ.")
    require_preprocessed_extras(base, base_episodes)
    require_preprocessed_extras(append, append_episodes)

    offset = max(row["episode_index"] for row in base_episodes) + 1
    index_offset = int(base_info["total_frames"])
    chunks_size = int(base_info["chunks_size"])

    shutil.copytree(base, output)

    appended_rows: list[dict[str, Any]] = []
    appended_stats_rows: list[dict[str, Any]] = []
    append_rows_by_episode = {row["episode_index"]: row for row in append_episodes}
    append_stats_by_episode = {row["episode_index"]: row for row in append_episode_stats}

    for old_episode_index in tqdm(sorted(append_rows_by_episode), desc="Appending episodes"):
        new_episode_index = old_episode_index + offset
        old_row = append_rows_by_episode[old_episode_index]
        new_row = json.loads(json.dumps(old_row))
        new_row["episode_index"] = new_episode_index
        appended_rows.append(new_row)
        appended_stats_rows.append(
            reindex_episode_stats(append_stats_by_episode[old_episode_index], new_episode_index, index_offset)
        )

        src_parquet = append / format_lerobot_path(
            append_info["data_path"], old_episode_index, None, int(append_info["chunks_size"])
        )
        dst_parquet = output / format_lerobot_path(base_info["data_path"], new_episode_index, None, chunks_size)
        reindex_parquet(src_parquet, dst_parquet, new_episode_index, index_offset)

        for key in video_keys(base_info):
            src_video = append / format_lerobot_path(
                append_info["video_path"], old_episode_index, key, int(append_info["chunks_size"])
            )
            dst_video = output / format_lerobot_path(base_info["video_path"], new_episode_index, key, chunks_size)
            if not src_video.is_file():
                raise FileNotFoundError(f"Missing source video: {src_video}")
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_video)
            copy_video_sidecars(src_video, dst_video)

        src_extra = append / "extras" / f"episode_{old_episode_index:06d}"
        dst_extra = output / "extras" / f"episode_{new_episode_index:06d}"
        if not src_extra.is_dir():
            raise FileNotFoundError(f"Missing source extras dir: {src_extra}")
        shutil.copytree(src_extra, dst_extra)
        update_extra_metadata(dst_extra, new_episode_index)

    all_episodes = base_episodes + appended_rows
    all_episode_stats = base_episode_stats + appended_stats_rows
    write_jsonl(output / "meta" / "episodes.jsonl", all_episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", all_episode_stats)

    merged_info = json.loads(json.dumps(base_info))
    merged_info["total_episodes"] = len(all_episodes)
    merged_info["total_frames"] = int(base_info["total_frames"]) + int(append_info["total_frames"])
    merged_info["total_videos"] = int(base_info["total_videos"]) + int(append_info["total_videos"])
    merged_info["total_chunks"] = math.ceil(merged_info["total_episodes"] / chunks_size)
    merged_info["splits"] = {"train": f"0:{merged_info['total_episodes']}"}
    write_json(output / "meta" / "info.json", merged_info)

    parquet_paths = sorted((output / "data").glob("chunk-*/episode_*.parquet"))
    write_json(output / "meta" / "stats.json", calculate_dataset_statistics(parquet_paths))

    print(f"Merged dataset written to: {output}")
    print(f"Episodes: {len(all_episodes)}")
    print(f"Frames: {merged_info['total_frames']}")


if __name__ == "__main__":
    main()
