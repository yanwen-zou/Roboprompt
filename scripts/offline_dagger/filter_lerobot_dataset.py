#!/usr/bin/env python3
"""Interactively filter a LeRobot dataset without task-progress checks.

Usage:
    python scripts/offline_dagger/filter_lerobot_dataset.py real_robot_data/active_teapot/lerobot

Controls:
    p: keep the current episode and move to the next one
    q: delete the current episode and move to the next one
    r: replay the current episode
    Esc: abort without writing anything

The source dataset is never modified. Kept episodes are written to a new
dataset with contiguous episode_index values and rebuilt frame indices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from review_rollout import (
    build_filtered_dataset,
    draw_hud,
    episode_video_path,
    read_json,
    read_jsonl,
    require_lerobot_root,
    resolve_output_path,
    select_video_key,
    validate_metadata,
)


DEFAULT_REPLAY_SPEED = 2.0


@dataclass(frozen=True)
class EpisodeDecision:
    episode_index: int
    keep: bool


def parse_episode_indices(value: str) -> set[int]:
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root.")
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
        "--episodes",
        default=None,
        help="Optional comma/range list to review, for example '0,3,8-12'. Defaults to all episodes.",
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
            "Default: <dataset parent>/filtered."
        ),
    )
    return parser.parse_args()


def replay_episode(
    video_path: Path,
    episode_index: int,
    review_position: int,
    review_count: int,
    length: int,
    fps: float,
) -> str:
    window_name = "filter_lerobot_dataset"
    delay_ms = max(1, int(round(1000.0 / fps)))

    while True:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        final_frame: np.ndarray | None = None
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
                    f"frames {length}",
                    "",
                    "p keep",
                    "q delete",
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
        cap.release()

        if decision in {"keep", "delete", "abort"}:
            return decision
        if decision == "replay":
            continue
        if final_frame is None:
            raise RuntimeError(f"Video has no readable frames: {video_path}")


def default_output_path(dataset: Path, output_arg: Path | None) -> Path:
    if output_arg is None:
        return dataset.parent / "filtered"
    return resolve_output_path(dataset, output_arg)


def confirm_apply(delete_indices: list[int], keep_indices: list[int], yes: bool, output: Path) -> bool:
    if yes:
        return True
    delete_text = ", ".join(str(index) for index in delete_indices)
    response = input(
        f"Write {len(keep_indices)} kept episode(s) to {output} "
        f"and skip {len(delete_indices)} deleted episode(s) [{delete_text}]? [y/N] "
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

    if args.episodes:
        requested = parse_episode_indices(args.episodes)
        review_episodes = [row for row in episodes if int(row["episode_index"]) in requested]
        found = {int(row["episode_index"]) for row in review_episodes}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested episode(s) not found: {missing}")
    else:
        review_episodes = episodes

    if not review_episodes:
        print("No episodes to review. Dataset unchanged.")
        return

    chosen_video_key = select_video_key(dataset, info, review_episodes, args.video_key)
    print(f"Reviewing {len(review_episodes)}/{len(episodes)} episode(s) from: {dataset}")
    print(f"Replay video key: {chosen_video_key}")
    print(f"Replay speed: {replay_speed:g}x ({replay_fps:g} FPS)")
    print("Controls: p=keep, q=delete, r=replay, Esc=abort")

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
            )
            if decision == "abort":
                print("Aborted. Dataset unchanged.")
                return
            keep = decision == "keep"
            decisions.append(EpisodeDecision(episode_index=episode_index, keep=keep))
            print(f"episode {episode_index}: {'keep' if keep else 'delete'}")
    finally:
        cv2.destroyAllWindows()

    reviewed = {item.episode_index for item in decisions}
    keep_indices = [item.episode_index for item in decisions if item.keep]
    keep_indices.extend(int(row["episode_index"]) for row in episodes if int(row["episode_index"]) not in reviewed)
    keep_indices = sorted(keep_indices)
    delete_indices = [item.episode_index for item in decisions if not item.keep]

    if not keep_indices:
        print("No episodes marked keep. Dataset unchanged.")
        return

    output = default_output_path(dataset, args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not confirm_apply(delete_indices, keep_indices, bool(args.yes), output):
        print("Review not confirmed. Dataset unchanged.")
        return

    build_filtered_dataset(dataset, output, info, episodes, episode_stats, keep_indices)
    print(f"Wrote {len(keep_indices)} kept episode(s): {output}")
    if delete_indices:
        print(f"Skipped {len(delete_indices)} deleted episode(s): {delete_indices}")


if __name__ == "__main__":
    main()
