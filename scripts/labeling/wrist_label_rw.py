#!/usr/bin/env python3
"""
Read real-world Flexiv episodes from a standard LeRobot dataset directory,
convert base-frame actions into the TCP local frame, save raw wrist-local
position deltas, normalize per-step wrist-local position deltas globally using
1st/99th quantiles, and write wrist_relative_action.npy.

The saved wrist_relative_action.npy is 3-D and already normalized to [-1, 1]
so that the dataloader can use it directly as prompt_local_motion without
re-normalizing.

The saved wrist_relative_action_raw.npy is 3-D raw wrist-local xyz. Dataloaders
that need an action-horizon prompt should sum this raw file over the horizon
and then normalize the summed vector.

Data format (real robot / Flexiv):
- observation.state: [tcp_x, tcp_y, tcp_z, tcp_qx, tcp_qy, tcp_qz, tcp_qw, gripper]
- action: [delta_pos(3), delta_rotvec(3), gripper]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOTS = [
    REPO_ROOT,
]
for path in reversed(PACKAGE_ROOTS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.utils.realworld_projection import quat_xyzw_to_mat

STATE_TCP_POSITION = slice(0, 3)
STATE_TCP_QUAT_XYZW = slice(3, 7)
ACTION_POSITION = slice(0, 3)
ACTION_ROTATION = slice(3, 6)
ACTION_GRIPPER = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert base-frame actions to TCP-frame relative actions for real-world episodes, "
            "save raw wrist-local position deltas, normalize per-step wrist-local position "
            "deltas with global q01/q99, and save wrist_relative_action.npy "
            "(3-D, already normalized to [-1, 1])."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to a standard LeRobot dataset root (contains data/, videos/, meta/, extras/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing wrist_relative_action.npy files.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def discover_episodes(dataset_root: Path) -> list[Path]:
    """Return list of parquet paths for every episode."""
    paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise ValueError(f"No episode parquet files found in {dataset_root / 'data'}")
    return paths


def load_episode_data(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
    states = np.stack(df["observation.state"].to_list()).astype(np.float64)
    actions = np.stack(df["action"].to_list()).astype(np.float64)
    return states, actions


def compute_tcp_relative_actions(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Convert base-frame action deltas into the current TCP local frame (7-D)."""
    frame_count = min(len(states), len(actions))
    if frame_count == 0:
        return np.zeros((0, 7), dtype=np.float64)

    states = np.asarray(states[:frame_count], dtype=np.float64)
    actions = np.asarray(actions[:frame_count], dtype=np.float64)

    local_actions = np.zeros((frame_count, 7), dtype=np.float64)

    for i in range(frame_count):
        tcp_rot = quat_xyzw_to_mat(states[i, STATE_TCP_QUAT_XYZW])

        # Position delta: rotate from base frame into TCP frame
        local_actions[i, :3] = tcp_rot.T @ actions[i, ACTION_POSITION]

        # Rotation delta (rotvec): rotate the axis from base frame into TCP frame
        local_actions[i, 3:6] = tcp_rot.T @ actions[i, ACTION_ROTATION]

        # Gripper stays unchanged
        local_actions[i, 6] = actions[i, ACTION_GRIPPER]

    return local_actions


def get_output_path(parquet_path: Path) -> Path:
    """Return output path for wrist_relative_action.npy.

    Standard layout: dataset_root/extras/episode_XXXXXX/wrist_relative_action.npy
    """
    dataset_root = parquet_path.parents[2]  # data/chunk-000/episode_XXXXXX.parquet -> dataset_root
    ep_name = parquet_path.stem
    output_dir = dataset_root / "extras" / ep_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "wrist_relative_action.npy"


def get_raw_output_path(parquet_path: Path) -> Path:
    output_path = get_output_path(parquet_path)
    return output_path.with_name("wrist_relative_action_raw.npy")


def compute_global_quantiles(parquet_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Compute global q01/q99 over wrist-local position (first 3 dims)."""
    all_local_actions: list[np.ndarray] = []
    for pp in parquet_paths:
        try:
            states, actions = load_episode_data(pp)
        except (FileNotFoundError, ValueError):
            continue
        local_actions = compute_tcp_relative_actions(states, actions)
        if local_actions.shape[0] > 0:
            all_local_actions.append(local_actions)
    if not all_local_actions:
        raise ValueError("No valid episodes found to compute global quantiles.")
    concat = np.concatenate(all_local_actions, axis=0)
    wrist_positions = concat[:, :3]
    q01 = np.quantile(wrist_positions, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(wrist_positions, 0.99, axis=0).astype(np.float32)
    return q01, q99


def process_episode(parquet_path: Path, q01: np.ndarray, q99: np.ndarray, overwrite: bool, verbose: bool) -> bool:
    """Process a single episode. Returns True if written/skipped successfully."""
    output_path = get_output_path(parquet_path)
    raw_output_path = get_raw_output_path(parquet_path)
    if output_path.exists() and raw_output_path.exists() and not overwrite:
        if verbose:
            print(f"Skipping {output_path} and {raw_output_path} (already exist)")
        return True

    states, actions = load_episode_data(parquet_path)
    local_actions = compute_tcp_relative_actions(states, actions)
    raw_xyz = local_actions[:, :3].astype(np.float32)
    denom = np.maximum(q99 - q01, 1e-6)
    normalized_xyz = (raw_xyz - q01) / denom * 2.0 - 1.0
    normalized_xyz = np.clip(normalized_xyz, -1.0, 1.0).astype(np.float32)
    np.save(raw_output_path, raw_xyz)
    np.save(output_path, normalized_xyz)
    if verbose:
        print(f"Saved {output_path} and {raw_output_path}  frames={len(normalized_xyz)}")
    return True


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset_dir).resolve()
    target_parquets = discover_episodes(dataset)

    if args.verbose:
        print(f"Dataset root: {dataset}")
        print(f"Found {len(target_parquets)} episodes")

    # ------------------------------------------------------------------
    # Compute global q01/q99 over the wrist-local position (first 3 dims)
    # ------------------------------------------------------------------
    q01, q99 = compute_global_quantiles(target_parquets)

    if args.verbose:
        print(f"Global wrist-local position statistics:")
        print(f"  q01: {q01}")
        print(f"  q99: {q99}")

    # ------------------------------------------------------------------
    # Process all episodes
    # ------------------------------------------------------------------
    for pp in target_parquets:
        process_episode(pp, q01, q99, args.overwrite, args.verbose)


if __name__ == "__main__":
    main()
