#!/usr/bin/env python3
"""Visualize LeRobot TCP/action trajectories in 3D.

Examples:
    python scripts/realworld/eval/visualize_lerobot_3d_trajectory.py \
        real_robot_data/active_teapot/lerobot --episodes 61,67,74

    python scripts/realworld/eval/visualize_lerobot_3d_trajectory.py \
        real_robot_data/active_teapot/lerobot --episodes 61 --mode both --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


def parse_episode_indices(value: str) -> list[int]:
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
    return sorted(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root containing meta/ and data/.")
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma/range list, for example '0,3,8-12'. Defaults to all episodes.",
    )
    parser.add_argument(
        "--mode",
        choices=("tcp", "action", "both"),
        default="tcp",
        help="tcp plots observation.state[:3]. action plots state[0,:3] + cumsum(action[:,:3]).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset>/trajectory_3d_vis.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving them.")
    parser.add_argument("--elev", type=float, default=25.0, help="3D view elevation in degrees.")
    parser.add_argument("--azim", type=float, default=-55.0, help="3D view azimuth in degrees.")
    parser.add_argument(
        "--axis-mode",
        choices=("global", "episode"),
        default="global",
        help="global uses one xyz scale for all selected episodes. episode rescales each figure independently.",
    )
    parser.add_argument(
        "--axis-padding-mm",
        type=float,
        default=5.0,
        help="Padding added to fixed xyz limits. Default: 5mm.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_lerobot_path(pattern: str, episode_index: int, chunks_size: int) -> Path:
    return Path(
        pattern.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
    )


def data_parquet_path(dataset: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    path = dataset / format_lerobot_path(str(info["data_path"]), episode_index, chunks_size)
    if path.is_file():
        return path

    episode_name = f"episode_{episode_index:06d}.parquet"
    candidates = sorted((dataset / "data").glob(f"chunk-*/{episode_name}"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find parquet for episode {episode_index}.")


def load_episode_trajectory(parquet_path: Path, mode: str) -> dict[str, np.ndarray]:
    columns = ["observation.state"]
    if mode in {"action", "both"}:
        columns.append("action")
    df = pd.read_parquet(parquet_path, columns=columns)

    state = np.stack(df["observation.state"].to_list()).astype(np.float64)
    if state.ndim != 2 or state.shape[1] < 3:
        raise ValueError(f"Expected observation.state with shape [T, >=3], got {state.shape} in {parquet_path}")

    trajectories = {"tcp": state[:, :3]}
    if mode in {"action", "both"}:
        actions = np.stack(df["action"].to_list()).astype(np.float64)
        if actions.ndim != 2 or actions.shape[1] < 3:
            raise ValueError(f"Expected action with shape [T, >=3], got {actions.shape} in {parquet_path}")
        action_xyz = actions[:, :3]
        trajectories["action"] = state[0, :3][None, :] + np.cumsum(action_xyz, axis=0)
    return trajectories


def equal_axis_limits(points_mm: np.ndarray, padding_mm: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    points_mm = np.asarray(points_mm, dtype=np.float64)
    if points_mm.ndim != 2 or points_mm.shape[1] != 3 or points_mm.size == 0:
        raise ValueError(f"Expected points with shape [N, 3], got {points_mm.shape}.")

    lower = np.nanmin(points_mm, axis=0)
    upper = np.nanmax(points_mm, axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.5 * max(float(np.max(upper - lower)), 1e-6) + float(padding_mm)
    return (
        (float(center[0] - radius), float(center[0] + radius)),
        (float(center[1] - radius), float(center[1] + radius)),
        (float(center[2] - radius), float(center[2] + radius)),
    )


def all_trajectory_points_mm(trajectories: dict[str, np.ndarray]) -> np.ndarray:
    points = [np.asarray(xyz, dtype=np.float64) * 1000.0 for xyz in trajectories.values() if xyz.size]
    if not points:
        raise ValueError("No trajectory points available.")
    return np.concatenate(points, axis=0)


def set_axis_limits(
    axis,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None,
    *,
    fallback_points_mm: np.ndarray,
    padding_mm: float,
) -> None:
    if limits is None:
        limits = equal_axis_limits(fallback_points_mm, padding_mm)

    axis.set_xlim3d(limits[0])
    axis.set_ylim3d(limits[1])
    axis.set_zlim3d(limits[2])


def plot_episode(
    *,
    episode_index: int,
    trajectories: dict[str, np.ndarray],
    output_path: Path,
    fps: float,
    elev: float,
    azim: float,
    axis_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None,
    axis_padding_mm: float,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")

    colors = {"tcp": "tab:blue", "action": "tab:orange"}
    for name, xyz in trajectories.items():
        if xyz.size == 0:
            continue
        xyz_mm = xyz * 1000.0
        axis.plot(xyz_mm[:, 0], xyz_mm[:, 1], xyz_mm[:, 2], color=colors[name], linewidth=2.0, label=name)
        axis.scatter(xyz_mm[0, 0], xyz_mm[0, 1], xyz_mm[0, 2], color="tab:green", s=45, label=f"{name} start")
        axis.scatter(xyz_mm[-1, 0], xyz_mm[-1, 1], xyz_mm[-1, 2], color="tab:red", s=45, label=f"{name} end")

    length = max(len(xyz) for xyz in trajectories.values())
    duration = length / fps if fps > 0 else 0.0
    axis.set_title(f"episode {episode_index:06d}  frames={length}  duration={duration:.2f}s")
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
    axis.view_init(elev=elev, azim=azim)
    axis.grid(True, alpha=0.25)
    set_axis_limits(
        axis,
        axis_limits,
        fallback_points_mm=all_trajectory_points_mm(trajectories),
        padding_mm=axis_padding_mm,
    )
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    dataset = args.dataset.expanduser().resolve()
    info_path = dataset / "meta" / "info.json"
    episodes_path = dataset / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"Expected LeRobot dataset root with meta/info.json and meta/episodes.jsonl: {dataset}")

    info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    available = [int(row["episode_index"]) for row in episodes]
    if args.episodes:
        selected = parse_episode_indices(args.episodes)
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"Requested episode(s) not found: {missing}")
    else:
        selected = available

    fps = float(info.get("fps", 0.0))
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else dataset / "trajectory_3d_vis"

    episode_trajectories: dict[int, dict[str, np.ndarray]] = {}
    all_points: list[np.ndarray] = []
    for episode_index in selected:
        parquet_path = data_parquet_path(dataset, info, episode_index)
        trajectories = load_episode_trajectory(parquet_path, args.mode)
        episode_trajectories[episode_index] = trajectories
        if args.axis_mode == "global":
            all_points.append(all_trajectory_points_mm(trajectories))

    axis_limits = None
    if args.axis_mode == "global":
        axis_limits = equal_axis_limits(np.concatenate(all_points, axis=0), args.axis_padding_mm)
        print(
            "Using global xyz limits: "
            f"x={axis_limits[0]}, y={axis_limits[1]}, z={axis_limits[2]} mm"
        )

    for episode_index in selected:
        trajectories = episode_trajectories[episode_index]
        output_path = output_dir / f"episode_{episode_index:06d}_{args.mode}_3d.png"
        plot_episode(
            episode_index=episode_index,
            trajectories=trajectories,
            output_path=output_path,
            fps=fps,
            elev=args.elev,
            azim=args.azim,
            axis_limits=axis_limits,
            axis_padding_mm=args.axis_padding_mm,
        )
        print(f"Saved episode {episode_index}: {output_path}")


if __name__ == "__main__":
    main()
