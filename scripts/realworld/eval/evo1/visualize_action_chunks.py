#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_action_chunks(path: Path) -> dict[str, np.ndarray]:
    if path.is_dir():
        path = path / "action_chunks.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Action chunk file not found: {path}")
    return dict(np.load(path))


def _chunk_values(chunks: np.ndarray, valid_lengths: np.ndarray, idx: int) -> np.ndarray:
    return chunks[idx, : int(valid_lengths[idx])]


def _metric(actions: np.ndarray, name: str) -> np.ndarray:
    if name == "xyz_norm":
        return np.linalg.norm(actions[:, : min(3, actions.shape[1])], axis=1)
    if name == "rot_norm":
        if actions.shape[1] <= 3:
            return np.zeros((actions.shape[0],), dtype=np.float32)
        return np.linalg.norm(actions[:, 3: min(6, actions.shape[1])], axis=1)
    if name == "motion_norm":
        return np.linalg.norm(actions[:, : min(6, actions.shape[1])], axis=1)
    if name == "abs_gripper":
        if actions.shape[1] <= 6:
            return np.zeros((actions.shape[0],), dtype=np.float32)
        return np.abs(actions[:, 6])
    raise ValueError(f"Unsupported metric: {name}")


def _relative_cumulative_xyz_distance(actions: np.ndarray) -> np.ndarray:
    if actions.shape[1] < 3:
        return np.zeros((actions.shape[0],), dtype=np.float32)
    xyz = np.asarray(actions[:, :3], dtype=np.float32)
    cumulative_xyz = np.cumsum(xyz, axis=0)
    relative_xyz = cumulative_xyz - cumulative_xyz[0:1]
    return np.linalg.norm(relative_xyz, axis=1)


def _save_xyz_distance_episode_timestep_overlay(data: dict[str, np.ndarray], output_dir: Path) -> None:
    chunks = data["action_chunks"]
    valid_lengths = data["valid_lengths"]
    episode_indices = data["episode_indices"]
    chunk_indices = data["chunk_indices"]
    start_steps = data["start_steps"]
    executed_steps = data["executed_steps"]
    if chunks.shape[2] < 3:
        return

    cmap = plt.get_cmap("tab20")

    for episode_idx in np.unique(episode_indices):
        row_indices = np.where(episode_indices == episode_idx)[0]
        fig, axis = plt.subplots(1, 1, figsize=(14, 5))
        y_max = 0.0
        for local_idx, row_idx in enumerate(row_indices):
            actions = _chunk_values(chunks, valid_lengths, row_idx)
            x = np.arange(actions.shape[0], dtype=np.int32) + int(start_steps[row_idx])
            exec_end = int(start_steps[row_idx]) + int(executed_steps[row_idx]) - 1
            y = _relative_cumulative_xyz_distance(actions)
            if y.size:
                y_max = max(y_max, float(np.nanmax(y)))
            axis.plot(
                x,
                y,
                color=cmap(local_idx % cmap.N),
                alpha=0.9,
                linewidth=2.2,
                marker="o",
                markersize=4.5,
                markeredgewidth=0.0,
                label=f"chunk {int(chunk_indices[row_idx])}",
            )
            axis.axvspan(
                int(start_steps[row_idx]) - 0.5,
                exec_end + 0.5,
                color="tab:green",
                alpha=0.018,
                linewidth=0,
            )

        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        axis.set_ylabel("relative cumulative 3D distance")
        axis.set_xlabel("episode timestep")
        axis.grid(True, alpha=0.2)
        axis.set_title(f"FastWAM relative cumulative xyz distance, episode={int(episode_idx)}")
        axis.set_ylim(0.0, max(y_max * 2.0, 1e-6))
        if len(row_indices) <= 20:
            axis.legend(ncol=4, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"episode_{int(episode_idx):03d}_xyz_distance_overlay.png", dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(1, 1, figsize=(14, 5))
    y_max = 0.0
    for row_idx in range(chunks.shape[0]):
        actions = _chunk_values(chunks, valid_lengths, row_idx)
        x = np.arange(actions.shape[0], dtype=np.int32) + int(start_steps[row_idx])
        exec_end = int(start_steps[row_idx]) + int(executed_steps[row_idx]) - 1
        y = _relative_cumulative_xyz_distance(actions)
        if y.size:
            y_max = max(y_max, float(np.nanmax(y)))
        axis.plot(
            x,
            y,
            color=cmap(int(chunk_indices[row_idx]) % cmap.N),
            alpha=0.75,
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            markeredgewidth=0.0,
        )
        axis.axvspan(
            int(start_steps[row_idx]) - 0.5,
            exec_end + 0.5,
            color="tab:green",
            alpha=0.012,
            linewidth=0,
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axis.set_ylabel("relative cumulative 3D distance")
    axis.set_xlabel("episode timestep")
    axis.grid(True, alpha=0.2)
    axis.set_title("FastWAM relative cumulative xyz distance, all episodes")
    axis.set_ylim(0.0, max(y_max * 2.0, 1e-6))
    fig.tight_layout()
    fig.savefig(output_dir / "all_episodes_xyz_distance_overlay.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved FastWAM rollout action chunks.")
    parser.add_argument("input", type=Path, help="Rollout run directory or action_chunks.npz path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data = _load_action_chunks(args.input)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.input / "action_chunk_vis" if args.input.is_dir() else args.input.parent / "action_chunk_vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_xyz_distance_episode_timestep_overlay(data, output_dir)

    print(f"Saved action chunk visualizations to {output_dir}")


if __name__ == "__main__":
    main()
