"""Rollout recording helpers for RoboCasa policy evaluation."""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import shutil
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper

import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils


@dataclasses.dataclass(frozen=True)
class RolloutRecording:
    env_name: str
    output_dir: Path
    episodes_dir: Path
    hdf5_path: Path
    env_info_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "env_name": self.env_name,
            "output_dir": str(self.output_dir),
            "episodes_dir": str(self.episodes_dir),
            "hdf5_path": str(self.hdf5_path),
            "env_info_path": str(self.env_info_path),
        }


@dataclasses.dataclass
class RolloutEpisodeCache:
    eye_in_hand_images: list[np.ndarray] = dataclasses.field(default_factory=list)
    agentview_left_images: list[np.ndarray] = dataclasses.field(default_factory=list)
    agentview_right_images: list[np.ndarray] = dataclasses.field(default_factory=list)
    states: list[np.ndarray] = dataclasses.field(default_factory=list)
    actions: list[np.ndarray] = dataclasses.field(default_factory=list)
    rewards: list[float] = dataclasses.field(default_factory=list)
    dones: list[bool] = dataclasses.field(default_factory=list)


def _sanitize_path_component(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip(
        "._-"
    )
    return sanitized or "unknown"


def _rollout_root() -> Path:
    data_root = os.environ.get("RP_DATA_ROOT")
    if not data_root:
        raise ValueError("RP_DATA_ROOT must be set when --args.record-rollout is enabled.")
    return Path(data_root) / "rollouts"


def _build_env_info(env_name: str, split: str, seed: int) -> str:
    controller_config = load_composite_controller_config(controller=None, robot="PandaOmron")
    env_info = {
        "env_name": env_name,
        "robots": "PandaOmron",
        "controller_configs": controller_config,
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "camera_widths": 256,
        "camera_heights": 256,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "camera_depths": False,
        "seed": seed,
        "generative_textures": None,
        "randomize_cameras": False,
        "translucent_robot": False,
    }

    if split == "target":
        env_info.update(
            {
                "obj_instance_split": "target",
                "layout_and_style_ids": list(zip(range(1, 11), range(1, 11))),
                "layout_ids": None,
                "style_ids": None,
            }
        )
    elif split == "pretrain":
        env_info.update(
            {
                "obj_instance_split": "pretrain",
                "layout_and_style_ids": None,
                "layout_ids": -2,
                "style_ids": -2,
            }
        )
    else:
        raise ValueError(f"Unsupported split for rollout recording: {split}")

    return json.dumps(env_info)


def start_rollout_recording(
    gym_env,
    *,
    env_name: str,
    split: str,
    seed: int,
    run_name: str,
) -> RolloutRecording:
    task_dir = _rollout_root() / _sanitize_path_component(env_name)
    output_dir = task_dir / _sanitize_path_component(run_name)
    episodes_dir = output_dir / "episodes"
    if episodes_dir.exists() and any(episodes_dir.iterdir()):
        raise ValueError(f"Rollout recording directory already contains episodes: {episodes_dir}")
    episodes_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(gym_env.env, DataCollectionWrapper):
        raise ValueError("RoboCasa gym env is already wrapped with DataCollectionWrapper.")

    gym_env.env = DataCollectionWrapper(
        gym_env.env,
        str(episodes_dir),
        use_env_xml_for_reset=True,
    )
    recording = RolloutRecording(
        env_name=env_name,
        output_dir=output_dir,
        episodes_dir=episodes_dir,
        hdf5_path=output_dir / "rollouts.hdf5",
        env_info_path=output_dir / "env_info.json",
    )
    recording.env_info_path.write_text(_build_env_info(env_name, split, seed))
    return recording


def _cached_lerobot_state_from_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["state.base_position"], dtype=np.float64),
            np.asarray(obs["state.base_rotation"], dtype=np.float64),
            np.asarray(obs["state.end_effector_position_relative"], dtype=np.float64),
            np.asarray(obs["state.end_effector_rotation_relative"], dtype=np.float64),
            np.asarray(obs["state.gripper_qpos"], dtype=np.float64),
        ],
        axis=0,
    )


def start_rollout_episode_cache() -> RolloutEpisodeCache:
    return RolloutEpisodeCache()


def append_rollout_cache_step(cache: RolloutEpisodeCache, obs: dict, action: np.ndarray) -> None:
    cache.eye_in_hand_images.append(np.asarray(obs["video.robot0_eye_in_hand"], dtype=np.uint8))
    cache.agentview_left_images.append(np.asarray(obs["video.robot0_agentview_left"], dtype=np.uint8))
    cache.agentview_right_images.append(np.asarray(obs["video.robot0_agentview_right"], dtype=np.uint8))
    cache.states.append(_cached_lerobot_state_from_obs(obs))
    cache.actions.append(np.asarray(action, dtype=np.float64)[:12])


def complete_rollout_cache_step(cache: RolloutEpisodeCache, *, reward: float, done: bool) -> None:
    cache.rewards.append(float(reward))
    cache.dones.append(bool(done))


def write_rollout_episode_cache(gym_env, cache: RolloutEpisodeCache) -> None:
    ep_directory = getattr(gym_env.env, "ep_directory", None)
    if ep_directory is None:
        raise ValueError("No rollout episode directory was created; did the episode execute at least one step?")
    if not cache.actions:
        raise ValueError("Cannot write an empty rollout episode cache.")
    if len(cache.actions) != len(cache.rewards) or len(cache.actions) != len(cache.dones):
        raise ValueError(
            f"Incomplete rollout episode cache: {len(cache.actions)} actions, "
            f"{len(cache.rewards)} rewards, {len(cache.dones)} dones."
        )

    dones = np.asarray(cache.dones, dtype=bool)
    dones[-1] = True
    np.savez_compressed(
        Path(ep_directory) / "cached_rollout_obs.npz",
        robot0_eye_in_hand_image=np.stack(cache.eye_in_hand_images),
        robot0_agentview_left_image=np.stack(cache.agentview_left_images),
        robot0_agentview_right_image=np.stack(cache.agentview_right_images),
        observation_state=np.stack(cache.states).astype(np.float64),
        actions=np.stack(cache.actions).astype(np.float64),
        rewards=np.asarray(cache.rewards, dtype=np.float32),
        dones=dones,
    )


def write_episode_stats(
    gym_env,
    *,
    success: bool,
    aborted: bool,
    episode_idx: int,
    num_steps: int,
) -> None:
    ep_directory = getattr(gym_env.env, "ep_directory", None)
    if ep_directory is None:
        raise ValueError(
            "No rollout episode directory was created; did the episode execute at least one step?"
        )

    stats = {
        "success": bool(success),
        "aborted": bool(aborted),
        "episode_idx": int(episode_idx),
        "num_steps": int(num_steps),
    }
    with open(Path(ep_directory) / "ep_stats.json", "w") as f:
        json.dump(stats, f, indent=4)


def _gather_rollout_episodes_as_hdf5(recording: RolloutRecording) -> Path:
    env_info = recording.env_info_path.read_text()
    ep_dirs = sorted(path for path in recording.episodes_dir.iterdir() if path.is_dir())
    if not ep_dirs:
        raise ValueError(f"No rollout episodes found under {recording.episodes_dir}.")

    with h5py.File(recording.hdf5_path, "w") as f:
        grp = f.create_group("data")
        num_eps = 0
        env_name = None

        for ep_directory in ep_dirs:
            states = []
            actions = []
            actions_abs = []
            success = False
            for state_file in sorted(glob(str(ep_directory / "state_*.npz"))):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])
                states.extend(dic["states"])
                success = success or bool(dic["successful"])
                for action_info in dic["action_infos"]:
                    actions.append(action_info["actions"])
                    if "actions_abs" in action_info:
                        actions_abs.append(action_info["actions_abs"])

            if len(states) == 0:
                continue
            del states[-1]
            if len(states) != len(actions):
                raise ValueError(
                    f"State/action length mismatch in {ep_directory}: {len(states)} states, {len(actions)} actions."
                )

            ep_data_grp = grp.create_group(f"demo_{num_eps}")
            ep_data_grp.attrs["model_file"] = (ep_directory / "model.xml").read_text()
            ep_meta_path = ep_directory / "ep_meta.json"
            if not ep_meta_path.exists():
                raise ValueError(f"Missing ep_meta.json for recorded rollout: {ep_directory}")
            ep_data_grp.attrs["ep_meta"] = ep_meta_path.read_text()
            ep_stats_path = ep_directory / "ep_stats.json"
            if ep_stats_path.exists():
                ep_data_grp.attrs["ep_stats"] = ep_stats_path.read_text()
            ep_data_grp.attrs["success"] = success

            ep_data_grp.create_dataset("states", data=np.asarray(states))
            ep_data_grp.create_dataset("actions", data=np.asarray(actions))
            if actions_abs:
                ep_data_grp.create_dataset("actions_abs", data=np.asarray(actions_abs))
            cache_path = ep_directory / "cached_rollout_obs.npz"
            if cache_path.exists():
                cache = np.load(cache_path)
                cache_actions = np.asarray(cache["actions"])
                if cache_actions.shape != np.asarray(actions).shape:
                    raise ValueError(
                        f"Cached action shape mismatch in {ep_directory}: "
                        f"{cache_actions.shape} cached vs {np.asarray(actions).shape} recorded."
                    )
                rollout_obs_grp = ep_data_grp.create_group("rollout_obs")
                for key in [
                    "robot0_eye_in_hand_image",
                    "robot0_agentview_left_image",
                    "robot0_agentview_right_image",
                    "observation_state",
                    "rewards",
                    "dones",
                ]:
                    rollout_obs_grp.create_dataset(key, data=np.asarray(cache[key]))
            num_eps += 1

        if num_eps == 0:
            raise ValueError(f"No non-empty rollout episodes found under {recording.episodes_dir}.")

        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["robocasa_version"] = getattr(robocasa, "__version__", "unknown")
        grp.attrs["robosuite_version"] = getattr(robosuite, "__version__", "unknown")
        grp.attrs["mujoco_version"] = getattr(mujoco, "__version__", "unknown")
        grp.attrs["env"] = env_name or recording.env_name
        grp.attrs["env_info"] = env_info

    DatasetUtils.convert_to_robomimic_format(str(recording.hdf5_path), filter_num_demos=None)
    return recording.hdf5_path


def _copy_ep_stats_to_lerobot_extras(recording: RolloutRecording) -> None:
    lerobot_extras_dir = recording.output_dir / "lerobot" / "extras"
    with h5py.File(recording.hdf5_path, "r") as f:
        for ep_idx, demo in enumerate(f["data"].keys()):
            ep_stats = f["data"][demo].attrs.get("ep_stats", None)
            if ep_stats is None:
                continue
            ep_dir = lerobot_extras_dir / f"episode_{ep_idx:06d}"
            if not ep_dir.exists():
                raise ValueError(f"Missing converted LeRobot extras directory: {ep_dir}")
            (ep_dir / "ep_stats.json").write_text(str(ep_stats))


def _cleanup_converted_rollout_intermediates(recording: RolloutRecording) -> None:
    lerobot_dir = recording.output_dir / "lerobot"
    if not lerobot_dir.exists():
        raise ValueError(f"Cannot clean rollout intermediates before LeRobot conversion exists: {lerobot_dir}")

    if recording.episodes_dir.exists():
        shutil.rmtree(recording.episodes_dir)
    if recording.hdf5_path.exists():
        recording.hdf5_path.unlink()


def finalize_rollout_recordings(recordings: list[dict[str, str]]) -> list[dict[str, str]]:
    from robocasa.scripts.dataset_scripts import convert_hdf5_lerobot

    results = []
    for payload in recordings:
        recording = RolloutRecording(
            env_name=payload["env_name"],
            output_dir=Path(payload["output_dir"]),
            episodes_dir=Path(payload["episodes_dir"]),
            hdf5_path=Path(payload["hdf5_path"]),
            env_info_path=Path(payload["env_info_path"]),
        )
        hdf5_path = _gather_rollout_episodes_as_hdf5(recording)
        convert_hdf5_lerobot.main(
            SimpleNamespace(
                raw_dataset_path=str(hdf5_path),
                camera_names=[
                    "robot0_eye_in_hand",
                    "robot0_agentview_left",
                    "robot0_agentview_right",
                ],
                camera_height=256,
                camera_width=256,
            )
        )
        _copy_ep_stats_to_lerobot_extras(recording)
        _cleanup_converted_rollout_intermediates(recording)
        results.append(
            {
                **recording.to_dict(),
                "lerobot_path": str(recording.output_dir / "lerobot"),
            }
        )
    return results
