import argparse
import json
import os
import pathlib
import shutil
import sys
import threading
import time
from typing import Any

# Must be set before importing robosuite / robocasa.
os.environ.setdefault("MUJOCO_GL", "osmesa")
if os.environ.get("MUJOCO_GL") == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[3]
for path in (
    WORKSPACE_ROOT / "openpi" / "src",
    WORKSPACE_ROOT / "openpi" / "packages" / "openpi-client" / "src",
    WORKSPACE_ROOT / "robocasa",
    WORKSPACE_ROOT / "robosuite",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cv2
import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401
from robocasa.utils.lerobot_utils import LerobotDatasetWrapper


FPS = 20
VIDEO_INFO = {
    "video.fps": FPS,
    "video.codec": "h264",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "has_audio": False,
}
TASK_NAME_ID = 1
TASK_DESCRIPTION_ID = 0


def _state_for_lerobot(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.gripper_qpos"],
        ),
        axis=0,
    ).astype(np.float64)


def _env_action_from_policy_action(action: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "action.end_effector_position": action[0:3].astype(np.float32),
        "action.end_effector_rotation": action[3:6].astype(np.float32),
        "action.gripper_close": action[6:7].astype(np.float32),
        "action.base_motion": action[7:11].astype(np.float32),
        "action.control_mode": action[11:12].astype(np.float32),
    }


def _policy_action_to_lerobot(action: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            action[7:11],
            action[11:12],
            action[0:3],
            action[3:6],
            action[6:7],
        ),
        axis=0,
    ).astype(np.float64)


def _make_dataset(output_root: pathlib.Path, repo_id: str, image_size: int, overwrite: bool):
    lerobot_root = output_root / "lerobot"
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    img_shape = (image_size, image_size, 3)
    dataset = LerobotDatasetWrapper.create(
        repo_id=repo_id,
        root=lerobot_root,
        robot_type="PandaOmron",
        fps=FPS,
        features={
            "observation.images.robot0_eye_in_hand": {
                "dtype": "video",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
                "video_info": VIDEO_INFO,
            },
            "observation.images.robot0_agentview_left": {
                "dtype": "video",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
                "video_info": VIDEO_INFO,
            },
            "observation.images.robot0_agentview_right": {
                "dtype": "video",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
                "video_info": VIDEO_INFO,
            },
            "annotation.human.task_description": {"dtype": "int64", "shape": (1,)},
            "annotation.human.task_name": {"dtype": "int64", "shape": (1,)},
            "observation.state": {"dtype": "float64", "shape": (16,)},
            "action": {"dtype": "float64", "shape": (12,)},
            "next.reward": {"dtype": "float32", "shape": (1,)},
            "next.done": {"dtype": "bool", "shape": (1,)},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return dataset


def _copy_groot_metadata(output_root: pathlib.Path) -> None:
    meta_dir = output_root / "lerobot" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    source_meta = _find_existing_robocasa_meta()
    shutil.copyfile(source_meta / "modality.json", meta_dir / "modality.json")
    shutil.copyfile(source_meta / "embodiment.json", meta_dir / "embodiment.json")


def _find_existing_robocasa_meta() -> pathlib.Path:
    candidates = []
    data_root = os.environ.get("RP_DATA_ROOT")
    if data_root:
        candidates.extend(pathlib.Path(data_root).glob("**/lerobot/meta/modality.json"))
    candidates.extend((WORKSPACE_ROOT / "robocasa_data_ckpt").glob("**/lerobot/meta/modality.json"))
    for modality_path in sorted(candidates):
        meta_dir = modality_path.parent
        if (meta_dir / "embodiment.json").exists():
            return meta_dir
    raise FileNotFoundError(
        "Could not find RoboCasa GROOT metadata. Expected modality.json and embodiment.json "
        "under $RP_DATA_ROOT/**/lerobot/meta or ./robocasa_data_ckpt/**/lerobot/meta."
    )


def _write_dataset_stats(output_root: pathlib.Path) -> None:
    try:
        from robocasa.utils.lerobot_utils import calculate_dataset_statistics
    except ImportError:
        return

    parquet_paths = sorted((output_root / "lerobot" / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        return
    stats = calculate_dataset_statistics(parquet_paths)
    with open(output_root / "lerobot" / "meta" / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")


def _append_task_name(output_root: pathlib.Path, task_name: str) -> None:
    tasks_path = output_root / "lerobot" / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line).get("task_index") == TASK_NAME_ID:
                return
        except json.JSONDecodeError:
            continue
    payload = {"task_index": TASK_NAME_ID, "task": task_name}
    with open(tasks_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _write_collection_metadata(output_root: pathlib.Path, args: argparse.Namespace, seed: int) -> None:
    extras_dir = output_root / "lerobot" / "extras"
    extras_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_name": args.env_name,
        "split": args.split,
        "seed": seed,
        "seed_source": "argument" if args.seed is not None else "time.time_ns",
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "pos_step": args.pos_step,
        "rot_step": args.rot_step,
    }
    with open(extras_dir / "collection_meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


class KeyboardState:
    def __init__(self):
        from pynput.keyboard import Key, Listener

        self._space_key = Key.space
        self._pressed: set[str] = set()
        self._events: list[str] = []
        self._lock = threading.Lock()
        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def pressed(self) -> set[str]:
        with self._lock:
            return set(self._pressed)

    def consume_events(self) -> list[str]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def stop(self) -> None:
        self._listener.stop()

    def _on_press(self, key) -> None:
        name = self._key_name(key)
        if name is None:
            return
        with self._lock:
            if name not in self._pressed:
                self._events.append(name)
            self._pressed.add(name)

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        if name is None:
            return
        with self._lock:
            self._pressed.discard(name)

    def _key_name(self, key) -> str | None:
        if key == self._space_key:
            return "space"
        char = getattr(key, "char", None)
        if char:
            return char.lower()
        return None


def _action_from_pressed_keys(
    pressed: set[str],
    pos_step: float,
    rot_step: float,
    gripper_closed: bool,
) -> tuple[np.ndarray, bool]:
    action = np.zeros(12, dtype=np.float32)
    action[6] = 1.0 if gripper_closed else 0.0

    action[0] = (float("w" in pressed) - float("s" in pressed)) * pos_step
    action[1] = (float("a" in pressed) - float("d" in pressed)) * pos_step
    action[2] = (float("r" in pressed) - float("f" in pressed)) * pos_step
    action[3] = (float("j" in pressed) - float("l" in pressed)) * rot_step
    action[4] = (float("i" in pressed) - float("k" in pressed)) * rot_step
    action[5] = (float("u" in pressed) - float("o" in pressed)) * rot_step

    handled = bool(np.any(action[:6]))
    return action, handled


def _draw_overlay(image: np.ndarray, episode: int, step: int, success: bool, gripper_closed: bool) -> np.ndarray:
    frame = image.copy()
    lines = [
        f"episode {episode} step {step} success={success} gripper={'closed' if gripper_closed else 'open'}",
        "hold keys together: w/s x, a/d y, r/f z | j/l roll, i/k pitch, u/o yaw",
        "space gripper | n save episode | x discard/reset | q quit",
    ]
    y = 22
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return frame


def _show_rgb_window(window_name: str, image: np.ndarray, display_scale: float) -> None:
    frame = image
    if display_scale != 1.0:
        frame = cv2.resize(
            frame,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _add_frame(dataset, obs, action: np.ndarray, reward: float, done: bool, prompt: str) -> None:
    dataset.add_frame(
        {
            "observation.images.robot0_eye_in_hand": np.asarray(obs["video.robot0_eye_in_hand"], dtype=np.uint8),
            "observation.images.robot0_agentview_left": np.asarray(obs["video.robot0_agentview_left"], dtype=np.uint8),
            "observation.images.robot0_agentview_right": np.asarray(obs["video.robot0_agentview_right"], dtype=np.uint8),
            "observation.state": _state_for_lerobot(obs),
            "action": _policy_action_to_lerobot(action),
            "annotation.human.task_description": np.asarray([TASK_DESCRIPTION_ID], dtype=np.int64),
            "annotation.human.task_name": np.asarray([TASK_NAME_ID], dtype=np.int64),
            "next.reward": np.asarray([reward], dtype=np.float32),
            "next.done": np.asarray([done], dtype=bool),
            "task": prompt,
        }
    )


def _flush_episode(dataset, records: list[tuple[dict[str, Any], np.ndarray, float, bool]], prompt: str) -> None:
    if not records:
        raise ValueError("Cannot save an empty episode.")
    records[-1] = (records[-1][0], records[-1][1], records[-1][2], True)
    for obs, action, reward, done in records:
        _add_frame(dataset, obs, action, reward, done, prompt)
    dataset.save_episode()


def _copy_obs_for_record(obs: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "video.robot0_eye_in_hand",
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    ]
    return {key: np.asarray(obs[key]).copy() for key in keys}


def _save_episode_extras(
    output_root: pathlib.Path,
    episode_idx: int,
    *,
    prompt: str,
    states: list[np.ndarray],
    policy_actions: list[np.ndarray],
    lerobot_actions: list[np.ndarray],
    rewards: list[float],
    dones: list[bool],
) -> None:
    extras_dir = output_root / "lerobot" / "extras" / f"episode_{episode_idx:06d}"
    extras_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        extras_dir / "raw_episode.npz",
        observation_state=np.asarray(states, dtype=np.float64),
        policy_order_actions=np.asarray(policy_actions, dtype=np.float64),
        lerobot_order_actions=np.asarray(lerobot_actions, dtype=np.float64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
    )
    with open(extras_dir / "ep_meta.json", "w", encoding="utf-8") as f:
        json.dump({"lang": prompt, "num_steps": len(policy_actions)}, f, indent=4)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Keyboard data collection for the custom RoboCasa StackCube env.")
    parser.add_argument("--env-name", default="StackCube")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default="data/robocasa_stack_cube_keyboard")
    parser.add_argument("--repo-id", default="robocasa/stack_cube_keyboard")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--pos-step", type=float, default=0.2)
    parser.add_argument("--rot-step", type=float, default=0.12)
    parser.add_argument("--display-scale", type=float, default=2.0)
    parser.add_argument("--record-idle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    seed = args.seed
    if seed is None:
        seed = time.time_ns() % (2**32 - 1)
    seed = int(seed)

    output_root = pathlib.Path(args.output_root)
    dataset = _make_dataset(output_root, args.repo_id, 256, args.overwrite)
    env = gym.make(f"robocasa/{args.env_name}", split=args.split, seed=seed)
    print(f"using env seed: {seed}")

    episode_idx = 0
    agentview_window = f"{args.env_name} agentview"
    wrist_window = f"{args.env_name} wrist"
    cv2.namedWindow(agentview_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(wrist_window, cv2.WINDOW_NORMAL)
    keyboard = KeyboardState()

    try:
        while episode_idx < args.episodes:
            obs, _ = env.reset()
            prompt = str(obs["annotation.human.task_description"])
            gripper_closed = False
            step_idx = 0
            episode_states: list[np.ndarray] = []
            episode_policy_actions: list[np.ndarray] = []
            episode_lerobot_actions: list[np.ndarray] = []
            episode_rewards: list[float] = []
            episode_dones: list[bool] = []
            episode_records: list[tuple[dict[str, Any], np.ndarray, float, bool]] = []
            latest_success = False

            while step_idx < args.max_steps:
                agentview_display = _draw_overlay(
                    obs["video.robot0_agentview_left"],
                    episode_idx,
                    step_idx,
                    latest_success,
                    gripper_closed,
                )
                wrist_display = _draw_overlay(
                    obs["video.robot0_eye_in_hand"],
                    episode_idx,
                    step_idx,
                    latest_success,
                    gripper_closed,
                )
                _show_rgb_window(agentview_window, agentview_display, args.display_scale)
                _show_rgb_window(wrist_window, wrist_display, args.display_scale)
                cv2.waitKey(max(1, int(1000 / FPS)))

                events = keyboard.consume_events()
                if "space" in events:
                    gripper_closed = not gripper_closed
                if "q" in events:
                    raise KeyboardInterrupt
                if "x" in events:
                    print(f"discard episode {episode_idx}")
                    break
                if "n" in events:
                    if episode_records:
                        episode_dones[-1] = True
                        _flush_episode(dataset, episode_records, prompt)
                        _save_episode_extras(
                            output_root,
                            episode_idx,
                            prompt=prompt,
                            states=episode_states,
                            policy_actions=episode_policy_actions,
                            lerobot_actions=episode_lerobot_actions,
                            rewards=episode_rewards,
                            dones=episode_dones,
                        )
                        print(f"saved episode {episode_idx} ({len(episode_policy_actions)} steps)")
                        episode_idx += 1
                    else:
                        print("empty episode; not saved")
                    break

                action, handled = _action_from_pressed_keys(
                    keyboard.pressed(),
                    args.pos_step,
                    args.rot_step,
                    gripper_closed,
                )
                gripper_toggled = "space" in events
                if not handled and not gripper_toggled and not args.record_idle:
                    continue
                record_obs = _copy_obs_for_record(obs)
                episode_states.append(_state_for_lerobot(obs))
                episode_policy_actions.append(action.astype(np.float64))
                episode_lerobot_actions.append(_policy_action_to_lerobot(action))

                obs, reward, _, _, info = env.step(_env_action_from_policy_action(action))
                done = bool(info.get("success", False))
                latest_success = done
                episode_rewards.append(float(reward))
                episode_dones.append(done)
                episode_records.append((record_obs, action.copy(), float(reward), done))
                step_idx += 1

                if done or step_idx >= args.max_steps:
                    episode_dones[-1] = True
                    _flush_episode(dataset, episode_records, prompt)
                    _save_episode_extras(
                        output_root,
                        episode_idx,
                        prompt=prompt,
                        states=episode_states,
                        policy_actions=episode_policy_actions,
                        lerobot_actions=episode_lerobot_actions,
                        rewards=episode_rewards,
                        dones=episode_dones,
                    )
                    print(f"saved episode {episode_idx} ({len(episode_policy_actions)} steps, success={done})")
                    episode_idx += 1
                    break
    except KeyboardInterrupt:
        print("collection stopped")
    finally:
        keyboard.stop()
        env.close()
        cv2.destroyAllWindows()

    _copy_groot_metadata(output_root)
    _append_task_name(output_root, args.env_name)
    _write_collection_metadata(output_root, args, seed)
    _write_dataset_stats(output_root)
    print(f"dataset written to {output_root / 'lerobot'}")


if __name__ == "__main__":
    main()
