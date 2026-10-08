import argparse
import json
import os
import time
from datetime import datetime

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_env import RobotEnv
from my_device.macros import CAM_SERIAL


def _get_next_demo_id(data_group: h5py.Group) -> int:
    demo_ids = []
    for name in data_group.keys():
        if not name.startswith("demo_"):
            continue
        try:
            demo_ids.append(int(name.split("_", 1)[1]))
        except ValueError:
            continue
    return (max(demo_ids) + 1) if demo_ids else 0


def _ensure_dataset_root(h5file: h5py.File, robot_env: RobotEnv, fps: float) -> h5py.Group:
    data_group = h5file.require_group("data")
    now = datetime.now()
    data_group.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
    data_group.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
    data_group.attrs["env"] = "FlexivRealTeleop"
    data_group.attrs["env_info"] = json.dumps(
        {
            "env_name": "FlexivRealTeleop",
            "robot": "Flexiv",
            "teleop_device": "Sigma7",
            "camera_names": [
                "robot0_agentview_left",
                "robot0_eye_in_hand",
            ],
            "camera_serials": list(robot_env.camera_serial),
            "fps": float(fps),
            "base_moving": False,
            "disable_gripper_cmd": bool(getattr(robot_env, "disable_gripper_cmd", False)),
            "action_semantics": "delta from observed TCP pose to teleop absolute target pose",
            "actions_abs_semantics": "teleop absolute target pose",
            "actions_teleop_delta_semantics": "delta between consecutive teleop target poses",
        }
    )
    return data_group


def _episode_meta(robot_env: RobotEnv, num_steps: int, task_description: str) -> str:
    return json.dumps(
        {
            "lang": task_description,
            "num_steps": int(num_steps),
            "camera_serials": list(robot_env.camera_serial),
            "base_moving": False,
            "disable_gripper_cmd": bool(getattr(robot_env, "disable_gripper_cmd", False)),
            "action_semantics": "delta from observed TCP pose to teleop absolute target pose",
            "actions_abs_semantics": "teleop absolute target pose",
            "actions_teleop_delta_semantics": "delta between consecutive teleop target poses",
            "source": "flexiv_real_teleop",
        }
    )


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def _build_state(tcp_pose_wxyz: np.ndarray, gripper_state: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(tcp_pose_wxyz[:3], dtype=np.float64),
            _quat_wxyz_to_xyzw(tcp_pose_wxyz[3:7]),
            np.asarray([gripper_state], dtype=np.float64),
        ],
        axis=0,
    )


def _build_delta_action_from_abs_target(current_tcp_pose_wxyz: np.ndarray, action_abs_wxyz: np.ndarray) -> np.ndarray:
    current_tcp_pose_wxyz = np.asarray(current_tcp_pose_wxyz, dtype=np.float64)
    action_abs_wxyz = np.asarray(action_abs_wxyz, dtype=np.float64)

    delta_pos = action_abs_wxyz[:3] - current_tcp_pose_wxyz[:3]
    current_rot = R.from_quat(_quat_wxyz_to_xyzw(current_tcp_pose_wxyz[3:7]))
    target_rot = R.from_quat(_quat_wxyz_to_xyzw(action_abs_wxyz[3:7]))
    delta_rotvec = (current_rot.inv() * target_rot).as_rotvec().astype(np.float64)
    gripper_action = np.asarray([action_abs_wxyz[7]], dtype=np.float64)
    return np.concatenate([delta_pos, delta_rotvec, gripper_action], axis=0)


def _build_teleop_delta_action(action_delta_wxyz: np.ndarray) -> np.ndarray:
    delta_pos = np.asarray(action_delta_wxyz[:3], dtype=np.float64)
    delta_quat_xyzw = _quat_wxyz_to_xyzw(action_delta_wxyz[3:7])
    delta_rotvec = R.from_quat(delta_quat_xyzw).as_rotvec().astype(np.float64)
    gripper_action = np.asarray([action_delta_wxyz[7]], dtype=np.float64)
    return np.concatenate([delta_pos, delta_rotvec, gripper_action], axis=0)


def _build_action_abs(action_abs_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(action_abs_wxyz[:3], dtype=np.float64),
            _quat_wxyz_to_xyzw(action_abs_wxyz[3:7]),
            np.asarray([action_abs_wxyz[7]], dtype=np.float64),
        ],
        axis=0,
    )


def _write_demo(
    h5file: h5py.File,
    robot_env: RobotEnv,
    episode: dict,
    fps: float,
    task_description: str,
) -> int:
    data_group = _ensure_dataset_root(h5file, robot_env, fps)
    demo_id = _get_next_demo_id(data_group)
    demo_group = data_group.create_group(f"demo_{demo_id}")
    demo_group.attrs["model_file"] = ""
    demo_group.attrs["ep_meta"] = _episode_meta(robot_env, len(episode["actions"]), task_description)

    demo_group.create_dataset("actions", data=episode["actions"], compression="gzip", compression_opts=4)
    demo_group.create_dataset("actions_abs", data=episode["actions_abs"], compression="gzip", compression_opts=4)
    demo_group.create_dataset(
        "actions_teleop_delta",
        data=episode["actions_teleop_delta"],
        compression="gzip",
        compression_opts=4,
    )
    demo_group.create_dataset("states", data=episode["states"], compression="gzip", compression_opts=4)

    obs_group = demo_group.create_group("obs")
    obs_group.create_dataset(
        "robot0_agentview_left_image",
        data=episode["robot0_agentview_left_image"],
        compression="gzip",
        compression_opts=4,
    )
    obs_group.create_dataset(
        "robot0_eye_in_hand_image",
        data=episode["robot0_eye_in_hand_image"],
        compression="gzip",
        compression_opts=4,
    )
    obs_group.create_dataset(
        "robot0_base_to_eef_pos",
        data=episode["robot0_base_to_eef_pos"],
        compression="gzip",
        compression_opts=4,
    )
    obs_group.create_dataset(
        "robot0_base_to_eef_quat",
        data=episode["robot0_base_to_eef_quat"],
        compression="gzip",
        compression_opts=4,
    )
    obs_group.create_dataset(
        "robot0_gripper_qpos",
        data=episode["robot0_gripper_qpos"],
        compression="gzip",
        compression_opts=4,
    )
    obs_group.create_dataset(
        "robot0_joint_pos",
        data=episode["robot0_joint_pos"],
        compression="gzip",
        compression_opts=4,
    )
    h5file.flush()
    return demo_id


def record(h5file: h5py.File, robot_env: RobotEnv, fps: float, task_description: str):
    env_images = []
    wrist_images = []
    tcp_pos = []
    tcp_quat = []
    gripper_qpos = []
    joint_pos = []
    states = []
    actions = []
    actions_abs = []
    actions_teleop_delta = []

    robot_env.keyboard.start = False
    robot_env.keyboard.discard = False
    robot_env.keyboard.finish = False
    robot_env.keyboard.manual_reset = False
    episode_start_time = None

    last_p = robot_env.robot.init_pose[:3]
    last_r = R.from_quat(robot_env.robot.init_pose[3:7], scalar_first=True)

    while not robot_env.keyboard.quit and not robot_env.keyboard.discard and not robot_env.keyboard.finish:
        if robot_env.keyboard.manual_reset:
            print("Manual reset requested.")
            robot_env.reset_robot()
            last_p = robot_env.robot.init_pose[:3]
            last_r = R.from_quat(robot_env.robot.init_pose[3:7], scalar_first=True)
            robot_env.keyboard.manual_reset = False
            continue

        transition_data, last_p, last_r = robot_env.human_teleop_step(last_p, last_r)
        if transition_data is None:
            continue

        if not robot_env.keyboard.start:
            continue

        if episode_start_time is None:
            episode_start_time = time.time()
            print("Episode start!")

        current_tcp_pose = np.asarray(transition_data["tcp_pose"], dtype=np.float64)
        current_joint_pos = np.asarray(transition_data["joint_pos"], dtype=np.float64)
        current_gripper_state = float(robot_env.gripper.get_gripper_state())
        action_abs_wxyz = np.asarray(transition_data["action_abs"], dtype=np.float64)

        env_images.append(transition_data["policy_env_img"].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        wrist_images.append(transition_data["policy_wrist_img"].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        tcp_pos.append(np.asarray(current_tcp_pose[:3], dtype=np.float64))
        tcp_quat.append(_quat_wxyz_to_xyzw(current_tcp_pose[3:7]))
        gripper_qpos.append(np.asarray([current_gripper_state], dtype=np.float64))
        joint_pos.append(current_joint_pos)
        states.append(_build_state(current_tcp_pose, current_gripper_state))
        actions.append(_build_delta_action_from_abs_target(current_tcp_pose, action_abs_wxyz))
        actions_abs.append(_build_action_abs(action_abs_wxyz))
        actions_teleop_delta.append(_build_teleop_delta_action(np.asarray(transition_data["action"], dtype=np.float64)))

    if not robot_env.keyboard.start or robot_env.keyboard.quit or robot_env.keyboard.discard:
        print("WARNING: discard the demo!")
        time.sleep(0.5)
        if episode_start_time is not None:
            elapsed_sec = time.time() - episode_start_time
            print(f"Discarded episode elapsed time: {elapsed_sec:.3f}s")
        return
    if not actions:
        print("WARNING: discard empty demo!")
        return

    episode = {
        "robot0_agentview_left_image": np.stack(env_images, axis=0),
        "robot0_eye_in_hand_image": np.stack(wrist_images, axis=0),
        "robot0_base_to_eef_pos": np.stack(tcp_pos, axis=0),
        "robot0_base_to_eef_quat": np.stack(tcp_quat, axis=0),
        "robot0_gripper_qpos": np.stack(gripper_qpos, axis=0),
        "robot0_joint_pos": np.stack(joint_pos, axis=0),
        "states": np.stack(states, axis=0),
        "actions": np.stack(actions, axis=0),
        "actions_abs": np.stack(actions_abs, axis=0),
        "actions_teleop_delta": np.stack(actions_teleop_delta, axis=0),
    }

    demo_id = _write_demo(h5file, robot_env, episode, fps, task_description)
    elapsed_sec = 0.0 if episode_start_time is None else time.time() - episode_start_time
    print("Saved demo", demo_id)
    print(f"Demo {demo_id} elapsed time: {elapsed_sec:.3f}s")


def main(args):
    if args.resolution is None:
        args.resolution = [224, 224]
    robot_env = RobotEnv(
        camera_serial=CAM_SERIAL,
        img_shape=[3] + args.resolution,
        fps=args.fps,
        disable_gripper_cmd=args.disable_gripper_cmd,
        z_freeze_clutch_threshold=args.z_freeze_clutch_threshold,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)
    h5_path = os.path.join(args.output, f"demo_im128_{timestamp}.hdf5")
    try:
        with h5py.File(h5_path, "a") as h5file:
            while not robot_env.keyboard.quit:
                print("start recording...")
                record(h5file, robot_env, args.fps, args.task_description)
                if not robot_env.keyboard.quit:
                    print("reset the environment...")
                    robot_env.reset_robot()
    finally:
        robot_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-res", "--resolution", nargs="+", type=int)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--task-description", type=str, required=True)
    parser.add_argument(
        "--disable-gripper-cmd",
        action="store_true",
        help="Do not send per-step Sigma gripper commands; record the current gripper width instead.",
    )
    parser.add_argument(
        "--z-freeze-clutch-threshold",
        type=float,
        default=None,
        help="If set, keep the current TCP z while the Logitech wheel clutch value is below this threshold.",
    )
    args = parser.parse_args()
    main(args)
