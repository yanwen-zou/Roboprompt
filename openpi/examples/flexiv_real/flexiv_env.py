from __future__ import annotations

import logging
import pathlib
import sys
import time
from typing import Any

import cv2
import numpy as np
from openpi_client.runtime import environment as _environment
from scipy.spatial.transform import Rotation as R
import torch
from torchvision.transforms import CenterCrop, Compose, Resize
from torchvision.transforms import InterpolationMode
from typing_extensions import override

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
HARDWARE_ROOT = REPO_ROOT / "hardware"
for path in (REPO_ROOT, HARDWARE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flexiv.robot import FlexivGripper, FlexivRobot  # noqa: E402
from my_device.camera import CameraD400  # noqa: E402
from my_device.keyboard import Keyboard  # noqa: E402
from my_device.macros import CAM_SERIAL  # noqa: E402


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


class FlexivInferenceEnv:

    def __init__(self, camera_serial: list[str], img_shape: tuple[int, int, int], fps: float) -> None:
        self.camera_serial = list(camera_serial)
        self.fps = float(fps)
        self.robot = FlexivRobot()
        self.gripper = FlexivGripper(self.robot)
        self.keyboard = Keyboard(allowed_keys=("q",))
        self._cameras = self._init_cameras()

        _, height, width = img_shape
        self._image_processor = Compose(
            [
                Resize((height + 8, width + 8), interpolation=InterpolationMode.BICUBIC),
                CenterCrop((height, width)),
            ]
        )

    def _init_cameras(self) -> dict[str, CameraD400]:
        if len(self.camera_serial) < 2:
            raise ValueError(f"Expected two camera serials for env/wrist cameras, got {self.camera_serial!r}.")
        return {
            "env": CameraD400(self.camera_serial[0]),
            # "env": CameraD400(self.camera_serial[2]),
            "wrist": CameraD400(self.camera_serial[1]),
        }

    def reset_robot(self, reset_pose: np.ndarray | None = None) -> dict[str, Any]:
        if reset_pose is not None:
            self.robot.send_tcp_pose(np.asarray(reset_pose, dtype=np.float64)[:7])
        else:
            self.robot.send_tcp_pose(self.robot.init_pose)
        time.sleep(2.0)
        self.gripper.move(self.gripper.max_width)
        time.sleep(0.5)
        return self.get_robot_state()

    def get_robot_state(self) -> dict[str, Any]:
        tcp_pose, joint_pos, _, _ = self.robot.get_robot_state()
        env_rgb, wrist_rgb = self._get_camera_frames()
        return {
            "tcp_pose": np.asarray(tcp_pose, dtype=np.float64),
            "joint_pos": np.asarray(joint_pos, dtype=np.float64),
            "policy_env_img": self._process_image(env_rgb),
            "policy_wrist_img": self._process_image(wrist_rgb),
            "env_img_raw": env_rgb.copy(),
            "wrist_img_raw": wrist_rgb.copy(),
        }

    def deploy_action(self, tcp_action: np.ndarray, gripper_action: float) -> None:
        self.robot.send_tcp_pose(np.asarray(tcp_action, dtype=np.float64))
        self.gripper.move(float(gripper_action))
        time.sleep(1.0 / self.fps)

    def _get_camera_frames(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            env_bgr, _ = self._cameras["env"].get_data()
            wrist_bgr, _ = self._cameras["wrist"].get_data()
        except Exception as exc:
            raise RuntimeError(f"Failed to read images from cameras: {exc}") from exc
        env_rgb = cv2.cvtColor(env_bgr, cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
        return env_rgb, wrist_rgb

    def _process_image(self, image_rgb: np.ndarray) -> np.ndarray:
        image = self._image_processor(torch.from_numpy(image_rgb.copy()).permute(2, 0, 1))
        return np.asarray(image, dtype=np.uint8)


class FlexivRealEnv(_environment.Environment):
    """Runtime environment wrapper for Flexiv real-world evaluation."""

    def __init__(
        self,
        reset_pose: list[float] | np.ndarray | None = None,
        render_height: int = 224,
        render_width: int = 224,
        fps: float = 5.0,
        prompt: str | None = None,
    ) -> None:
        self._env = FlexivInferenceEnv(
            camera_serial=list(CAM_SERIAL),
            img_shape=(3, render_height, render_width),
            fps=fps,
        )
        self._reset_pose = None if reset_pose is None else np.asarray(reset_pose, dtype=np.float64)
        self._prompt = prompt
        self._state_data: dict[str, Any] | None = None

    @override
    def reset(self) -> None:
        self._clear_keyboard_flags()
        self._state_data = self._env.reset_robot(self._reset_pose)
        logging.info("Episode reset complete. Use `p` in the rollout window for prompts and `q` to quit.")

    @override
    def is_episode_complete(self) -> bool:
        keyboard = self._env.keyboard
        return bool(keyboard.quit)

    @override
    def get_observation(self) -> dict:
        if self._state_data is None:
            raise RuntimeError("State is not set. Call reset() first.")

        obs = {
            "observation/state": self._build_state(self._state_data),
            "observation/image": self._get_policy_image(self._state_data["policy_env_img"]),
            "observation/wrist_image": self._get_policy_image(self._state_data["policy_wrist_img"]),
        }
        if self._prompt is not None:
            obs["prompt"] = self._prompt
        return obs

    @override
    def apply_action(self, action: dict) -> None:
        raw_action = np.array(action["actions"], dtype=np.float64, copy=True)
        if raw_action.shape != (7,):
            raise ValueError(f"Expected Flexiv action with shape (7,), got {raw_action.shape}.")
        if not np.all(np.isfinite(raw_action)):
            raise ValueError(f"Action contains non-finite values: {raw_action}")

        if self._state_data is None:
            raise RuntimeError("State is not set. Call reset() first.")

        if "target_pose_wxyz" in action:
            target_pose = np.asarray(action["target_pose_wxyz"], dtype=np.float64)
            if target_pose.shape != (7,):
                raise ValueError(f"Expected target_pose_wxyz with shape (7,), got {target_pose.shape}.")
            if not np.all(np.isfinite(target_pose)):
                raise ValueError(f"target_pose_wxyz contains non-finite values: {target_pose}")
        else:
            tcp_pose = np.asarray(self._state_data["tcp_pose"], dtype=np.float64)
            target_pose = self._compose_target_pose(tcp_pose, raw_action)
        gripper_action = float(np.clip(raw_action[6], 0.0, self._env.gripper.max_width))

        self._env.deploy_action(target_pose, gripper_action)
        self._state_data = self._env.get_robot_state()

    def _clear_keyboard_flags(self) -> None:
        keyboard = self._env.keyboard
        keyboard.start = False
        keyboard.finish = False
        keyboard.discard = False
        keyboard.success = False
        keyboard.fail = False
        keyboard.quit = False
        keyboard.manual_reset = False

    def _build_state(self, state_data: dict[str, Any]) -> np.ndarray:
        tcp_pose = np.asarray(state_data["tcp_pose"], dtype=np.float64)
        gripper_state = float(self._env.gripper.get_gripper_state())
        return np.concatenate(
            [
                tcp_pose[:3],
                _quat_wxyz_to_xyzw(tcp_pose[3:7]),
                np.asarray([gripper_state], dtype=np.float64),
            ],
            axis=0,
        )

    def _get_policy_image(self, image: Any) -> np.ndarray:
        if image is None:
            raise RuntimeError("Camera frame is unavailable.")
        if hasattr(image, "detach"):
            image = image.detach()
        if hasattr(image, "cpu"):
            image = image.cpu()
        return np.asarray(image, dtype=np.uint8)

    def get_latest_env_image(self, *, raw: bool = True) -> np.ndarray:
        if self._state_data is None:
            raise RuntimeError("State is not set. Call reset() first.")
        key = "env_img_raw" if raw else "policy_env_img"
        image = self._state_data.get(key)
        if image is None:
            raise RuntimeError(f"Camera frame '{key}' is unavailable.")
        return self._get_policy_image(image).copy()

    def _compose_target_pose(self, tcp_pose_wxyz: np.ndarray, action: np.ndarray) -> np.ndarray:
        delta_pos = np.array(action[:3], dtype=np.float64, copy=True)
        delta_rotvec = np.array(action[3:6], dtype=np.float64, copy=True)

        current_pos = np.array(tcp_pose_wxyz[:3], dtype=np.float64, copy=True)
        current_rot = R.from_quat(_quat_wxyz_to_xyzw(tcp_pose_wxyz[3:7]))
        target_pos = current_pos + delta_pos
        target_rot = current_rot * R.from_rotvec(delta_rotvec)

        return np.concatenate(
            [
                target_pos.astype(np.float64),
                _quat_xyzw_to_wxyz(target_rot.as_quat()),
            ],
            axis=0,
        )


def add_observation_target_poses(result: dict, observation_state: np.ndarray) -> dict:
    """Attach absolute target poses for actions trained as observation-to-target deltas."""
    if "actions" not in result:
        return result

    actions = np.asarray(result["actions"], dtype=np.float64)
    if actions.ndim == 1:
        actions_2d = actions[None, :]
        squeeze = True
    elif actions.ndim == 2:
        actions_2d = actions
        squeeze = False
    else:
        raise ValueError(f"Expected actions with shape [D] or [T, D], got {actions.shape}.")
    if actions_2d.shape[-1] < 6:
        raise ValueError(f"Expected action dim >= 6, got {actions_2d.shape}.")

    state = np.asarray(observation_state, dtype=np.float64)
    if state.shape[0] < 7:
        raise ValueError(f"Expected observation state dim >= 7, got {state.shape}.")

    anchor_pos = state[:3]
    anchor_rot = R.from_quat(state[3:7])
    target_poses = []
    for raw_action in actions_2d:
        target_pos = anchor_pos + raw_action[:3]
        target_rot = anchor_rot * R.from_rotvec(raw_action[3:6])
        target_poses.append(
            np.concatenate(
                [
                    target_pos.astype(np.float64),
                    _quat_xyzw_to_wxyz(target_rot.as_quat()),
                ],
                axis=0,
            )
        )

    updated = dict(result)
    updated["target_pose_wxyz"] = np.stack(target_poses, axis=0)
    if squeeze:
        updated["target_pose_wxyz"] = updated["target_pose_wxyz"][0]
    return updated
