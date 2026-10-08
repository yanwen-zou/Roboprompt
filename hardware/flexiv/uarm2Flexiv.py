#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import argparse
import logging
import pathlib
import re
import sys
import threading
import time
from typing import Any

import numpy as np

try:
    import tyro
except ImportError:
    tyro = None


def _find_repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "hardware").is_dir() and (parent / "openpi").is_dir():
            return parent
    raise RuntimeError("Cannot find Roboprompt repo root from this script path.")


REPO_ROOT = _find_repo_root()
HARDWARE_ROOT = REPO_ROOT / "hardware"
OPENPI_ROOT = REPO_ROOT / "openpi"
OPENPI_SRC_ROOT = OPENPI_ROOT / "src"
OPENPI_CLIENT_ROOT = OPENPI_ROOT / "packages" / "openpi-client" / "src"
for path in (REPO_ROOT, HARDWARE_ROOT, OPENPI_ROOT, OPENPI_SRC_ROOT, OPENPI_CLIENT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _rotation_cls() -> Any:
    from scipy.spatial.transform import Rotation

    return Rotation


@dataclasses.dataclass
class TeleopConfig:
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    read_hz: float = 100.0
    fps: float = 10.0

    # The current uarm hardware has 8 servos: 0..6 map to Flexiv's 7 joints,
    # and 7 maps to the gripper. Keep cartesian_delta available for older rigs.
    control_mode: str = "joint"
    joint_channels: tuple[int, int, int, int, int, int, int] = (0, 1, 2, 3, 4, 5, 6)
    joint_scale: tuple[float, float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    joint_sign: tuple[float, float, float, float, float, float, float] = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    max_joint_step: float = 0.08

    # Cartesian delta mode channels. Only used when --control-mode cartesian_delta.
    xyz_channels: tuple[int, int, int] = (0, 1, 2)
    rot_channels: tuple[int, int, int] = (3, 4, 5)
    gripper_channel: int = 7

    # Relative mode scales angle changes between control ticks into Flexiv
    # action deltas. Tune these first on a slow, clear workspace.
    xyz_scale: tuple[float, float, float] = (0.0006, 0.0006, 0.0006)
    rot_scale: tuple[float, float, float] = (0.006, 0.006, 0.006)
    xyz_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rot_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)

    max_xyz_step: float = 0.012
    max_rot_step: float = 0.08
    gripper_min_deg: float = -20.0
    gripper_max_deg: float = 0
    invert_gripper: bool = False

    home_on_init: bool = True
    reset_on_start: bool = False


class UArmServoReader:
    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        servo_ids: tuple[int, ...] = tuple(range(7)),
        timeout: float = 0.1,
    ) -> None:
        self._port = port
        self._baudrate = int(baudrate)
        self._servo_ids = tuple(int(idx) for idx in servo_ids)
        import serial

        self._ser = serial.Serial(self._port, self._baudrate, timeout=timeout)
        size = max(self._servo_ids) + 1
        self._zero_angles = np.zeros(size, dtype=np.float64)
        self._angles = np.zeros(size, dtype=np.float64)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        logging.info("Opened uarm serial port %s at %d baud.", self._port, self._baudrate)
        self._init_servos()

    def close(self) -> None:
        self.stop()
        self._ser.close()

    def start(self, *, hz: float) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, kwargs={"hz": hz}, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_angles(self) -> np.ndarray:
        with self._lock:
            return self._angles.copy()

    def _send_command(self, cmd: str) -> str:
        self._ser.write(cmd.encode("ascii"))
        time.sleep(0.008)
        return self._ser.read_all().decode("ascii", errors="ignore")

    @staticmethod
    def _pwm_to_angle(response: str, servo_num: int, *, pwm_min: int = 500, pwm_max: int = 2500) -> float | None:
        match = re.search(f"#{servo_num:03d}P(\\d{{4}})", response)
        if not match:
            return None
        pwm_val = int(match.group(1))
        return (pwm_val - pwm_min) / float(pwm_max - pwm_min) * 270.0

    def _read_servo_angle(self, servo_id: int) -> float | None:
        response = self._send_command(f"#{servo_id:03d}PRAD!")
        return self._pwm_to_angle(response.strip(), servo_id)

    def _init_servos(self) -> None:
        self._send_command("#000PVER!")
        for servo_id in self._servo_ids:
            self._send_command("#000PCSK!")
            self._send_command(f"#{servo_id:03d}PULK!")
            angle = self._read_servo_angle(servo_id)
            self._zero_angles[servo_id] = 0.0 if angle is None else angle
        logging.info("uarm zero calibration complete: %s", np.round(self._zero_angles, 3).tolist())

    def _read_loop(self, *, hz: float) -> None:
        dt = 1.0 / max(float(hz), 1e-6)
        last_valid = np.zeros_like(self._angles)
        while not self._stop_event.is_set():
            new_angles = last_valid.copy()
            for servo_id in self._servo_ids:
                angle = self._read_servo_angle(servo_id)
                if angle is None:
                    logging.debug("No uarm response for servo %d.", servo_id)
                    continue
                new_angles[servo_id] = angle - self._zero_angles[servo_id]
            with self._lock:
                self._angles = new_angles.copy()
            last_valid = new_angles
            time.sleep(dt)


class DirectFlexivBackend:
    def __init__(self, *, fps: float, home_on_init: bool) -> None:
        from flexiv.robot import FlexivGripper, FlexivRobot

        self._fps = float(fps)
        self.robot = FlexivRobot(home=home_on_init)
        self.gripper = FlexivGripper(self.robot)

    def reset(self) -> None:
        self.robot.send_tcp_pose(self.robot.init_pose)
        time.sleep(2.0)
        self.gripper.move(self.gripper.max_width)
        time.sleep(0.5)

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray(self.robot.get_joint_pos(), dtype=np.float64)

    def apply_action(self, action: dict[str, np.ndarray]) -> None:
        if "_joint_target" in action:
            self.robot.send_joint_pose(np.asarray(action["_joint_target"], dtype=np.float64))
            self.gripper.move(float(action["_gripper_width"]))
            time.sleep(1.0 / max(self._fps, 1e-6))
            return

        raw_action = np.asarray(action["actions"], dtype=np.float64)
        tcp_pose = np.asarray(self.robot.get_tcp_pose(), dtype=np.float64)
        target_pose = self._compose_target_pose(tcp_pose, raw_action)
        self.robot.send_tcp_pose(target_pose)
        self.gripper.move(float(raw_action[6]))
        time.sleep(1.0 / max(self._fps, 1e-6))

    @staticmethod
    def _compose_target_pose(tcp_pose_wxyz: np.ndarray, action: np.ndarray) -> np.ndarray:
        rotation = _rotation_cls()
        current_rot = rotation.from_quat(_quat_wxyz_to_xyzw(tcp_pose_wxyz[3:7]))
        target_rot = current_rot * rotation.from_rotvec(np.asarray(action[3:6], dtype=np.float64))
        return np.concatenate(
            [
                np.asarray(tcp_pose_wxyz[:3], dtype=np.float64) + np.asarray(action[:3], dtype=np.float64),
                _quat_xyzw_to_wxyz(target_rot.as_quat()),
            ]
        )


class UArmFlexivTeleop:
    def __init__(self, config: TeleopConfig, *, env: Any | None = None) -> None:
        self.config = config
        if config.control_mode not in {"joint", "cartesian_delta"}:
            raise ValueError(
                f"Unsupported control_mode {config.control_mode!r}; "
                "expected 'joint' or 'cartesian_delta'."
            )
        self.reader = UArmServoReader(
            port=config.serial_port,
            baudrate=config.baudrate,
            servo_ids=self._servo_ids(config),
        )
        self.reader.start(hz=config.read_hz)
        self.backend = env if env is not None else DirectFlexivBackend(fps=config.fps, home_on_init=config.home_on_init)
        self._prev_angles: np.ndarray | None = None
        self._anchor_angles: np.ndarray | None = None
        self._anchor_joint_pos: np.ndarray | None = None
        self._last_joint_cmd: np.ndarray | None = None

    def close(self) -> None:
        self.reader.close()

    def reset_motion_reference(self) -> None:
        angles = self.reader.get_angles()
        self._prev_angles = angles.copy()
        self._anchor_angles = angles.copy()
        self._anchor_joint_pos = self._get_backend_joint_pos()
        self._last_joint_cmd = self._anchor_joint_pos.copy()

    def reset_backend(self) -> None:
        if hasattr(self.backend, "reset"):
            self.backend.reset()
        self.reset_motion_reference()

    def next_action(self) -> dict[str, np.ndarray]:
        if self.config.control_mode == "joint":
            return self._next_joint_action()
        return self._next_cartesian_delta_action()

    def step(self) -> dict[str, np.ndarray]:
        action = self.next_action()
        self.apply_action(action)
        return action

    def apply_action(self, action: dict[str, np.ndarray]) -> None:
        if "_joint_target" in action:
            self._apply_joint_action(action)
            return
        self.backend.apply_action(action)

    def record_action_from_transition(self, before_observation: dict, command: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.config.control_mode != "joint":
            return {"actions": np.asarray(command["actions"], dtype=np.float64)}
        if not hasattr(self.backend, "get_observation"):
            return {"actions": np.asarray(command["actions"], dtype=np.float64)}

        after_observation = self.backend.get_observation()
        before_state = np.asarray(before_observation["observation/state"], dtype=np.float64)
        after_state = np.asarray(after_observation["observation/state"], dtype=np.float64)
        rotation = _rotation_cls()
        before_rot = rotation.from_quat(before_state[3:7])
        after_rot = rotation.from_quat(after_state[3:7])
        delta_rotvec = (before_rot.inv() * after_rot).as_rotvec()
        action = np.concatenate(
            [
                after_state[:3] - before_state[:3],
                delta_rotvec,
                np.asarray([float(command["_gripper_width"])], dtype=np.float64),
            ]
        )
        return {"actions": action.astype(np.float64)}

    @staticmethod
    def _servo_ids(config: TeleopConfig) -> tuple[int, ...]:
        if config.control_mode == "joint":
            ids = config.joint_channels + (config.gripper_channel,)
        else:
            ids = config.xyz_channels + config.rot_channels + (config.gripper_channel,)
        return tuple(sorted(set(int(idx) for idx in ids)))

    def _next_cartesian_delta_action(self) -> dict[str, np.ndarray]:
        angles = self.reader.get_angles()
        if self._prev_angles is None:
            self._prev_angles = angles.copy()
        delta_angles = angles - self._prev_angles
        self._prev_angles = angles.copy()

        xyz = delta_angles[list(self.config.xyz_channels)]
        rot = delta_angles[list(self.config.rot_channels)]
        xyz_delta = xyz * np.asarray(self.config.xyz_scale) * np.asarray(self.config.xyz_sign)
        rot_delta = rot * np.asarray(self.config.rot_scale) * np.asarray(self.config.rot_sign)
        xyz_delta = np.clip(xyz_delta, -self.config.max_xyz_step, self.config.max_xyz_step)
        rot_delta = np.clip(rot_delta, -self.config.max_rot_step, self.config.max_rot_step)
        gripper_width = self._map_gripper_width(float(angles[self.config.gripper_channel]))

        action = np.concatenate([xyz_delta, rot_delta, np.asarray([gripper_width], dtype=np.float64)])
        return {"actions": action.astype(np.float64)}

    def _next_joint_action(self) -> dict[str, np.ndarray]:
        angles = self.reader.get_angles()
        if self._anchor_angles is None or self._anchor_joint_pos is None or self._last_joint_cmd is None:
            self.reset_motion_reference()
        assert self._anchor_angles is not None
        assert self._anchor_joint_pos is not None
        assert self._last_joint_cmd is not None

        joint_angles = angles[list(self.config.joint_channels)]
        anchor_angles = self._anchor_angles[list(self.config.joint_channels)]
        desired_joint = self._anchor_joint_pos + np.deg2rad(joint_angles - anchor_angles) * np.asarray(
            self.config.joint_scale,
            dtype=np.float64,
        ) * np.asarray(self.config.joint_sign, dtype=np.float64)
        delta = np.clip(
            desired_joint - self._last_joint_cmd,
            -float(self.config.max_joint_step),
            float(self.config.max_joint_step),
        )
        joint_target = self._last_joint_cmd + delta
        self._last_joint_cmd = joint_target.copy()

        gripper_width = self._map_gripper_width(float(angles[self.config.gripper_channel]))
        record_action = np.concatenate(
            [
                np.zeros(6, dtype=np.float64),
                np.asarray([gripper_width], dtype=np.float64),
            ]
        )
        return {
            "actions": record_action,
            "_joint_target": joint_target.astype(np.float64),
            "_gripper_width": np.asarray(gripper_width, dtype=np.float64),
        }

    def _get_backend_joint_pos(self) -> np.ndarray:
        if hasattr(self.backend, "get_joint_pos"):
            return np.asarray(self.backend.get_joint_pos(), dtype=np.float64)
        backend_env = getattr(self.backend, "_env", None)
        if backend_env is not None and hasattr(backend_env, "robot"):
            return np.asarray(backend_env.robot.get_joint_pos(), dtype=np.float64)
        raise RuntimeError("Backend does not expose Flexiv joint position.")

    def _apply_joint_action(self, action: dict[str, np.ndarray]) -> None:
        joint_target = np.asarray(action["_joint_target"], dtype=np.float64)
        gripper_width = float(action["_gripper_width"])
        backend_env = getattr(self.backend, "_env", None)
        if backend_env is not None and hasattr(backend_env, "robot"):
            backend_env.robot.send_joint_pose(joint_target)
            backend_env.gripper.move(gripper_width)
            time.sleep(1.0 / max(float(self.config.fps), 1e-6))
            self.backend._state_data = backend_env.get_robot_state()
            return
        self.backend.apply_action(action)

    def _map_gripper_width(self, angle_deg: float) -> float:
        low = float(self.config.gripper_min_deg)
        high = float(self.config.gripper_max_deg)
        ratio = (angle_deg - low) / max(high - low, 1e-6)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if self.config.invert_gripper:
            ratio = 1.0 - ratio
        backend_env = getattr(self.backend, "_env", None)
        gripper = getattr(backend_env, "gripper", None)
        if gripper is None:
            gripper = getattr(self.backend, "gripper", None)
        max_width = float(getattr(gripper, "max_width", 0.085))
        return ratio * max_width


def main(config: TeleopConfig) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    teleop = UArmFlexivTeleop(config)
    try:
        if config.reset_on_start:
            teleop.reset_backend()
        else:
            teleop.reset_motion_reference()
        print("uarm -> Flexiv teleop started. Press Ctrl+C to stop.", flush=True)
        while True:
            teleop.step()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        teleop.close()


def add_teleop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--serial-port", default=TeleopConfig.serial_port)
    parser.add_argument("--baudrate", type=int, default=TeleopConfig.baudrate)
    parser.add_argument("--read-hz", type=float, default=TeleopConfig.read_hz)
    parser.add_argument("--fps", type=float, default=TeleopConfig.fps)
    parser.add_argument("--control-mode", choices=("joint", "cartesian_delta"), default=TeleopConfig.control_mode)
    parser.add_argument("--joint-channels", type=int, nargs=7, default=list(TeleopConfig.joint_channels))
    parser.add_argument("--joint-scale", type=float, nargs=7, default=list(TeleopConfig.joint_scale))
    parser.add_argument("--joint-sign", type=float, nargs=7, default=list(TeleopConfig.joint_sign))
    parser.add_argument("--max-joint-step", type=float, default=TeleopConfig.max_joint_step)
    parser.add_argument("--xyz-channels", type=int, nargs=3, default=list(TeleopConfig.xyz_channels))
    parser.add_argument("--rot-channels", type=int, nargs=3, default=list(TeleopConfig.rot_channels))
    parser.add_argument("--gripper-channel", type=int, default=TeleopConfig.gripper_channel)
    parser.add_argument("--xyz-scale", type=float, nargs=3, default=list(TeleopConfig.xyz_scale))
    parser.add_argument("--rot-scale", type=float, nargs=3, default=list(TeleopConfig.rot_scale))
    parser.add_argument("--xyz-sign", type=float, nargs=3, default=list(TeleopConfig.xyz_sign))
    parser.add_argument("--rot-sign", type=float, nargs=3, default=list(TeleopConfig.rot_sign))
    parser.add_argument("--max-xyz-step", type=float, default=TeleopConfig.max_xyz_step)
    parser.add_argument("--max-rot-step", type=float, default=TeleopConfig.max_rot_step)
    parser.add_argument("--gripper-min-deg", type=float, default=TeleopConfig.gripper_min_deg)
    parser.add_argument("--gripper-max-deg", type=float, default=TeleopConfig.gripper_max_deg)
    parser.add_argument("--invert-gripper", action=argparse.BooleanOptionalAction, default=TeleopConfig.invert_gripper)
    parser.add_argument("--home-on-init", action=argparse.BooleanOptionalAction, default=TeleopConfig.home_on_init)
    parser.add_argument("--reset-on-start", action=argparse.BooleanOptionalAction, default=TeleopConfig.reset_on_start)


def teleop_config_from_namespace(ns: argparse.Namespace) -> TeleopConfig:
    return TeleopConfig(
        serial_port=ns.serial_port,
        baudrate=ns.baudrate,
        read_hz=ns.read_hz,
        fps=ns.fps,
        control_mode=ns.control_mode,
        joint_channels=tuple(ns.joint_channels),
        joint_scale=tuple(ns.joint_scale),
        joint_sign=tuple(ns.joint_sign),
        max_joint_step=ns.max_joint_step,
        xyz_channels=tuple(ns.xyz_channels),
        rot_channels=tuple(ns.rot_channels),
        gripper_channel=ns.gripper_channel,
        xyz_scale=tuple(ns.xyz_scale),
        rot_scale=tuple(ns.rot_scale),
        xyz_sign=tuple(ns.xyz_sign),
        rot_sign=tuple(ns.rot_sign),
        max_xyz_step=ns.max_xyz_step,
        max_rot_step=ns.max_rot_step,
        gripper_min_deg=ns.gripper_min_deg,
        gripper_max_deg=ns.gripper_max_deg,
        invert_gripper=ns.invert_gripper,
        home_on_init=ns.home_on_init,
        reset_on_start=ns.reset_on_start,
    )


def _argparse_cli() -> None:
    parser = argparse.ArgumentParser()
    add_teleop_args(parser)
    main(teleop_config_from_namespace(parser.parse_args()))


if __name__ == "__main__":
    if tyro is not None:
        tyro.cli(main)
    else:
        _argparse_cli()
