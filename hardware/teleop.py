from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

CSV_FIELDS = [
    "experiment",
    "success",
    "duration_sec",
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_experiment_rows(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []

    rows = []
    with log_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            experiment = row.get("experiment", "")
            if experiment in {"total", "success_rate"}:
                continue
            if not experiment:
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "success": row.get("success", ""),
                    "duration_sec": row.get("duration_sec", ""),
                }
            )
    return rows


def _next_experiment_id(log_path: Path) -> int:
    max_id = 0
    for row in _read_experiment_rows(log_path):
        try:
            max_id = max(max_id, int(row["experiment"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max_id + 1


def _write_csv_with_summary(log_path: Path, rows: list[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    valid_rows = []
    for row in rows:
        try:
            duration = float(row["duration_sec"])
        except (KeyError, TypeError, ValueError):
            duration = 0.0
        success = str(row.get("success", "")).strip()
        valid_rows.append(
            {
                "experiment": row.get("experiment", ""),
                "success": success,
                "duration_sec": f"{duration:.6f}",
            }
        )

    total_count = len(valid_rows)
    success_count = sum(1 for row in valid_rows if row["success"] == "1")
    total_duration = sum(float(row["duration_sec"]) for row in valid_rows)
    success_rate = success_count / total_count if total_count else 0.0
    avg_duration = total_duration / total_count if total_count else 0.0

    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(valid_rows)
        writer.writerow(
            {
                "experiment": "total",
                "success": f"{success_count}/{total_count}",
                "duration_sec": f"{total_duration:.6f}",
            }
        )
        writer.writerow(
            {
                "experiment": "success_rate",
                "success": f"{success_rate:.6f}",
                "duration_sec": f"avg={avg_duration:.6f}",
            }
        )


def _append_csv(log_path: Path, row: dict) -> None:
    rows = _read_experiment_rows(log_path)

    if row["success"] is None:
        success = ""
    else:
        success = "1" if row["success"] else "0"

    rows.append(
        {
            "experiment": int(row["trajectory_id"]),
            "success": success,
            "duration_sec": float(row["elapsed_sec"]),
        }
    )
    _write_csv_with_summary(log_path, rows)


class SigmaFlexivTeleop:
    def __init__(self, *, fps: float, home_on_init: bool, use_pedal: bool) -> None:
        import numpy as np
        import pygame
        from scipy.spatial.transform import Rotation as R

        from flexiv.robot import FlexivGripper, FlexivRobot
        from my_device.keyboard import Keyboard
        from my_device.logitechG29_wheel import Controller
        from my_device.sigma import Sigma7

        self._np = np
        self._pygame = pygame
        self._rotation = R
        self.fps = float(fps)
        self.robot = FlexivRobot(home=home_on_init)
        self.sigma = Sigma7()
        self.gripper = FlexivGripper(self.robot)
        self.keyboard = Keyboard()
        self.use_pedal = bool(use_pedal)
        self.last_throttle = False
        self.controller = None
        self._pygame.init()
        if self.use_pedal:
            self.controller = Controller(0)

    def close(self) -> None:
        self.keyboard.kill_listener()
        self.robot.stop()
        self.sigma.close()
        self._pygame.quit()

    def reset_robot(self) -> None:
        self.robot.send_tcp_pose(self.robot.init_pose)
        time.sleep(2.0)
        self.gripper.move(self.gripper.max_width)
        time.sleep(0.5)
        print("Reset!", flush=True)

    def reset_motion_reference(self):
        self.sigma.reset()
        self.last_throttle = False
        last_p = self.robot.init_pose[:3]
        last_r = self._rotation.from_quat(self.robot.init_pose[3:7], scalar_first=True)
        return last_p, last_r

    def reset_episode_flags(self) -> None:
        self.keyboard.start = False
        self.keyboard.finish = False
        self.keyboard.discard = False
        self.keyboard.success = False
        self.keyboard.fail = False
        self.keyboard.manual_reset = False

    def step(self, last_p, last_r):
        start_time = time.time()

        diff_p, diff_r, width = self.sigma.get_control()
        target_p = self.robot.init_pose[:3] + diff_p
        target_r = self._rotation.from_quat(self.robot.init_pose[3:7], scalar_first=True) * diff_r
        last_p = target_p
        last_r = target_r

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.keyboard.quit = True

        if self.use_pedal and self.controller is not None:
            throttle = self.controller.get_throttle()
            if throttle < -0.9:
                if not self.last_throttle:
                    self.sigma.detach()
                    self.last_throttle = True
                return False, last_p, last_r

        if self.last_throttle:
            self.last_throttle = False
            self.sigma.resume()
            last_p, last_r, _ = self.sigma.get_control()
            last_p = last_p + self.robot.init_pose[:3]
            last_r = self._rotation.from_quat(self.robot.init_pose[3:7], scalar_first=True) * last_r
            return False, last_p, last_r

        target_pose = self._np.concatenate((target_p, target_r.as_quat(scalar_first=True)), axis=0)
        self.robot.send_tcp_pose(target_pose)
        self.gripper.move_from_sigma(width)
        time.sleep(max(1.0 / self.fps - (time.time() - start_time), 0.0))
        return True, last_p, last_r


def teleop_once(
    teleop: SigmaFlexivTeleop,
    *,
    trajectory_id: int,
    task_description: str,
    fps: float,
    log_interrupted: bool,
) -> tuple[dict | None, bool]:
    teleop.reset_episode_flags()
    last_p, last_r = teleop.reset_motion_reference()

    started = False
    started_at = ""
    start_time = 0.0
    num_steps = 0

    print(
        "Ready for teleop. Press 's' to start timing, 'j' to mark success, "
        "'k' to mark failure, 'f' to finish unlabeled, 'd' to discard, 'q' to quit.",
        flush=True,
    )

    while True:
        if teleop.keyboard.manual_reset:
            print("Manual reset requested.", flush=True)
            teleop.reset_robot()
            last_p, last_r = teleop.reset_motion_reference()
            teleop.keyboard.manual_reset = False
            continue

        command_applied, last_p, last_r = teleop.step(last_p, last_r)

        if teleop.keyboard.quit:
            if started and log_interrupted:
                return _build_log_row(
                    trajectory_id=trajectory_id,
                    task_description=task_description,
                    fps=fps,
                    started_at=started_at,
                    start_time=start_time,
                    num_steps=num_steps,
                    status="interrupted",
                    success=None,
                ), True
            return None, True

        if not started:
            if not teleop.keyboard.start:
                continue
            started = True
            started_at = _now_iso()
            start_time = time.time()
            print(f"Trajectory {trajectory_id} started.", flush=True)

        if command_applied:
            num_steps += 1

        if teleop.keyboard.discard:
            elapsed_sec = time.time() - start_time
            print(f"Discarded trajectory {trajectory_id}. Elapsed: {elapsed_sec:.3f}s", flush=True)
            return None, False

        if teleop.keyboard.success:
            row = _build_log_row(
                trajectory_id=trajectory_id,
                task_description=task_description,
                fps=fps,
                started_at=started_at,
                start_time=start_time,
                num_steps=num_steps,
                status="success",
                success=True,
            )
            return row, False

        if teleop.keyboard.fail:
            row = _build_log_row(
                trajectory_id=trajectory_id,
                task_description=task_description,
                fps=fps,
                started_at=started_at,
                start_time=start_time,
                num_steps=num_steps,
                status="failure",
                success=False,
            )
            return row, False

        if teleop.keyboard.finish:
            row = _build_log_row(
                trajectory_id=trajectory_id,
                task_description=task_description,
                fps=fps,
                started_at=started_at,
                start_time=start_time,
                num_steps=num_steps,
                status="unlabeled",
                success=None,
            )
            return row, False


def _build_log_row(
    *,
    trajectory_id: int,
    task_description: str,
    fps: float,
    started_at: str,
    start_time: float,
    num_steps: int,
    status: str,
    success: bool | None,
) -> dict:
    ended_at = _now_iso()
    elapsed_sec = time.time() - start_time
    return {
        "trajectory_id": int(trajectory_id),
        "task_description": task_description,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": round(float(elapsed_sec), 6),
        "num_control_steps": int(num_steps),
        "fps": float(fps),
        "status": status,
        "success": success,
    }


def main(args: argparse.Namespace) -> None:
    log_path = Path(args.log_file)
    trajectory_id = _next_experiment_id(log_path)

    teleop = SigmaFlexivTeleop(
        fps=args.fps,
        home_on_init=args.home_on_init,
        use_pedal=not args.no_pedal,
    )
    try:
        if args.reset_on_start:
            teleop.reset_robot()

        while not teleop.keyboard.quit:
            row, should_quit = teleop_once(
                teleop,
                trajectory_id=trajectory_id,
                task_description=args.task_description,
                fps=args.fps,
                log_interrupted=args.log_interrupted,
            )
            if row is not None:
                _append_csv(log_path, row)
                print(
                    f"Logged trajectory {trajectory_id}: "
                    f"status={row['status']}, success={row['success']}, "
                    f"elapsed={row['elapsed_sec']:.3f}s",
                    flush=True,
                )
                trajectory_id += 1

            if should_quit:
                break

            if args.reset_between:
                print("Resetting robot before next trajectory.", flush=True)
                teleop.reset_robot()
            else:
                time.sleep(0.2)
    finally:
        teleop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, default=os.path.join("real_robot_data", "teleop", "result.csv"))
    parser.add_argument("--task-description", type=str, default="")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--no-home-on-init", action="store_false", dest="home_on_init", default=True)
    parser.add_argument("--no-pedal", action="store_true")
    parser.add_argument("--reset-on-start", action="store_true")
    parser.add_argument("--reset-between", action="store_true")
    parser.add_argument("--log-interrupted", action="store_true")
    main(parser.parse_args())
