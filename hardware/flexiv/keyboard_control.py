import os
import select
import sys
import termios
import time
import tty
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flexiv.robot import FlexivRobot


def build_keymap(step: float) -> Dict[str, Tuple[int, float]]:
    """Create a key mapping from keyboard input to joint deltas."""
    return {
        "q": (0, step),
        "a": (0, -step),
        "w": (1, step),
        "s": (1, -step),
        "e": (2, step),
        "d": (2, -step),
        "r": (3, step),
        "f": (3, -step),
        "t": (4, step),
        "g": (4, -step),
        "y": (5, step),
        "h": (5, -step),
        "u": (6, step),
        "j": (6, -step),
    }


def print_operation_guide(step: float) -> None:
    print("Flexiv keyboard control")
    print("Press one key at a time in this terminal or focused OpenCV window.")
    print("ESC: exit")
    print("c: capture calibration frame")
    print("+: increase joint step")
    print("-: decrease joint step")
    print("q/a: joint 1 +/-")
    print("w/s: joint 2 +/-")
    print("e/d: joint 3 +/-")
    print("r/f: joint 4 +/-")
    print("t/g: joint 5 +/-")
    print("y/h: joint 6 +/-")
    print("u/j: joint 7 +/-")
    print(f"Current step: {step:.3f} rad")


def read_key(timeout: float = 0.1) -> str:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return ""
    return sys.stdin.read(1)


def main() -> None:
    robot = FlexivRobot(home=False)
    robot.switch_mode("joint")

    try:
        current_joints = robot.get_joint_pos().astype(float)
    except Exception as exc:
        raise RuntimeError("Failed to read current joint positions.") from exc

    step = 0.01  # radians
    keymap = build_keymap(step)
    print_operation_guide(step)

    stdin_fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(stdin_fd)
    tty.setcbreak(stdin_fd)

    try:
        while True:
            key = read_key()
            if not key:
                continue
            if key == "\x1b":
                break
            if key in ("+", "="):
                step = min(0.2, step + 0.005)
                keymap = build_keymap(step)
                continue
            if key in ("-", "_"):
                step = max(0.001, step - 0.005)
                keymap = build_keymap(step)
                continue
            if key not in keymap:
                continue

            joint_idx, delta = keymap[key]
            current_joints[joint_idx] += delta
            robot.send_joint_pose(current_joints.tolist())
            time.sleep(0.02)
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
        robot.stop()


if __name__ == "__main__":
    main()
