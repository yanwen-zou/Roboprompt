#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import argparse
from datetime import datetime
import json
import logging
import pathlib
import queue
import select
import sys
import termios
import threading
import time
import tty
from typing import Any

try:
    import tyro
except ImportError:
    tyro = None

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from uarm2Flexiv import REPO_ROOT, TeleopConfig, UArmFlexivTeleop  # noqa: E402
from uarm2Flexiv import add_teleop_args, teleop_config_from_namespace  # noqa: E402

OPENPI_ROOT = REPO_ROOT / "openpi"
OPENPI_SRC_ROOT = OPENPI_ROOT / "src"
OPENPI_CLIENT_ROOT = OPENPI_ROOT / "packages" / "openpi-client" / "src"
HARDWARE_ROOT = REPO_ROOT / "hardware"
for path in (REPO_ROOT, HARDWARE_ROOT, OPENPI_ROOT, OPENPI_SRC_ROOT, OPENPI_CLIENT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

@dataclasses.dataclass
class Args(TeleopConfig):
    num_episodes: int = 1
    max_episode_steps: int = 0
    output_dir: str = "output_uarm"
    task: str | None = None
    render_height: int = 224
    render_width: int = 224
    reset_before_first_episode: bool = True
    reset_between_episodes: bool = False


class EpisodeKeyListener:
    def __init__(self) -> None:
        self._events: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> "EpisodeKeyListener":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def pop_event(self) -> str | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def _loop(self) -> None:
        if not sys.stdin.isatty():
            self._line_loop()
            return
        while not self._stop_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            char = sys.stdin.read(1)
            event = self._event_from_char(char)
            if event is not None:
                self._events.put(event)

    def _line_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                line = input()
            except EOFError:
                self._stop_event.set()
                return
            text = line.strip().lower()
            if text == "":
                self._events.put("start")
            elif text in {"f", "success"}:
                self._events.put("success")
            elif text in {"d", "fail", "failure"}:
                self._events.put("failure")
            elif text in {"q", "quit"}:
                self._events.put("quit")

    @staticmethod
    def _event_from_char(char: str) -> str | None:
        if char in {"\n", "\r"}:
            return "start"
        char = char.lower()
        if char == "f":
            return "success"
        if char == "d":
            return "failure"
        if char == "q":
            return "quit"
        return None


def _write_timings(run_dir: pathlib.Path, timings: list[dict[str, Any]]) -> None:
    with open(run_dir / "uarm_episode_times.json", "w", encoding="utf-8") as f:
        json.dump({"episodes": timings}, f, indent=2)
        f.write("\n")


def _sync_env_state(env: Any) -> None:
    env._state_data = env._env.get_robot_state()


def main(args: Args) -> None:
    from examples.flexiv_real import flexiv_env as _flexiv_env
    from examples.flexiv_real import recorder as _recorder
    from examples.flexiv_real.main import _allocate_run_dir

    logging.basicConfig(level=logging.INFO, force=True)
    if args.num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {args.num_episodes}.")

    task_prompt = args.task or "uarm teleoperation"
    run_dir = _allocate_run_dir(REPO_ROOT / args.output_dir)
    metadata = {
        "control": "uarm2Flexiv",
        "task": task_prompt,
        "teleop_config": dataclasses.asdict(args),
    }
    logging.info("Recording uarm teleop episodes to %s", run_dir)

    env = _flexiv_env.FlexivRealEnv(
        reset_pose=None,
        render_height=args.render_height,
        render_width=args.render_width,
        fps=args.fps,
        prompt=task_prompt,
    )
    teleop = UArmFlexivTeleop(args, env=env)
    recorder = _recorder.LeRobotRolloutRecorder(
        output_dir=run_dir,
        fps=int(args.fps),
        server_metadata=metadata,
        default_task=task_prompt,
        render_height=args.render_height,
        render_width=args.render_width,
        steer=False,
    )
    timings: list[dict[str, Any]] = []
    episode_idx = 0
    active = False
    start_monotonic = 0.0
    start_wall = ""
    step_idx = 0

    try:
        if args.reset_before_first_episode:
            env.reset()
        else:
            _sync_env_state(env)
        teleop.reset_motion_reference()
        print("uarm teleop is live.", flush=True)
        print("Enter: start recording | f: success | d: fail | q: quit", flush=True)

        with EpisodeKeyListener() as keys:
            while episode_idx < args.num_episodes:
                event = keys.pop_event()
                if event == "quit" or env.is_episode_complete():
                    logging.info("Quit requested.")
                    break
                if not active and event == "start":
                    if episode_idx > 0 and args.reset_between_episodes:
                        env.reset()
                    teleop.reset_motion_reference()
                    recorder.on_episode_start()
                    active = True
                    start_monotonic = time.monotonic()
                    start_wall = datetime.now().isoformat(timespec="seconds")
                    step_idx = 0
                    logging.info("Episode %d/%d started.", episode_idx + 1, args.num_episodes)
                    continue

                observation = env.get_observation()
                command = teleop.next_action()
                teleop.apply_action(command)

                if not active:
                    continue

                action = teleop.record_action_from_transition(observation, command)
                recorder.on_step(observation, action)
                step_idx += 1
                end_event = event if event in {"success", "failure"} else None
                if args.max_episode_steps > 0 and step_idx >= args.max_episode_steps:
                    end_event = "max_steps"
                if end_event is None:
                    continue

                duration = time.monotonic() - start_monotonic
                outcome = "success" if end_event == "success" else "failure"
                reason = f"{end_event}_key" if end_event in {"success", "failure"} else "max_episode_steps"
                recorder.on_episode_end(outcome=outcome, end_reason=reason)
                timing = {
                    "episode_index": episode_idx,
                    "outcome": outcome,
                    "end_reason": reason,
                    "duration_sec": duration,
                    "num_steps": step_idx,
                    "start_time": start_wall,
                    "end_time": datetime.now().isoformat(timespec="seconds"),
                }
                timings.append(timing)
                _write_timings(run_dir, timings)
                logging.info(
                    "Episode %d ended as %s: %.3f sec, %d steps.",
                    episode_idx + 1,
                    outcome,
                    duration,
                    step_idx,
                )
                print(
                    f"Episode {episode_idx + 1}: {outcome}, {duration:.3f}s, {step_idx} steps. "
                    "Press Enter for next episode.",
                    flush=True,
                )
                episode_idx += 1
                active = False
    finally:
        if active:
            recorder.discard_episode(reason="interrupted")
        recorder.finalize()
        teleop.close()
        _write_timings(run_dir, timings)


def _args_from_namespace(ns: argparse.Namespace) -> Args:
    base = dataclasses.asdict(teleop_config_from_namespace(ns))
    return Args(
        **base,
        num_episodes=ns.num_episodes,
        max_episode_steps=ns.max_episode_steps,
        output_dir=ns.output_dir,
        task=ns.task,
        render_height=ns.render_height,
        render_width=ns.render_width,
        reset_before_first_episode=ns.reset_before_first_episode,
        reset_between_episodes=ns.reset_between_episodes,
    )


def _argparse_cli() -> None:
    parser = argparse.ArgumentParser()
    add_teleop_args(parser)
    parser.add_argument("--num-episodes", type=int, default=Args.num_episodes)
    parser.add_argument("--max-episode-steps", type=int, default=Args.max_episode_steps)
    parser.add_argument("--output-dir", default=Args.output_dir)
    parser.add_argument("--task", default=Args.task)
    parser.add_argument("--render-height", type=int, default=Args.render_height)
    parser.add_argument("--render-width", type=int, default=Args.render_width)
    parser.add_argument(
        "--reset-before-first-episode",
        action=argparse.BooleanOptionalAction,
        default=Args.reset_before_first_episode,
    )
    parser.add_argument(
        "--reset-between-episodes",
        action=argparse.BooleanOptionalAction,
        default=Args.reset_between_episodes,
    )
    main(_args_from_namespace(parser.parse_args()))


if __name__ == "__main__":
    if tyro is not None:
        tyro.cli(main)
    else:
        _argparse_cli()
