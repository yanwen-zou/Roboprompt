import os

# Must set before importing robosuite/robocasa.
os.environ["MUJOCO_GL"] = "egl"

import collections
import dataclasses
from datetime import datetime
import logging
import pathlib
import re
import sys

import cv2
import gymnasium as gym
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
import tqdm
import tyro

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, EXAMPLES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main_utils import overlay_predicted_action_chunk, render_rollout_window
from rollout_recording import (
    append_rollout_cache_step,
    complete_rollout_cache_step,
    finalize_rollout_recordings,
    start_rollout_episode_cache,
    start_rollout_recording,
    write_episode_stats,
    write_rollout_episode_cache,
)
from utils import apply_action_perturbation
from utils import apply_smooth_action_perturbation

ACTION_PERTURBATION_SCALE = 0.5
ACTION_PERTURBATION_CONTROL_POINTS = 4
ACTION_PERTURBATION_EVERY_N_CHUNKS = 5
PERTURBED_REPLAN_STEPS = 25
ACTION_PERTURBATION_BLEND_IN_STEPS = 5
ACTION_PERTURBATION_BLEND_OUT_STEPS = 5


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized or "unknown_policy"


def _get_task_set_label(task_sets: list[str] | None) -> str:
    if not task_sets:
        return "unknown_tasks"
    return _sanitize_path_component("-".join(task_sets))


def _get_policy_run_name(checkpoint_info: dict[str, str] | None, task_sets: list[str] | None, now: datetime) -> str:
    if checkpoint_info is None:
        policy_label = "unknown_policy"
    else:
        policy_label = (
            checkpoint_info.get("config")
            or checkpoint_info.get("label")
            or pathlib.Path(checkpoint_info.get("dir", "")).name
            or "unknown_policy"
        )
    return f"{_sanitize_path_component(policy_label)}_{_get_task_set_label(task_sets)}_{now.strftime('%Y-%m-%d-%H-%M')}"


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    onscreen: bool = False
    action_perturbation: bool = True
    overlay_action_scale: float = 0.05
    overlay_camera_name: str = "robot0_agentview_left"

    split: str = "pretrain"
    num_trials: int = 50
    task_set: list = None

    checkpoint: str | None = None
    seed: int = 7
    convert_after: bool = True


def _get_checkpoint_info(client, checkpoint_override: str | None) -> dict[str, str] | None:
    checkpoint_info: dict[str, str] = {}
    if checkpoint_override:
        checkpoint_info["label"] = checkpoint_override

    try:
        server_metadata = client.get_server_metadata()
    except Exception as exc:
        logging.warning("Could not fetch server metadata: %s", exc)
        server_metadata = None

    if isinstance(server_metadata, dict):
        checkpoint_dir = server_metadata.get("checkpoint_dir")
        checkpoint_config = server_metadata.get("checkpoint_config")
        if checkpoint_dir:
            checkpoint_info["dir"] = checkpoint_dir
        if checkpoint_config:
            checkpoint_info["config"] = checkpoint_config

    return checkpoint_info or None


def record_main(args: Args) -> None:
    if args.task_set is None:
        raise ValueError("--args.task-set must be provided.")
    rng = np.random.default_rng(args.seed)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    checkpoint_info = _get_checkpoint_info(client, args.checkpoint)
    run_name = _get_policy_run_name(checkpoint_info, args.task_set, datetime.now())

    all_env_names = []
    for task in args.task_set:
        all_env_names.extend(TASK_SET_REGISTRY[task])

    recordings = []
    for env_name in all_env_names:
        recordings.append(
            record_env(
                env_name=env_name,
                split=args.split,
                num_trials=args.num_trials,
                resize_size=args.resize_size,
                replan_steps=args.replan_steps,
                host=args.host,
                port=args.port,
                seed=args.seed,
                onscreen=args.onscreen,
                action_perturbation=args.action_perturbation,
                overlay_action_scale=args.overlay_action_scale,
                overlay_camera_name=args.overlay_camera_name,
                run_name=run_name,
                rng=rng,
            )
        )

    if args.convert_after:
        converted = finalize_rollout_recordings(recordings)
        for item in converted:
            logging.info("[main-record] Converted %s", item["lerobot_path"])


def record_env(
    *,
    env_name: str,
    split: str,
    num_trials: int,
    resize_size: int,
    replan_steps: int,
    host: str,
    port: int,
    seed: int,
    onscreen: bool,
    action_perturbation: bool,
    overlay_action_scale: float,
    overlay_camera_name: str,
    run_name: str,
    rng: np.random.Generator,
) -> dict[str, str]:
    if split not in ["pretrain", "target"]:
        raise ValueError(f"Unsupported split: {split}")
    if ACTION_PERTURBATION_EVERY_N_CHUNKS < 1:
        raise ValueError(
            "ACTION_PERTURBATION_EVERY_N_CHUNKS must be at least 1, "
            f"got {ACTION_PERTURBATION_EVERY_N_CHUNKS}."
        )
    if PERTURBED_REPLAN_STEPS < 1:
        raise ValueError(f"PERTURBED_REPLAN_STEPS must be at least 1, got {PERTURBED_REPLAN_STEPS}.")
    horizon = int(get_task_horizon(env_name) * 1.5)
    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    env = gym.make(f"robocasa/{env_name}", split=split, seed=seed)
    recording = start_rollout_recording(
        env.unwrapped,
        env_name=env_name,
        split=split,
        seed=seed,
        run_name=run_name,
    )

    for episode_idx in tqdm.tqdm(range(num_trials), desc=env_name):
        obs, _ = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()
        latest_model_action_chunk = np.zeros((0, 12), dtype=np.float32)
        latest_model_action_index = 0
        model_action_chunk_count = 0
        rollout_cache = start_rollout_episode_cache()
        t = 0
        done = False
        aborted = False

        while t < horizon:
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
            )

            if not action_plan:
                state = np.concatenate(
                    (
                        obs["state.end_effector_position_relative"],
                        obs["state.end_effector_rotation_relative"],
                        obs["state.base_position"],
                        obs["state.base_rotation"],
                        obs["state.gripper_qpos"],
                    ),
                    axis=0,
                )
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": task_lang,
                }

                action_chunk = np.asarray(client.infer(element)["actions"], dtype=np.float32)
                model_action_chunk_count += 1
                should_perturb = model_action_chunk_count % ACTION_PERTURBATION_EVERY_N_CHUNKS == 0
                should_perturb = action_perturbation and should_perturb
                current_replan_steps = PERTURBED_REPLAN_STEPS if should_perturb else replan_steps
                if len(action_chunk) < current_replan_steps:
                    raise ValueError(
                        f"We want to replan every {current_replan_steps} steps, "
                        f"but policy only predicts {len(action_chunk)} steps."
                    )
                action_chunk = action_chunk[:current_replan_steps]
                prev_tail_action = None
                if len(latest_model_action_chunk) > 0:
                    prev_tail_action = latest_model_action_chunk[-1]
                if should_perturb:
                    action_chunk = apply_smooth_action_perturbation(
                        action_chunk,
                        rng=rng,
                        scale=ACTION_PERTURBATION_SCALE,
                        control_points=ACTION_PERTURBATION_CONTROL_POINTS,
                        prev_tail_action=prev_tail_action,
                        blend_in_steps=ACTION_PERTURBATION_BLEND_IN_STEPS,
                        blend_out_steps=ACTION_PERTURBATION_BLEND_OUT_STEPS,
                    )
                latest_model_action_chunk = action_chunk
                latest_model_action_index = 0
                action_plan.extend(action_chunk)

            action_chunk_for_overlay = latest_model_action_chunk[latest_model_action_index:]
            action = action_plan.popleft()
            if onscreen:
                display_image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                display_image_bgr = overlay_predicted_action_chunk(
                    image_bgr=display_image_bgr,
                    env=env.unwrapped,
                    obs=obs,
                    action_chunk=action_chunk_for_overlay,
                    camera_name=overlay_camera_name,
                    action_scale=overlay_action_scale,
                )
                render_rollout_window(display_image_bgr, env_name, episode_idx, t)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logging.info("Aborting current episode from rollout window.")
                    aborted = True
                    break
            # Keep recording data clean: overlay is only drawn into the onscreen display frame above.
            append_rollout_cache_step(rollout_cache, obs, action)
            obs, reward, done, _, info = env.step(convert_action(action))
            done = info["success"]
            complete_rollout_cache_step(rollout_cache, reward=reward, done=done)
            latest_model_action_index += 1
            if done:
                break
            t += 1

        write_episode_stats(
            env.unwrapped,
            success=done,
            aborted=aborted,
            episode_idx=episode_idx,
            num_steps=t + 1,
        )
        write_rollout_episode_cache(env.unwrapped, rollout_cache)
        logging.info("[%s] recorded episode %d (%s)", env_name, episode_idx, "success" if done else "failure")

    if onscreen:
        cv2.destroyWindow("RoboCasa Rollout")
    env.env.close()
    del env.env
    del env
    return recording.to_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(record_main)
