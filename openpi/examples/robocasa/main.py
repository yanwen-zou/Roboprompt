import os
# Must set before importing robosuite/robocasa
os.environ["MUJOCO_GL"] = "egl"

import collections
import dataclasses
import logging
import pathlib
import sys
import imageio
from datetime import datetime
import cv2
import numpy as np
import re
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import time
import json
import robocasa.utils.robomimic.robomimic_dataset_utils as FileUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
import robocasa
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
import gymnasium as gym
from robocasa.utils.env_utils import convert_action

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_utils import (
    InteractivePromptState,
    compose_interactive_replay_frame,
    compose_rollout_replay_frame,
    format_prompt_for_log,
    format_sample_kwargs_for_log,
    overlay_predicted_action_chunk,
    overlay_predicted_action_chunks,
    policy_inference_steps_from_metadata,
    render_rollout_window,
    render_prompt_window,
    select_replan_steps_for_prompt,
)
from scripts.realworld.eval.eval_ui.interactive_labeling import collect_prompt_payload_from_images


REPLAY_FPS = 20
INFERENCE_SLEEP_SECONDS = 0.5


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
    task_set_label = _get_task_set_label(task_sets)
    return f"{_sanitize_path_component(policy_label)}_{task_set_label}_{now.strftime('%Y-%m-%d-%H-%M')}"


def _make_eval_run_root(
    log_dir: str,
    checkpoint_info: dict[str, str] | None,
    task_sets: list[str] | None,
    now: datetime | None = None,
) -> pathlib.Path:
    now = now or datetime.now()
    return pathlib.Path(log_dir) / "evals" / _get_policy_run_name(checkpoint_info, task_sets, now)


def _write_summary(run_root, split, task_sets, env_summaries, checkpoint_info):
    run_root = pathlib.Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    total_episodes = sum(summary["num_episodes"] for summary in env_summaries)
    total_successes = sum(summary["num_successes"] for summary in env_summaries)
    success_rate = (
        float(total_successes) / float(total_episodes) if total_episodes else 0.0
    )
    mean_task_success_rate = (
        float(sum(summary["success_rate"] for summary in env_summaries)) / float(len(env_summaries))
        if env_summaries
        else 0.0
    )

    summary = {
        "split": split,
        "task_sets": task_sets,
        "checkpoint": checkpoint_info,
        "run_root": str(run_root),
        "num_tasks": len(env_summaries),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": success_rate,
        "mean_task_success_rate": mean_task_success_rate,
        "tasks": env_summaries,
    }

    summary_path = run_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"[summary] Wrote summary to {summary_path}")
    logging.info(
        "[summary] %d tasks, %d/%d successes (%.1f%%)",
        len(env_summaries),
        total_successes,
        total_episodes,
        success_rate * 100.0,
    )

    return summary


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 15
    prompt_replan_steps: int = 25
    onscreen: bool = False
    overlay_action_scale: float = 0.05
    overlay_camera_name: str = "robot0_agentview_left"
    interactive_prompt: bool = False
    inference_iterations: int = 10
    phase2_steps: float = 0.6
    max_phase2_steps: float = 10.0

    split: str = "pretrain"
    num_trials: int = 50  # Number of rollouts per task
    task_set: list[str] | None = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    log_dir: str | None = None
    checkpoint: str | None = None
    force: bool = False

    seed: int = 7  # Random Seed (for reproducibility)


def _get_server_metadata(client) -> dict:
    try:
        server_metadata = client.get_server_metadata()
    except Exception as exc:
        logging.warning("Could not fetch server metadata: %s", exc)
        return {}
    return server_metadata if isinstance(server_metadata, dict) else {}


def _get_checkpoint_info(server_metadata: dict, checkpoint_override: str | None) -> dict[str, str] | None:
    checkpoint_info: dict[str, str] = {}
    if checkpoint_override:
        checkpoint_info["label"] = checkpoint_override

    checkpoint_dir = server_metadata.get("checkpoint_dir")
    checkpoint_config = server_metadata.get("checkpoint_config")
    if checkpoint_dir:
        checkpoint_info["dir"] = checkpoint_dir
    if checkpoint_config:
        checkpoint_info["config"] = checkpoint_config

    return checkpoint_info or None


def _phase2_policy_type(metadata: dict) -> str:
    return str(metadata.get("phase2_policy_type", "")).strip().lower()


def eval_main(args: Args) -> None:
    if args.interactive_prompt and not args.onscreen:
        raise ValueError("interactive_prompt requires onscreen=True so the prompt hotkey/window can be used.")

    # Set random seed
    np.random.seed(args.seed)
    
    split = args.split
    log_dir = args.log_dir
    num_trials = args.num_trials
    resize_size = args.resize_size
    replan_steps = args.replan_steps
    prompt_replan_steps = args.prompt_replan_steps
    host = args.host
    port = args.port
    client = _websocket_client_policy.WebsocketClientPolicy(host, port)
    if args.inference_iterations < 1:
        raise ValueError(f"inference_iterations must be at least 1, got {args.inference_iterations}.")
    server_metadata = _get_server_metadata(client)
    max_phase2_steps = policy_inference_steps_from_metadata(server_metadata, default=args.max_phase2_steps)
    emit_num_steps = _phase2_policy_type(server_metadata) == "openpi"
    checkpoint_info = _get_checkpoint_info(server_metadata, args.checkpoint)
    run_root = _make_eval_run_root(args.log_dir, checkpoint_info, args.task_set)

    all_env_names = []
    for task in args.task_set:
        env_names = TASK_SET_REGISTRY[task]
        all_env_names.extend(env_names)

    env_summaries = []
    for env_name in all_env_names:
        # try:
        env_summary = eval_env(
            env_name,
            split,
            run_root,
            num_trials,
            resize_size,
            replan_steps,
            prompt_replan_steps,
            host,
            port,
            args.seed,
            args.force,
            checkpoint_info,
            args.onscreen,
            args.overlay_action_scale,
            args.overlay_camera_name,
            args.interactive_prompt,
            args.inference_iterations,
            args.phase2_steps,
            max_phase2_steps,
            emit_num_steps,
        )
        if env_summary is not None:
            env_summaries.append(env_summary)
        # except Exception as e:
        #     print("Exception!")
        #     print(e)

    _write_summary(run_root, split, args.task_set, env_summaries, checkpoint_info)
def eval_env(
    env_name,
    split,
    run_root,
    num_trials,
    resize_size,
    replan_steps,
    prompt_replan_steps,
    host,
    port,
    seed,
    force,
    checkpoint_info,
    onscreen,
    overlay_action_scale,
    overlay_camera_name,
    interactive_prompt,
    inference_iterations,
    phase2_steps,
    max_phase2_steps,
    emit_num_steps,
):
    # set args based on task
    assert split in ["pretrain", "target"]
    task_horizon = get_task_horizon(env_name)
    # set dataset path and horizon
    horizon = int(task_horizon * 1.5) # the policy moves slow so give the policy extra time

    log_path = pathlib.Path(run_root) / split / env_name
    stats_path = log_path / "stats.json"

    if stats_path.exists() and not force:
        print(
            f"{env_name}/{split}, existing stats found at {stats_path}. "
            "Skipping. Pass --args.force true to rerun."
        )
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return {
            "env_name": env_name,
            "log_path": str(log_path),
            "num_episodes": stats["num_episodes"],
            "num_successes": int(round(stats["num_episodes"] * stats["success_rate"])),
            "success_rate": stats["success_rate"],
            "checkpoint": stats.get("checkpoint", checkpoint_info),
            "used_existing_stats": True,
        }

    log_path.mkdir(parents=True, exist_ok=True)


    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    # Get task
    env = gym.make(f"robocasa/{env_name}", split=split, seed=seed)
    # Start episodes
    task_episodes, task_successes = 0, 0
    current_phase2_steps = float(np.clip(phase2_steps, 0.0, max_phase2_steps))
    for episode_idx in tqdm.tqdm(range(num_trials)):

        # Reset environment
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()

        # Setup
        t = 0
        done = False
        aborted = False
        replay_images = []
        prompt_state = InteractivePromptState()
        infer_count = 0
        latest_model_action_chunk = np.zeros((0, 12), dtype=np.float32)
        latest_model_action_chunks = []
        latest_model_action_index = 0

        logging.info(f"Starting episode {task_episodes+1}...")
        while t < horizon:
            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, resize_size, resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
            )

            if onscreen and not action_plan:
                render_rollout_window(
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    env_name,
                    episode_idx,
                    t,
                    enable_prompt_hotkey=interactive_prompt,
                )
                key = cv2.waitKey(1) & 0xFF
                if interactive_prompt and key == ord("p"):
                    prompt_payload = collect_prompt_payload_from_images(
                        base_image_rgb=img,
                        wrist_image_rgb=wrist_img,
                        phase2_steps=current_phase2_steps,
                        max_phase2_steps=max_phase2_steps,
                    )
                    current_phase2_steps = float(
                        prompt_payload.get("sample_kwargs", {}).get("phase2_steps", current_phase2_steps)
                    )
                    prompt_state.update(prompt_payload)
                    action_plan.clear()
                    latest_model_action_chunk = np.zeros((0, 12), dtype=np.float32)
                    latest_model_action_chunks = []
                    latest_model_action_index = 0
                    continue
                if key == ord("q"):
                    logging.info("Aborting current episode from rollout window.")
                    aborted = True
                    break

            # Save preprocessed image for replay video
            # replay_images.append(img)

            if not action_plan:
                state = np.concatenate(
                    (
                        obs["state.end_effector_position_relative"],
                        obs["state.end_effector_rotation_relative"],
                        obs["state.base_position"],
                        obs["state.base_rotation"],
                        obs["state.gripper_qpos"],
                    ), axis=0
                )
                # state = np.ascontiguousarray(state)
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": task_lang,
                }
                prompt_inputs = prompt_state.build_model_inputs()
                sample_kwargs = prompt_state.build_sample_kwargs(
                    policy_inference_steps=max_phase2_steps,
                    emit_num_steps=emit_num_steps,
                )
                element.update(prompt_inputs)
                infer_count += 1
                logging.info(
                    "[infer %d] task prompt: %s",
                    infer_count,
                    task_lang,
                )
                logging.info(
                    "[infer %d] interactive prompt: %s",
                    infer_count,
                    format_prompt_for_log(prompt_inputs),
                )
                logging.info(
                    "[infer %d] sample kwargs: %s",
                    infer_count,
                    format_sample_kwargs_for_log(sample_kwargs),
                )
                if onscreen:
                    render_prompt_window(
                        {**prompt_state.build_display_inputs(), "sample_kwargs": sample_kwargs},
                        env_name,
                        episode_idx,
                        infer_count,
                    )

                # Query the model multiple times and execute only the final sampled action chunk.
                if onscreen:
                    sleep_frame = compose_rollout_replay_frame(
                        image_bgr=cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        prompt_state=prompt_state,
                        interactive_prompt=interactive_prompt,
                    )
                    replay_images.extend(
                        sleep_frame.copy()
                        for _ in range(max(1, int(round(INFERENCE_SLEEP_SECONDS * REPLAY_FPS))))
                    )
                time.sleep(INFERENCE_SLEEP_SECONDS)
                current_replan_steps = select_replan_steps_for_prompt(
                    prompt_inputs=prompt_inputs,
                    replan_steps=replan_steps,
                    prompt_replan_steps=prompt_replan_steps,
                )
                if current_replan_steps != replan_steps:
                    logging.info(
                        "[infer %d] active motion prompt detected, using prompt_replan_steps=%d",
                        infer_count,
                        current_replan_steps,
                    )
                action_chunks = []
                for iteration_idx in range(inference_iterations):
                    action_chunk = np.asarray(
                        client.infer(element, sample_kwargs=sample_kwargs or None)["actions"],
                        dtype=np.float32,
                    )
                    if len(action_chunk) < current_replan_steps:
                        raise ValueError(
                            f"We want to replan every {current_replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps "
                            f"on inference iteration {iteration_idx + 1}/{inference_iterations}."
                        )
                    action_chunks.append(action_chunk[:current_replan_steps])
                action_chunk = action_chunks[-1]
                latest_model_action_chunks = action_chunks
                if len(action_chunk) < current_replan_steps:
                    raise ValueError(
                        f"We want to replan every {current_replan_steps} steps, "
                        f"but policy only predicts {len(action_chunk)} steps."
                    )
                latest_model_action_chunk = action_chunk
                latest_model_action_index = 0
                action_plan.extend(action_chunk)
                prompt_state.add_prompt_mem()
            start_time = time.time()
            if onscreen:
                if latest_model_action_index < len(latest_model_action_chunk):
                    action_chunk_for_overlay = latest_model_action_chunk[latest_model_action_index:]
                    action_chunks_for_overlay = [
                        action_chunk[latest_model_action_index:]
                        for action_chunk in latest_model_action_chunks
                    ]
                else:
                    action_chunk_for_overlay = np.asarray(list(action_plan), dtype=np.float32)
                    action_chunks_for_overlay = [action_chunk_for_overlay]
                if len(action_chunks_for_overlay) == 1:
                    display_image_bgr = overlay_predicted_action_chunk(
                        image_bgr=cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        env=env.unwrapped,
                        obs=obs,
                        action_chunk=action_chunk_for_overlay,
                        camera_name=overlay_camera_name,
                        action_scale=overlay_action_scale,
                    )
                else:
                    display_image_bgr = overlay_predicted_action_chunks(
                        image_bgr=cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        env=env.unwrapped,
                        obs=obs,
                        action_chunks=action_chunks_for_overlay,
                        camera_name=overlay_camera_name,
                        action_scale=overlay_action_scale,
                    )
                render_rollout_window(
                    display_image_bgr,
                    env_name,
                    episode_idx,
                    t,
                    enable_prompt_hotkey=interactive_prompt,
                )
                key = cv2.waitKey(1) & 0xFF
                if interactive_prompt and key == ord("p"):
                    prompt_payload = collect_prompt_payload_from_images(
                        base_image_rgb=img,
                        wrist_image_rgb=wrist_img,
                        phase2_steps=current_phase2_steps,
                        max_phase2_steps=max_phase2_steps,
                    )
                    current_phase2_steps = float(
                        prompt_payload.get("sample_kwargs", {}).get("phase2_steps", current_phase2_steps)
                    )
                    prompt_state.update(prompt_payload)
                    action_plan.clear()
                    latest_model_action_chunk = np.zeros((0, 12), dtype=np.float32)
                    latest_model_action_chunks = []
                    latest_model_action_index = 0
                    continue
                if key == ord("q"):
                    logging.info("Aborting current episode from rollout window.")
                    aborted = True
                    break
                if t % 2 == 0 or t == horizon - 1:
                    replay_images.append(
                        compose_rollout_replay_frame(
                            image_bgr=display_image_bgr,
                            prompt_state=prompt_state,
                            interactive_prompt=interactive_prompt,
                        )
                    )
            action = action_plan.popleft()
            latest_model_action_index += 1
            action = convert_action(action)
            # Execute action in environment
            obs, reward, done, truncated, info = env.step(action)
            done = info["success"] # for robocasa, usuccess entry in info
            if onscreen:
                if done:
                    replay_images.append(
                        compose_rollout_replay_frame(
                            image_bgr=display_image_bgr,
                            prompt_state=prompt_state,
                            interactive_prompt=interactive_prompt,
                        )
                    )
            else:
                replay_img = env.render()
                replay_img = np.ascontiguousarray(replay_img)
                replay_img = image_tools.convert_to_uint8(
                    replay_img
                )
                if interactive_prompt:
                    replay_img = compose_interactive_replay_frame(replay_img, prompt_state)
                if t % 2 == 0 or t == horizon - 1 or done:
                    replay_images.append(replay_img)
            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1
            end_time = time.time()
            # logging.info(f"Step {t} executed in {end_time - start_time:.2f} seconds.")
        task_episodes += 1
        total_episodes += 1

        # Save a replay video of the episode
        suffix = "success" if done else ("aborted" if aborted else "failure")
        imageio.mimwrite(
            log_path / f"rollout_{episode_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=REPLAY_FPS,
        )

        # Log current results
        episode_result = "SUCCESS" if done else ("ABORTED" if aborted else "FAILURE")
        logging.info("Episode result: %s", episode_result)
        prompt_count_message = (
            f"[{env_name}] Episode {episode_idx} interactive UI prompt count: "
            f"{prompt_state.prompt_update_count}"
        )
        logging.info(prompt_count_message)
        print(prompt_count_message, flush=True)
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"[{env_name}] Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"[{env_name}] Total episodes: {total_episodes}")
    print()
    with open(stats_path, "w") as f:
        stats = {
            "num_episodes": total_episodes,
            "num_successes": total_successes,
            "success_rate": float(total_successes) / float(total_episodes),
            "checkpoint": checkpoint_info,
            "split": split,
            "env_name": env_name,
        }
        json.dump(stats, f, indent=4)

    # close and delete the env
    if onscreen:
        cv2.destroyWindow("RoboCasa Rollout")
        try:
            cv2.destroyWindow("RoboCasa Prompt")
        except cv2.error:
            pass
    env.env.close()
    del env.env
    del env

    return {
        "env_name": env_name,
        "log_path": str(log_path),
        "num_episodes": total_episodes,
        "num_successes": total_successes,
        "success_rate": float(total_successes) / float(total_episodes),
        "checkpoint": checkpoint_info,
        "used_existing_stats": False,
    }




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_main)
