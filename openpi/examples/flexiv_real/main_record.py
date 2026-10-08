from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys
from typing import Any

import numpy as np
import tyro

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_SRC_ROOT = OPENPI_ROOT / "src"
OPENPI_CLIENT_ROOT = OPENPI_ROOT / "packages" / "openpi-client" / "src"
preferred_paths = [str(REPO_ROOT), str(OPENPI_ROOT), str(OPENPI_SRC_ROOT), str(OPENPI_CLIENT_ROOT)]
sys.path[:] = [path for path in sys.path if path not in preferred_paths]
sys.path[:0] = preferred_paths

from openpi_client import action_chunk_broker
from openpi_client import base_policy as _base_policy
from openpi_client import websocket_client_policy as _websocket_client_policy
from typing_extensions import override

from examples.flexiv_real import flexiv_env as _flexiv_env
from examples.flexiv_real import recorder as _recorder
from examples.flexiv_real.main import _allocate_run_dir
from examples.utils import apply_action_perturbation
from examples.utils import apply_smooth_action_perturbation
from scripts.realworld.eval.eval_ui.interactive_labeling import collect_prompt_payload_from_images
from scripts.utils.interactive_prompt import InteractivePromptState
from scripts.utils.interactive_prompt import render_prompt_window


ACTION_PERTURBATION_SCALE = 0.02
ACTION_PERTURBATION_CONTROL_POINTS = 4
ACTION_PERTURBATION_EVERY_N_CHUNKS = 5
ACTION_PERTURBATION_BLEND_IN_STEPS = 8
ACTION_PERTURBATION_BLEND_OUT_STEPS = 5


class PerturbedActionChunkPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        rng: np.random.Generator,
        enabled: bool,
        scale: float,
        control_points: int,
        every_n_chunks: int,
        blend_in_steps: int = ACTION_PERTURBATION_BLEND_IN_STEPS,
        blend_out_steps: int = ACTION_PERTURBATION_BLEND_OUT_STEPS,
    ) -> None:
        self._policy = policy
        self._rng = rng
        self._enabled = bool(enabled)
        self._scale = float(scale)
        self._control_points = int(control_points)
        self._every_n_chunks = int(every_n_chunks)
        self._blend_in_steps = int(blend_in_steps)
        self._blend_out_steps = int(blend_out_steps)
        self._chunk_count = 0
        self._last_chunk: np.ndarray | None = None

    @override
    def infer(self, obs: dict, *, sample_kwargs: dict[str, Any] | None = None) -> dict:
        result = self._policy.infer(obs, sample_kwargs=sample_kwargs)
        self._chunk_count += 1
        should_perturb = self._enabled and self._chunk_count % self._every_n_chunks == 0

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected model action chunk with shape [T, D], got {actions.shape}.")

        # Capture the tail of the previous chunk before we overwrite it.
        prev_tail_action = None
        if self._last_chunk is not None and len(self._last_chunk) > 0:
            prev_tail_action = self._last_chunk[-1]

        if not should_perturb:
            # Still remember this unperturbed chunk for the next call's blend-in.
            self._last_chunk = actions.copy()
            return result

        logging.info("Applying smooth action perturbation to model action chunk %d.", self._chunk_count)

        # Apply perturbation with smooth boundaries.
        perturbed_actions = apply_smooth_action_perturbation(
            actions,
            rng=self._rng,
            scale=self._scale,
            control_points=self._control_points,
            prev_tail_action=prev_tail_action,
            blend_in_steps=self._blend_in_steps,
            blend_out_steps=self._blend_out_steps,
        )

        # Remember the perturbed chunk so the next call can blend from it.
        self._last_chunk = perturbed_actions.copy()

        return {
            **result,
            "actions": perturbed_actions,
        }

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._chunk_count = 0
        self._last_chunk = None


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 15
    fps: float = 5.0

    num_episodes: int = 10
    max_episode_steps: int = 300

    render_height: int = 224
    render_width: int = 224
    task: str | None = None
    output_dir: str = "output_noise"

    seed: int = 7
    action_perturbation: bool = True
    action_perturbation_scale: float = ACTION_PERTURBATION_SCALE
    action_perturbation_control_points: int = ACTION_PERTURBATION_CONTROL_POINTS
    action_perturbation_every_n_chunks: int = ACTION_PERTURBATION_EVERY_N_CHUNKS
    action_perturbation_blend_in_steps: int = ACTION_PERTURBATION_BLEND_IN_STEPS
    action_perturbation_blend_out_steps: int = ACTION_PERTURBATION_BLEND_OUT_STEPS


def main(args: Args) -> None:
    if args.action_perturbation_every_n_chunks < 1:
        raise ValueError(
            "action_perturbation_every_n_chunks must be at least 1, "
            f"got {args.action_perturbation_every_n_chunks}."
        )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    task_prompt = (
        args.task
        or metadata.get("default_prompt")
        or metadata.get("prompt")
        or metadata.get("task_description")
    )
    run_dir = _allocate_run_dir(REPO_ROOT / args.output_dir)
    logging.info("Recording perturbed rollouts to %s", run_dir)

    perturbation_metadata = {
        "enabled": args.action_perturbation,
        "scale": args.action_perturbation_scale,
        "control_points": args.action_perturbation_control_points,
        "every_n_chunks": args.action_perturbation_every_n_chunks,
        "blend_in_steps": args.action_perturbation_blend_in_steps,
        "blend_out_steps": args.action_perturbation_blend_out_steps,
        "seed": args.seed,
    }
    recorder_metadata = {
        **metadata,
        "action_perturbation": perturbation_metadata,
    }

    env = _flexiv_env.FlexivRealEnv(
        reset_pose=metadata.get("reset_pose"),
        render_height=args.render_height,
        render_width=args.render_width,
        fps=args.fps,
        prompt=task_prompt,
    )
    policy = PerturbedActionChunkPolicy(
        ws_client_policy,
        rng=np.random.default_rng(args.seed),
        enabled=args.action_perturbation,
        scale=args.action_perturbation_scale,
        control_points=args.action_perturbation_control_points,
        every_n_chunks=args.action_perturbation_every_n_chunks,
        blend_in_steps=args.action_perturbation_blend_in_steps,
        blend_out_steps=args.action_perturbation_blend_out_steps,
    )
    broker = action_chunk_broker.ActionChunkBroker(
        policy=policy,
        action_horizon=args.action_horizon,
    )
    max_episode_steps = args.max_episode_steps if args.max_episode_steps > 0 else float("inf")
    current_recorder = None

    try:
        for episode_idx in range(args.num_episodes):
            episode_dir = run_dir / f"episode_{episode_idx:06d}"
            logging.info(
                "Starting episode %d/%d; recording to %s",
                episode_idx + 1,
                args.num_episodes,
                episode_dir,
            )
            current_recorder = _recorder.LeRobotRolloutRecorder(
                output_dir=episode_dir,
                fps=args.fps,
                server_metadata=recorder_metadata,
                default_task=task_prompt,
                render_height=args.render_height,
                render_width=args.render_width,
                steer=False,
            )
            env.reset()
            broker.reset()
            current_recorder.on_episode_start()

            step_idx = 0
            while not env.is_episode_complete() and step_idx < max_episode_steps:
                observation = env.get_observation()
                needs_new_inference = broker.needs_new_inference()
                action = broker.infer(observation, sample_kwargs={"enable_evo1_steerer": False})
                env.apply_action(action)
                current_recorder.on_step(
                    observation,
                    action,
                )
                if needs_new_inference:
                    logging.debug("Started new perturbed action chunk at step %d.", step_idx)
                step_idx += 1

            current_recorder.on_episode_end()
            current_recorder.finalize()
            current_recorder = None

        env.reset()
    finally:
        if current_recorder is not None:
            current_recorder.finalize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
