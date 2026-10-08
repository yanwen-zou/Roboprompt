from __future__ import annotations

import dataclasses
import functools
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


LOGGER = logging.getLogger("openpi.vis_train")


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str
    dataset_path: str
    output_dir: str | None = None
    exp_name: str = "vis_train"
    filter_key: str | None = None
    max_steps: int | None = None
    add_promp: bool = True
    use_noop_weight_loader: bool = False
    skip_norm_stats: bool = False
    use_sample_actions: bool = False
    action_scale: float = 0.05
    left_overlay_name: str = "left_camera_pred_gt_overlay.mp4"
    seed: int = 0


def _init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def _init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


@at.typecheck
def _train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        loss_output = model.compute_loss(rng, observation, actions, train=True)
        if isinstance(loss_output, tuple):
            chunked_loss, metrics = loss_output
        else:
            chunked_loss = loss_output
            metrics = {}
        return jnp.mean(chunked_loss), metrics

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(metrics)
    return new_state, info


@at.typecheck
def _forward_step(
    config: _config.TrainConfig,
    use_sample_actions: bool,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Forward-only step for visualization (saves memory by skipping gradients)."""
    model = nnx.merge(state.model_def, state.params)
    observation, actions = batch

    info: dict[str, at.Array] = {
        "loss": jnp.array(0.0),
        "grad_norm": jnp.array(0.0),
        "param_norm": jnp.array(0.0),
    }

    if use_sample_actions:
        model.eval()
        pred_actions = model.sample_actions(rng, observation, num_steps=10)
        info["dropout"] = jnp.array(-1.0)
        info["pred_actions"] = pred_actions
        return info

    model.train()
    train_rng = jax.random.fold_in(rng, state.step)
    loss_output = model.compute_loss(train_rng, observation, actions, train=True)
    if isinstance(loss_output, tuple):
        chunked_loss, metrics = loss_output
    else:
        chunked_loss = loss_output
        metrics = {}
    info = {"loss": jnp.mean(chunked_loss), "grad_norm": jnp.array(0.0), "param_norm": jnp.array(0.0)}
    info.update(metrics)

    return info


def _override_config(args: Args) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)
    data = config.data
    if not hasattr(data, "data_dirs"):
        raise ValueError(f"Config {args.config_name!r} does not support overriding local Groot data dirs.")

    data_kwargs = dict(
        data_dirs=[{"path": args.dataset_path, "filter_key": args.filter_key}],
        dataset_weights=None,
        add_promp=args.add_promp,
    )
    data = dataclasses.replace(data, **data_kwargs)

    weight_loader = _weight_loaders.NoOpWeightLoader() if args.use_noop_weight_loader else config.weight_loader
    return dataclasses.replace(
        config,
        data=data,
        exp_name=args.exp_name,
        batch_size=jax.device_count(),
        num_workers=1,
        wandb_enabled=False,
        weight_loader=weight_loader,
        fsdp_devices=jax.device_count(),
    )


def _first_episode_step_indices(dataset) -> tuple[int, list[int]]:
    if not hasattr(dataset, "trajectory_ids") or not hasattr(dataset, "all_steps"):
        raise ValueError("vis_train currently supports datasets with trajectory_ids/all_steps, e.g. Groot single datasets.")

    first_trajectory_id = int(dataset.trajectory_ids[0])
    step_indices = [i for i, (trajectory_id, _) in enumerate(dataset.all_steps) if int(trajectory_id) == first_trajectory_id]
    if not step_indices:
        raise ValueError(f"No steps found for first trajectory {first_trajectory_id}.")
    return first_trajectory_id, step_indices


def _batched_sample(sample: dict) -> dict:
    return jax.tree.map(lambda x: jnp.expand_dims(jnp.asarray(x), axis=0), sample)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if min_value >= -1.01 and max_value <= 1.01:
            image = ((image + 1.0) * 127.5).clip(0, 255)
        elif min_value >= -0.01 and max_value <= 1.01:
            image = (image * 255.0).clip(0, 255)
        else:
            image = np.clip(image, 0, 255)
    return np.asarray(image, dtype=np.uint8)


def _project_points_from_world_to_camera_unclipped(
    points: np.ndarray,
    world_to_camera_transform: np.ndarray,
) -> np.ndarray:
    """Project world points to integer (height, width) pixels without clipping to the image bounds."""
    points = np.asarray(points)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with last dimension 3, got shape {points.shape}")
    world_to_camera_transform = np.asarray(world_to_camera_transform)
    if world_to_camera_transform.shape != (4, 4):
        raise ValueError(f"Expected world_to_camera_transform shape (4, 4), got {world_to_camera_transform.shape}")

    ones_pad = np.ones(points.shape[:-1] + (1,), dtype=points.dtype)
    points_h = np.concatenate((points, ones_pad), axis=-1)
    mat_reshape = [1] * len(points.shape[:-1]) + [4, 4]
    projected = np.matmul(world_to_camera_transform.reshape(mat_reshape), points_h[..., None])[..., 0]
    z = projected[..., 2:3]
    if np.any(np.abs(z) <= 1e-12):
        raise ValueError("Cannot project points with near-zero camera depth.")
    projected = projected[..., :2] / z
    pixels_xy = np.rint(projected).astype(np.int32)
    return np.concatenate((pixels_xy[..., 1:2], pixels_xy[..., 0:1]), axis=-1)


def _project_action_chunk_to_left_pixels(
    raw_sample: dict,
    actions: np.ndarray,
    action_scale: float,
    horizon: int | None = None,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < 3:
        raise ValueError(f"Expected action chunk with shape [horizon, action_dim>=3], got {actions.shape}")
    if horizon is not None:
        actions = actions[: min(int(horizon), len(actions))]

    required_keys = (
        "prompt_action_q01",
        "prompt_action_q99",
        "prompt_base_rot",
        "prompt_tcp_world_pos",
        "prompt_world_to_camera",
        "prompt_camera_resolution",
    )
    missing_keys = [key for key in required_keys if key not in raw_sample]
    if missing_keys:
        raise KeyError(f"raw_sample missing keys required for action projection: {missing_keys}")

    q01 = np.asarray(raw_sample["prompt_action_q01"], dtype=np.float32)[:3]
    q99 = np.asarray(raw_sample["prompt_action_q99"], dtype=np.float32)[:3]
    action_xyz = 0.5 * (actions[:, :3] + 1.0) * (q99 - q01 + 1e-6) + q01

    base_rot = np.asarray(raw_sample["prompt_base_rot"], dtype=np.float32)
    world_step_delta = action_xyz @ base_rot.T * float(action_scale)
    world_traj = np.asarray(raw_sample["prompt_tcp_world_pos"], dtype=np.float32)[None, :] + np.cumsum(
        world_step_delta,
        axis=0,
    )

    return _project_points_from_world_to_camera_unclipped(
        world_traj,
        np.asarray(raw_sample["prompt_world_to_camera"], dtype=np.float32),
    )


def _prompt_2d_drag_to_left_pixels(raw_sample: dict) -> np.ndarray:
    required_keys = (
        "prompt_2d_drag",
        "prompt_camera_resolution",
        "prompt_tcp_world_pos",
        "prompt_world_to_camera",
    )
    missing_keys = [key for key in required_keys if key not in raw_sample]
    if missing_keys:
        raise KeyError(f"raw_sample missing keys required for 2d drag visualization: {missing_keys}")

    start_hw = _project_points_from_world_to_camera_unclipped(
        np.asarray(raw_sample["prompt_tcp_world_pos"], dtype=np.float32)[None, :],
        np.asarray(raw_sample["prompt_world_to_camera"], dtype=np.float32),
    )[0].astype(np.float32)
    drag_xy = np.nan_to_num(
        np.asarray(raw_sample["prompt_2d_drag"], dtype=np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    drag_xy = np.clip(drag_xy, -1.0, 1.0)
    camera_hw = np.asarray(raw_sample["prompt_camera_resolution"], dtype=np.float32)
    drag_hw = np.asarray(
        [
            drag_xy[1] * max(camera_hw[0] - 1.0, 1.0),
            drag_xy[0] * max(camera_hw[1] - 1.0, 1.0),
        ],
        dtype=np.float32,
    )
    end_hw = start_hw + drag_hw
    pixels_hw = np.nan_to_num(np.stack([start_hw, end_hw], axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    return np.rint(pixels_hw).astype(np.int32)


def _draw_pixels_polyline(
    image: np.ndarray,
    pixels_hw: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    pixels_hw = np.asarray(pixels_hw, dtype=np.int32)
    if pixels_hw.ndim != 2 or pixels_hw.shape[-1] != 2 or len(pixels_hw) == 0:
        return
    points_xy = [(int(w), int(h)) for h, w in pixels_hw]
    if len(points_xy) == 1:
        cv2.circle(image, points_xy[0], 4, color, -1, cv2.LINE_AA)
        return
    cv2.polylines(image, [np.asarray(points_xy, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
    cv2.circle(image, points_xy[0], 4, color, -1, cv2.LINE_AA)
    cv2.circle(image, points_xy[-1], 5, color, -1, cv2.LINE_AA)
    cv2.arrowedLine(image, points_xy[0], points_xy[-1], color, max(1, thickness), cv2.LINE_AA, tipLength=0.18)


def _format_vector(value: np.ndarray) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{x:+.4f}" for x in arr.tolist()) + "]"


def _pad_frame_to_shape(frame: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    h, w = frame.shape[:2]
    if h > target_h or w > target_w:
        raise ValueError(f"Cannot pad frame shape {(h, w)} to smaller target shape {target_hw}.")
    if h == target_h and w == target_w:
        return frame
    padded = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    padded[:h, :w] = frame
    return padded


def _render_left_overlay_frame(
    raw_sample: dict,
    model_left_image: np.ndarray,
    transformed_sample: dict,
    pred_pixels: np.ndarray | None,
    gt_pixels: np.ndarray | None,
    drag_gt_pixels: np.ndarray | None,
    info: dict[str, Any],
    episode_step: int,
    total_steps: int,
) -> np.ndarray:
    frame = _to_uint8_image(model_left_image).copy()
    if gt_pixels is not None:
        _draw_pixels_polyline(frame, gt_pixels, color=(80, 220, 110), thickness=1)
    if pred_pixels is not None:
        _draw_pixels_polyline(frame, pred_pixels, color=(255, 80, 80), thickness=1)
    if drag_gt_pixels is not None:
        _draw_pixels_polyline(frame, drag_gt_pixels, color=(255, 190, 40), thickness=3)
    image_panels = [frame]

    prompt_image_masks = raw_sample.get("prompt_image_masks")
    traj_prompt_mask = bool(np.asarray(prompt_image_masks["prompt_0"])) if isinstance(prompt_image_masks, dict) else False
    if traj_prompt_mask:
        prompt_image = _to_uint8_image(transformed_sample["prompt_images"]["prompt_0"]).copy()
        image_h = max(panel.shape[0] for panel in (*image_panels, prompt_image))
        image_panels = [_pad_frame_to_shape(panel, (image_h, panel.shape[1])) for panel in image_panels]
        prompt_image = _pad_frame_to_shape(prompt_image, (image_h, prompt_image.shape[1]))
        image_panels.append(prompt_image)

    image_row = np.concatenate(image_panels, axis=1)

    # Create info panel on the right
    info_panel_width = 400
    h, w = image_row.shape[:2]
    info_panel = np.zeros((h, info_panel_width, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    line_height = 16
    x_offset = 10
    y_offset = 25

    lines = [
        f"Step: {episode_step + 1} / {total_steps}",
    ]

    for k, v in info.items():
        if k in ("loss", "pred_actions") or "norm" in k or "loss" in k or "valid" in k:
            continue
        if np.ndim(v) == 0:
            lines.append(f"{k}: {float(v):.6f}")

    raw_local_motion = raw_sample.get("prompt_local_motion", raw_sample.get("prompt_wrist_relative_action"))
    model_local_motion = transformed_sample.get("prompt_local_motion", transformed_sample.get("prompt_wrist_relative_action"))
    model_local_motion_mask = transformed_sample.get(
        "prompt_local_motion_mask", transformed_sample.get("prompt_wrist_relative_action_mask", False)
    )
    raw_global_motion = raw_sample.get("prompt_global_motion", raw_sample.get("prompt_primitive_cmd"))
    model_global_motion = transformed_sample.get("prompt_global_motion", transformed_sample.get("prompt_primitive_cmd"))
    model_global_motion_mask = transformed_sample.get(
        "prompt_global_motion_mask", transformed_sample.get("prompt_primitive_cmd_mask", False)
    )
    raw_drag = raw_sample.get("prompt_2d_drag")
    model_drag = transformed_sample.get("prompt_2d_drag")
    model_drag_mask = transformed_sample.get("prompt_2d_drag_mask", False)
    prompt_src_frame = raw_sample.get("prompt_source_frame_index", "n/a")
    lines.extend(
        [
            "",
            f"Prompt src frame: {prompt_src_frame}",
            f"Traj mask: {traj_prompt_mask}",
            f"Global mask: {bool(np.asarray(model_global_motion_mask))}",
            # f"Global raw: {_format_vector(raw_global_motion if raw_global_motion is not None else np.zeros(3, dtype=np.float32))}",
            f"Global model: {_format_vector(model_global_motion if model_global_motion is not None else np.zeros(3, dtype=np.float32))}",
            f"Local mask: {bool(np.asarray(model_local_motion_mask))}",
            # f"Local raw: {_format_vector(raw_local_motion if raw_local_motion is not None else np.zeros(3, dtype=np.float32))}",
            f"Local model: {_format_vector(model_local_motion if model_local_motion is not None else np.zeros(3, dtype=np.float32))}",
            f"2D drag mask: {bool(np.asarray(model_drag_mask))}",
            # f"2D drag raw: {_format_vector(raw_drag if raw_drag is not None else np.zeros(2, dtype=np.float32))}",
            f"2D drag model: {_format_vector(model_drag if model_drag is not None else np.zeros(2, dtype=np.float32))}",
        ]
    )

    prompt = raw_sample.get("prompt", "")
    if prompt:
        lines.append("")
        lines.append("Prompt:")
        max_chars = 55
        prompt_str = str(prompt)
        for i in range(0, len(prompt_str), max_chars):
            lines.append(prompt_str[i:i + max_chars])

    for line in lines:
        cv2.putText(info_panel, line, (x_offset, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += line_height
        if y_offset > h - 20:
            break

    return np.concatenate([image_row, info_panel], axis=1)

def _write_video(frames: list[np.ndarray], output_path: Path, fps: int = 10) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(frame.shape[0] for frame in frames)
    width = max(frame.shape[1] for frame in frames)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for frame in frames:
            frame = _pad_frame_to_shape(frame, (height, width))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main(args: Args) -> None:
    _init_logging()
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("vis_train_output") / timestamp
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _override_config(args)
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed_dataset = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=args.skip_norm_stats)
    transform_fn = transformed_dataset._transform

    trajectory_id, step_indices = _first_episode_step_indices(raw_dataset)
    if args.max_steps is not None:
        step_indices = step_indices[: args.max_steps]

    LOGGER.info("Visualizing %d steps from first trajectory %d", len(step_indices), trajectory_id)

    rng = jax.random.key(args.seed)
    rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state, train_state_sharding = _init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)

    p_forward_step = jax.jit(
        functools.partial(_forward_step, config, args.use_sample_actions),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    overlay_horizon = int(getattr(config.model, "drag_2d_horizon", config.model.action_horizon))
    LOGGER.info(
        "Projecting overlay action chunks with horizon=%d and action_scale=%.4f",
        overlay_horizon,
        args.action_scale,
    )

    left_overlay_frames: list[np.ndarray] = []
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for episode_step, dataset_index in enumerate(step_indices):
        LOGGER.info(
            "[Step %d/%d] dataset_index=%d — loading sample...",
            episode_step + 1,
            len(step_indices),
            int(dataset_index),
        )
        raw_sample = raw_dataset[dataset_index]
        transformed_sample = transform_fn(raw_sample)

        # Debug: print non-numeric fields
        flat = jax.tree_util.tree_leaves_with_path(transformed_sample)
        for path, value in flat:
            key = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
            if isinstance(value, (str, bytes)) or (hasattr(value, 'dtype') and value.dtype.kind in ('U', 'S', 'O')):
                LOGGER.info("[DEBUG] Non-numeric field: %s = %r (type=%s)", key, value, type(value))

        batched = _batched_sample(transformed_sample)
        gt_action_chunk = np.asarray(batched["actions"])[0]
        observation = _model.Observation.from_dict(batched)
        actions = jnp.asarray(batched["actions"])

        # Replicate single sample to match device count for FSDP sharding.
        num_devices = jax.device_count()
        observation = jax.tree.map(lambda x: jnp.repeat(x, num_devices, axis=0), observation)
        actions = jnp.repeat(actions, num_devices, axis=0)

        LOGGER.info(
            "[Step %d/%d] running forward...",
            episode_step + 1,
            len(step_indices),
        )
        with sharding.set_mesh(mesh):
            info = p_forward_step(rng, train_state, (observation, actions))

        LOGGER.info(
            "[Step %d/%d] gathering results...",
            episode_step + 1,
            len(step_indices),
        )
        info_raw = jax.device_get(info)
        info_np: dict[str, Any] = {}
        for k, v in info_raw.items():
            if np.ndim(v) == 0:
                info_np[k] = float(v)
            else:
                info_np[k] = np.asarray(v)

        LOGGER.info(
            "[Step %d/%d] rendering frame...",
            episode_step + 1,
            len(step_indices),
        )

        pred_pixels = None
        if "pred_actions" in info_np:
            pred_actions = np.asarray(info_np["pred_actions"])[0]
            pred_pixels = _project_action_chunk_to_left_pixels(
                raw_sample,
                pred_actions,
                action_scale=args.action_scale,
                horizon=overlay_horizon,
            )
            LOGGER.info(
                "[Step %d/%d] pred_pixels[:10]=%s",
                episode_step + 1,
                len(step_indices),
                np.array2string(np.asarray(pred_pixels)[:10], precision=1),
            )

        gt_pixels = _project_action_chunk_to_left_pixels(
            raw_sample,
            gt_action_chunk,
            action_scale=args.action_scale,
            horizon=overlay_horizon,
        )
        LOGGER.info(
            "[Step %d/%d] gt_pixels[:10]=%s",
            episode_step + 1,
            len(step_indices),
            np.array2string(np.asarray(gt_pixels)[:10], precision=1),
        )

        drag_gt_pixels = None
        drag_mask = bool(np.asarray(raw_sample.get("prompt_2d_drag_mask", False)))
        if drag_mask:
            drag_gt_pixels = _prompt_2d_drag_to_left_pixels(raw_sample)
            LOGGER.info(
                "[Step %d/%d] drag_gt_pixels=%s",
                episode_step + 1,
                len(step_indices),
                np.array2string(np.asarray(drag_gt_pixels), precision=1),
            )

        frame = _render_left_overlay_frame(
            raw_sample,
            model_left_image=np.asarray(transformed_sample["image"]["base_0_rgb"]),
            transformed_sample=transformed_sample,
            pred_pixels=pred_pixels,
            gt_pixels=gt_pixels,
            drag_gt_pixels=drag_gt_pixels,
            info=info_np,
            episode_step=episode_step,
            total_steps=len(step_indices),
        )
        left_overlay_frames.append(frame)

        # Save individual frame image
        frame_path = frames_dir / f"frame_{episode_step:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        LOGGER.info("[Step %d/%d] saved frame image to %s", episode_step + 1, len(step_indices), frame_path)
        # Debug prints for projection diagnostics.
        if "pred_actions" in info_np:
            pred_xyz = np.asarray(info_np["pred_actions"])[..., :3]
            LOGGER.info(
                "[Step %d/%d] pred_actions xyz min=%.4f max=%.4f mean=%.4f",
                episode_step + 1,
                len(step_indices),
                float(np.min(pred_xyz)),
                float(np.max(pred_xyz)),
                float(np.mean(pred_xyz)),
            )
        if "prompt_action_q01" in raw_sample:
            q01 = np.asarray(raw_sample["prompt_action_q01"])[:3]
            q99 = np.asarray(raw_sample["prompt_action_q99"])[:3]
            LOGGER.info(
                "[Step %d/%d] action_q01=%s action_q99=%s",
                episode_step + 1,
                len(step_indices),
                np.array2string(q01, precision=4),
                np.array2string(q99, precision=4),
            )

        LOGGER.info(
            "[Step %d/%d] done — loss=%.4f, saved to %s",
            episode_step + 1,
            len(step_indices),
            info_np["loss"],
            output_dir / args.left_overlay_name,
        )

    _write_video(left_overlay_frames, output_dir / args.left_overlay_name)
    LOGGER.info("Saved %d overlay frames (video + individual images) to %s", len(left_overlay_frames), output_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
