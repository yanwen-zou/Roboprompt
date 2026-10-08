import math

import numpy as np


def sample_bezier_position_noise(
    *,
    num_steps: int,
    scale: float,
    control_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if scale < 0.0:
        raise ValueError(f"action_perturbation_scale must be non-negative, got {scale}.")
    if control_points < 2:
        raise ValueError(f"action_perturbation_control_points must be at least 2, got {control_points}.")
    if scale == 0.0:
        return np.zeros((num_steps, 3), dtype=np.float32)

    points = rng.normal(loc=0.0, scale=scale, size=(control_points, 3)).astype(np.float32)
    points[0] = 0.0
    u_values = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    degree = control_points - 1
    noise = np.zeros((num_steps, 3), dtype=np.float32)
    for idx in range(control_points):
        basis = math.comb(degree, idx) * (u_values**idx) * ((1.0 - u_values) ** (degree - idx))
        noise += basis[:, None] * points[idx]
    return noise


def apply_action_perturbation(
    action_chunk: np.ndarray,
    *,
    rng: np.random.Generator,
    scale: float,
    control_points: int,
) -> np.ndarray:
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk with shape [T, D], got {action_chunk.shape}.")
    if action_chunk.shape[1] < 3:
        raise ValueError(f"Expected action dimension >= 3 for eef position perturbation, got {action_chunk.shape[1]}.")

    perturbed = action_chunk.copy()
    perturbed[:, :3] += sample_bezier_position_noise(
        num_steps=len(perturbed),
        scale=scale,
        control_points=control_points,
        rng=rng,
    )
    perturbed[:, :3] = np.clip(perturbed[:, :3], -1.0, 1.0)
    return perturbed


def apply_smooth_action_perturbation(
    action_chunk: np.ndarray,
    *,
    rng: np.random.Generator,
    scale: float,
    control_points: int,
    prev_tail_action: np.ndarray | None = None,
    blend_in_steps: int = 8,
    blend_out_steps: int = 5,
) -> np.ndarray:
    """Apply position perturbation with smooth blend-in/out to avoid chunk-boundary corners.

    Blend-in: the first `blend_in_steps` actions are interpolated from `prev_tail_action`
    (the last action of the previous chunk, i.e. the prior velocity vector) toward the
    fully perturbed action using cosine easing.

    Blend-out: the last `blend_out_steps` actions are interpolated from the fully perturbed
    action back toward the original unperturbed action, so the next normal chunk can
    seamlessly continue the trajectory.
    """
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk with shape [T, D], got {action_chunk.shape}.")
    if action_chunk.shape[1] < 3:
        raise ValueError(f"Expected action dimension >= 3 for eef position perturbation, got {action_chunk.shape[1]}.")

    T = len(action_chunk)
    perturbed = action_chunk.copy()
    perturbed[:, :3] += sample_bezier_position_noise(
        num_steps=T,
        scale=scale,
        control_points=control_points,
        rng=rng,
    )
    perturbed[:, :3] = np.clip(perturbed[:, :3], -1.0, 1.0)

    # --- Blend-in: smooth transition from previous chunk's tail action ---
    if prev_tail_action is not None and blend_in_steps > 0:
        prev_tail_action = np.asarray(prev_tail_action, dtype=np.float32)
        blend = min(blend_in_steps, T)
        for i in range(blend):
            t = i / max(1, blend - 1)
            w = 0.5 * (1.0 - np.cos(np.pi * t))  # cosine ease-in-out, 0 -> 1
            perturbed[i] = (1.0 - w) * prev_tail_action + w * perturbed[i]

    # --- Blend-out: smooth transition back to original unperturbed action ---
    if blend_out_steps > 0:
        blend = min(blend_out_steps, T)
        for i in range(blend):
            t = i / max(1, blend - 1)
            w = 0.5 * (1.0 - np.cos(np.pi * t))  # cosine ease-in-out, 0 -> 1
            idx = T - blend + i
            perturbed[idx] = (1.0 - w) * perturbed[idx] + w * action_chunk[idx]

    return perturbed
