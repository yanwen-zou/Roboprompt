from __future__ import annotations

import logging
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy

from openpi import transforms
from steering.backends.evo1 import Evo1Backend
from steering.config import SteeringConfig
from steering.runtime import SteeredPhase2Runtime
from steering.schemas import Phase1Action


LOGGER = logging.getLogger(__name__)
_GRIPPER_ACTION_INDEX = 6


def _format_action_debug(name: str, actions: np.ndarray) -> str:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        return f"{name}: shape={actions.shape}"
    xyz = actions[:, : min(3, actions.shape[-1])]
    xyz_norm = np.linalg.norm(xyz, axis=-1) if xyz.shape[-1] else np.zeros(actions.shape[0], dtype=np.float32)
    xyz_adj = np.diff(xyz, axis=0) if len(xyz) >= 2 else np.zeros((0, xyz.shape[-1]), dtype=np.float32)
    xyz_second = np.diff(xyz, n=2, axis=0) if len(xyz) >= 3 else np.zeros((0, xyz.shape[-1]), dtype=np.float32)
    first_dims = actions[:, : min(7, actions.shape[-1])]
    first_dims_text = np.array2string(first_dims, precision=6, suppress_small=False, threshold=10000)
    parts = [
        f"{name}: shape={actions.shape}",
        f"xyz_min={np.min(xyz, axis=0).tolist() if xyz.size else []}",
        f"xyz_max={np.max(xyz, axis=0).tolist() if xyz.size else []}",
        f"xyz_norm_mean={float(np.mean(xyz_norm)):.6f}",
        f"xyz_norm_max={float(np.max(xyz_norm)):.6f}",
        f"xyz_adj_abs_mean={float(np.mean(np.abs(xyz_adj))):.6f}" if xyz_adj.size else "xyz_adj_abs_mean=0.000000",
        f"xyz_second_abs_mean={float(np.mean(np.abs(xyz_second))):.6f}"
        if xyz_second.size
        else "xyz_second_abs_mean=0.000000",
        f"first{first_dims.shape[-1]}_seq={first_dims_text}",
    ]
    if actions.shape[-1] > 6:
        parts.append(f"gripper_minmax=({float(np.min(actions[:, 6])):.6f}, {float(np.max(actions[:, 6])):.6f})")
    return " ".join(parts)


class Pi05ActionAdapter:
    """Adapt raw phase-1 action chunks into OpenPI/pi0.5-normalized actions."""

    def __init__(
        self,
        config: SteeringConfig,
        *,
        action_norm_stats: transforms.NormStats | None,
        use_quantile_norm: bool,
    ):
        self._config = config
        self._action_norm_stats = action_norm_stats
        self._use_quantile_norm = use_quantile_norm

    def to_phase1_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 1:
            horizon = self._config.action_horizon
            action_dim = self._config.max_action_dim
            if horizon is None:
                raise ValueError("Cannot reshape flat phase-1 actions without action_horizon.")
            actions = actions.reshape(horizon, action_dim)
        elif actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        elif actions.ndim != 2:
            raise ValueError(f"Expected phase-1 actions with rank 1, 2, or single-batch rank 3, got {actions.shape}.")

        target_horizon = self._config.action_horizon or actions.shape[0]
        target_dim = self._config.action_dim or actions.shape[-1]
        actions = self._fit_shape(actions, target_horizon, target_dim)
        normalized = self._normalize_for_pi05(actions).astype(np.float32)
        # if normalized.shape[-1] > _GRIPPER_ACTION_INDEX:
        #     normalized[..., _GRIPPER_ACTION_INDEX] = 0.0  # Use downstream policy mean for gripper init.
        return normalized

    @staticmethod
    def _fit_shape(actions: np.ndarray, horizon: int, action_dim: int) -> np.ndarray:
        out = np.zeros((horizon, action_dim), dtype=np.float32)
        horizon_n = min(horizon, actions.shape[0])
        dim_n = min(action_dim, actions.shape[-1])
        out[:horizon_n, :dim_n] = actions[:horizon_n, :dim_n]
        return out

    def _normalize_for_pi05(self, actions: np.ndarray) -> np.ndarray:
        stats = self._action_norm_stats
        if stats is None:
            return actions
        normalized = np.array(actions, dtype=np.float32, copy=True)
        if self._use_quantile_norm:
            if stats.q01 is None or stats.q99 is None:
                raise ValueError("pi0.5 action quantile stats are required for quantile normalization.")
            q01 = np.asarray(stats.q01, dtype=np.float32)
            q99 = np.asarray(stats.q99, dtype=np.float32)
            dim = min(actions.shape[-1], q01.shape[-1])
            normalized[..., :dim] = (
                (actions[..., :dim] - q01[..., :dim]) / (q99[..., :dim] - q01[..., :dim] + 1e-6) * 2.0 - 1.0
            )
            return normalized
        mean = np.asarray(stats.mean, dtype=np.float32)
        std = np.asarray(stats.std, dtype=np.float32)
        dim = min(actions.shape[-1], mean.shape[-1])
        normalized[..., :dim] = (actions[..., :dim] - mean[..., :dim]) / (std[..., :dim] + 1e-6)
        return normalized


class OpenPIPhase2Policy:
    """OpenPI implementation of phase-2 refinement from phase-1 actions."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        config: SteeringConfig,
        *,
        action_norm_stats: transforms.NormStats | None,
        use_quantile_norm: bool,
    ):
        self._policy = policy
        self._config = config
        self._action_adapter = Pi05ActionAdapter(
            config,
            action_norm_stats=action_norm_stats,
            use_quantile_norm=use_quantile_norm,
        )
        self._policy_sample_kwargs = dict(getattr(policy, "_sample_kwargs", {}) or {})
        self._metadata = {
            **getattr(policy, "metadata", {}),
            "steerer": {
                "enabled": True,
                "mode": config.mode,
                "checkpoint_dir": config.checkpoint_dir,
                "enable_by_default": config.enable_by_default,
            },
            f"{_legacy_mode_key(config.mode)}_steerer": {
                "enabled": True,
                "checkpoint_dir": config.checkpoint_dir,
                "enable_by_default": config.enable_by_default,
            },
        }

    def infer_direct(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None,
        sample_kwargs: dict[str, Any],
    ) -> dict:
        sample_kwargs = dict(sample_kwargs)
        phase1_actions = sample_kwargs.pop("phase1_actions", None)
        if phase1_actions is None:
            return self._policy.infer(obs, noise=noise, sample_kwargs=sample_kwargs)

        raw_actions = np.asarray(phase1_actions, dtype=np.float32)
        policy_norm_actions = self._action_adapter.to_phase1_actions(raw_actions)
        sample_kwargs["phase1_actions"] = policy_norm_actions
        result = self._policy.infer(obs, noise=noise, sample_kwargs=sample_kwargs)
        result["phase1_actions_raw"] = raw_actions
        result["policy_norm_actions"] = policy_norm_actions.astype(np.float32)
        _set_phase2_debug(result, {**self._policy_sample_kwargs, **sample_kwargs})
        result["steerer"] = {
            "enabled": False,
            "reused_phase1_actions": True,
            "phase2_steps": sample_kwargs.get("phase2_steps"),
        }
        return result

    def refine(
        self,
        obs: dict,
        *,
        phase1_action: Phase1Action,
        phase2_steps: float,
        noise: np.ndarray | None,
        sample_kwargs: dict[str, Any],
    ) -> dict:
        raw_actions = np.asarray(phase1_action.actions, dtype=np.float32)
        policy_norm_actions = self._action_adapter.to_phase1_actions(raw_actions)
        log_parts = [_format_action_debug("raw_denormalized", raw_actions)]
        if phase1_action.model_actions is not None:
            log_parts.insert(0, _format_action_debug("model_normalized", phase1_action.model_actions))
        # LOGGER.info("%s steerer action debug | %s", phase1_action.mode.upper(), " | ".join(log_parts))

        sample_kwargs.update(
            {
                "phase1_actions": policy_norm_actions,
                "phase2_steps": phase2_steps,
            }
        )
        result = self._policy.infer(obs, noise=noise, sample_kwargs=sample_kwargs)
        _set_phase2_debug(result, {**self._policy_sample_kwargs, **sample_kwargs})
        legacy_key = _legacy_mode_key(phase1_action.mode)
        result["steerer"] = {
            "enabled": True,
            "mode": phase1_action.mode,
            "phase2_steps": phase2_steps,
        }
        result[f"{legacy_key}_steerer"] = {
            "enabled": True,
            "phase2_steps": phase2_steps,
        }
        steerer_model_actions = (
            raw_actions.astype(np.float32)
            if phase1_action.model_actions is None
            else np.asarray(phase1_action.model_actions, dtype=np.float32)
        )
        result["phase1_actions_model"] = steerer_model_actions
        result["phase1_actions_raw"] = raw_actions.astype(np.float32)
        result["policy_norm_actions"] = policy_norm_actions.astype(np.float32)
        result["steerer_phase1_actions"] = steerer_model_actions
        result["steerer_phase1_actions_raw"] = raw_actions.astype(np.float32)
        result[f"{legacy_key}_phase1_actions"] = steerer_model_actions
        result[f"{legacy_key}_phase1_actions_raw"] = raw_actions.astype(np.float32)
        result[f"{legacy_key}_policy_norm_actions"] = policy_norm_actions.astype(np.float32)
        return result

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class EvoOpenPISteeredPolicy(SteeredPhase2Runtime):
    """OpenPI phase-2 policy composed with an Evo-1 phase-1 backend."""

    def __init__(
        self,
        pi05_policy: _base_policy.BasePolicy,
        config: SteeringConfig,
        *,
        action_norm_stats: transforms.NormStats | None,
        use_quantile_norm: bool,
    ):
        backend = Evo1Backend(config)
        phase2_policy = OpenPIPhase2Policy(
            pi05_policy,
            backend.config,
            action_norm_stats=action_norm_stats,
            use_quantile_norm=use_quantile_norm,
        )
        super().__init__(
            backend=backend,
            phase2_policy=phase2_policy,
            config=backend.config,
            enable_keys=("enable_evo1_steerer", "enable_steerer"),
        )



def create_steered_openpi_policy(
    policy: _base_policy.BasePolicy,
    config: SteeringConfig,
    *,
    action_norm_stats: transforms.NormStats | None,
    use_quantile_norm: bool,
) -> _base_policy.BasePolicy:
    mode = config.mode.lower()
    if mode == "evo":
        return EvoOpenPISteeredPolicy(
            policy,
            config,
            action_norm_stats=action_norm_stats,
            use_quantile_norm=use_quantile_norm,
        )
    raise ValueError(f"Unsupported steerer mode '{config.mode}'. Expected 'evo'.")


def _legacy_mode_key(mode: str) -> str:
    return "evo1" if mode.lower() == "evo" else mode.lower()


def _set_phase2_debug(result: dict[str, Any], sample_kwargs: dict[str, Any]) -> None:
    if "phase2_steps" not in sample_kwargs and "steerer_phase2_steps" not in sample_kwargs:
        return
    phase2_steps = float(sample_kwargs.get("phase2_steps", sample_kwargs.get("steerer_phase2_steps")))
    num_steps = float(sample_kwargs.get("num_steps", 10.0))
    if not np.isfinite(num_steps) or num_steps <= 0.0:
        num_steps = 10.0
    noise_level = float(np.clip(phase2_steps / num_steps, 0.0, 1.0))
    effective_noise_level = noise_level
    random_noise_ratio = float(sample_kwargs.get("random_noise_ratio", 0.0) or 0.0)
    result["phase2_debug"] = {
        "policy_inference_steps": num_steps,
        "phase2_steps": phase2_steps,
        "phase2_noise_level": noise_level,
        "phase2_effective_noise_level": effective_noise_level,
        "phase2_effective_denoise_steps": effective_noise_level * num_steps,
        "frs": bool(sample_kwargs.get("frs", False)),
        "random_noise_ratio": random_noise_ratio,
    }
