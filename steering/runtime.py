from __future__ import annotations

from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from steering.config import SteeringConfig
from steering.schemas import Phase1Action


class SteeredPhase2Runtime(_base_policy.BasePolicy):
    """Compose a phase-1 steerer backend with a phase-2 downstream policy."""

    def __init__(
        self,
        *,
        backend: Any,
        phase2_policy: Any,
        config: SteeringConfig,
        enable_keys: tuple[str, ...],
    ):
        self._backend = backend
        self._phase2_policy = phase2_policy
        self._config = config
        self._enable_keys = enable_keys

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> dict:
        sample_kwargs = dict(sample_kwargs or {})
        enabled = self._pop_enabled(sample_kwargs)
        if not enabled:
            return self._phase2_policy.infer_direct(obs, noise=noise, sample_kwargs=sample_kwargs)

        phase2_steps = self._pop_phase2_steps(sample_kwargs)
        hardcode_phase1_actions = sample_kwargs.pop("phase1_actions", None)
        phase1_action = self._backend.predict(obs)
        if hardcode_phase1_actions is not None:
            phase1_action = self._combine_phase1_actions(phase1_action, hardcode_phase1_actions)
        return self._phase2_policy.refine(
            obs,
            phase1_action=phase1_action,
            phase2_steps=phase2_steps,
            noise=noise,
            sample_kwargs=sample_kwargs,
        )

    def _pop_enabled(self, sample_kwargs: dict[str, Any]) -> bool:
        enabled: bool | None = None
        for key in self._enable_keys:
            if key in sample_kwargs:
                value = bool(sample_kwargs.pop(key))
                if enabled is None:
                    enabled = value
        if enabled is not None:
            return enabled
        return self._config.enable_by_default

    @staticmethod
    def _pop_phase2_steps(sample_kwargs: dict[str, Any]) -> float:
        if "steerer_phase2_steps" in sample_kwargs:
            return float(sample_kwargs.pop("steerer_phase2_steps"))
        if "phase2_steps" in sample_kwargs:
            return float(sample_kwargs.pop("phase2_steps"))
        raise ValueError("Steered phase-2 runtime requires phase2_steps in sample_kwargs.")

    @staticmethod
    def _combine_phase1_actions(phase1_action: Phase1Action, extra_actions: Any) -> Phase1Action:
        backend_actions = np.asarray(phase1_action.actions, dtype=np.float32)
        hardcode_actions = np.asarray(extra_actions, dtype=np.float32)
        if backend_actions.ndim != 2:
            raise ValueError(f"Expected backend phase1 actions [T,D], got {backend_actions.shape}.")
        if hardcode_actions.ndim == 3 and hardcode_actions.shape[0] == 1:
            hardcode_actions = hardcode_actions[0]
        if hardcode_actions.ndim != 2:
            raise ValueError(f"Expected hardcode phase1 actions [T,D], got {hardcode_actions.shape}.")

        horizon = max(backend_actions.shape[0], hardcode_actions.shape[0])
        action_dim = max(backend_actions.shape[1], hardcode_actions.shape[1])
        combined_actions = np.zeros((horizon, action_dim), dtype=np.float32)
        combined_actions[: backend_actions.shape[0], : backend_actions.shape[1]] += backend_actions
        combined_actions[: hardcode_actions.shape[0], : hardcode_actions.shape[1]] += hardcode_actions

        metadata = dict(phase1_action.metadata)
        metadata["combined_with_hardcode_phase1"] = True
        return Phase1Action(
            actions=combined_actions,
            mode=phase1_action.mode,
            model_actions=phase1_action.model_actions,
            metadata=metadata,
        )

    def reset(self) -> None:
        for component in (self._backend, self._phase2_policy):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._phase2_policy.metadata)
        backend_metadata = dict(getattr(self._backend, "metadata", {}) or {})
        if backend_metadata:
            backend_metadata.setdefault("enabled", True)
            metadata.setdefault("steerer", backend_metadata)
            mode = backend_metadata.get("mode")
            if mode:
                metadata.setdefault(f"{mode}_steerer", backend_metadata)
            if mode == "evo":
                metadata.setdefault("evo1_steerer", backend_metadata)
        return metadata
