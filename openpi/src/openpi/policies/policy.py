from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_rtc = getattr(model, "sample_actions_rtc", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("frs",))
            self._sample_actions_rtc = (
                nnx_utils.module_jit(model.sample_actions_rtc, static_argnames=("prefix_attention_schedule",))
                if hasattr(model, "sample_actions_rtc")
                else None
            )
            self._rng = rng or jax.random.key(0)
        self._prev_actions: np.ndarray | None = None
        self._metadata.setdefault("openpi_rtc", {"supported": self._sample_actions_rtc is not None})

    def _shift_prefix_actions(self, action_chunk: np.ndarray, time_base: int) -> np.ndarray:
        action_chunk = np.asarray(action_chunk)
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected prefix action chunk with 2 dims, got shape {action_chunk.shape}.")
        if time_base < 0:
            time_base = 0
        if time_base >= action_chunk.shape[0]:
            raise ValueError(f"time_base {time_base} must be less than action chunk length {action_chunk.shape[0]}.")
        remainder = action_chunk[time_base + 1 :]
        pad_length = action_chunk.shape[0] - remainder.shape[0]
        if pad_length <= 0:
            return remainder
        pad = np.zeros((pad_length, action_chunk.shape[1]), dtype=action_chunk.dtype)
        return np.concatenate([remainder, pad], axis=0)

    def _prepare_rtc_prefix_actions(self, obs: dict, time_base: int) -> np.ndarray | None:
        if self._prev_actions is None:
            return None
        prefix_actions = np.asarray(self._prev_actions)
        if prefix_actions.ndim == 1:
            prefix_actions = prefix_actions[None, ...]
        if prefix_actions.ndim == 3 and prefix_actions.shape[0] == 1:
            prefix_actions = prefix_actions[0]
        if prefix_actions.ndim != 2:
            raise ValueError(f"Expected cached action chunk with 2 dims, got shape {prefix_actions.shape}.")

        rtc_inputs = jax.tree.map(lambda x: x, obs)
        rtc_inputs["actions"] = prefix_actions
        rtc_inputs = self._input_transform(rtc_inputs)
        prefix_actions = np.asarray(rtc_inputs["actions"])
        if prefix_actions.ndim == 3 and prefix_actions.shape[0] == 1:
            prefix_actions = prefix_actions[0]
        return self._shift_prefix_actions(prefix_actions, time_base)

    @override
    def infer(  # type: ignore[misc]
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        time_base: int | None = None,
        rtc_config: dict[str, Any] | None = None,
    ) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        merged_sample_kwargs = dict(self._sample_kwargs)
        if sample_kwargs is not None:
            merged_sample_kwargs.update(sample_kwargs)
        if time_base is None:
            time_base = merged_sample_kwargs.pop("time_base", None)
        else:
            merged_sample_kwargs.pop("time_base", None)
        if rtc_config is None:
            rtc_config = merged_sample_kwargs.pop("rtc_config", None)
        else:
            merged_sample_kwargs.pop("rtc_config", None)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            merged_sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        use_rtc = time_base is not None and rtc_config is not None
        rtc_disabled_by_phase2 = any(
            key in merged_sample_kwargs
            for key in ("phase1_actions", "phase2_steps", "steerer_phase2_steps")
        )
        start_time = time.monotonic()
        if use_rtc and not rtc_disabled_by_phase2:
            if self._sample_actions_rtc is None:
                raise ValueError("OpenPI RTC was requested, but this model does not support sample_actions_rtc.")
            prefix_actions = self._prepare_rtc_prefix_actions(obs, int(time_base))
            if prefix_actions is None:
                actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **merged_sample_kwargs)
                rtc_info = {"enabled": False, "reason": "missing_prev_actions"}
            else:
                rtc_sample_kwargs = {
                    key: value
                    for key, value in merged_sample_kwargs.items()
                    if key not in {"frs", "phase1_actions", "phase2_steps", "random_noise_ratio", "steerer_phase2_steps"}
                }
                rtc_sample_kwargs.update(
                    {
                        "prefix_actions": (
                            torch.from_numpy(np.asarray(prefix_actions)).to(self._pytorch_device)
                            if self._is_pytorch_model
                            else jnp.asarray(prefix_actions)
                        ),
                        "inference_delay": int(rtc_config["inference_delay"]),
                        "prefix_attention_horizon": int(rtc_config["prefix_attention_horizon"]),
                        "max_guidance_weight": float(rtc_config["max_guidance_weight"]),
                    }
                )
                if "prefix_attention_schedule" in rtc_config:
                    rtc_sample_kwargs["prefix_attention_schedule"] = str(rtc_config["prefix_attention_schedule"])
                actions = self._sample_actions_rtc(sample_rng_or_pytorch_device, observation, **rtc_sample_kwargs)
                rtc_info = {
                    "enabled": True,
                    "time_base": int(time_base),
                    "inference_delay": int(rtc_config["inference_delay"]),
                    "prefix_attention_horizon": int(rtc_config["prefix_attention_horizon"]),
                }
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **merged_sample_kwargs)
            rtc_info = {"enabled": False, "reason": "phase2_sample_kwargs"} if use_rtc else None
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if "actions" in outputs:
            self._prev_actions = np.asarray(outputs["actions"]).copy()
        if rtc_info is not None:
            outputs["openpi_rtc"] = rtc_info
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @override
    def reset(self) -> None:
        self._prev_actions = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict, *, sample_kwargs: dict[str, Any] | None = None) -> dict:  # type: ignore[misc]
        if sample_kwargs is None:
            results = self._policy.infer(obs)
        else:
            results = self._policy.infer(obs, sample_kwargs=sample_kwargs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._policy, "metadata", {})
