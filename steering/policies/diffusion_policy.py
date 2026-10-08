from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import pathlib
import sys
from typing import Any

import dill
import numpy as np
from PIL import Image
import torch

from steering.backends.evo1 import Evo1Backend
from steering.config import SteeringConfig
from steering.runtime import SteeredPhase2Runtime
from steering.schemas import Phase1Action


def _find_repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "diffusion_policy" / "diffusion_policy").is_dir():
            return parent
    raise FileNotFoundError("Could not find diffusion_policy/diffusion_policy from the steering source tree.")


def _ensure_diffusion_policy_on_path() -> pathlib.Path:
    repo_root = _find_repo_root()
    path = repo_root / "diffusion_policy"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return repo_root


@dataclasses.dataclass(frozen=True)
class DiffusionPolicyRuntimeConfig:
    checkpoint_path: str
    config: str | None = None
    device: str = "cuda"
    use_ema: bool = True
    num_inference_steps: int | None = None
    frs: bool = False
    use_ddim: bool = False


class DiffusionPolicyPhase2Policy:
    """Diffusion Policy downstream policy with optional phase-1 action initialization."""

    def __init__(
        self,
        policy: torch.nn.Module,
        runtime_config: DiffusionPolicyRuntimeConfig,
        cfg: Any,
        checkpoint_path: pathlib.Path,
    ):
        self._policy = policy
        self._runtime_config = runtime_config
        self._cfg = cfg
        self._checkpoint_path = checkpoint_path
        self._shape_meta = cfg.task.shape_meta
        self._device = torch.device(runtime_config.device)
        self._n_obs_steps = int(getattr(policy, "n_obs_steps", cfg.n_obs_steps))
        self._action_horizon = int(getattr(policy, "horizon", cfg.horizon))
        self._action_dim = int(getattr(policy, "action_dim", self._shape_meta["action"]["shape"][0]))
        self._rgb_keys = [
            key for key, attr in self._shape_meta["obs"].items() if attr.get("type", "low_dim") == "rgb"
        ]
        self._lowdim_keys = [
            key for key, attr in self._shape_meta["obs"].items() if attr.get("type", "low_dim") == "low_dim"
        ]
        self._metadata = {
            "phase2_policy_type": "diffusion_policy",
            "phase2_policy_config": runtime_config.config,
            "phase2_policy_dir": str(checkpoint_path),
            "diffusion_policy": {
                "checkpoint_path": str(checkpoint_path),
                "use_ema": runtime_config.use_ema,
                "num_inference_steps": runtime_config.num_inference_steps,
                "frs": bool(runtime_config.frs),
                "scheduler": _diffusion_policy_scheduler_name(policy),
                "use_ddim": bool(runtime_config.use_ddim),
                "n_obs_steps": self._n_obs_steps,
                "action_horizon": self._action_horizon,
                "action_dim": self._action_dim,
            },
        }
        action_norm_stats = _diffusion_policy_action_norm_stats(policy)
        if action_norm_stats is not None:
            self._metadata["action_norm_stats"] = action_norm_stats
            self._metadata["use_quantile_norm"] = False

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def infer_direct(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None,
        sample_kwargs: dict[str, Any],
    ) -> dict:
        del noise
        sample_kwargs = dict(sample_kwargs)
        phase1_actions = sample_kwargs.pop("phase1_actions", None)
        phase2_steps = self._pop_optional_phase2_steps(sample_kwargs)
        if phase1_actions is not None:
            if phase2_steps is None:
                phase2_steps = 0.0
            if float(phase2_steps) <= 0.0 and not self._uses_frs_noise(sample_kwargs):
                return self._phase1_passthrough(
                    Phase1Action(actions=np.asarray(phase1_actions, dtype=np.float32), mode="reused"),
                    phase2_steps=float(phase2_steps),
                    action_horizon=int(sample_kwargs.get("action_horizon", self._action_horizon)),
                )
            if phase2_steps is not None:
                sample_kwargs.setdefault("num_inference_steps", self._default_num_inference_steps())
                sample_kwargs.setdefault("refinement_steps", self._normalize_num_inference_steps(phase2_steps))
            normalized_phase1 = self._normalize_action_chunk(
                phase1_actions,
                horizon=int(sample_kwargs.get("action_horizon", self._action_horizon)),
            )
            result = self._infer(obs, past_action=normalized_phase1, sample_kwargs=sample_kwargs)
            self._set_phase2_debug(result, sample_kwargs)
            result["steerer"] = {
                "enabled": False,
                "reused_phase1_actions": True,
                "phase2_steps": phase2_steps,
            }
            phase1_actions_array = np.asarray(phase1_actions, dtype=np.float32)
            policy_norm_actions = self._squeeze_single_batch_action(
                normalized_phase1.detach().cpu().numpy().astype(np.float32)
            )
            result["phase1_actions_raw"] = phase1_actions_array
            result["policy_norm_actions"] = policy_norm_actions
            result["steerer_phase1_actions"] = phase1_actions_array
            result["steerer_phase1_actions_raw"] = phase1_actions_array
            result["steerer_phase1_actions_normalized"] = policy_norm_actions
            return result
        if phase2_steps is not None:
            sample_kwargs.setdefault("num_inference_steps", self._normalize_num_inference_steps(phase2_steps))
        return self._infer(obs, past_action=None, sample_kwargs=sample_kwargs)

    def refine(
        self,
        obs: dict,
        *,
        phase1_action: Phase1Action,
        phase2_steps: float,
        noise: np.ndarray | None,
        sample_kwargs: dict[str, Any],
    ) -> dict:
        del noise
        sample_kwargs = dict(sample_kwargs)
        sample_kwargs.pop("phase1_actions", None)
        if float(phase2_steps) <= 0.0 and not self._uses_frs_noise(sample_kwargs):
            return self._phase1_passthrough(
                phase1_action,
                phase2_steps=phase2_steps,
                action_horizon=int(sample_kwargs.get("action_horizon", self._action_horizon)),
            )
        sample_kwargs.setdefault("num_inference_steps", self._default_num_inference_steps())
        sample_kwargs.setdefault("refinement_steps", self._normalize_num_inference_steps(phase2_steps))
        normalized_phase1 = self._normalize_action_chunk(
            phase1_action.actions,
            horizon=int(sample_kwargs.get("action_horizon", self._action_horizon)),
        )
        result = self._infer(obs, past_action=normalized_phase1, sample_kwargs=sample_kwargs)
        self._set_phase2_debug(result, sample_kwargs)
        result["steerer"] = {
            "enabled": True,
            "mode": phase1_action.mode,
            "phase2_steps": phase2_steps,
        }
        phase1_actions_array = np.asarray(phase1_action.actions, dtype=np.float32)
        steerer_model_actions = (
            phase1_actions_array
            if phase1_action.model_actions is None
            else np.asarray(phase1_action.model_actions, dtype=np.float32)
        )
        normalized_phase1_array = self._squeeze_single_batch_action(
            normalized_phase1.detach().cpu().numpy().astype(np.float32)
        )
        result["phase1_actions_model"] = steerer_model_actions
        result["phase1_actions_raw"] = phase1_actions_array
        result["policy_norm_actions"] = normalized_phase1_array
        result["steerer_phase1_actions"] = steerer_model_actions
        result["steerer_phase1_actions_raw"] = phase1_actions_array
        result["steerer_phase1_actions_normalized"] = normalized_phase1_array
        if phase1_action.model_actions is not None:
            result["steerer_phase1_model_actions"] = np.asarray(phase1_action.model_actions, dtype=np.float32)
        return result

    def _infer(
        self,
        obs: Mapping[str, Any],
        *,
        past_action: torch.Tensor | None,
        sample_kwargs: dict[str, Any],
    ) -> dict:
        sample_kwargs = dict(sample_kwargs)
        num_inference_steps = sample_kwargs.pop("num_inference_steps", self._default_num_inference_steps())
        refinement_steps = sample_kwargs.pop("refinement_steps", None)
        action_horizon = int(sample_kwargs.pop("action_horizon", self._action_horizon))
        random_noise_ratio = float(sample_kwargs.pop("random_noise_ratio", 0.0) or 0.0)
        frs = bool(sample_kwargs.pop("frs", self._runtime_config.frs))
        if sample_kwargs:
            raise ValueError(f"Unsupported Diffusion Policy sample kwargs: {sorted(sample_kwargs.keys())}")

        obs_dict = self._obs_to_policy_input(obs)
        if past_action is not None:
            obs_dict["past_action"] = self._fit_action_tensor(past_action, horizon=action_horizon, dim=self._action_dim)

        old_num_inference_steps = getattr(self._policy, "num_inference_steps", None)
        had_refinement_steps = hasattr(self._policy, "refinement_steps")
        old_refinement_steps = getattr(self._policy, "refinement_steps", None)
        had_frs = hasattr(self._policy, "frs")
        old_frs = getattr(self._policy, "frs", None)
        had_random_noise_ratio = hasattr(self._policy, "random_noise_ratio")
        old_random_noise_ratio = getattr(self._policy, "random_noise_ratio", None)
        if num_inference_steps is not None:
            self._policy.num_inference_steps = self._normalize_num_inference_steps(num_inference_steps)
        if refinement_steps is not None:
            self._policy.refinement_steps = self._normalize_num_inference_steps(refinement_steps)
        self._policy.frs = frs
        self._policy.random_noise_ratio = random_noise_ratio
        try:
            with torch.no_grad():
                pred = self._policy.predict_action(obs_dict)
        finally:
            if old_num_inference_steps is not None:
                self._policy.num_inference_steps = old_num_inference_steps
            if refinement_steps is not None:
                if had_refinement_steps:
                    self._policy.refinement_steps = old_refinement_steps
                else:
                    delattr(self._policy, "refinement_steps")
            if had_frs:
                self._policy.frs = old_frs
            else:
                delattr(self._policy, "frs")
            if had_random_noise_ratio:
                self._policy.random_noise_ratio = old_random_noise_ratio
            else:
                delattr(self._policy, "random_noise_ratio")

        actions = pred["action"].detach().to(dtype=torch.float32, device="cpu").numpy()[0]
        result = {
            "actions": actions.astype(np.float32),
        }
        if "action_pred" in pred:
            result["diffusion_policy_action_pred"] = (
                pred["action_pred"].detach().to(dtype=torch.float32, device="cpu").numpy()[0].astype(np.float32)
            )
        return result

    def _phase1_passthrough(
        self,
        phase1_action: Phase1Action,
        *,
        phase2_steps: float,
        action_horizon: int | None = None,
    ) -> dict:
        horizon = int(self._action_horizon if action_horizon is None else action_horizon)
        actions = self._fit_action_array(
            np.asarray(phase1_action.actions, dtype=np.float32),
            horizon=horizon,
            dim=self._action_dim,
        )
        normalized_phase1 = self._normalize_action_chunk(actions, horizon=horizon)
        normalized_phase1_array = self._squeeze_single_batch_action(
            normalized_phase1.detach().cpu().numpy().astype(np.float32)
        )
        result = {
            "actions": actions.astype(np.float32),
            "steerer": {
                "enabled": phase1_action.mode != "reused",
                "mode": phase1_action.mode,
                "phase2_steps": phase2_steps,
                "passthrough": True,
            },
            "phase1_actions_raw": actions.astype(np.float32),
            "policy_norm_actions": normalized_phase1_array,
            "steerer_phase1_actions": actions.astype(np.float32),
            "steerer_phase1_actions_raw": actions.astype(np.float32),
            "steerer_phase1_actions_normalized": normalized_phase1_array,
            "phase2_debug": {
                "policy_inference_steps": float(self._default_num_inference_steps() or 0.0),
                "phase2_steps": float(phase2_steps),
                "phase2_noise_level": 0.0,
                "phase2_effective_noise_level": 0.0,
                "phase2_effective_denoise_steps": 0.0,
                "frs": False,
                "random_noise_ratio": 0.0,
            },
        }
        if phase1_action.model_actions is not None:
            result["phase1_actions_model"] = np.asarray(phase1_action.model_actions, dtype=np.float32)
            result["steerer_phase1_model_actions"] = np.asarray(phase1_action.model_actions, dtype=np.float32)
        return result

    def _set_phase2_debug(self, result: dict[str, Any], sample_kwargs: Mapping[str, Any]) -> None:
        num_steps = float(sample_kwargs.get("num_inference_steps", 0.0) or 0.0)
        refinement_steps = sample_kwargs.get("refinement_steps")
        if refinement_steps is None or num_steps <= 0.0:
            return
        phase2_steps = float(refinement_steps)
        noise_level = float(np.clip(phase2_steps / num_steps, 0.0, 1.0))
        random_noise_ratio = float(sample_kwargs.get("random_noise_ratio", 0.0) or 0.0)
        effective_noise_level = noise_level
        frs = bool(sample_kwargs.get("frs", self._runtime_config.frs))
        result["phase2_debug"] = {
            "policy_inference_steps": num_steps,
            "phase2_steps": phase2_steps,
            "phase2_noise_level": noise_level,
            "phase2_effective_noise_level": effective_noise_level,
            "phase2_effective_denoise_steps": effective_noise_level * num_steps,
            "frs": frs,
            "random_noise_ratio": random_noise_ratio,
        }

    def _default_num_inference_steps(self) -> int | None:
        if self._runtime_config.num_inference_steps is not None:
            return self._runtime_config.num_inference_steps
        value = getattr(self._policy, "num_inference_steps", None)
        if value is None:
            return None
        return self._normalize_num_inference_steps(value)

    def _obs_to_policy_input(self, obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        result = {}
        image_dict = self._get_image_dict(obs)
        for key in self._rgb_keys:
            value = self._lookup_obs_value(obs, image_dict, key)
            result[key] = self._image_to_tensor(value, self._shape_meta["obs"][key])
        for key in self._lowdim_keys:
            value = self._lookup_obs_value(obs, {}, key)
            result[key] = self._lowdim_to_tensor(value, self._shape_meta["obs"][key])
        return result

    def _lookup_obs_value(self, obs: Mapping[str, Any], image_dict: Mapping[str, Any], key: str) -> Any:
        attr = self._shape_meta["obs"][key]
        source_key = attr.get("lerobot_key", key)
        candidates = [
            key,
            source_key,
            source_key.removeprefix("observation.images."),
            source_key.replace("observation.", "observation/"),
        ]
        if attr.get("type", "low_dim") == "low_dim":
            candidates.extend(["state", "observation/state"])
        for candidate in candidates:
            if candidate in image_dict:
                return image_dict[candidate]
            if candidate in obs:
                return obs[candidate]
        raise KeyError(f"Diffusion Policy missing observation key {key!r} (source {source_key!r}).")

    def _image_to_tensor(self, value: Any, attr: Mapping[str, Any]) -> torch.Tensor:
        shape = tuple(int(v) for v in attr["shape"])
        frames = np.asarray(value)
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4:
            raise ValueError(f"Expected image obs rank 3 or 4, got {frames.shape}.")
        frames = self._fit_time(frames, self._n_obs_steps)
        processed = [self._prepare_image(frame, shape=shape) for frame in frames]
        array = np.stack(processed, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0).to(self._device)

    def _lowdim_to_tensor(self, value: Any, attr: Mapping[str, Any]) -> torch.Tensor:
        dim = int(attr["shape"][0])
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[None]
        if array.ndim != 2:
            raise ValueError(f"Expected low-dim obs rank 1 or 2, got {array.shape}.")
        array = self._fit_last_dim(array, dim)
        array = self._fit_time(array, self._n_obs_steps)
        return torch.from_numpy(array.astype(np.float32)).unsqueeze(0).to(self._device)

    def _normalize_action_chunk(self, actions: np.ndarray, *, horizon: int) -> torch.Tensor:
        action = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self._device)
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 2:
            raise ValueError(f"Expected phase-1 action chunk [T,D], got {tuple(action.shape)}")
        action = self._fit_action_tensor(action.unsqueeze(0), horizon=horizon, dim=self._action_dim)[0]
        return self._policy.normalizer["action"].normalize(action).unsqueeze(0)

    @staticmethod
    def _fit_action_array(action: np.ndarray, *, horizon: int, dim: int) -> np.ndarray:
        array = np.asarray(action, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected phase-1 action chunk [T,D], got {array.shape}")
        out = np.zeros((horizon, dim), dtype=np.float32)
        horizon_n = min(horizon, array.shape[0])
        dim_n = min(dim, array.shape[-1])
        out[:horizon_n, :dim_n] = array[:horizon_n, :dim_n]
        if 0 < horizon_n < horizon:
            out[horizon_n:, :dim_n] = array[horizon_n - 1, :dim_n]
        return out

    @staticmethod
    def _fit_action_tensor(action: torch.Tensor, *, horizon: int, dim: int) -> torch.Tensor:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        out = torch.zeros((action.shape[0], horizon, dim), device=action.device, dtype=action.dtype)
        horizon_n = min(horizon, action.shape[1])
        dim_n = min(dim, action.shape[2])
        out[:, :horizon_n, :dim_n] = action[:, :horizon_n, :dim_n]
        return out

    @staticmethod
    def _squeeze_single_batch_action(action: np.ndarray) -> np.ndarray:
        if action.ndim == 3 and action.shape[0] == 1:
            return action[0]
        return action

    @staticmethod
    def _normalize_num_inference_steps(value: float | int) -> int:
        return max(1, int(round(float(value))))

    @staticmethod
    def _pop_optional_phase2_steps(sample_kwargs: dict[str, Any]) -> float | None:
        if "steerer_phase2_steps" in sample_kwargs:
            return float(sample_kwargs.pop("steerer_phase2_steps"))
        if "phase2_steps" in sample_kwargs:
            return float(sample_kwargs.pop("phase2_steps"))
        return None

    def _uses_frs_noise(self, sample_kwargs: Mapping[str, Any]) -> bool:
        return (
            bool(sample_kwargs.get("frs", self._runtime_config.frs))
            and float(sample_kwargs.get("random_noise_ratio", 0.0) or 0.0) > 0.0
        )

    @staticmethod
    def _fit_last_dim(array: np.ndarray, dim: int) -> np.ndarray:
        if array.shape[-1] > dim:
            raise ValueError(f"Input dim {array.shape[-1]} exceeds configured dim {dim}.")
        if array.shape[-1] == dim:
            return array
        out = np.zeros((*array.shape[:-1], dim), dtype=np.float32)
        out[..., : array.shape[-1]] = array
        return out

    @staticmethod
    def _fit_time(array: np.ndarray, length: int) -> np.ndarray:
        if array.shape[0] == length:
            return array
        if array.shape[0] > length:
            return array[-length:]
        pad = np.repeat(array[:1], length - array.shape[0], axis=0)
        return np.concatenate([pad, array], axis=0)

    @staticmethod
    def _get_image_dict(obs: Mapping[str, Any]) -> dict[str, Any]:
        image_dict = obs.get("image")
        images = dict(image_dict) if isinstance(image_dict, Mapping) else {}
        aliases = {
            "agentview_image": ("robot0_agentview_left", "base_0_rgb", "observation.images.robot0_agentview_left"),
            "wrist_image": ("robot0_eye_in_hand", "left_wrist_0_rgb", "wrist_0_rgb", "observation.images.robot0_eye_in_hand"),
            "robot0_agentview_left": ("agentview_image", "base_0_rgb"),
            "robot0_eye_in_hand": ("wrist_image", "left_wrist_0_rgb", "wrist_0_rgb"),
        }
        for target, sources in aliases.items():
            if target not in images:
                for source in sources:
                    if source in images:
                        images[target] = images[source]
                        break
        candidates = {
            "agentview_image": obs.get("observation/image"),
            "wrist_image": obs.get("observation/wrist_image"),
            "robot0_agentview_left": obs.get("observation/image"),
            "robot0_eye_in_hand": obs.get("observation/wrist_image"),
        }
        for key, value in candidates.items():
            if key not in images and value is not None:
                images[key] = value
        return images

    @classmethod
    def _prepare_image(cls, image: np.ndarray, *, shape: tuple[int, int, int]) -> np.ndarray:
        channels, height, width = shape
        if channels != 3:
            raise ValueError(f"Only RGB images are supported, got shape {shape}.")
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"Expected RGB image rank 3, got {array.shape}.")
        if array.shape[-1] != 3 and array.shape[0] == 3:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got {array.shape}.")
        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            if array.min() < 0.0:
                array = (array + 1.0) / 2.0
            if array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return cls._center_crop_resize(array, width=width, height=height).transpose(2, 0, 1)

    @staticmethod
    def _center_crop_resize(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
        pil = Image.fromarray(image).convert("RGB")
        src_w, src_h = pil.size
        scale = max(width / src_w, height / src_h)
        resized = pil.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        resized_w, resized_h = resized.size
        left = max((resized_w - width) // 2, 0)
        top = max((resized_h - height) // 2, 0)
        return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _diffusion_policy_action_norm_stats(policy: torch.nn.Module) -> dict[str, list[float]] | None:
    normalizer = getattr(policy, "normalizer", None)
    if normalizer is None:
        return None
    try:
        action_stats = normalizer["action"].get_input_stats()
    except Exception:
        return None
    stats = {}
    for key in ("min", "max", "mean", "std"):
        if key not in action_stats:
            continue
        stats[key] = _tensor_like_to_float_list(action_stats[key])
    return stats or None


def _tensor_like_to_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        array = value.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    return [float(item) for item in np.asarray(array, dtype=np.float32).reshape(-1)]


class EvoDiffusionPolicySteeredPolicy(SteeredPhase2Runtime):
    def __init__(self, phase2_policy: DiffusionPolicyPhase2Policy, config: SteeringConfig):
        config = dataclasses.replace(
            config,
            action_horizon=phase2_policy.action_horizon,
            action_dim=phase2_policy.action_dim,
        )
        super().__init__(
            backend=Evo1Backend(config),
            phase2_policy=phase2_policy,
            config=config,
            enable_keys=("enable_evo1_steerer", "enable_steerer"),
        )



def create_steered_diffusion_policy(
    runtime_config: DiffusionPolicyRuntimeConfig,
    steerer_config: SteeringConfig,
) -> SteeredPhase2Runtime:
    phase2_policy = load_diffusion_policy_phase2_policy(runtime_config)
    mode = steerer_config.mode.lower()
    if mode == "evo":
        return EvoDiffusionPolicySteeredPolicy(phase2_policy, steerer_config)
    raise ValueError(f"Unsupported steerer mode '{steerer_config.mode}'. Expected 'evo'.")


def load_diffusion_policy_phase2_policy(runtime_config: DiffusionPolicyRuntimeConfig) -> DiffusionPolicyPhase2Policy:
    _ensure_diffusion_policy_on_path()
    import hydra

    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    checkpoint_path = _resolve_checkpoint_path(runtime_config.checkpoint_path)
    payload = _torch_load_checkpoint(checkpoint_path)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload)
    policy = workspace.ema_model if runtime_config.use_ema and getattr(workspace, "ema_model", None) is not None else workspace.model
    if runtime_config.use_ddim:
        _replace_diffusion_policy_scheduler_with_ddim(policy)
    policy.to(runtime_config.device)
    policy.eval()
    if runtime_config.num_inference_steps is not None:
        policy.num_inference_steps = DiffusionPolicyPhase2Policy._normalize_num_inference_steps(
            runtime_config.num_inference_steps
        )
    return DiffusionPolicyPhase2Policy(
        policy=policy,
        runtime_config=runtime_config,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
    )


def _resolve_checkpoint_path(path: str) -> pathlib.Path:
    candidate = pathlib.Path(path).expanduser()
    if candidate.is_file():
        return candidate
    for child in (
        candidate / "checkpoints" / "latest.ckpt",
        candidate / "latest.ckpt",
    ):
        if child.is_file():
            return child
    raise FileNotFoundError(f"Could not find Diffusion Policy checkpoint from {path!r}.")


def _torch_load_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    try:
        return torch.load(path.open("rb"), pickle_module=dill, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")


def _replace_diffusion_policy_scheduler_with_ddim(policy: torch.nn.Module) -> None:
    if not hasattr(policy, "noise_scheduler"):
        raise AttributeError("Diffusion Policy does not expose noise_scheduler; cannot enable DDIM.")
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    policy.noise_scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)


def _diffusion_policy_scheduler_name(policy: torch.nn.Module) -> str:
    scheduler = getattr(policy, "noise_scheduler", None)
    if scheduler is None:
        return ""
    return type(scheduler).__name__
