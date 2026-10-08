from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import logging
import pathlib
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

from steering.backends.evo1 import Evo1Backend
from steering.config import SteeringConfig
from steering.runtime import SteeredPhase2Runtime
from steering.schemas import Phase1Action


LOGGER = logging.getLogger(__name__)


def _find_repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "FastWAM" / "src" / "fastwam").is_dir():
            return parent
    raise FileNotFoundError("Could not find FastWAM/src/fastwam from the steering source tree.")


def _ensure_fastwam_on_path() -> pathlib.Path:
    repo_root = _find_repo_root()
    for path in (repo_root / "FastWAM" / "src", repo_root / "FastWAM"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


@dataclasses.dataclass(frozen=True)
class FastWAMRuntimeConfig:
    checkpoint_path: str
    dataset_stats_path: str
    config: str
    device: str = "cuda"
    mixed_precision: str = "bf16"
    num_inference_steps: int = 20
    text_cfg_scale: float = 1.0
    action_cfg_scale: float = 1.0
    sigma_shift: float | None = None
    seed: int | None = None
    rand_device: str = "cpu"
    tiled: bool = False
    frs: bool = False
    negative_prompt: str = ""
    prompt_template: str = "A video recorded from a robot's point of view executing the following instruction: {task}"


class FastWAMPhase2Policy:
    """FastWAM downstream policy that can condition on an upstream phase-1 action chunk."""

    def __init__(self, model: torch.nn.Module, processor: Any, runtime_config: FastWAMRuntimeConfig, cfg: Any):
        self._model = model
        self._processor = processor
        self._runtime_config = runtime_config
        self._cfg = cfg
        self._action_horizon = int(cfg.data.train.num_frames) - 1
        action_meta = cfg.data.train.shape_meta.action
        if len(action_meta) != 1:
            raise ValueError("FastWAM steering currently expects one action entry in shape_meta.")
        self._action_dim = int(action_meta[0]["shape"])
        self._num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1
        self._video_size = tuple(int(v) for v in cfg.data.train.video_size)
        self._concat_multi_camera = str(cfg.data.train.get("concat_multi_camera", "horizontal"))
        self._metadata = {
            "phase2_policy_type": "fastwam",
            "phase2_policy_config": runtime_config.config,
            "phase2_policy_dir": runtime_config.checkpoint_path,
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
            "fastwam": {
                "checkpoint_path": runtime_config.checkpoint_path,
                "dataset_stats_path": runtime_config.dataset_stats_path,
                "num_video_frames": self._num_video_frames,
                "action_horizon": self._action_horizon,
                "action_dim": self._action_dim,
                "frs": bool(runtime_config.frs),
            },
        }

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
            phase1_action = Phase1Action(actions=np.asarray(phase1_actions, dtype=np.float32), mode="reused")
            action_horizon = int(sample_kwargs.get("action_horizon", self._action_horizon))
            if phase2_steps is None:
                phase2_steps = 0.0
            if float(phase2_steps) <= 0.0 and not self._uses_random_noise(sample_kwargs):
                result = self._phase1_passthrough(
                    phase1_action,
                    phase2_steps=float(phase2_steps),
                    action_horizon=action_horizon,
                )
            else:
                sample_kwargs.setdefault("num_inference_steps", self._runtime_config.num_inference_steps)
                sample_kwargs.setdefault("refinement_steps", self._normalize_num_inference_steps(phase2_steps))
                result = self._infer(obs, phase1_action=phase1_action, sample_kwargs=sample_kwargs)
                phase1_actions_array = np.asarray(phase1_actions, dtype=np.float32)
                horizon = self._result_action_horizon(result)
                phase1_actions_array = self._pad_action_array_temporal(phase1_actions_array, horizon=horizon)
                result["phase1_actions_raw"] = phase1_actions_array
                result["steerer_phase1_actions"] = phase1_actions_array
                result["steerer_phase1_actions_raw"] = phase1_actions_array
            result["steerer"] = {
                "enabled": False,
                "reused_phase1_actions": True,
                "phase2_steps": phase2_steps,
            }
            return result
        if phase2_steps is not None:
            sample_kwargs.setdefault("num_inference_steps", self._normalize_num_inference_steps(phase2_steps))
        return self._infer(obs, phase1_action=None, sample_kwargs=sample_kwargs)

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
        action_horizon = int(sample_kwargs.get("action_horizon", self._action_horizon))
        if float(phase2_steps) <= 0.0 and not self._uses_random_noise(sample_kwargs):
            return self._phase1_passthrough(
                phase1_action,
                phase2_steps=phase2_steps,
                action_horizon=action_horizon,
            )

        sample_kwargs.setdefault("num_inference_steps", self._runtime_config.num_inference_steps)
        sample_kwargs.setdefault("refinement_steps", self._normalize_num_inference_steps(phase2_steps))
        result = self._infer(obs, phase1_action=phase1_action, sample_kwargs=sample_kwargs)
        result["steerer"] = {
            "enabled": True,
            "mode": phase1_action.mode,
            "phase2_steps": phase2_steps,
        }
        phase1_actions = np.asarray(phase1_action.actions, dtype=np.float32)
        horizon = self._result_action_horizon(result)
        phase1_actions = self._pad_action_array_temporal(phase1_actions, horizon=horizon)
        steerer_model_actions = (
            phase1_actions
            if phase1_action.model_actions is None
            else np.asarray(phase1_action.model_actions, dtype=np.float32)
        )
        steerer_model_actions = self._pad_action_array_temporal(steerer_model_actions, horizon=horizon)
        legacy_key = "evo1" if phase1_action.mode == "evo" else phase1_action.mode
        result["phase1_actions_model"] = steerer_model_actions
        result["phase1_actions_raw"] = phase1_actions
        result["steerer_phase1_actions"] = steerer_model_actions
        result["steerer_phase1_actions_raw"] = phase1_actions
        result[f"{legacy_key}_phase1_actions"] = steerer_model_actions
        result[f"{legacy_key}_phase1_actions_raw"] = phase1_actions
        if phase1_action.model_actions is not None:
            result["steerer_phase1_model_actions"] = steerer_model_actions
        return result

    def _phase1_passthrough(
        self,
        phase1_action: Phase1Action,
        *,
        phase2_steps: float,
        action_horizon: int | None = None,
    ) -> dict:
        phase1_actions = np.asarray(phase1_action.actions, dtype=np.float32)
        horizon = int(self._action_horizon if action_horizon is None else action_horizon)
        action_meta = self._processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("FastWAM steering currently expects one action entry in shape_meta.")
        expected_dim = int(action_meta[0]["shape"])
        actions = self._fit_action_array(
            phase1_actions,
            horizon=horizon,
            dim=expected_dim,
        )
        legacy_key = "evo1" if phase1_action.mode == "evo" else phase1_action.mode
        model_actions = self._normalize_action_chunk(actions, horizon=horizon)[0]
        steerer_model_actions = (
            actions.astype(np.float32)
            if phase1_action.model_actions is None
            else np.asarray(phase1_action.model_actions, dtype=np.float32)
        )
        horizon = actions.shape[0]
        phase1_actions_padded = self._pad_action_array_temporal(phase1_actions, horizon=horizon)
        steerer_model_actions = self._pad_action_array_temporal(steerer_model_actions, horizon=horizon)
        result = {
            "actions": actions.astype(np.float32),
            "model_actions": model_actions
            .detach()
            .to(dtype=torch.float32, device="cpu")
            .numpy()
            .astype(np.float32),
            "steerer": {
                "enabled": True,
                "mode": phase1_action.mode,
                "phase2_steps": phase2_steps,
                "passthrough": True,
            },
            "phase1_actions_model": steerer_model_actions,
            "phase1_actions_raw": phase1_actions_padded,
            "steerer_phase1_actions": steerer_model_actions,
            "steerer_phase1_actions_raw": phase1_actions_padded,
            f"{legacy_key}_phase1_actions": steerer_model_actions,
            f"{legacy_key}_phase1_actions_raw": phase1_actions_padded,
        }
        if phase1_action.model_actions is not None:
            result["steerer_phase1_model_actions"] = steerer_model_actions
        return result

    def _infer(
        self,
        obs: Mapping[str, Any],
        *,
        phase1_action: Phase1Action | None,
        sample_kwargs: dict[str, Any],
    ) -> dict:
        image = self._obs_to_image_tensor(obs)
        proprio = self._obs_to_proprio(obs)
        prompt = self._obs_to_prompt(obs)
        action_horizon = int(sample_kwargs.pop("action_horizon", self._action_horizon))
        if action_horizon != self._action_horizon:
            LOGGER.debug(
                "FastWAM inference action_horizon=%d differs from configured/trained horizon=%d.",
                action_horizon,
                self._action_horizon,
            )
        init_action = None
        if phase1_action is not None:
            init_action = self._normalize_action_chunk(phase1_action.actions, horizon=action_horizon)

        num_inference_steps = sample_kwargs.pop("num_inference_steps", None)
        if num_inference_steps is None:
            num_inference_steps = sample_kwargs.pop("num_steps", self._runtime_config.num_inference_steps)
        else:
            sample_kwargs.pop("num_steps", None)

        kwargs = {
            "prompt": prompt,
            "input_image": image,
            "num_frames": self._num_video_frames,
            "action": None,
            "init_action": init_action,
            "action_horizon": action_horizon,
            "proprio": proprio,
            "negative_prompt": str(sample_kwargs.pop("negative_prompt", self._runtime_config.negative_prompt)),
            "text_cfg_scale": float(sample_kwargs.pop("text_cfg_scale", self._runtime_config.text_cfg_scale)),
            "action_cfg_scale": float(sample_kwargs.pop("action_cfg_scale", self._runtime_config.action_cfg_scale)),
            "num_inference_steps": self._normalize_num_inference_steps(num_inference_steps),
            "sigma_shift": sample_kwargs.pop("sigma_shift", self._runtime_config.sigma_shift),
            "seed": sample_kwargs.pop("seed", self._runtime_config.seed),
            "rand_device": str(sample_kwargs.pop("rand_device", self._runtime_config.rand_device)),
            "tiled": bool(sample_kwargs.pop("tiled", self._runtime_config.tiled)),
            "refinement_steps": sample_kwargs.pop("refinement_steps", None),
            "frs": bool(sample_kwargs.pop("frs", self._runtime_config.frs)),
            "random_noise_ratio": self._pop_random_noise_ratio(sample_kwargs),
            # Disable FastWAM's debug consistency check; online serving only needs policy inference.
            "test_action_with_infer_action": False,
        }
        with torch.no_grad():
            pred = self._model.infer(**kwargs)

        model_actions = pred["action"].detach().to(dtype=torch.float32, device="cpu")
        raw_actions = self._denormalize_action_chunk(model_actions)[0]
        result = {
            "actions": raw_actions.astype(np.float32),
            "model_actions": model_actions.numpy().astype(np.float32),
        }
        self._set_phase2_debug(result, kwargs)
        return result

    @staticmethod
    def _set_phase2_debug(result: dict[str, Any], sample_kwargs: Mapping[str, Any]) -> None:
        num_steps = float(sample_kwargs.get("num_inference_steps", 0.0) or 0.0)
        refinement_steps = sample_kwargs.get("refinement_steps")
        if refinement_steps is None or num_steps <= 0.0:
            return
        phase2_steps = float(refinement_steps)
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

    def _uses_random_noise(self, sample_kwargs: Mapping[str, Any]) -> bool:
        return (
            bool(sample_kwargs.get("frs", self._runtime_config.frs))
            and float(sample_kwargs.get("random_noise_ratio", 0.0) or 0.0) > 0.0
        )

    @staticmethod
    def _pop_random_noise_ratio(sample_kwargs: dict[str, Any]) -> float:
        return float(sample_kwargs.pop("random_noise_ratio", 0.0))

    def _obs_to_prompt(self, obs: Mapping[str, Any]) -> str:
        prompt = obs.get("prompt")
        if prompt is None:
            prompt = getattr(self._processor, "default_prompt", None)
        if prompt is None:
            prompt = ""
        if not isinstance(prompt, str):
            prompt = np.asarray(prompt).item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
        if "{task}" in self._runtime_config.prompt_template:
            return self._runtime_config.prompt_template.format(task=prompt)
        return str(prompt)

    def _obs_to_proprio(self, obs: Mapping[str, Any]) -> torch.Tensor:
        state = obs.get("state", obs.get("observation/state"))
        if state is None:
            raise KeyError("FastWAM policy expects 'observation/state' or 'state' in obs.")
        state = torch.as_tensor(np.asarray(state, dtype=np.float32))
        if state.ndim == 1:
            state = state.unsqueeze(0)

        state_meta = self._processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("FastWAM steering currently expects one state entry in shape_meta.")
        state_key = state_meta[0]["key"]
        expected_dim = int(state_meta[0]["shape"])
        state = self._fit_last_dim(state, expected_dim)
        batch = {"state": {state_key: state}}
        batch = self._processor.action_state_transform(batch)
        batch = self._processor.normalizer.forward(batch)
        return batch["state"][state_key].to(device=self._model.device, dtype=self._model.torch_dtype)

    def _normalize_action_chunk(self, actions: np.ndarray, *, horizon: int) -> torch.Tensor:
        action = torch.as_tensor(np.asarray(actions, dtype=np.float32))
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 2:
            raise ValueError(f"Expected phase-1 action chunk [T,D], got {tuple(action.shape)}")

        action_meta = self._processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("FastWAM steering currently expects one action entry in shape_meta.")
        action_key = action_meta[0]["key"]
        expected_dim = int(action_meta[0]["shape"])
        action = self._fit_action_shape(action, horizon, expected_dim)
        normalizer = self._processor.normalizer.normalizers["action"][action_key]
        action = normalizer.forward(action).unsqueeze(0)
        return action.to(device=self._model.device, dtype=self._model.torch_dtype)

    def _denormalize_action_chunk(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected FastWAM action [B,T,D], got {tuple(action.shape)}")
        action_meta = self._processor.shape_meta["action"]
        action_key = action_meta[0]["key"]
        normalizer = self._processor.normalizer.normalizers["action"][action_key]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()

    def _obs_to_image_tensor(self, obs: Mapping[str, Any]) -> torch.Tensor:
        image_dict = self._get_image_dict(obs)
        camera_tensors = []
        for idx, meta in enumerate(self._processor.shape_meta["images"][: self._processor.num_output_cameras]):
            key = meta["key"]
            if key not in image_dict:
                raise KeyError(f"FastWAM policy missing image key '{key}' in obs.")
            shape = meta["shape"]
            height, width = int(shape[1]), int(shape[2])
            camera_tensors.append(self._resize_rgb(image_dict[key], width=width, height=height, camera_idx=idx))

        if len(camera_tensors) == 1:
            image = camera_tensors[0]
        elif self._concat_multi_camera == "horizontal":
            image = np.concatenate(camera_tensors, axis=1)
        elif self._concat_multi_camera == "vertical":
            image = np.concatenate(camera_tensors, axis=0)
        else:
            raise ValueError(f"Unsupported FastWAM concat_multi_camera={self._concat_multi_camera!r}.")

        expected_h, expected_w = self._video_size
        if image.shape[:2] != (expected_h, expected_w):
            image = self._center_crop_resize(image, width=expected_w, height=expected_h)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self._model.device, dtype=self._model.torch_dtype)
        return tensor * (2.0 / 255.0) - 1.0

    @staticmethod
    def _get_image_dict(obs: Mapping[str, Any]) -> dict[str, Any]:
        image_dict = obs.get("image")
        if isinstance(image_dict, Mapping):
            images = dict(image_dict)
            aliases = {
                "robot0_agentview_left": ("base_0_rgb",),
                "robot0_eye_in_hand": ("left_wrist_0_rgb", "wrist_0_rgb"),
                "robot0_agentview_right": ("right_wrist_0_rgb",),
                "base_0_rgb": ("robot0_agentview_left",),
                "left_wrist_0_rgb": ("robot0_eye_in_hand",),
                "right_wrist_0_rgb": ("robot0_agentview_right",),
            }
            for target, sources in aliases.items():
                if target not in images:
                    for source in sources:
                        if source in images:
                            images[target] = images[source]
                            break
            return images
        candidates = {
            "robot0_agentview_left": obs.get("observation/image"),
            "robot0_eye_in_hand": obs.get("observation/wrist_image"),
            "robot0_agentview_right": obs.get("observation/right_wrist_image"),
            "base_0_rgb": obs.get("observation/image"),
            "left_wrist_0_rgb": obs.get("observation/wrist_image"),
            "right_wrist_0_rgb": obs.get("observation/right_wrist_image"),
        }
        return {key: value for key, value in candidates.items() if value is not None}

    @classmethod
    def _resize_rgb(cls, image: Any, *, width: int, height: int, camera_idx: int) -> np.ndarray:
        del camera_idx
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"Expected RGB image with rank 3, got {array.shape}.")
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
        return cls._center_crop_resize(array, width=width, height=height)

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

    @staticmethod
    def _fit_last_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        if tensor.shape[-1] > dim:
            raise ValueError(f"Input dim {tensor.shape[-1]} exceeds configured dim {dim}.")
        if tensor.shape[-1] < dim:
            pad_shape = (*tensor.shape[:-1], dim - tensor.shape[-1])
            tensor = torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=-1)
        return tensor

    @staticmethod
    def _fit_action_shape(action: torch.Tensor, horizon: int, dim: int) -> torch.Tensor:
        out = torch.zeros((horizon, dim), dtype=action.dtype)
        horizon_n = min(horizon, action.shape[0])
        dim_n = min(dim, action.shape[-1])
        out[:horizon_n, :dim_n] = action[:horizon_n, :dim_n]
        if 0 < horizon_n < horizon:
            out[horizon_n:, :dim_n] = action[horizon_n - 1, :dim_n]
        return out

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
    def _result_action_horizon(result: Mapping[str, Any]) -> int:
        actions = result.get("actions")
        if isinstance(actions, np.ndarray) and actions.ndim >= 1:
            return int(actions.shape[0])
        return 0

    @staticmethod
    def _pad_action_array_temporal(action: np.ndarray, *, horizon: int) -> np.ndarray:
        array = np.asarray(action, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected phase-1 action chunk [T,D], got {array.shape}")
        if horizon <= 0 or array.shape[0] >= horizon:
            return array.astype(np.float32, copy=False)
        if array.shape[0] == 0:
            return np.zeros((horizon, array.shape[-1]), dtype=np.float32)
        pad = np.repeat(array[-1:, :], horizon - array.shape[0], axis=0)
        return np.concatenate([array, pad], axis=0).astype(np.float32, copy=False)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim


class EvoFastWAMSteeredPolicy(SteeredPhase2Runtime):
    def __init__(self, phase2_policy: FastWAMPhase2Policy, config: SteeringConfig):
        super().__init__(
            backend=Evo1Backend(config),
            phase2_policy=phase2_policy,
            config=config,
            enable_keys=("enable_evo1_steerer", "enable_steerer"),
        )



def create_steered_fastwam_policy(
    runtime_config: FastWAMRuntimeConfig,
    steerer_config: SteeringConfig,
) -> SteeredPhase2Runtime:
    phase2_policy = load_fastwam_phase2_policy(runtime_config)
    steerer_config = dataclasses.replace(
        steerer_config,
        action_horizon=phase2_policy.action_horizon,
        action_dim=phase2_policy.action_dim,
    )
    mode = steerer_config.mode.lower()
    if mode == "evo":
        return EvoFastWAMSteeredPolicy(phase2_policy, steerer_config)
    raise ValueError(f"Unsupported steerer mode '{steerer_config.mode}'. Expected 'evo'.")


def load_fastwam_phase2_policy(runtime_config: FastWAMRuntimeConfig) -> FastWAMPhase2Policy:
    _ensure_fastwam_on_path()
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.config_resolvers import register_default_resolvers
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    register_default_resolvers()
    cfg = _load_fastwam_cfg(runtime_config.config, OmegaConf, initialize_config_dir, compose)
    cfg.model.load_text_encoder = True

    model_dtype = _mixed_precision_to_dtype(runtime_config.mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=runtime_config.device)
    model.load_checkpoint(runtime_config.checkpoint_path)
    model = model.to(runtime_config.device).eval()

    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(runtime_config.dataset_stats_path))
    return FastWAMPhase2Policy(model=model, processor=processor, runtime_config=runtime_config, cfg=cfg)


def _load_fastwam_cfg(config: str, OmegaConf: Any, initialize_config_dir: Any, compose: Any) -> Any:
    path = pathlib.Path(config).expanduser()
    if path.is_file():
        return OmegaConf.load(path)

    repo_root = _find_repo_root()
    config_dir = repo_root / "FastWAM" / "configs"
    task_name = config[:-5] if config.endswith(".yaml") else config
    if task_name.startswith("task/"):
        task_name = task_name.removeprefix("task/")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="train", overrides=[f"task={task_name}"])


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision {mixed_precision!r}. Expected no, fp16, or bf16.")
