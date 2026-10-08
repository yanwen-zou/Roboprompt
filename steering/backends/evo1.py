from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
import pathlib
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from steering.config import SteeringConfig
from steering.schemas import Phase1Action


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Evo1Inputs:
    images: list[torch.Tensor]
    image_mask: torch.Tensor
    prompt: str
    state: torch.Tensor
    action_mask: torch.Tensor
    debug: dict[str, Any]
    raw_images: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Evo1ModelBundle:
    model: torch.nn.Module
    normalizer: "Evo1Normalizer"
    device: torch.device
    config: dict[str, Any]


class Evo1Normalizer:
    """Min/max normalizer used by Evo-1-style checkpoints."""

    def __init__(self, stats_or_path: dict[str, Any] | str | pathlib.Path, *, state_dim: int, action_dim: int):
        if isinstance(stats_or_path, str | pathlib.Path):
            with pathlib.Path(stats_or_path).open() as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain one robot key, got {list(stats.keys())}.")
        robot_stats = next(iter(stats.values()))
        self.state_dim = int(torch.as_tensor(robot_stats["observation.state"]["min"]).shape[-1])
        self.action_dim = int(torch.as_tensor(robot_stats["action"]["min"]).shape[-1])
        self.state_min = self._pad(robot_stats["observation.state"]["min"], state_dim, name="observation.state/min")
        self.state_max = self._pad(robot_stats["observation.state"]["max"], state_dim, name="observation.state/max")
        self.action_min = self._pad(robot_stats["action"]["min"], action_dim, name="action/min")
        self.action_max = self._pad(robot_stats["action"]["max"], action_dim, name="action/max")

    @staticmethod
    def _pad(values: Any, dim: int, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.shape[-1] > dim:
            raise ValueError(f"{name} length {tensor.shape[-1]} exceeds configured dim {dim}.")
        if tensor.shape[-1] < dim:
            tensor = torch.cat([tensor, torch.zeros(dim - tensor.shape[-1], dtype=torch.float32)], dim=-1)
        return tensor

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        dim = state.shape[-1]
        state_min = self.state_min[:dim].to(state.device, dtype=state.dtype)
        state_max = self.state_max[:dim].to(state.device, dtype=state.dtype)
        return torch.clamp(2.0 * (state - state_min) / (state_max - state_min + 1e-8) - 1.0, -1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


class Evo1InputAdapter:
    """Adapt canonical/OpenPI-style observations into Evo-1 tensors."""

    def __init__(self, config: SteeringConfig, *, device: torch.device):
        self._config = config
        self._device = device

    def __call__(self, obs: Mapping[str, Any], normalizer: Evo1Normalizer) -> Evo1Inputs:
        raw_input_debug = self._raw_input_debug(obs)
        image_dict = self._get_image_dict(obs)

        image_masks = obs.get("image_mask") or {}
        if not isinstance(image_masks, Mapping):
            image_masks = {}
        images = []
        masks = []
        image_debug = []
        for key in self._config.image_keys[: self._config.max_views]:
            if key in image_dict:
                image = self._image_to_tensor(image_dict[key])
                mask = bool(np.asarray(image_masks.get(key, True)).item())
                images.append(image)
                masks.append(mask)
                image_debug.append(self._image_debug(key, image, mask))
            else:
                image = torch.zeros((3, self._config.image_size, self._config.image_size), device=self._device)
                images.append(image)
                masks.append(False)
                image_debug.append(self._image_debug(key, image, False))

        raw_state_np = np.asarray(self._get_state(obs), dtype=np.float32)
        state = torch.as_tensor(raw_state_np, device=self._device)
        if state.ndim == 1:
            state = state[None, :]
        normalized_state = normalizer.normalize_state(state)
        normalized_state = self._pad_last_dim(normalized_state, self._config.max_state_dim, name="state")

        prompt = obs.get("prompt", self._config.prompt)
        if prompt is None and self._has_interactive_prompt(obs):
            prompt = ""
        if prompt is None:
            raise ValueError("Evo-1 steerer requires a text prompt or an interactive prompt in obs.")
        if not isinstance(prompt, str):
            prompt = np.asarray(prompt).item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

        action_mask = torch.zeros((1, self._config.max_action_dim), dtype=torch.int32, device=self._device)
        action_dim = self._config.action_dim or self._config.max_action_dim
        action_mask[:, : min(action_dim, self._config.max_action_dim)] = 1
        state_cpu = torch_to_numpy(state[0])
        normalized_state_real_cpu = torch_to_numpy(normalized_state[0, : state.shape[-1]])
        normalized_state_cpu = torch_to_numpy(normalized_state[0])
        debug = {
            "obs_keys": sorted(str(key) for key in obs.keys()),
            "image_dict_keys": sorted(str(key) for key in image_dict.keys()),
            "raw_input_debug": raw_input_debug,
            "image_debug": image_debug,
            "raw_state_shape": tuple(raw_state_np.shape),
            "raw_state_first8": np.asarray(raw_state_np).reshape(-1)[:8].astype(np.float32),
            "padded_state_first8": state_cpu[:8],
            "normalized_state_first8": normalized_state_cpu[:8],
            "normalized_state_real_min": float(np.min(normalized_state_real_cpu)),
            "normalized_state_real_max": float(np.max(normalized_state_real_cpu)),
            "normalized_state_real_clipped_count": int(np.count_nonzero(np.abs(normalized_state_real_cpu) >= 0.9999)),
            "normalized_state_padded_tail_abs_sum": float(np.sum(np.abs(normalized_state_cpu[state.shape[-1] :]))),
            "prompt": prompt,
            "action_mask_shape": tuple(action_mask.shape),
            "action_mask_sum": int(action_mask.sum().item()),
        }

        return Evo1Inputs(
            images=images,
            image_mask=torch.as_tensor(masks, dtype=torch.int32, device=self._device),
            prompt=prompt,
            state=normalized_state,
            action_mask=action_mask,
            debug=debug,
            raw_images=self._raw_input_images(obs),
        )

    def _get_image_dict(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        image_dict = obs.get("image")
        if isinstance(image_dict, Mapping):
            return self._with_prompt_image_override(dict(image_dict), obs)

        openpi_images = {
            "base_0_rgb": obs.get("observation/image"),
        }
        openpi_images = {key: value for key, value in openpi_images.items() if value is not None}
        if openpi_images:
            return self._with_prompt_image_override(openpi_images, obs)

        raise ValueError(
            "Evo1Backend expects image fields such as 'observation/image', or obs['image'] as a camera mapping."
        )

    @staticmethod
    def _with_prompt_image_override(image_dict: dict[str, Any], obs: Mapping[str, Any]) -> dict[str, Any]:
        prompt_images = obs.get("prompt_images")
        prompt_image_masks = obs.get("prompt_image_masks")
        if not isinstance(prompt_images, Mapping) or not isinstance(prompt_image_masks, Mapping):
            return image_dict
        if "prompt_0" not in prompt_images:
            return image_dict
        prompt_mask = prompt_image_masks.get("prompt_0", False)
        if bool(np.any(np.asarray(prompt_mask, dtype=np.bool_))):
            image_dict["base_0_rgb"] = prompt_images["prompt_0"]
            image_dict["prompt_0_rgb"] = prompt_images.get("prompt_overlay_0", prompt_images["prompt_0"])
        return image_dict

    @staticmethod
    def _get_state(obs: Mapping[str, Any]) -> Any:
        if "state" in obs:
            return obs["state"]
        if "observation/state" in obs:
            return obs["observation/state"]
        raise KeyError("Evo1Backend expects 'observation/state' or 'state' in obs.")

    @staticmethod
    def _has_interactive_prompt(obs: Mapping[str, Any]) -> bool:
        prompt_keys = (
            "prompt_images",
            "prompt_global_motion",
            "prompt_local_motion",
            "prompt_2d_drag",
            "prompt_primitive_cmd",
            "prompt_wrist_relative_action",
        )
        return any(key in obs for key in prompt_keys)

    def _image_to_tensor(self, image: Any) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {array.shape}.")
        if array.shape[-1] != 3 and array.shape[0] == 3:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] != 3:
            raise ValueError(f"Expected HWC or CHW RGB image, got shape {array.shape}.")
        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            if array.min() < 0.0:
                array = (array + 1.0) / 2.0
            if array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array).resize((self._config.image_size, self._config.image_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.asarray(pil, dtype=np.float32)).permute(2, 0, 1) / 255.0
        return tensor.to(self._device)

    @staticmethod
    def _image_debug(key: str, image: torch.Tensor, mask: bool) -> dict[str, Any]:
        array = torch_to_numpy(image)
        digest = hashlib.sha1(array.tobytes()).hexdigest()[:10]
        return {
            "key": key,
            "mask": mask,
            "shape": tuple(array.shape),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "sha1": digest,
        }

    @staticmethod
    def _pad_last_dim(tensor: torch.Tensor, dim: int, *, name: str) -> torch.Tensor:
        if tensor.shape[-1] > dim:
            raise ValueError(f"{name} dim {tensor.shape[-1]} exceeds configured dim {dim}.")
        if tensor.shape[-1] < dim:
            pad_shape = (*tensor.shape[:-1], dim - tensor.shape[-1])
            tensor = torch.cat([tensor, torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)], dim=-1)
        return tensor

    @classmethod
    def _raw_input_debug(cls, obs: Mapping[str, Any]) -> dict[str, Any]:
        prompt_images = obs.get("prompt_images")
        prompt_image_masks = obs.get("prompt_image_masks")
        prompt_mask = False
        if isinstance(prompt_image_masks, Mapping):
            prompt_mask = bool(np.any(np.asarray(prompt_image_masks.get("prompt_0", False), dtype=np.bool_)))

        debug: dict[str, Any] = {
            "prompt_image_mask_prompt_0": prompt_mask,
            "observation_image": cls._raw_image_debug(obs.get("observation/image")),
            "observation_wrist_image": cls._raw_image_debug(obs.get("observation/wrist_image")),
            "observation_right_wrist_image": cls._raw_image_debug(obs.get("observation/right_wrist_image")),
            "prompt_0": None,
            "prompt_overlay_0": None,
        }
        if isinstance(prompt_images, Mapping):
            debug["prompt_0"] = cls._raw_image_debug(prompt_images.get("prompt_0"))
            debug["prompt_overlay_0"] = cls._raw_image_debug(prompt_images.get("prompt_overlay_0"))
        return debug

    @staticmethod
    def _raw_input_images(obs: Mapping[str, Any]) -> dict[str, Any]:
        images = {
            "observation_image": obs.get("observation/image"),
            "observation_wrist_image": obs.get("observation/wrist_image"),
            "observation_right_wrist_image": obs.get("observation/right_wrist_image"),
        }
        prompt_images = obs.get("prompt_images")
        if isinstance(prompt_images, Mapping):
            images["prompt_0"] = prompt_images.get("prompt_0")
            images["prompt_overlay_0"] = prompt_images.get("prompt_overlay_0")
        return {key: value for key, value in images.items() if value is not None}

    @staticmethod
    def _raw_image_debug(image: Any) -> dict[str, Any] | None:
        if image is None:
            return None
        array = np.asarray(image)
        if array.size == 0:
            return {"shape": tuple(array.shape), "dtype": str(array.dtype), "empty": True}
        image_array = array
        if image_array.ndim == 3 and image_array.shape[-1] != 3 and image_array.shape[0] == 3:
            image_array = np.moveaxis(image_array, 0, -1)
        flat = image_array.astype(np.float32, copy=False)
        return {
            "shape": tuple(array.shape),
            "canonical_shape": tuple(image_array.shape),
            "dtype": str(array.dtype),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "sha1": hashlib.sha1(np.ascontiguousarray(image_array).tobytes()).hexdigest()[:10],
        }


class Evo1Backend:
    """Phase-1 steerer backend backed by an Evo-1 checkpoint."""

    mode = "evo"

    def __init__(self, config: SteeringConfig):
        self._bundle = load_evo1_bundle(config)
        self.config = dataclasses.replace(
            config,
            image_keys=tuple(self._bundle.config.get("image_keys", config.image_keys)),
            image_size=int(self._bundle.config.get("image_size", config.image_size)),
            max_views=max(int(self._bundle.config.get("max_views", config.max_views)), config.max_views),
            max_state_dim=int(self._bundle.config.get("state_dim", config.max_state_dim)),
            max_action_dim=int(self._bundle.config.get("per_action_dim", config.max_action_dim)),
        )
        input_config = dataclasses.replace(
            self.config,
            action_dim=min(int(self._bundle.normalizer.action_dim), self.config.max_action_dim),
        )
        self._input_adapter = Evo1InputAdapter(input_config, device=self._bundle.device)
        self._input_dump_enabled = os.environ.get("EVO1_INPUT_DUMP", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        self._input_dump_dir = pathlib.Path(os.environ.get("EVO1_INPUT_DUMP_DIR", "output/evo1/infer_inputs")).expanduser()
        self._input_dump_max = int(os.environ.get("EVO1_INPUT_DUMP_MAX", "100"))
        self._input_dump_index = 0
        self._generator = None if self.config.inference_seed is not None else _create_torch_generator(None, self._bundle.device)
        LOGGER.info(
            "EVO1_NORM_STATS checkpoint_dir=%s state_dim=%d action_dim=%d "
            "state_min[:8]=%s state_max[:8]=%s action_min[:7]=%s action_max[:7]=%s",
            self.config.checkpoint_dir,
            self._bundle.normalizer.state_dim,
            self._bundle.normalizer.action_dim,
            _array_preview(self._bundle.normalizer.state_min[:8]),
            _array_preview(self._bundle.normalizer.state_max[:8]),
            _array_preview(self._bundle.normalizer.action_min[:7]),
            _array_preview(self._bundle.normalizer.action_max[:7]),
        )
        if self._input_dump_enabled:
            LOGGER.info("EVO1_INPUT_DUMP dir=%s max=%d", self._input_dump_dir, self._input_dump_max)
        LOGGER.info(
            "EVO1_INFERENCE_SEED seed=%s deterministic_per_predict=%s",
            self.config.inference_seed,
            self.config.inference_seed is not None,
        )

    def predict(self, obs: Mapping[str, Any]) -> Phase1Action:
        evo_inputs = self._input_adapter(obs, self._bundle.normalizer)
        LOGGER.info(
            "EVO1_INPUT obs_keys=%s image_dict_keys=%s images=%s prompt=%r raw_state_shape=%s "
            "raw_state[:8]=%s norm_state[:8]=%s norm_state_real_min=%.6f norm_state_real_max=%.6f "
            "norm_state_real_clipped_count=%d norm_state_padded_tail_abs_sum=%.6f "
            "image_mask=%s action_mask_shape=%s action_mask_sum=%d",
            evo_inputs.debug["obs_keys"],
            evo_inputs.debug["image_dict_keys"],
            evo_inputs.debug["image_debug"],
            evo_inputs.debug["prompt"],
            evo_inputs.debug["raw_state_shape"],
            _array_preview(evo_inputs.debug["raw_state_first8"]),
            _array_preview(evo_inputs.debug["normalized_state_first8"]),
            evo_inputs.debug["normalized_state_real_min"],
            evo_inputs.debug["normalized_state_real_max"],
            evo_inputs.debug["normalized_state_real_clipped_count"],
            evo_inputs.debug["normalized_state_padded_tail_abs_sum"],
            torch_to_numpy(evo_inputs.image_mask).astype(np.int32).tolist(),
            evo_inputs.debug["action_mask_shape"],
            evo_inputs.debug["action_mask_sum"],
        )
        self._dump_input(evo_inputs)
        with torch.no_grad():
            with torch.autocast(
                device_type=self._bundle.device.type,
                dtype=torch.bfloat16,
                enabled=self._bundle.device.type == "cuda",
            ):
                action = self._bundle.model.run_inference(
                    images=evo_inputs.images,
                    image_mask=evo_inputs.image_mask,
                    prompt=evo_inputs.prompt,
                    state_input=evo_inputs.state,
                    action_mask=evo_inputs.action_mask,
                    generator=self._prediction_generator(),
                )
        action_dim = int(self._bundle.config.get("per_action_dim", self.config.max_action_dim))
        action = action.reshape(1, -1, action_dim)
        normalized_actions = torch_to_numpy(action[0])
        raw_actions = torch_to_numpy(self._bundle.normalizer.denormalize_action(action)[0])
        debug_norm_xyz = normalized_actions[:, :3]
        debug_xyz = raw_actions[:, :3]
        debug_delta_xyz = np.diff(debug_xyz, axis=0)
        debug_delta_norm = np.linalg.norm(debug_delta_xyz, axis=-1)
        debug_accel_norm = np.linalg.norm(np.diff(debug_delta_xyz, axis=0), axis=-1)
        LOGGER.info(
            "EVO1_ACTION norm_xyz[:5]=%s norm_min=%.6f norm_max=%.6f norm_saturated_count=%d "
            "denorm_xyz[:5]=%s delta_xyz[:4]=%s delta_norm[:8]=%s "
            "delta_norm_mean=%.6f delta_norm_max=%.6f accel_norm_mean=%.6f accel_norm_max=%.6f",
            np.array2string(
                debug_norm_xyz[:5],
                precision=5,
                suppress_small=False,
                max_line_width=240,
                threshold=debug_norm_xyz[:5].size,
            ),
            float(np.min(normalized_actions)),
            float(np.max(normalized_actions)),
            int(np.count_nonzero(np.abs(normalized_actions) >= 0.9999)),
            np.array2string(
                debug_xyz[:5],
                precision=5,
                suppress_small=False,
                max_line_width=240,
                threshold=debug_xyz[:5].size,
            ),
            np.array2string(
                debug_delta_xyz[:4],
                precision=5,
                suppress_small=False,
                max_line_width=240,
                threshold=debug_delta_xyz[:4].size,
            ),
            np.array2string(debug_delta_norm[:8], precision=5, suppress_small=False, max_line_width=240),
            float(np.mean(debug_delta_norm)) if debug_delta_norm.size else 0.0,
            float(np.max(debug_delta_norm)) if debug_delta_norm.size else 0.0,
            float(np.mean(debug_accel_norm)) if debug_accel_norm.size else 0.0,
            float(np.max(debug_accel_norm)) if debug_accel_norm.size else 0.0,
        )
        return Phase1Action(actions=raw_actions, mode=self.mode)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "checkpoint_dir": self.config.checkpoint_dir,
            "enable_by_default": self.config.enable_by_default,
            "inference_seed": self.config.inference_seed,
            "action_norm_stats": {
                "min": self._bundle.normalizer.action_min.detach().cpu().float().numpy().tolist(),
                "max": self._bundle.normalizer.action_max.detach().cpu().float().numpy().tolist(),
            },
        }

    def _dump_input(self, evo_inputs: Evo1Inputs) -> None:
        if not self._input_dump_enabled or self._input_dump_index >= self._input_dump_max:
            return
        dump_idx = self._input_dump_index
        self._input_dump_index += 1

        try:
            self._input_dump_dir.mkdir(parents=True, exist_ok=True)
            stem = f"input_{dump_idx:06d}"
            image_paths = []
            for image_idx, (image, info) in enumerate(zip(evo_inputs.images, evo_inputs.debug["image_debug"])):
                key = str(info["key"])
                image_path = self._input_dump_dir / f"{stem}_{image_idx:02d}_{key}.png"
                _tensor_image_to_pil(image).save(image_path)
                image_paths.append(str(image_path))

            raw_image_paths = {}
            for key, image in evo_inputs.raw_images.items():
                image_path = self._input_dump_dir / f"{stem}_raw_{key}.png"
                _raw_image_to_pil(image).save(image_path)
                raw_image_paths[key] = str(image_path)

            panel_path = self._input_dump_dir / f"{stem}.png"
            _save_input_panel(evo_inputs, panel_path)

            payload = {
                "index": dump_idx,
                "panel_path": str(panel_path),
                "image_paths": image_paths,
                "raw_image_paths": raw_image_paths,
                "checkpoint_dir": self.config.checkpoint_dir,
                "prompt": evo_inputs.prompt,
                "image_mask": torch_to_numpy(evo_inputs.image_mask).astype(np.int32).tolist(),
                "action_mask": torch_to_numpy(evo_inputs.action_mask[0]).astype(np.int32).tolist(),
                "state": torch_to_numpy(evo_inputs.state[0]).tolist(),
                "debug": evo_inputs.debug,
            }
            json_path = self._input_dump_dir / f"{stem}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(_json_ready(payload), f, ensure_ascii=False, indent=2)
            LOGGER.info("EVO1_INPUT_DUMP_SAVED panel=%s json=%s", panel_path, json_path)
        except Exception:
            LOGGER.exception("Failed to dump Evo1 input")

    def _prediction_generator(self) -> torch.Generator:
        if self.config.inference_seed is None:
            if self._generator is None:
                self._generator = _create_torch_generator(None, self._bundle.device)
            return self._generator
        return _create_torch_generator(self.config.inference_seed, self._bundle.device)


def _load_evo1_class(evo1_repo_dir: pathlib.Path) -> type[torch.nn.Module]:
    evo1_py = evo1_repo_dir / "scripts" / "Evo1.py"
    if not evo1_py.is_file():
        raise FileNotFoundError(f"Missing Evo-1 module: {evo1_py}")

    spec = importlib.util.spec_from_file_location("_steering_external_evo1", evo1_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Evo-1 module spec from {evo1_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.EVO1
    except AttributeError as exc:
        raise ImportError(f"Evo-1 module {evo1_py} does not define EVO1") from exc


def _find_evo1_repo_dir() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        evo1_repo_dir = parent / "Evo-1"
        if (evo1_repo_dir / "scripts" / "Evo1.py").is_file():
            return evo1_repo_dir
    raise FileNotFoundError("Could not find Evo-1/scripts/Evo1.py from the steering source tree.")


def _create_torch_generator(seed: int | None, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    return generator


def _load_state_dict(ckpt_dir: pathlib.Path) -> dict[str, Any]:
    checkpoint_file = ckpt_dir / "mp_rank_00_model_states.pt"
    checkpoint_meta_path = ckpt_dir / "checkpoint.json"
    if checkpoint_meta_path.is_file():
        with checkpoint_meta_path.open() as f:
            checkpoint_meta = json.load(f)
        checkpoint_file = ckpt_dir / str(checkpoint_meta.get("checkpoints", checkpoint_file.name))

    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    return checkpoint.get("module", checkpoint)


def load_evo1_bundle(config: SteeringConfig) -> Evo1ModelBundle:
    ckpt_dir = pathlib.Path(config.checkpoint_dir).expanduser()
    evo1_cls = _load_evo1_class(_find_evo1_repo_dir())

    with (ckpt_dir / "config.json").open() as f:
        evo_config = json.load(f)
    evo_config = dict(evo_config)
    if vlm_name := os.environ.get("EVO_VLM"):
        evo_config["vlm_name"] = vlm_name
    evo_config["device"] = config.device
    evo_config["finetune_vlm"] = False
    evo_config["finetune_action_head"] = False
    evo_config["num_inference_timesteps"] = config.num_inference_timesteps

    device = torch.device(config.device)
    model = evo1_cls(evo_config).eval()
    model.load_state_dict(_load_state_dict(ckpt_dir), strict=True)
    model = model.to(device)

    model_state_dim = int(evo_config.get("state_dim", config.max_state_dim))
    model_action_dim = int(evo_config.get("per_action_dim", config.max_action_dim))
    normalizer = Evo1Normalizer(
        ckpt_dir / "norm_stats.json",
        state_dim=model_state_dim,
        action_dim=model_action_dim,
    )
    return Evo1ModelBundle(model=model, normalizer=normalizer, device=device, config=evo_config)


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().to(torch.float32).numpy()


def _array_preview(values: Any, *, precision: int = 6) -> str:
    array = np.asarray(values, dtype=np.float32)
    return np.array2string(array, precision=precision, suppress_small=False, max_line_width=240)


def _tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    array = torch_to_numpy(image)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray((array * 255.0).round().astype(np.uint8)).convert("RGB")


def _raw_image_to_pil(image: Any) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected raw image with 3 dims, got shape {array.shape}.")
    if array.shape[-1] != 3 and array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected raw RGB image with 3 channels, got shape {array.shape}.")
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.min() < 0.0:
            array = (array + 1.0) / 2.0
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(array)).convert("RGB")


def _save_input_panel(evo_inputs: Evo1Inputs, path: pathlib.Path) -> None:
    images = [_tensor_image_to_pil(image) for image in evo_inputs.images]
    if not images:
        images = [Image.new("RGB", (224, 224), color=(0, 0, 0))]

    base_w, base_h = images[0].size
    header_h = 18
    tile_w = base_w
    tile_h = base_h + header_h
    canvas = Image.new("RGB", (tile_w * len(images), tile_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    image_mask = torch_to_numpy(evo_inputs.image_mask).astype(np.int32).tolist()
    for idx, image in enumerate(images):
        image = image.resize((base_w, base_h))
        x0 = idx * tile_w
        info = evo_inputs.debug["image_debug"][idx]
        active = bool(image_mask[idx]) if idx < len(image_mask) else False
        draw.text(
            (x0 + 6, 4),
            f"{info['key']} mask={int(active)} sha1={info['sha1']}",
            fill=(230, 230, 230) if active else (140, 140, 140),
        )
        canvas.paste(image, (x0, header_h))

    panel_w = 460
    panel = Image.new("RGB", (panel_w, tile_h), color=(30, 30, 30))
    panel_draw = ImageDraw.Draw(panel)
    lines = [
        "=== MASK ===",
        f"image_mask: {image_mask}",
        f"action_mask_sum: {evo_inputs.debug['action_mask_sum']}",
        "=== STATE ===",
        f"raw_shape: {evo_inputs.debug['raw_state_shape']}",
        f"raw[:8]: {_array_preview(evo_inputs.debug['raw_state_first8'], precision=4)}",
        f"norm[:8]: {_array_preview(evo_inputs.debug['normalized_state_first8'], precision=4)}",
        f"norm_real_min/max: {evo_inputs.debug['normalized_state_real_min']:.4f} / {evo_inputs.debug['normalized_state_real_max']:.4f}",
        f"norm_real_clipped: {evo_inputs.debug['normalized_state_real_clipped_count']}",
        f"pad_tail_abs_sum: {evo_inputs.debug['normalized_state_padded_tail_abs_sum']:.6f}",
        "=== IMAGE STATS ===",
    ]
    for info in evo_inputs.debug["image_debug"]:
        lines.append(
            f"{info['key']}: mean={info['mean']:.4f} std={info['std']:.4f} min={info['min']:.3f} max={info['max']:.3f}"
        )
    lines.extend(["=== PROMPT ===", *list(_wrap_text(str(evo_inputs.prompt), width=62))])

    y = 4
    line_h = 11
    for line in lines:
        if y + line_h > tile_h:
            break
        panel_draw.text((8, y), line, fill=(220, 220, 220))
        y += line_h

    output = Image.new("RGB", (canvas.width + panel_w, tile_h))
    output.paste(canvas, (0, 0))
    output.paste(panel, (canvas.width, 0))
    output.save(path)


def _wrap_text(text: str, *, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return torch_to_numpy(value).tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
