import dataclasses
import logging
import re
from typing import Literal, Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing weights matching the regex (e.g., LoRA or new modules like prompt encoders).
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Pi0ControlTowerWeightLoader(WeightLoader):
    """Loads the base Pi0/Pi0.5 tower and optionally seeds a future control tower.

    The base tower is initialized from `base_weight_loader`. If a control tower is
    present in the target params, it can additionally be initialized from either:
    - the official original PaliGemma checkpoint
    - another trained checkpoint containing a `PaliGemma` subtree (e.g. Pi0.5)

    Control-tower remapping is a no-op until those target parameters exist.
    """

    base_weight_loader: WeightLoader
    control_tower_init: Literal["none", "original_paligemma", "checkpoint"] = "none"
    control_tower_params_path: str | None = None
    base_prefix: str = "PaliGemma"
    control_tower_prefix: str = "ControlTower"

    def load(self, params: at.Params) -> at.Params:
        merged = self.base_weight_loader.load(params)
        # Keep all newly introduced control-tower params (control tower, injectors, etc.)
        # even when the base loader only tolerates a narrower missing-key regex.
        merged = _merge_params(merged, params, missing_regex=".*")

        control_params = self._load_control_source()
        if control_params is None:
            return merged

        return _merge_params_by_prefix(
            loaded_params=control_params,
            params=merged,
            source_prefix=self.base_prefix,
            target_prefix=self.control_tower_prefix,
        )

    def _load_control_source(self) -> at.Params | None:
        if self.control_tower_init == "none":
            return None
        if self.control_tower_init == "original_paligemma":
            return _load_original_paligemma_params()
        if self.control_tower_init == "checkpoint":
            if self.control_tower_params_path is None:
                raise ValueError("control_tower_params_path must be set when control_tower_init='checkpoint'.")
            return _model.restore_params(download.maybe_download(self.control_tower_params_path), restore_type=np.ndarray)
        raise ValueError(f"Unsupported control_tower_init: {self.control_tower_init}")


@dataclasses.dataclass(frozen=True)
class Pi0MotionTowerWeightLoader(WeightLoader):
    """Loads a base Pi0/Pi0.5 checkpoint and keeps newly initialized motion-tower params."""

    base_weight_loader: WeightLoader

    def load(self, params: at.Params) -> at.Params:
        loaded_params = self.base_weight_loader.load(params)
        return _merge_params(loaded_params, params, missing_regex=".*")


def _load_original_paligemma_params() -> at.Params:
    path = download.maybe_download(
        "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
    )
    with path.open("rb") as f:
        flat_params = dict(np.load(f, allow_pickle=False))
    return {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params_by_prefix(
    loaded_params: at.Params,
    params: at.Params,
    *,
    source_prefix: str,
    target_prefix: str,
) -> at.Params:
    """Copies matching params from one subtree into another subtree when present."""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    result = dict(flat_ref)

    source_prefix_with_sep = f"{source_prefix}/"
    target_prefix_with_sep = f"{target_prefix}/"

    for key, value in flat_loaded.items():
        if key == source_prefix:
            mapped_key = target_prefix
        elif key.startswith(source_prefix_with_sep):
            mapped_key = target_prefix_with_sep + key[len(source_prefix_with_sep) :]
        else:
            continue

        if mapped_key not in flat_ref:
            continue
        result[mapped_key] = value.astype(flat_ref[mapped_key].dtype) if value.dtype != flat_ref[mapped_key].dtype else value

    return flax.traverse_util.unflatten_dict(result, sep="/")
