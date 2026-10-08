"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_control_tower_config as pi0_control_tower_config
import openpi.models.pi0_motion_tower_config as pi0_motion_tower_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.groot_utils.groot_openpi_dataset as _groot_openpi_dataset

from robocasa.macros import DATASET_BASE_PATH
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
from robocasa.utils.dataset_registry_utils import get_ds_soup


def _robocasa_data_path(rel_path: str) -> str:
    """Returns the full path to a RoboCasa dataset."""
    return os.path.join(DATASET_BASE_PATH, rel_path)


_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _workspace_data_path(rel_path: str) -> str:
    """Returns an absolute path rooted at the Roboprompt workspace."""
    return str(_WORKSPACE_ROOT / rel_path)


_RP_DATA_ROOT = os.environ.get("RP_DATA_ROOT", str(_WORKSPACE_ROOT / "robocasa_data_ckpt"))
_ACTIVE_GLASS_DATA_ROOT = os.environ.get("ACTIVE_GLASS_DATA_ROOT", os.path.join(_RP_DATA_ROOT, "active_glass"))
_PI05_ROBOCASA_PRETRAIN50_CKPT = os.path.join(
    _RP_DATA_ROOT,
    "ckpt_pi05/pi05_robocasa_pretrain50/pretrain50_no_overlay/49999/params",
)

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class OnlineDatasetConfig:
    root: str | Sequence[str] | None = None
    # If true, online LeRobot roots that contain a frame-level `denoise_step`
    # column are filtered to samples with denoise_step < max denoise steps.
    filter_denoise_step_lt_max: bool = False
    # Optional override for the online denoise-step filter threshold. If unset,
    # the loader reads max_phase2_steps/num_steps from extras/dataset_meta_server.json.
    denoise_step_max: float | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional local root(s) for LeRobot datasets. When set, the dataset is loaded from
    # `local_dataset_root` instead of the default Hugging Face cache location.
    local_dataset_root: str | Sequence[str] | None = None
    # Optional split local roots. If online roots are provided, training samples
    # alternate 50/50 between offline and online roots. If online roots are empty,
    # offline roots are sampled normally.
    offline_local_dataset_root: str | Sequence[str] | None = None
    online_local_dataset_root: str | Sequence[str] | OnlineDatasetConfig | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Temporary Flexiv option: replace dataset actions with deltas derived from
    # consecutive observation.state values.
    state_action: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()
    # Action dimension for padding (used by Groot datasets)
    action_dim: int | None = None
    # Multi-dataset support for Groot datasets
    data_dirs: list[str] | None = None
    offline_data_dirs: list[str] | None = None
    online_data_dirs: list[str] | None = None
    dataset_weights: list[float] | None = None
    
    # If true, will add prompt-style overlays on left camera images during training.
    add_promp: bool = False
    primitive_cmd_apply_prob: float = 0.5
    drag_2d_apply_prob: float = 0.5


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                input_transforms = [
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    ),
                ]
                input_transforms.append(_transforms.PadStatesAndActions(model_config.action_dim))
                return _transforms.Group(
                    inputs=input_transforms,
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                input_transforms = [
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=model_config.discrete_state_input,
                    ),
                ]
                input_transforms.append(_transforms.PadStatesAndActions(model_config.action_dim))
                return _transforms.Group(
                    inputs=input_transforms,
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFlexivDataConfig(DataConfigFactory):
    """
    Data config for local Flexiv realworld datasets converted into standard LeRobot format.
    """

    local_dataset_root: str | Sequence[str] = tyro.MISSING
    offline_local_dataset_root: str | Sequence[str] | None = None
    online_local_dataset_root: str | Sequence[str] | OnlineDatasetConfig | None = None
    # Deprecated compatibility fields. Prefer OnlineDatasetConfig(...).
    online_filter_denoise_step_lt_max: bool = False
    online_denoise_step_max: float | None = None
    default_prompt: str | None = None
    state_action: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.robot0_agentview_left",
                        "observation/wrist_image": "observation.images.robot0_eye_in_hand",
                        "observation/state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
        data_transform_inputs = []
        if self.state_action:
            data_transform_inputs.append(_transforms.FlexivStateDeltaActions())
        data_transform_inputs.append(libero_policy.LiberoInputs(model_type=model_config.model_type))
        data_transforms = _transforms.Group(
            inputs=data_transform_inputs,
            outputs=[libero_policy.LiberoOutputs()],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        online_local_dataset_root = self.online_local_dataset_root
        if not isinstance(online_local_dataset_root, OnlineDatasetConfig):
            online_local_dataset_root = OnlineDatasetConfig(
                root=online_local_dataset_root,
                filter_denoise_step_lt_max=self.online_filter_denoise_step_lt_max,
                denoise_step_max=self.online_denoise_step_max,
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            local_dataset_root=self.local_dataset_root,
            offline_local_dataset_root=self.offline_local_dataset_root,
            online_local_dataset_root=online_local_dataset_root,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            prompt_from_task=True,
            state_action=self.state_action,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaDataConfig(DataConfigFactory):
    """Config for training on Groot datasets."""
    
    repo_id: str | None = None
    
    data_dirs: Any | None = None
    offline_data_dirs: Any | None = None
    online_data_dirs: Any | None = None
    dataset_weights: list[float] | None = None
    
    action_dim: int | None = None
    
    # If true, will add prompt-style overlays on left camera images during training.
    add_promp: bool = False
    primitive_cmd_apply_prob: float = 0.5
    drag_2d_apply_prob: float = 0.5
    
    # Explicitly set asset_id for norm_stats saving. If not provided, will be auto-generated from data_dirs.
    asset_id: str | None = None
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group()

        data_transforms = _transforms.Group(
            inputs=[
                robocasa_policy.RobocasaInputs(
                    model_type=model_config.model_type,
                    include_prompt_images=not isinstance(
                        model_config,
                        pi0_motion_tower_config.Pi0MotionTowerConfig,
                    ),
                )
            ],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        base = self.create_base_config(assets_dirs, model_config)
        if self.asset_id is not None:
            base = dataclasses.replace(
                base,
                asset_id=self.asset_id,
                norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), self.asset_id),
            )

        # If assets stats are missing, derive stats from the Groot/LeRobot dataset metadata.
        fallback_norm_stats = None
        fallback_asset_id = None
        stats_data_dirs = self.data_dirs
        if self.offline_data_dirs is not None or self.online_data_dirs:
            stats_data_dirs = list(self.offline_data_dirs or []) + list(self.online_data_dirs or [])
        if base.norm_stats is None and stats_data_dirs and len(stats_data_dirs) > 0:
            try:
                if len(stats_data_dirs) == 1:
                    d = stats_data_dirs[0]
                    # Support both string path and dict with 'path' key
                    ds_meta = {"path": d} if isinstance(d, str) else d
                    norm_stats = _groot_openpi_dataset._load_norm_stats_from_groot_dataset(ds_meta)
                    if norm_stats is not None:
                        fallback_norm_stats = norm_stats
                        fallback_asset_id = self.asset_id or f"robocasa_{pathlib.Path(ds_meta['path']).name}"
                        logging.info(f"Loaded norm stats from local data dir: {d}")
                else:
                    # Support both string paths and dicts
                    ds_metas = [{"path": d} if isinstance(d, str) else d for d in stats_data_dirs]
                    norm_stats = _groot_openpi_dataset._load_norm_stats_from_groot_mixture_dataset(ds_metas)
                    if norm_stats is not None:
                        fallback_norm_stats = norm_stats
                        # Generate asset_id from mixture name or use a hash of data dirs
                        if self.asset_id:
                            fallback_asset_id = self.asset_id
                        else:
                            # Try to extract a meaningful name from first data dir
                            first_path = ds_metas[0]["path"] if isinstance(ds_metas[0], dict) else str(ds_metas[0])
                            fallback_asset_id = f"robocasa_mix_{pathlib.Path(first_path).name}"
                        logging.info(f"Loaded combined norm stats from {len(stats_data_dirs)} data dirs")
            except FileNotFoundError as exc:
                logging.info(f"Norm stats not found in configured data dirs, skipping dataset fallback: {exc}")
 
        return dataclasses.replace(
            base,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_dim=model_config.action_dim,
            data_dirs=self.data_dirs,
            offline_data_dirs=self.offline_data_dirs,
            online_data_dirs=self.online_data_dirs,
            dataset_weights=self.dataset_weights,
            norm_stats=base.norm_stats or fallback_norm_stats,
            asset_id=base.asset_id or fallback_asset_id,
            add_promp=self.add_promp,
            primitive_cmd_apply_prob=self.primitive_cmd_apply_prob,
            drag_2d_apply_prob=self.drag_2d_apply_prob,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),

    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir=os.environ.get("DROID_RLDS_DATA_ROOT", os.path.join(_RP_DATA_ROOT, "droid")),
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05_rp_robocasa_interactive",
        model=pi0_control_tower_config.Pi0ControlTowerConfig(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=[
                {
                    "path": _robocasa_data_path("v1.0/target/composite/PrepareCoffee/20250812/lerobot"),
                    "filter_key": None,
                }
            ],
            add_promp=True,
        ),
        batch_size=1,
        num_workers=0,
        num_train_steps=10,
        save_interval=100,
        overwrite=True,
        exp_name="debug_pi05_rp_robocasa_interactive",
        wandb_enabled=False,
        weight_loader=weight_loaders.NoOpWeightLoader(),
    ),
    #
    # RoboCasa dataset configs.
    #
    TrainConfig(
        name="pi0_robocasa_target50",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target50"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=500000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_finetune_target_atomic_seen",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("INSERT_CKPTPOINT_HERE"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_finetune_target_composite_seen",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_seen"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("INSERT_CKPTPOINT_HERE"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_finetune_target_composite_unseen",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_unseen"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("INSERT_CKPTPOINT_HERE"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_atomic_seen",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_atomic_seen_random_weight_init",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_atomic_seen_paligemma_init",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_composite_seen",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_seen"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_composite_unseen",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_unseen"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_pretrain_human300_mg60",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human300_mg60"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_pretrain_human300",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human300"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi05_robocasa_pretrain50",
        model=pi0_config.Pi0Config(
            pi05=True,
            max_token_len=128
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human50"],
            asset_id="robocasa_pretrain50",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=50000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=50000,
        save_interval=5000,
        keep_period=10000,
        batch_size=128,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_rp_robocasa_pretrain50",
        model=pi0_control_tower_config.Pi0ControlTowerConfig(
            pi05=True,
            main_tower_dropout_prob=0.5,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human50"],
            asset_id="robocasa_pretrain50",
            add_promp=True,
        ),
        weight_loader=weight_loaders.Pi0ControlTowerWeightLoader(
            base_weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            control_tower_init="original_paligemma",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=50000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=50000,
        save_interval=5000,
        keep_period=10000,
        batch_size=128,
        num_workers=16,
    ),
    #
    # Custom RoboCasa config for single task finetuning.
    #

    TrainConfig(
        name="pi05_rp_robocasa_prepare_coffee_finetune",
        model=pi0_control_tower_config.Pi0ControlTowerConfig(
            pi05=True,
            main_tower_dropout_prob=0.5,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=[
                {
                    "path": _robocasa_data_path("v1.0/target/composite/PrepareCoffee/20250812/lerobot"),
                    "filter_key": None,
                }
            ],
            # Set to True to add prompt-style overlays on left camera images.
            add_promp=True,
        ),
        weight_loader=weight_loaders.Pi0ControlTowerWeightLoader(
            base_weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            control_tower_init="checkpoint",
            control_tower_params_path="gs://openpi-assets/checkpoints/pi05_base/params",
        ),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_rp_pretrain50_finetune_control_only",
        model=pi0_control_tower_config.Pi0ControlTowerConfig(
            pi05=True,
            main_tower_dropout_prob=0.5,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human50"],
            asset_id="robocasa_pretrain50",
            add_promp=True,
        ),
        weight_loader=weight_loaders.Pi0ControlTowerWeightLoader(
            base_weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_ROBOCASA_PRETRAIN50_CKPT),
            control_tower_init="original_paligemma",
        ),
        freeze_filter=nnx.All(
            nnx.Param,
            nnx_utils.PathRegex(".*PaliGemma.*"), # freeze main tower (prefix)
        ),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_rp_pretrain50_finetune_control_action_expert",
        model=pi0_control_tower_config.Pi0ControlTowerConfig(
            pi05=True,
            main_tower_dropout_prob=0.5,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human50"],
            asset_id="robocasa_pretrain50",
            add_promp=True,
        ),
        weight_loader=weight_loaders.Pi0ControlTowerWeightLoader(
            base_weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_ROBOCASA_PRETRAIN50_CKPT),
            control_tower_init="original_paligemma",
        ),
        freeze_filter=nnx.All(
            nnx.Param,
            nnx_utils.PathRegex(".*PaliGemma.*"),
            nnx.Not(nnx_utils.PathRegex(".*PaliGemma.*llm.*_1.*")),
        ),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_motion_tower_pretrain50",
        model=pi0_motion_tower_config.Pi0MotionTowerConfig(
            pi05=True,
            main_tower_dropout_prob=0.5,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human50"],
            asset_id="robocasa_pretrain50",
            add_promp=True,
        ),
        weight_loader=weight_loaders.Pi0MotionTowerWeightLoader(
            base_weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_ROBOCASA_PRETRAIN50_CKPT),
        ),
        freeze_filter=nnx.All(
            nnx.Param,
            nnx_utils.PathRegex(".*PaliGemma.*"),
        ),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_pretrain_motion",
        model=pi0_motion_tower_config.Pi0MotionTowerConfig(
            pi05=True,
            max_token_len=128,
            main_tower_dropout_prob=0,
            local_motion_aux_loss_weight=3.0,
            global_motion_aux_loss_weight=1.0,
            drag_2d_aux_loss_weight=5.0,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=get_ds_soup(
                split="pretrain",
                task_set="pretrain_motion",
                source="human",
                demo_fraction=1,
            ),
            asset_id="robocasa_pretrain_motion",
            add_promp=True,
        ),
        weight_loader=weight_loaders.Pi0MotionTowerWeightLoader(
            base_weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_finetune_motion_freeze_all",
        model=pi0_motion_tower_config.Pi0MotionTowerConfig(
            pi05=True,
            main_tower_dropout_prob=0,
            local_motion_aux_loss_weight=3.0,
            global_motion_aux_loss_weight=2.0,
            drag_2d_aux_loss_weight=5.0,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=[
                # {
                #     "path": _robocasa_data_path("v1.0/rollouts/CloseMicrowave_2026-05-06-17-52/lerobot"),
                #     "filter_key": None,
                # },
                {
                    "path": _robocasa_data_path("v1.0/rollouts/CloseToasterOvenDoor_2026-05-06-17-52/lerobot"),
                    "filter_key": None,
                },
                {
                    "path": _robocasa_data_path("v1.0/rollouts/PickPlaceCounterToCabinet/pi05_pretrain_motion_pretrain_cabinet_2026-05-16-10-53/lerobot"),
                    "filter_key": None,
                }
            ],
            asset_id="robocasa_finetune_motion",
            add_promp=True,
        ),
        weight_loader=weight_loaders.Pi0MotionTowerWeightLoader(
            base_weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(_RP_DATA_ROOT, "pi05_robocasa_pretrain50/params")),
        ),
        freeze_filter=nnx.All(
            nnx.Param,
            nnx.Any(
                nnx_utils.PathRegex(".*PaliGemma.*"),
                nnx_utils.PathRegex(".*action_in_proj.*"),
                nnx_utils.PathRegex(".*time_mlp_in.*"),
                nnx_utils.PathRegex(".*time_mlp_out.*"),
                nnx_utils.PathRegex(".*action_out_proj.*"),
            ),
        ),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),

    TrainConfig(
        name="pi05_flexiv_bread",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_bread",
            local_dataset_root=os.path.join(_RP_DATA_ROOT, "real_robot_data/bread"),
            offline_local_dataset_root=os.path.join(_RP_DATA_ROOT, "real_robot_data/bread"),
            online_local_dataset_root=OnlineDatasetConfig(root=[]),
            default_prompt="cook the bread",
            assets=AssetsConfig(
                asset_id="flexiv_bread",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_cup",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_cup",
            local_dataset_root=os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup"),
            offline_local_dataset_root=os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup"),
            online_local_dataset_root=OnlineDatasetConfig(root=[]),
            default_prompt="hang the cup on the rack",
            assets=AssetsConfig(
                assets_dir=os.path.join(_RP_DATA_ROOT, "real_robot_data/cup"),
                asset_id="lerobot-cup",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_cup_dagger1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_cup",
            local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup-dagger1"),
            ],
            offline_local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup"),
            ],
            online_local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup-dagger1"),
            ],
            default_prompt="hang the cup on the rack",
            assets=AssetsConfig(
                assets_dir=os.path.join(
                    _RP_DATA_ROOT,
                    "pi05/pi05_flexiv_cup/cup_0819_base/29999/assets",
                ),
                asset_id="lerobot-cup",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(
                _RP_DATA_ROOT,
                "pi05/pi05_flexiv_cup/cup_0819_base/29999/params",
            )
        ),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=10000,
        save_interval=2500,
        keep_period=2500,
        batch_size=64,
        num_workers=16,
    ),
        TrainConfig(
        name="pi05_flexiv_cup_dagger2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_cup",
            local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup-dagger1"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup-dagger2"),
            ],
            offline_local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup-dagger1"),
            ],
            online_local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/cup/lerobot-cup-dagger2"),
            ],
            default_prompt="hang the cup on the rack",
            assets=AssetsConfig(
                assets_dir=os.path.join(
                    _RP_DATA_ROOT,
                    "pi05/pi05_flexiv_cup/cup_0819_base/29999/assets",
                ),
                asset_id="lerobot-cup",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(
                _RP_DATA_ROOT,
                "pi05/pi05_flexiv_cup_dagger1/cup-openpi-dagger1/9999/params",
            )
        ),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=10000,
        save_interval=2500,
        keep_period=2500,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_maze",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_maze",
            local_dataset_root=os.path.join(_RP_DATA_ROOT, "real_robot_data/maze/lerobot-maze"),
            offline_local_dataset_root=os.path.join(_RP_DATA_ROOT, "real_robot_data/maze/lerobot-maze"),
            online_local_dataset_root=OnlineDatasetConfig(root=[]),
            default_prompt="push cube to the red flag through the maze",
            assets=AssetsConfig(
                assets_dir=os.path.join(_RP_DATA_ROOT, "real_robot_data/maze"),
                asset_id="flexiv_maze",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_maze_dagger1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_maze",
            local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/maze"),
            ],
            offline_local_dataset_root=os.path.join(_RP_DATA_ROOT, "real_robot_data/maze"),
            online_local_dataset_root=OnlineDatasetConfig(
                root=os.path.join(_RP_DATA_ROOT, "real_robot_data/maze/lerobot-maze-dagger1"),
                filter_denoise_step_lt_max=False,
            ),
            default_prompt="push cube to the red flag through the maze",
            assets=AssetsConfig(
                assets_dir=os.path.join(
                    _RP_DATA_ROOT,
                    "pi05/pi05_flexiv_maze/maze_0819_base/29999/assets",
                ),
                asset_id="flexiv_maze",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(
                _RP_DATA_ROOT,
                "pi05/pi05_flexiv_maze/maze_0819_base/29999/params",
            )
        ),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=10000,
        save_interval=2500,
        keep_period=2500,
        batch_size=64,
        num_workers=16,
    ),
        TrainConfig(
        name="pi05_flexiv_maze_dagger2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_maze",
            local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/maze"),
            ],
            offline_local_dataset_root=os.path.join(_RP_DATA_ROOT, "real_robot_data/maze"),
            online_local_dataset_root=OnlineDatasetConfig(
                root=os.path.join(_RP_DATA_ROOT, "real_robot_data/maze/lerobot-maze-dagger2"),
                filter_denoise_step_lt_max=False,
            ),
            default_prompt="push cube to the red flag through the maze",
            assets=AssetsConfig(
                assets_dir=os.path.join(
                    _RP_DATA_ROOT,
                    "pi05/pi05_flexiv_maze/maze_0819_base/29999/assets",
                ),
                asset_id="flexiv_maze",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(
                _RP_DATA_ROOT,
                "pi05/pi05_flexiv_maze_dagger1/maze_dagger1/9999/params",
            )
        ),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=10000,
        save_interval=2500,
        keep_period=2500,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_book",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_book",
            local_dataset_root=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "book"),
            default_prompt="put book into shelf",
            assets=AssetsConfig(
                assets_dir=_ACTIVE_GLASS_DATA_ROOT,
                asset_id="book",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "openpi_assets"),
        checkpoint_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "checkpoints"),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_active_glass_bread",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_active_glass_bread",
            local_dataset_root=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "bread"),
            default_prompt="put bread into toaster",
            assets=AssetsConfig(
                assets_dir=_ACTIVE_GLASS_DATA_ROOT,
                asset_id="bread",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "openpi_assets"),
        checkpoint_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "checkpoints"),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    # Derive delta actions from the original bread observation.state sequences.
    TrainConfig(
        name="pi05_flexiv_active_glass_bread_state_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_active_glass_bread_state_action",
            local_dataset_root=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "bread"),
            default_prompt="put bread into toaster",
            state_action=True,
            assets=AssetsConfig(
                asset_id="bread_state_action",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "openpi_assets"),
        checkpoint_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "checkpoints"),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_active_glass_teapot",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_active_glass_teapot",
            local_dataset_root=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "teapot"),
            default_prompt="pour water into cup",
            assets=AssetsConfig(
                assets_dir=_ACTIVE_GLASS_DATA_ROOT,
                asset_id="teapot",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "openpi_assets"),
        checkpoint_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "checkpoints"),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_active_glass_teapot_state_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_active_glass_teapot_state_action",
            local_dataset_root=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "teapot_state_action"),
            default_prompt="pour water into cup",
            assets=AssetsConfig(
                assets_dir=_ACTIVE_GLASS_DATA_ROOT,
                asset_id="teapot_state_action",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "openpi_assets"),
        checkpoint_base_dir=os.path.join(_ACTIVE_GLASS_DATA_ROOT, "checkpoints"),
        num_train_steps=30000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_toaster_right_dagger1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_bread",
            local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right_var"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right-ext"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/pick-bread-dagger"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/toaster_right_openpi_dagger1"),
            ],
            offline_local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right_var"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right-ext"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/pick-bread-dagger"),
            ],
            online_local_dataset_root=OnlineDatasetConfig(
                root=[
                    os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/toaster_right_openpi_dagger1"),
                ],
                filter_denoise_step_lt_max=True,
            ),
            default_prompt="insert the bread into the left slot of the toaster",
            assets=AssetsConfig(
                assets_dir=os.path.join(
                    _RP_DATA_ROOT,
                    "pi05/pi05_flexiv_toaster_right/pi05_flexiv_toaster_right_25000/assets",
                ),
                asset_id="flexiv_toaster_right",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(
                _RP_DATA_ROOT,
                "pi05/pi05_flexiv_toaster_right/pi05_flexiv_toaster_right_25000/params",
            )
        ),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=10000,
        save_interval=2500,
        keep_period=2500,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_toaster_right_state_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_bread",
            local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right_var"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right-ext"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/pick-bread-dagger"),
            ],
            default_prompt="cook the bread",
            state_action=True,
            assets=AssetsConfig(
                asset_id="flexiv_toaster_right_state_action",
            ),
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=10000,
        save_interval=2500,
        keep_period=2500,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_toaster_right_dagger2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_bread",
            local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right_var"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right-ext"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/pick-bread-dagger"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/toaster_right_openpi_dagger1"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/toaster_right_openpi_dagger2"),
            ],
            offline_local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right_var"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right-ext"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/pick-bread-dagger"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/toaster_right_openpi_dagger1"),
            ],
            online_local_dataset_root=OnlineDatasetConfig(
                root=[
                    os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/toaster_right_openpi_dagger2"),
                ],
                filter_denoise_step_lt_max=True,
            ),
            default_prompt="cook the bread",
            assets=AssetsConfig(
                assets_dir=os.path.join(
                    _RP_DATA_ROOT,
                    "pi05/pi05_flexiv_toaster_right_dagger1/"
                    "toaster_right_openpi_dagger1_from_25000/9999/assets",
                ),
                asset_id="flexiv_toaster_right",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(
                _RP_DATA_ROOT,
                "pi05/pi05_flexiv_toaster_right_dagger1/"
                "toaster_right_openpi_dagger1_from_25000/9999/params",
            )
        ),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=10000,
        save_interval=2500,
        keep_period=2500,
        batch_size=64,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_flexiv_toaster_right_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotFlexivDataConfig(
            repo_id="realworld/flexiv_bread",
            local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right_var"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right-ext"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right-rollout1"),
            ],
            offline_local_dataset_root=[
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right_var"),
                os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right-ext"),
            ],
            online_local_dataset_root=OnlineDatasetConfig(
                root=[
                    os.path.join(_RP_DATA_ROOT, "real_robot_data/bread/lerobot-toaster-right-rollout1"),
                ],
                filter_denoise_step_lt_max=True,
            ),
            default_prompt="cook the bread",
            assets=AssetsConfig(
                assets_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets", "pi05_flexiv_toaster_right"),
                asset_id="flexiv_toaster_right",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(_RP_DATA_ROOT, "pi05/pi05_flexiv_toaster_right/toaster_right_0622/29999/params")
        ),
        assets_base_dir=os.path.join(_RP_DATA_ROOT, "pi05", "assets"),
        checkpoint_base_dir=os.path.join(_RP_DATA_ROOT, "pi05"),
        num_train_steps=10000,
        save_interval=2500,
        keep_period=2500,
        batch_size=64,
        num_workers=16,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
