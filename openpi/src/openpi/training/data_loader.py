from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ModuleNotFoundError:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

import openpi.groot_utils.groot_openpi_dataset as _groot_openpi_dataset
import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class BalancedOfflineOnlineDataset(Dataset):
    """Expose offline and online datasets through a 50/50 logical index space."""

    def __init__(self, offline_dataset: Dataset, online_dataset: Dataset | None = None):
        if len(offline_dataset) == 0:
            raise ValueError("BalancedOfflineOnlineDataset requires at least one offline sample.")
        if online_dataset is not None and len(online_dataset) == 0:
            raise ValueError("BalancedOfflineOnlineDataset requires at least one online sample.")
        self._offline_dataset = offline_dataset
        self._online_dataset = online_dataset

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        if self._online_dataset is None:
            return self._offline_dataset[idx]
        if idx % 2 == 0:
            return self._offline_dataset[(idx // 2) % len(self._offline_dataset)]
        return self._online_dataset[(idx // 2) % len(self._online_dataset)]

    def __len__(self) -> int:
        if self._online_dataset is None:
            return len(self._offline_dataset)
        return 2 * max(len(self._offline_dataset), len(self._online_dataset))


class BalancedOfflineOnlineSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: Dataset, seed: int = 0):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        half_len = len(self.dataset) // 2
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        offline_order = torch.randperm(half_len, generator=generator).tolist()
        online_order = torch.randperm(half_len, generator=generator).tolist()
        indices = []
        for offline_idx, online_idx in zip(offline_order, online_order):
            indices.append(2 * offline_idx)
            indices.append(2 * online_idx + 1)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    local_dataset_root = getattr(data_config, "local_dataset_root", None)
    
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # 1) groot datasets
    offline_data_dirs = getattr(data_config, "offline_data_dirs", None)
    online_data_dirs = getattr(data_config, "online_data_dirs", None)
    if offline_data_dirs is not None or online_data_dirs:
        data_dirs = _as_list(offline_data_dirs if offline_data_dirs is not None else data_config.data_dirs)
        online_dirs = _as_list(online_data_dirs)
        if not data_dirs:
            raise ValueError("At least one offline data dir is required when online data dirs are configured.")
        data_dirs = data_dirs + online_dirs
        dataset_weights = None
        if online_dirs:
            dataset_weights = (
                [0.5 / len(data_dirs[: -len(online_dirs)])] * len(data_dirs[: -len(online_dirs)])
                + [0.5 / len(online_dirs)] * len(online_dirs)
            )
    else:
        data_dirs = getattr(data_config, "data_dirs", None)
        dataset_weights = getattr(data_config, "dataset_weights", None)

    if data_dirs:
        add_promp = getattr(data_config, "add_promp", False)
        primitive_cmd_apply_prob = float(getattr(data_config, "primitive_cmd_apply_prob", 0.5))
        drag_2d_apply_prob = float(getattr(data_config, "drag_2d_apply_prob", 0.5))
        if len(data_dirs) == 1:
            return _groot_openpi_dataset.GrootOpenpiSingleDataset(
                dataset_meta=data_dirs[0],
                action_horizon=action_horizon,
                add_promp=add_promp,
                primitive_cmd_apply_prob=primitive_cmd_apply_prob,
                drag_2d_apply_prob=drag_2d_apply_prob,
            )
        elif len(data_dirs) > 1:
            return _groot_openpi_dataset.GrootOpenpiMultiDataset(
                dataset_meta_list=data_dirs,
                dataset_weights=dataset_weights,
                dataset_weights_alpha=0.4,
                action_horizon=action_horizon,
                add_promp=add_promp,
                primitive_cmd_apply_prob=primitive_cmd_apply_prob,
                drag_2d_apply_prob=drag_2d_apply_prob,
            )
        else:
            raise ValueError

    # Standard (openpi) LeRobot dataset loading
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")

    offline_local_dataset_root = getattr(data_config, "offline_local_dataset_root", None)
    online_local_dataset_root = getattr(data_config, "online_local_dataset_root", None)
    online_filter_denoise_step_lt_max = False
    online_denoise_step_max = None
    if isinstance(online_local_dataset_root, _config.OnlineDatasetConfig):
        online_filter_denoise_step_lt_max = bool(online_local_dataset_root.filter_denoise_step_lt_max)
        online_denoise_step_max = online_local_dataset_root.denoise_step_max
        online_local_dataset_root = online_local_dataset_root.root
    else:
        online_filter_denoise_step_lt_max = bool(
            getattr(data_config, "online_filter_denoise_step_lt_max", False)
        )
        online_denoise_step_max = getattr(data_config, "online_denoise_step_max", None)
    if offline_local_dataset_root is not None or online_local_dataset_root:
        offline_roots = _resolve_local_lerobot_dataset_roots(
            offline_local_dataset_root if offline_local_dataset_root is not None else local_dataset_root
        )
        online_roots = _resolve_local_lerobot_dataset_roots(online_local_dataset_root)
        if not offline_roots:
            raise ValueError("At least one offline local dataset root is required.")
        offline_dataset = _create_lerobot_dataset_group(
            offline_roots,
            repo_id=repo_id,
            action_horizon=action_horizon,
            action_sequence_keys=data_config.action_sequence_keys,
            prompt_from_task=data_config.prompt_from_task,
            state_action=data_config.state_action,
        )
        online_dataset = None
        if online_roots:
            online_dataset = _create_lerobot_dataset_group(
                online_roots,
                repo_id=repo_id,
                action_horizon=action_horizon,
                action_sequence_keys=data_config.action_sequence_keys,
                prompt_from_task=data_config.prompt_from_task,
                state_action=data_config.state_action,
                filter_denoise_step_lt_max=online_filter_denoise_step_lt_max,
                denoise_step_max=online_denoise_step_max,
            )
        return BalancedOfflineOnlineDataset(offline_dataset, online_dataset)

    dataset_roots = _resolve_local_lerobot_dataset_roots(local_dataset_root)
    if dataset_roots:
        logging.info("Loading %d local LeRobot datasets from %s", len(dataset_roots), local_dataset_root)
        return _create_lerobot_dataset_group(
            dataset_roots,
            repo_id=repo_id,
            action_horizon=action_horizon,
            action_sequence_keys=data_config.action_sequence_keys,
            prompt_from_task=data_config.prompt_from_task,
            state_action=data_config.state_action,
        )

    return _create_single_lerobot_dataset(
        repo_id=repo_id,
        local_dataset_root=local_dataset_root,
        action_horizon=action_horizon,
        action_sequence_keys=data_config.action_sequence_keys,
        prompt_from_task=data_config.prompt_from_task,
        state_action=data_config.state_action,
    )


def _create_lerobot_dataset_group(
    dataset_roots: Sequence[str],
    *,
    repo_id: str,
    action_horizon: int,
    action_sequence_keys: Sequence[str],
    prompt_from_task: bool,
    state_action: bool = False,
    filter_denoise_step_lt_max: bool = False,
    denoise_step_max: float | None = None,
) -> Dataset:
    datasets: list[Dataset] = []
    for dataset_root in dataset_roots:
        dataset = _create_single_lerobot_dataset(
            repo_id=repo_id,
            local_dataset_root=dataset_root,
            action_horizon=action_horizon,
            action_sequence_keys=action_sequence_keys,
            prompt_from_task=prompt_from_task,
            state_action=state_action,
        )
        if filter_denoise_step_lt_max:
            dataset = _filter_lerobot_dataset_by_denoise_step(
                dataset,
                pathlib.Path(dataset_root),
                denoise_step_max=denoise_step_max,
            )
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return typing.cast(Dataset, torch.utils.data.ConcatDataset(datasets))


def _resolve_local_lerobot_dataset_roots(local_dataset_root: str | Sequence[str] | None) -> list[str]:
    """Expand local root(s) into concrete LeRobot dataset directories."""
    if local_dataset_root is None:
        return []

    if isinstance(local_dataset_root, str):
        root_paths = [pathlib.Path(local_dataset_root)]
    else:
        root_paths = [pathlib.Path(root) for root in local_dataset_root]

    dataset_roots: list[str] = []
    for root_path in root_paths:
        if not root_path.exists():
            continue

        if (root_path / "meta" / "info.json").exists():
            dataset_roots.append(str(root_path))
            continue

        dataset_roots.extend(
            sorted(
                str(child)
                for child in root_path.iterdir()
                if child.is_dir() and child.name.startswith("lerobot") and (child / "meta" / "info.json").exists()
            )
        )
    return dataset_roots


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _create_single_lerobot_dataset(
    *,
    repo_id: str,
    local_dataset_root: str | None,
    action_horizon: int,
    action_sequence_keys: Sequence[str],
    prompt_from_task: bool,
    state_action: bool = False,
) -> Dataset:
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=local_dataset_root)
    delta_timestamps = {key: [t / dataset_meta.fps for t in range(action_horizon)] for key in action_sequence_keys}
    if state_action:
        delta_timestamps["observation.state"] = [t / dataset_meta.fps for t in range(action_horizon + 1)]
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=local_dataset_root,
        delta_timestamps=delta_timestamps,
    )

    if prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def _filter_lerobot_dataset_by_denoise_step(
    dataset: Dataset,
    dataset_root: pathlib.Path,
    *,
    denoise_step_max: float | None,
) -> Dataset:
    threshold = _resolve_online_denoise_step_max(dataset_root, denoise_step_max)
    if threshold is None:
        logging.warning(
            "Skipping online denoise-step filtering for %s because no max denoise step could be resolved.",
            dataset_root,
        )
        return dataset

    indices = _online_denoise_step_indices(dataset_root, threshold)
    if indices is None:
        logging.warning(
            "Skipping online denoise-step filtering for %s because its parquet files do not contain denoise_step.",
            dataset_root,
        )
        return dataset
    if not indices:
        raise ValueError(
            f"Online dataset {dataset_root} has no samples with denoise_step < {threshold:g}."
        )

    logging.info(
        "Filtered online dataset %s by denoise_step < %.3f: %d/%d samples kept.",
        dataset_root,
        threshold,
        len(indices),
        len(dataset),
    )
    return typing.cast(Dataset, torch.utils.data.Subset(dataset, indices))


def _resolve_online_denoise_step_max(dataset_root: pathlib.Path, override: float | None) -> float | None:
    if override is not None:
        try:
            value = float(override)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value > 0 else None

    metadata_path = dataset_root / "extras" / "dataset_meta_server.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Could not parse server metadata at %s.", metadata_path)
        return None

    candidates = [
        metadata.get("max_phase2_steps"),
        metadata.get("phase2_max_steps"),
        metadata.get("policy_inference_steps"),
        metadata.get("num_steps"),
        metadata.get("num_inference_steps"),
    ]
    for nested_key in ("openpi", "diffusion_policy", "fastwam"):
        nested = metadata.get(nested_key)
        if isinstance(nested, dict):
            candidates.extend(
                [
                    nested.get("max_phase2_steps"),
                    nested.get("phase2_max_steps"),
                    nested.get("policy_inference_steps"),
                    nested.get("num_steps"),
                    nested.get("num_inference_steps"),
                ]
            )

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    return None


def _online_denoise_step_indices(dataset_root: pathlib.Path, threshold: float) -> list[int] | None:
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No LeRobot parquet files found under {dataset_root / 'data'}.")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to filter online LeRobot datasets by denoise_step.") from exc

    kept_indices: list[int] = []
    row_offset = 0
    saw_denoise_step = False
    for parquet_path in parquet_paths:
        try:
            values = pd.read_parquet(parquet_path, columns=["denoise_step"])["denoise_step"].to_list()
        except Exception:
            try:
                row_count = len(pd.read_parquet(parquet_path, columns=["frame_index"]))
            except Exception:
                row_count = len(pd.read_parquet(parquet_path))
            row_offset += row_count
            continue

        saw_denoise_step = True
        for local_idx, value in enumerate(values):
            denoise_step = _as_scalar_denoise_step(value)
            if denoise_step is not None and denoise_step < threshold:
                kept_indices.append(row_offset + local_idx)
        row_offset += len(values)

    return kept_indices if saw_denoise_step else None


def _as_scalar_denoise_step(value: typing.Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.size == 0:
            return None
        value = array.reshape(-1)[0]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    if sampler is None and shuffle and _contains_balanced_offline_online_dataset(dataset):
        sampler = BalancedOfflineOnlineSampler(dataset, seed=seed)
        shuffle = False

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def _contains_balanced_offline_online_dataset(dataset) -> bool:
    if isinstance(dataset, BalancedOfflineOnlineDataset):
        return True
    inner_dataset = getattr(dataset, "_dataset", None)
    if inner_dataset is None:
        return False
    return _contains_balanced_offline_online_dataset(inner_dataset)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    def _safe_put(x):
                        arr = np.asarray(x)
                        if arr.dtype.kind in ('U', 'S', 'O'):
                            return x
                        return jax.make_array_from_process_local_data(self._sharding, x)
                    yield jax.tree.map(_safe_put, batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
