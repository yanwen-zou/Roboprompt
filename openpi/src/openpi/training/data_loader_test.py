import dataclasses
import json

import jax
import numpy as np
import pandas as pd

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_balanced_offline_online_dataset():
    class TinyDataset:
        def __init__(self, prefix: str, length: int):
            self.prefix = prefix
            self.length = length

        def __getitem__(self, index):
            return f"{self.prefix}{index}"

        def __len__(self):
            return self.length

    dataset = _data_loader.BalancedOfflineOnlineDataset(
        TinyDataset("offline", 3),
        TinyDataset("online", 1),
    )

    assert len(dataset) == 6
    assert [dataset[i].startswith("offline") for i in range(len(dataset))] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert [dataset[i] for i in range(len(dataset))] == [
        "offline0",
        "online0",
        "offline1",
        "online0",
        "offline2",
        "online0",
    ]

    sampler = _data_loader.BalancedOfflineOnlineSampler(dataset, seed=0)
    sampler_indices = list(sampler)
    assert len(sampler_indices) == len(dataset)
    assert [idx % 2 == 0 for idx in sampler_indices] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_online_denoise_step_threshold_from_metadata(tmp_path):
    root = tmp_path / "online"
    (root / "extras").mkdir(parents=True)
    (root / "extras" / "dataset_meta_server.json").write_text(
        json.dumps({"max_phase2_steps": 20.0}),
        encoding="utf-8",
    )

    assert _data_loader._resolve_online_denoise_step_max(root, None) == 20.0
    assert _data_loader._resolve_online_denoise_step_max(root, 10.0) == 10.0


def test_online_denoise_step_indices_keep_only_below_threshold(tmp_path):
    root = tmp_path / "online"
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "denoise_step": [20.0, 19.5, 0.0, np.nan],
            "frame_index": [0, 1, 2, 3],
        }
    ).to_parquet(data_dir / "episode_000000.parquet", index=False)

    assert _data_loader._online_denoise_step_indices(root, 20.0) == [1, 2]
    assert _data_loader._online_denoise_step_indices(root, 1.0) == [2]


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
