# Training Entry Points

Complete [Quick Start](../README.md#quick-start) first. Paths below describe expected layouts, not datasets or weights bundled with this checkout.

`steering/` is an inference-only composition layer. It loads trained checkpoints and normalization assets, then serves:

```text
steerer -> phase-1 action -> policy -> phase-2 action
```

Training remains owned by each model stack:

- Evo-1 trains through `scripts/realworld/train/evo1/`.
- OpenPI trains through `openpi/scripts/train.py`.
- FastWAM trains through `FastWAM/scripts/train.py` and `FastWAM/scripts/train_zero*.sh`.
- Diffusion Policy trains through `diffusion_policy/train.py`.

Default training outputs are under the shared data root:

```text
$RP_DATA_ROOT/evo1
$RP_DATA_ROOT/pi05
$RP_DATA_ROOT/fastwam
$RP_DATA_ROOT/diffusion_policy
```

## Evo-1

Use the bash launchers under `scripts/realworld/train/evo1/`.

Noise-data two-stage run:

```bash
export RP_DATA_ROOT=/path/to/data_root
bash scripts/realworld/train/evo1/noise_train.sh
```

The default checkpoint layout is:

```text
$RP_DATA_ROOT/evo1/<run_name>/step_final
```


## OpenPI

Use OpenPI's training entrypoint with a registered config from `openpi/src/openpi/training/config.py`.

First compute normalization statistics for the training data:

```bash
export RP_DATA_ROOT=/path/to/data_root
export OPENPI_DATA_HOME=/path/to/openpi_cache

python openpi/scripts/compute_norm_stats.py --config-name pi05_flexiv_bread
```

For `pi05_flexiv_bread`, this writes:

```text
$RP_DATA_ROOT/pi05/assets/pi05_flexiv_bread/flexiv_bread/norm_stats.json
```

Then launch training:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python openpi/scripts/train.py pi05_flexiv_bread \
  --exp_name pi05_flexiv_bread
```

For `pi05_flexiv_bread`, checkpoints default to:

```text
$RP_DATA_ROOT/pi05/pi05_flexiv_bread/<exp_name>
```

For a different layout, add or copy a `TrainConfig` in `openpi/src/openpi/training/config.py`, change its `data.offline_local_dataset_root` / `data.online_local_dataset_root` for standard LeRobot data, or `data.offline_data_dirs` / `data.online_data_dirs` for Groot data, then rerun `compute_norm_stats.py` for that config before training. If online data is provided, the OpenPI torch loader samples 50% offline and 50% online; if online data is empty, it samples 100% offline.

## Diffusion Policy

Use the Diffusion Policy Hydra config from the repo root:

```bash
export RP_DATA_ROOT=/path/to/data_root
python diffusion_policy/train.py \
  --config-name=train_diffusion_transformer_hybrid_flexiv_bread_right
```

For Flexiv LeRobot v2.1 image configs, split data with:

```yaml
offline_dataset_paths:
  - /path/to/offline_lerobot_a
  - /path/to/offline_lerobot_b
online_dataset_paths:
  - /path/to/online_lerobot_a
```

When `online_dataset_paths` is non-empty, `LeRobotV21ImageDataset` exposes a 50% offline / 50% online training sample stream. When it is empty, training remains 100% offline. The bread-right config treats:

```text
$RP_DATA_ROOT/real_robot_data/bread/toaster_right_dp_dagger1
```

as online data.

## FastWAM

Use FastWAM's Accelerate launchers from inside `FastWAM/`.

First precompute text embeddings for the dataset prompts. The bread config stores them next to the bread datasets.

```bash
export RP_DATA_ROOT=/path/to/data_root
cd FastWAM
python scripts/precompute_text_embeds.py task=bread_joint_2cam224_1e-4
```

Then launch training:

```bash
bash scripts/train_zero1.sh 8 task=bread_joint_2cam224_1e-4
```

FastWAM writes to:

```text
$RP_DATA_ROOT/fastwam/<task_name>/<run_id>
```

The bread text embedding cache is derived from `offline_dataset_dirs` and lives next to the selected dataset group:

```text
$RP_DATA_ROOT/real_robot_data/bread/text_embeds_cache/fastwam
```

Dataset selection is Hydra-config-driven. The bread data config is:

```text
FastWAM/configs/data/bread.yaml
```

By default it trains on three LeRobot datasets under the shared data root:

```yaml
offline_dataset_dirs:
  - ${oc.env:RP_DATA_ROOT}/real_robot_data/bread/lerobot-pot
  - ${oc.env:RP_DATA_ROOT}/real_robot_data/bread/lerobot-toaster-left
  - ${oc.env:RP_DATA_ROOT}/real_robot_data/bread/lerobot-toaster-right
online_dataset_dirs: []
```

Pass different dataset dirs with a Hydra override. Use the same override for `precompute_text_embeds.py` and training. If `online_dataset_dirs` is non-empty, FastWAM samples 50% offline and 50% online; if it is empty, it samples 100% offline:

```bash
RP_DATA_ROOT=/path/to/data_root \
python scripts/precompute_text_embeds.py \
  task=bread_joint_2cam224_1e-4 \
  "data.train.offline_dataset_dirs=[/path/to/offline_a,/path/to/offline_b]" \
  "data.train.online_dataset_dirs=[/path/to/online_a]"

RP_DATA_ROOT=/path/to/data_root \
bash scripts/train_zero1.sh 8 \
  task=bread_joint_2cam224_1e-4 \
  "data.train.offline_dataset_dirs=[/path/to/offline_a,/path/to/offline_b]" \
  "data.train.online_dataset_dirs=[/path/to/online_a]"
```

With that override, the text embedding cache is placed under the common parent of the selected offline datasets, and prompts from both offline and online dirs are written into that cache:

```text
/path/to/text_embeds_cache/fastwam
```

For first training, leave `pretrained_norm_stats` unset. FastWAM computes `dataset_stats.json` during dataset initialization and writes it into the run output directory. Serving/eval should use that `dataset_stats.json`, not Evo-1/OpenPI `norm_stats.json`.
