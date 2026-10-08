# How to Add a Phase-2 Policy

## Goal

A phase-2 policy is the downstream policy in the steering runtime. It consumes the current observation and may consume an upstream `Phase1Action`.

The runtime shape is:

```text
client obs
  |
  v
SteeredPhase2Runtime.infer(...)
  |
  |-- phase1 backend: Evo-1, ...
  |      obs -> Phase1Action
  |
  |-- phase2 policy: OpenPI, FastWAM, ...
         obs + optional Phase1Action -> final action chunk
```

Keep the boundary strict:

- `steering/backends/*`: phase-1 action producers only.
- `steering/policies/*`: downstream phase-2 policies.
- `steering/runtime.py`: generic composition layer.

Do not put a downstream policy into `steering/backends` just because it is a model with its own checkpoint. A backend should mean "this produces `Phase1Action`".

## Required Interface

A phase-2 policy object should implement:

```python
def infer_direct(
    self,
    obs: dict,
    *,
    noise: np.ndarray | None,
    sample_kwargs: dict[str, Any],
) -> dict:
    ...

def refine(
    self,
    obs: dict,
    *,
    phase1_action: Phase1Action,
    phase2_steps: float,
    noise: np.ndarray | None,
    sample_kwargs: dict[str, Any],
) -> dict:
    ...

@property
def metadata(self) -> dict[str, Any]:
    ...
```

`infer_direct()` is used when steering is disabled. `refine()` is used when steering is enabled and the runtime has already called the phase-1 backend.

## Files to Touch

For a new phase-2 policy named `foo`:

```text
steering/policies/foo.py
steering/policies/__init__.py
steering/scripts/serve_policy.py
```

Usually do not touch:

```text
steering/backends/*
steering/runtime.py
```

Only edit `steering/runtime.py` if the generic composition contract itself needs to change.

## Adapter Responsibilities

The phase-2 policy adapter owns all translation between runtime obs/action conventions and the downstream model's conventions.

It should handle:

1. Model checkpoint loading.
2. Model config loading.
3. Observation key mapping.
4. Image resizing and camera ordering.
5. Prompt/task text mapping.
6. State/proprio normalization.
7. Phase-1 action normalization.
8. Final action denormalization.
9. Metadata for the server.

The main invariant:

```text
phase-1 backends never know which phase-2 policy consumes their actions
```

## Phase-1 Action Conditioning

`Phase1Action.actions` should be treated as raw robot/client action units unless that backend explicitly documents otherwise.

A downstream phase-2 policy usually expects its own normalized action space. The adapter must:

1. Receive `Phase1Action.actions`.
2. Pad or crop to the downstream policy horizon.
3. Pad or crop to the downstream policy action dimension.
4. Normalize using the downstream policy's stats/normalizer.
5. Pass the normalized tensor into the downstream model.
6. Denormalize the downstream model output before returning to the server client.

Do not feed raw phase-1 actions directly into a normalized downstream model.

Do not return normalized model outputs directly to the robot/client.

## Stats and Normalization

Stats are policy-specific. Do not assume every model uses the same file format or even the same normalization type.

Before adding a phase-2 policy, answer:

- What stats file does training produce?
- What stats file does inference require?
- Are actions normalized by min/max, quantiles, or z-score?
- Are state/proprio and action normalized together or independently?
- Are action stats stepwise or global?
- Are gripper dimensions treated differently from delta pose dimensions?

The adapter should load the downstream policy's own stats, not the phase-1 backend's stats.

## Example: FastWAM

FastWAM is a phase-2 policy, not a backend. Its integration lives in:

```text
steering/policies/fastwam.py
```

The phase-1 backend can still be Evo-1:

```text
Evo-1 backend -> Phase1Action -> FastWAM phase-2 policy
```

### FastWAM Data Keys

The Flexiv LeRobot dataset produced by `scripts/realworld/collect_demos/convert_hdf5_to_lerobot.py` stores:

```text
observation.images.robot0_agentview_left
observation.images.robot0_eye_in_hand
observation.images.robot0_agentview_right  # optional
observation.state
action
```

FastWAM's LeRobot wrapper maps `shape_meta` entries as:

```text
images[].key = foo  -> observation.images.foo
state.key = default -> observation.state
action.key = default -> action
```

So Flexiv should use:

```yaml
shape_meta:
  images:
    - key: robot0_agentview_left
    - key: robot0_eye_in_hand
  action:
    - key: default
  state:
    - key: default
```

Do not copy LIBERO camera keys such as `image` and `wrist_image`; those point at different LeRobot feature names.

### FastWAM Stats

FastWAM does not use the same normalization file schema as Evo-1/OpenPI.

Common existing stats:

```text
norm_stats.json
  robot_name:
    observation.state:
      min/max
    action:
      min/max
```

FastWAM stats:

```text
dataset_stats.json
  state:
    default:
      global_min/global_max/global_mean/global_std/...
  action:
    default:
      global_min/global_max/global_mean/global_std/...
```

For first training, leave `pretrained_norm_stats` unset in the train dataset config. FastWAM computes stats during dataset initialization and saves `dataset_stats.json` into the run output directory.

For validation, evaluation, and serving, pass the training run's `dataset_stats.json`.

Do not pass Evo-1/OpenPI `norm_stats.json` to FastWAM.

Recommended Flexiv setting:

```yaml
norm_default_mode: min/max
use_stepwise_action_norm: false
```

### FastWAM Text Embeddings

FastWAM training expects cached text context tensors:

```text
sample["context"]
sample["context_mask"]
```

The dataset reads them from `text_embedding_cache_dir`. Run FastWAM's text embedding precompute step before training, using the same data config and instruction strings. Missing text embeddings fail dataset sampling before model training starts.

### FastWAM Horizon

Current Flexiv config:

```yaml
num_frames: 33
action_video_freq_ratio: 4
```

This gives:

```text
action_horizon = 32
num_video_frames = 9
```

FastWAM video tokenization requires:

```text
num_video_frames % 4 == 1
```

If `num_frames` or `action_video_freq_ratio` changes, preserve that constraint and make sure phase-1 action chunks can be padded/cropped to the same horizon.

### FastWAM Training

Example:

```bash
export RP_DATA_ROOT=/path/to/data_root
cd FastWAM
python scripts/precompute_text_embeds.py task=bread_joint_2cam224_1e-4
bash scripts/train_zero1.sh 8 task=bread_joint_2cam224_1e-4
```

### FastWAM Serving

Example:

```bash
python steering/scripts/serve_policy.py \
  --policy.type=fastwam \
  --policy.config=bread_joint_2cam224_1e-4 \
  --policy.dir=/path/to/fastwam_checkpoint.pt \
  --policy.stats=/path/to/dataset_stats.json \
  --steerer.mode=evo \
  --steerer.ckpt=/path/to/evo_or_hit_phase1_ckpt \
  --steerer.steps=20
```

`--steerer.*` configures the upstream phase-1 action producer. `--policy.*` configures the downstream phase-2 policy.

## Observation Mapping

Support the canonical obs shape already used by the server:

```text
observation/image
observation/wrist_image
observation/right_wrist_image
observation/state
prompt
```

For models with named camera maps, also support:

```python
obs["image"] = {
    "camera_name": ...,
}
```

When a phase-2 policy uses different camera names from the phase-1 backend, add aliases in the policy adapter, not in the backend.

FastWAM aliases:

```text
base_0_rgb        <-> robot0_agentview_left
left_wrist_0_rgb  <-> robot0_eye_in_hand
right_wrist_0_rgb <-> robot0_agentview_right
```

## Server Integration

Add a `--policy.type=<name>` branch in `steering/scripts/serve_policy.py`.

The server should construct:

1. A `SteeringConfig` for the phase-1 backend.
2. A policy-specific runtime config for the downstream phase-2 policy.
3. A `SteeredPhase2Runtime` composed from both.

For OpenPI this is currently:

```text
create_steered_openpi_policy(...)
```

For FastWAM this is:

```text
create_steered_fastwam_policy(...)
```

If the policy can also run without phase-1 conditioning, implement that path in `infer_direct()`.

## Checklist

When adding a phase-2 policy:

1. Add `steering/policies/<name>.py`.
2. Keep phase-1 model loading in `steering/backends` only if it returns `Phase1Action`.
3. Implement `infer_direct()`.
4. Implement `refine()`.
5. Normalize phase-1 actions into the downstream policy's expected action space.
6. Denormalize final policy outputs into robot/client action units.
7. Map obs keys and image camera names inside the policy adapter.
8. Load downstream policy stats from its own format.
9. Add `--policy.type=<name>` in `steering/scripts/serve_policy.py`.
10. Export public factory/config objects in `steering/policies/__init__.py`.
11. Run `python3 -m py_compile` on changed steering files.
