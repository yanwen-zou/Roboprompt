"""Configuration for the RoboPrompt Pi0 variant."""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_control_tower import Pi0ControlTower


@dataclasses.dataclass(frozen=True)
class Pi0ControlTowerConfig(pi0_config.Pi0Config):
    """`Pi0` config with prompt-side observation inputs for the RoboPrompt control tower."""

    # The RoboCasa prompt overlay is generated from `observation/image`, which
    # RobocasaInputs maps to `base_0_rgb`.
    control_injection_image_key: str = "base_0_rgb"
    # 1-based Gemma block indices where control residuals are injected.
    control_injection_layers: tuple[int, ...] = (1, 6, 11, 16)

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0ControlTower":
        from openpi.models.pi0_control_tower import Pi0ControlTower
        return Pi0ControlTower(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """Get input specification including control-tower prompt fields."""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        prompt_image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        prompt_image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                prompt_images={
                    "prompt_0": prompt_image_spec,
                },
                prompt_image_masks={
                    "prompt_0": prompt_image_mask_spec,
                },
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter_control_tower(self):
        """Return the current freeze filter for the clean pre-control baseline."""
        return self.get_freeze_filter()
