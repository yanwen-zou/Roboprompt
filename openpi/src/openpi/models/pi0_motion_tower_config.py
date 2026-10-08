"""Configuration for the RoboPrompt Pi0 motion-tower variant."""

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
    from openpi.models.pi0_motion_tower import Pi0MotionTower


@dataclasses.dataclass(frozen=True)
class Pi0MotionTowerConfig(pi0_config.Pi0Config):
    """`Pi0` config with lightweight motion-prompt conditioning."""

    motion_width: int = 1024
    motion_injection_layers: tuple[int, ...] = (1, 6, 11, 16)
    local_motion_aux_loss_weight: float = 0.0
    global_motion_aux_loss_weight: float = 0.0
    global_motion_horizon: int = 25
    global_motion_threshold: float = 0.2
    drag_2d_aux_loss_weight: float = 0.0
    drag_2d_horizon: int = 25
    drag_2d_action_scale: float = 0.05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0MotionTower":
        from openpi.models.pi0_motion_tower import Pi0MotionTower

        return Pi0MotionTower(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

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
                prompt_local_motion=jax.ShapeDtypeStruct([batch_size, 3], jnp.float32),
                prompt_local_motion_mask=jax.ShapeDtypeStruct([batch_size], bool),
                prompt_global_motion=jax.ShapeDtypeStruct([batch_size, 3], jnp.float32),
                prompt_global_motion_mask=jax.ShapeDtypeStruct([batch_size], bool),
                prompt_2d_drag=jax.ShapeDtypeStruct([batch_size, 2], jnp.float32),
                prompt_2d_drag_mask=jax.ShapeDtypeStruct([batch_size], bool),
                prompt_world_to_camera=jax.ShapeDtypeStruct([batch_size, 4, 4], jnp.float32),
                prompt_camera_resolution=jax.ShapeDtypeStruct([batch_size, 2], jnp.float32),
                prompt_tcp_world_pos=jax.ShapeDtypeStruct([batch_size, 3], jnp.float32),
                prompt_tcp_rot=jax.ShapeDtypeStruct([batch_size, 3, 3], jnp.float32),
                prompt_base_rot=jax.ShapeDtypeStruct([batch_size, 3, 3], jnp.float32),
                prompt_action_q01=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                prompt_action_q99=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                prompt_wrist_action_q01=jax.ShapeDtypeStruct([batch_size, 3], jnp.float32),
                prompt_wrist_action_q99=jax.ShapeDtypeStruct([batch_size, 3], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
