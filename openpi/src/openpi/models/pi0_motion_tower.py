"""Pi0 model with a lightweight motion-prompt tower."""

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask, posemb_sincos
from openpi.models.pi0_control_tower import _dropout_main_tower_input
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at


class Pi0MotionTower(Pi0):
    """Pi0 variant that FiLM-conditions suffix expert layers with motion prompts."""

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        action_expert_config = _gemma.get_config(config.action_expert_variant)
        motion_width = getattr(config, "motion_width", action_expert_config.width)
        if motion_width <= 0:
            raise ValueError(f"motion_width must be positive, got {motion_width}.")

        self.MotionTower = nnx.Dict(
            wrist_proj=nnx.Linear(3, motion_width, rngs=rngs),
            primitive_proj=nnx.Linear(3, motion_width, rngs=rngs),
            drag_2d_proj=nnx.Linear(2, motion_width, rngs=rngs),
            fuse_in=nnx.Linear(motion_width, motion_width, rngs=rngs),
            fuse_out=nnx.Linear(motion_width, motion_width, rngs=rngs),
        )
        self._motion_injection_layer_indices = tuple(
            layer - 1
            for layer in getattr(config, "motion_injection_layers", (1, 6, 11, 16))
            if 1 <= layer <= action_expert_config.depth
        )
        if not self._motion_injection_layer_indices:
            raise ValueError("motion_injection_layers must contain at least one valid layer.")
        self.motion_layer_film_projs = nnx.Dict(
            **{
                f"layer_{layer}": nnx.Linear(
                    motion_width,
                    2 * action_expert_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
                for layer in self._motion_injection_layer_indices
            }
        )
        self.motion_layer_mlp_in = nnx.Dict(
            **{
                f"layer_{layer}": nnx.Linear(motion_width, motion_width, rngs=rngs)
                for layer in self._motion_injection_layer_indices[:-1]
            }
        )
        self.motion_layer_mlp_out = nnx.Dict(
            **{
                f"layer_{layer}": nnx.Linear(motion_width, motion_width, rngs=rngs)
                for layer in self._motion_injection_layer_indices[:-1]
            }
        )
        self._motion_width = motion_width
        self._motion_depth = action_expert_config.depth
        self._main_tower_dropout_prob = getattr(config, "main_tower_dropout_prob", 0.0)
        self._local_motion_aux_loss_weight = getattr(
            config, "local_motion_aux_loss_weight", getattr(config, "wrist_rel_action_aux_loss_weight", 0.0)
        )
        self._global_motion_aux_loss_weight = getattr(
            config, "global_motion_aux_loss_weight", getattr(config, "primitive_cmd_aux_loss_weight", 0.0)
        )
        self._global_motion_horizon = getattr(
            config, "global_motion_horizon", getattr(config, "primitive_cmd_horizon", 25)
        )
        self._global_motion_threshold = getattr(
            config, "global_motion_threshold", getattr(config, "primitive_cmd_threshold", 0.2)
        )
        self._drag_2d_aux_loss_weight = getattr(config, "drag_2d_aux_loss_weight", 0.0)
        self._drag_2d_horizon = getattr(config, "drag_2d_horizon", 25)
        self._drag_2d_action_scale = getattr(config, "drag_2d_action_scale", 0.05)

    def _embed_motion_condition(
        self,
        obs: _model.Observation,
        *,
        apply: bool = True,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "b d"], at.Float[at.Array, " b"]]:
        batch_size = obs.state.shape[0]
        local_cond = None
        local_active = jnp.zeros((batch_size,), dtype=obs.state.dtype)
        if obs.prompt_local_motion is not None:
            local_motion = obs.prompt_local_motion
            if local_motion.shape[-1] != 3:
                raise ValueError(f"Expected prompt_local_motion dim 3, got {local_motion.shape[-1]}.")
            if obs.prompt_local_motion_mask is None:
                raise ValueError("prompt_local_motion_mask must be provided with prompt_local_motion.")
            local_mask = obs.prompt_local_motion_mask.astype(local_motion.dtype)
            local_cond = self.MotionTower.wrist_proj(local_motion) * local_mask[:, None]
            local_active = (local_mask > 0).astype(obs.state.dtype)

        global_cond = None
        global_active = jnp.zeros((batch_size,), dtype=obs.state.dtype)
        if obs.prompt_global_motion is not None:
            global_motion = obs.prompt_global_motion
            if global_motion.shape[-1] != 3:
                raise ValueError(f"Expected prompt_global_motion dim 3, got {global_motion.shape[-1]}.")
            if obs.prompt_global_motion_mask is None:
                raise ValueError("prompt_global_motion_mask must be provided with prompt_global_motion.")
            global_mask = obs.prompt_global_motion_mask.astype(global_motion.dtype)
            global_cond = self.MotionTower.primitive_proj(global_motion) * global_mask[:, None]
            global_active = (global_mask > 0).astype(obs.state.dtype)

        drag_cond = None
        drag_active = jnp.zeros((batch_size,), dtype=obs.state.dtype)
        if obs.prompt_2d_drag is not None:
            drag = obs.prompt_2d_drag
            if drag.shape[-1] != 2:
                raise ValueError(f"Expected prompt_2d_drag dim 2, got {drag.shape[-1]}.")
            if obs.prompt_2d_drag_mask is None:
                raise ValueError("prompt_2d_drag_mask must be provided with prompt_2d_drag.")
            drag_mask = obs.prompt_2d_drag_mask.astype(drag.dtype)
            drag_cond = self.MotionTower.drag_2d_proj(drag) * drag_mask[:, None]
            drag_active = (drag_mask > 0).astype(obs.state.dtype)

        if local_cond is None and global_cond is None and drag_cond is None and train:
            raise ValueError("Pi0MotionTower requires prompt_local_motion, prompt_global_motion, or prompt_2d_drag.")

        if local_cond is None:
            local_cond = jnp.zeros((batch_size, self._motion_width), dtype=obs.state.dtype)
        if global_cond is None:
            global_cond = jnp.zeros((batch_size, self._motion_width), dtype=obs.state.dtype)
        if drag_cond is None:
            drag_cond = jnp.zeros((batch_size, self._motion_width), dtype=obs.state.dtype)

        cond = local_cond + global_cond + drag_cond
        cond = self.MotionTower.fuse_in(cond)
        cond = nnx.swish(cond)
        cond = self.MotionTower.fuse_out(cond)
        cond = nnx.swish(cond)
        motion_active = jnp.maximum(jnp.maximum(local_active, global_active), drag_active)
        if not apply:
            motion_active = jnp.zeros_like(motion_active)
        cond = cond * motion_active[:, None]
        return cond, motion_active

    def _suffix_layer_films(
        self,
        obs: _model.Observation,
        suffix_tokens: at.Float[at.Array, "b s d"],
        *,
        apply_motion: bool = True,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "l b s two_d"], dict[str, at.Array]]:
        motion_cond, motion_active = self._embed_motion_condition(obs, apply=apply_motion, train=train)
        layer_films = jnp.zeros(
            (
                self._motion_depth,
                suffix_tokens.shape[0],
                suffix_tokens.shape[1],
                2 * suffix_tokens.shape[-1],
            ),
            dtype=suffix_tokens.dtype,
        )
        coeff = motion_active.astype(suffix_tokens.dtype)
        scale_norm = jnp.array(0.0, dtype=jnp.float32)
        shift_norm = jnp.array(0.0, dtype=jnp.float32)
        cond = motion_cond
        for i, layer in enumerate(self._motion_injection_layer_indices):
            scale_shift = self.motion_layer_film_projs[f"layer_{layer}"](cond).astype(suffix_tokens.dtype)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)
            scale = scale * coeff[:, None]
            shift = shift * coeff[:, None]
            scale_norm = scale_norm + jnp.linalg.norm(scale, axis=-1).mean()
            shift_norm = shift_norm + jnp.linalg.norm(shift, axis=-1).mean()
            scale_shift = jnp.concatenate([scale, shift], axis=-1)
            scale_shift = einops.repeat(scale_shift, "b d -> b s d", s=suffix_tokens.shape[1])
            layer_films = layer_films.at[layer].set(scale_shift)
            if i < len(self._motion_injection_layer_indices) - 1:
                cond = self.motion_layer_mlp_in[f"layer_{layer}"](cond)
                cond = nnx.swish(cond)
                cond = self.motion_layer_mlp_out[f"layer_{layer}"](cond)
                cond = nnx.swish(cond)

        num_layers = jnp.asarray(len(self._motion_injection_layer_indices), dtype=jnp.float32)
        info = {
            "motion_cond_norm": jnp.linalg.norm(motion_cond, axis=-1).mean(),
            "motion_layer_film_scale_norm": scale_norm / num_layers,
            "motion_layer_film_shift_norm": shift_norm / num_layers,
        }
        return layer_films, info

    def _apply_motion_film(
        self,
        obs: _model.Observation,
        action_tokens: at.Float[at.Array, "b s d"],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "b s d"], dict[str, at.Array]]:
        """Backward-compatible helper used by tests and diagnostics.

        The main forward path injects FiLM inside the suffix tower. For standalone
        inspection, we aggregate the per-layer FiLM coefficients into a single
        affine transform on the provided action tokens.
        """
        layer_films, info = self._suffix_layer_films(obs, action_tokens, train=train)
        total_film = jnp.sum(layer_films, axis=0)
        scale, shift = jnp.split(total_film, 2, axis=-1)
        conditioned_tokens = action_tokens * (1.0 + scale) + shift
        compat_info = {
            **info,
            "motion_film_scale_norm": info["motion_layer_film_scale_norm"],
            "motion_film_shift_norm": info["motion_layer_film_shift_norm"],
        }
        return conditioned_tokens, compat_info

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        train: bool = False,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _compute_local_motion_aux_loss(
        self,
        obs: _model.Observation,
        pred_actions: _model.Actions,
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        if obs.prompt_local_motion is None:
            raise ValueError("prompt_local_motion is required for local motion aux loss.")
        if obs.prompt_local_motion_mask is None:
            raise ValueError("prompt_local_motion_mask is required for local motion aux loss.")
        if obs.prompt_tcp_rot is None:
            raise ValueError("prompt_tcp_rot is required for local motion aux loss.")
        if obs.prompt_action_q01 is None or obs.prompt_action_q99 is None:
            raise ValueError("prompt_action_q01 and prompt_action_q99 are required for local motion aux loss.")
        if obs.prompt_wrist_action_q01 is None or obs.prompt_wrist_action_q99 is None:
            raise ValueError(
                "prompt_wrist_action_q01 and prompt_wrist_action_q99 are required for local motion aux loss."
            )
        if pred_actions.shape[-1] < 3:
            raise ValueError(f"Expected pred action dim at least 3, got {pred_actions.shape[-1]}.")
        if obs.prompt_local_motion.shape[-1] != 3:
            raise ValueError(f"Expected prompt_local_motion dim 3, got {obs.prompt_local_motion.shape[-1]}.")

        action_q01 = obs.prompt_action_q01
        action_q99 = obs.prompt_action_q99
        wrist_q01 = obs.prompt_wrist_action_q01
        wrist_q99 = obs.prompt_wrist_action_q99
        if action_q01.shape[-1] < 3 or action_q99.shape[-1] < 3:
            raise ValueError("prompt_action_q01 and prompt_action_q99 must contain at least xyz action dims.")

        pred_base_xyz = (pred_actions[:, 0, :3] + 1.0) / 2.0 * (
            action_q99[:, :3] - action_q01[:, :3] + 1e-6
        ) + action_q01[:, :3]
        pred_wrist_xyz = jnp.einsum("bji,bj->bi", obs.prompt_tcp_rot, pred_base_xyz)
        pred_wrist_xyz_norm = (pred_wrist_xyz - wrist_q01) / (wrist_q99 - wrist_q01 + 1e-6) * 2.0 - 1.0

        local_mask = obs.prompt_local_motion_mask.astype(pred_wrist_xyz_norm.dtype)
        sq_error = jnp.mean(jnp.square(pred_wrist_xyz_norm - obs.prompt_local_motion), axis=-1)
        valid_count = jnp.maximum(jnp.sum(local_mask), 1.0)
        aux_loss = sq_error * local_mask
        aux_mse = jnp.sum(aux_loss) / valid_count

        info = {
            "local_motion_aux_loss_valid": self._local_motion_aux_loss_weight * aux_mse,
        }
        return aux_loss, info

    def _compute_global_motion_aux_loss(
        self,
        obs: _model.Observation,
        pred_actions: _model.Actions,
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        if obs.prompt_global_motion is None:
            raise ValueError("prompt_global_motion is required for global motion aux loss.")
        if obs.prompt_global_motion_mask is None:
            raise ValueError("prompt_global_motion_mask is required for global motion aux loss.")
        if pred_actions.shape[-1] < 3:
            raise ValueError(f"Expected pred action dim at least 3, got {pred_actions.shape[-1]}.")
        if obs.prompt_global_motion.shape[-1] != 3:
            raise ValueError(f"Expected prompt_global_motion dim 3, got {obs.prompt_global_motion.shape[-1]}.")

        horizon = min(self._global_motion_horizon, pred_actions.shape[-2])
        pred_cmd_continuous = jnp.clip(jnp.sum(pred_actions[:, :horizon, :3], axis=1), -1.0, 1.0)
        pred_cmd_discrete = jnp.where(
            jnp.abs(pred_cmd_continuous) >= self._global_motion_threshold,
            jnp.sign(pred_cmd_continuous),
            0.0,
        )
        pred_cmd = pred_cmd_continuous + jax.lax.stop_gradient(pred_cmd_discrete - pred_cmd_continuous)
        global_mask = obs.prompt_global_motion_mask.astype(pred_cmd.dtype)
        sq_error = jnp.mean(jnp.square(pred_cmd - obs.prompt_global_motion), axis=-1)
        aux_loss = sq_error * global_mask
        valid_count = jnp.maximum(jnp.sum(global_mask), 1.0)

        info = {
            "global_motion_aux_loss_valid": self._global_motion_aux_loss_weight * jnp.sum(aux_loss) / valid_count,
        }
        return aux_loss, info

    def _project_world_points_to_xy(
        self,
        points_world: at.Float[at.Array, "b n three"],
        obs: _model.Observation,
    ) -> at.Float[at.Array, "b n two"]:
        if obs.prompt_world_to_camera is None:
            raise ValueError("prompt_world_to_camera is required for 2d drag aux loss.")
        if obs.prompt_camera_resolution is None:
            raise ValueError("prompt_camera_resolution is required for 2d drag aux loss.")
        ones = jnp.ones((*points_world.shape[:-1], 1), dtype=points_world.dtype)
        points_h = jnp.concatenate([points_world, ones], axis=-1)
        camera_points = jnp.einsum("bij,bnj->bni", obs.prompt_world_to_camera, points_h)
        z = camera_points[..., 2:3]
        z = jnp.where(jnp.abs(z) > 1e-6, z, jnp.where(z >= 0.0, 1e-6, -1e-6))
        xy_pixels = camera_points[..., :2] / z
        hw = obs.prompt_camera_resolution.astype(points_world.dtype)
        denom_xy = jnp.maximum(jnp.stack([hw[:, 1] - 1.0, hw[:, 0] - 1.0], axis=-1), 1.0)
        xy_unit = xy_pixels / denom_xy[:, None, :]
        return jnp.clip(jnp.nan_to_num(xy_unit, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    def _compute_2d_drag_aux_loss(
        self,
        obs: _model.Observation,
        pred_actions: _model.Actions,
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        if obs.prompt_2d_drag is None:
            raise ValueError("prompt_2d_drag is required for 2d drag aux loss.")
        if obs.prompt_2d_drag_mask is None:
            raise ValueError("prompt_2d_drag_mask is required for 2d drag aux loss.")
        if obs.prompt_tcp_world_pos is None:
            raise ValueError("prompt_tcp_world_pos is required for 2d drag aux loss.")
        if obs.prompt_base_rot is None:
            raise ValueError("prompt_base_rot is required for 2d drag aux loss.")
        if obs.prompt_action_q01 is None or obs.prompt_action_q99 is None:
            raise ValueError("prompt_action_q01 and prompt_action_q99 are required for 2d drag aux loss.")
        if pred_actions.shape[-1] < 3:
            raise ValueError(f"Expected pred action dim at least 3, got {pred_actions.shape[-1]}.")
        if obs.prompt_2d_drag.shape[-1] != 2:
            raise ValueError(f"Expected prompt_2d_drag dim 2, got {obs.prompt_2d_drag.shape[-1]}.")

        horizon = min(self._drag_2d_horizon, pred_actions.shape[-2])
        action_q01 = obs.prompt_action_q01
        action_q99 = obs.prompt_action_q99
        pred_base_xyz = (pred_actions[:, :horizon, :3] + 1.0) / 2.0 * (
            action_q99[:, None, :3] - action_q01[:, None, :3] + 1e-6
        ) + action_q01[:, None, :3]
        pred_world_xyz = jnp.einsum("bti,bji->btj", pred_base_xyz, obs.prompt_base_rot) * self._drag_2d_action_scale
        pred_end_world = obs.prompt_tcp_world_pos + jnp.sum(pred_world_xyz, axis=1)
        points_world = jnp.stack([obs.prompt_tcp_world_pos, pred_end_world], axis=1)
        points_xy = self._project_world_points_to_xy(points_world, obs)
        pred_drag = points_xy[:, 1] - points_xy[:, 0]

        drag_mask = obs.prompt_2d_drag_mask.astype(pred_drag.dtype)
        sq_error = jnp.mean(jnp.square(pred_drag - obs.prompt_2d_drag), axis=-1)
        aux_loss = sq_error * drag_mask
        valid_count = jnp.maximum(jnp.sum(drag_mask), 1.0)
        info = {
            "drag_2d_aux_loss_valid": self._drag_2d_aux_loss_weight * jnp.sum(aux_loss) / valid_count,
        }
        return aux_loss, info

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng, dropout_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        should_dropout = False
        if train and self._main_tower_dropout_prob > 0.0:
            should_dropout = jax.random.bernoulli(dropout_rng, p=self._main_tower_dropout_prob)
            observation = jax.lax.cond(
                should_dropout,
                _dropout_main_tower_input,
                lambda x: x,
                observation,
            )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_layer_films, motion_info = self._suffix_layer_films(
            observation, suffix_tokens, train=train
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            layer_films=[None, suffix_layer_films],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        pred_actions = x_t - time_expanded * v_t
        info = {
            **motion_info,
            "suffix_out_norm": jnp.linalg.norm(suffix_out, axis=-1).mean(),
        }
        if self._local_motion_aux_loss_weight > 0.0:
            local_aux_loss, local_aux_info = self._compute_local_motion_aux_loss(observation, pred_actions)
            weighted_local_aux_loss = self._local_motion_aux_loss_weight * local_aux_loss
            loss = loss + weighted_local_aux_loss[:, None]
            info.update(local_aux_info)
            info["_local_motion_aux_loss"] = jnp.mean(weighted_local_aux_loss)
        if self._global_motion_aux_loss_weight > 0.0:
            global_aux_loss, global_aux_info = self._compute_global_motion_aux_loss(observation, pred_actions)
            weighted_global_aux_loss = self._global_motion_aux_loss_weight * global_aux_loss
            loss = loss + weighted_global_aux_loss[:, None]
            info.update(global_aux_info)
            info["_global_motion_aux_loss"] = jnp.mean(weighted_global_aux_loss)
        if self._drag_2d_aux_loss_weight > 0.0:
            drag_aux_loss, drag_aux_info = self._compute_2d_drag_aux_loss(observation, pred_actions)
            weighted_drag_aux_loss = self._drag_2d_aux_loss_weight * drag_aux_loss
            loss = loss + weighted_drag_aux_loss[:, None]
            info.update(drag_aux_info)
            info["_drag_2d_aux_loss"] = jnp.mean(weighted_drag_aux_loss)
        return loss, info

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        phase2_steps: int = 5,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        phase2_steps = jnp.clip(jnp.asarray(phase2_steps), 0, num_steps)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def denoise(initial_x, start_time, *, apply_motion: bool):
            def step(carry):
                x_t, time = carry
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation,
                    x_t,
                    jnp.broadcast_to(time, batch_size),
                )
                suffix_layer_films, _ = self._suffix_layer_films(
                    observation, suffix_tokens, apply_motion=apply_motion
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                assert full_attn_mask.shape == (
                    batch_size,
                    suffix_tokens.shape[1],
                    prefix_tokens.shape[1] + suffix_tokens.shape[1],
                )
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],  # prefix tokens are cached
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                    layer_films=[None, suffix_layer_films],
                )
                assert prefix_out is None
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return x_t + dt * v_t, time + dt

            def cond(carry):
                x_t, time = carry
                return time >= -dt / 2

            x_0, _ = jax.lax.while_loop(cond, step, (initial_x, start_time))
            return x_0

        phase1_actions = denoise(noise, 1.0, apply_motion=True)
        phase2_time = phase2_steps.astype(phase1_actions.dtype) / jnp.asarray(num_steps, dtype=phase1_actions.dtype)
        phase2_rng = jax.random.fold_in(rng, 1)
        phase2_noise = jax.random.normal(phase2_rng, phase1_actions.shape)
        phase2_x_t = phase2_time * phase2_noise + (1.0 - phase2_time) * phase1_actions
        return denoise(phase2_x_t, phase2_time, apply_motion=False)
