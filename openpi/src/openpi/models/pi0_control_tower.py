"""RoboPrompt Pi0 model with a prompt-conditioned control tower."""

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at


def _dropout_main_tower_input(obs: _model.Observation) -> _model.Observation:
    """Drop only the main-tower prefix inputs, while preserving state and prompt-side inputs."""
    import dataclasses

    zero_images = {k: jnp.zeros_like(v) for k, v in obs.images.items()}
    zero_image_masks = {k: jnp.zeros_like(v, dtype=jnp.bool_) for k, v in obs.image_masks.items()}
    zero_tokenized_prompt = jnp.zeros_like(obs.tokenized_prompt) if obs.tokenized_prompt is not None else None
    zero_tokenized_prompt_mask = (
        jnp.zeros_like(obs.tokenized_prompt_mask, dtype=jnp.bool_)
        if obs.tokenized_prompt_mask is not None
        else None
    )
    return dataclasses.replace(
        obs,
        images=zero_images,
        image_masks=zero_image_masks,
        tokenized_prompt=zero_tokenized_prompt,
        tokenized_prompt_mask=zero_tokenized_prompt_mask,
    )


class Pi0ControlTower(Pi0):
    """Pi0 variant that injects prompt-tower features into the base PaliGemma tower."""

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        control_llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config],
                embed_dtype=config.dtype,
            )
        )
        control_llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        control_img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        control_img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.ControlTower = nnx.Dict(llm=control_llm, img=control_img)

        self.control_zero_projs = nnx.Dict(
            **{
                f"layer_{layer}": nnx.Linear(
                    paligemma_config.width,
                    paligemma_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
                for layer in range(paligemma_config.depth)
            }
        )
        self.control_text_seq_mlp_in = nnx.Linear(config.max_token_len, config.max_token_len, rngs=rngs)
        self.control_text_seq_mlp_out = nnx.Linear(
            config.max_token_len,
            config.max_token_len,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self._control_depth = paligemma_config.depth
        self._control_width = paligemma_config.width
        self._control_injection_image_key = getattr(config, "control_injection_image_key", "base_0_rgb")
        self._main_tower_dropout_prob = getattr(config, "main_tower_dropout_prob", 0.0)
        self._control_injection_layer_indices = tuple(
            layer - 1
            for layer in getattr(config, "control_injection_layers", (1, 6, 11, 16))
            if 1 <= layer <= self._control_depth
        )

    @override
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        tokens, input_mask, ar_mask, _, _ = self._embed_prefix_with_spans(obs)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def _embed_prefix_with_spans(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        dict[str, tuple[int, int]],
        tuple[int, int] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        image_spans = {}
        text_span = None
        token_offset = 0

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]
            image_spans[name] = (token_offset, token_offset + image_tokens.shape[1])
            token_offset += image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
            text_span = (token_offset, token_offset + tokenized_inputs.shape[1])

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, image_spans, text_span

    def embed_control_prompt(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "l b s d"] | None, at.Float[at.Array, "l b s d"] | None]:
        tokens = []
        input_mask = []
        ar_mask = []
        image_span = None
        text_span = None
        token_offset = 0

        if obs.prompt_images is not None:
            prompt_image_masks = obs.prompt_image_masks or {}
            image_start = token_offset
            for name in obs.prompt_images:
                image_tokens, _ = self.ControlTower.img(obs.prompt_images[name], train=False)
                tokens.append(image_tokens)
                image_mask = prompt_image_masks.get(name)
                if image_mask is None:
                    image_mask = jnp.ones((image_tokens.shape[0],), dtype=jnp.bool_)
                input_mask.append(einops.repeat(image_mask, "b -> b s", s=image_tokens.shape[1]))
                ar_mask += [False] * image_tokens.shape[1]
                token_offset += image_tokens.shape[1]
            image_span = (image_start, token_offset)

        if not tokens:
            return None, None

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (control_out,), _, (control_layer_outputs,) = self.ControlTower.llm(
            [tokens], mask=attn_mask, positions=positions, return_layer_outputs=True
        )
        control_layer_outputs = control_layer_outputs * input_mask[None, ..., None].astype(control_layer_outputs.dtype)
        image_out = (
            control_layer_outputs[:, :, image_span[0] : image_span[1]]
            if image_span is not None
            else None
        )
        text_out = (
            control_layer_outputs[:, :, text_span[0] : text_span[1]]
            if text_span is not None
            else None
        )
        return image_out, text_out

    def control_layer_injections(
        self,
        obs: _model.Observation,
        target_tokens: at.Float[at.Array, "b s d"],
        *,
        image_target_span: tuple[int, int] | None,
        text_target_span: tuple[int, int] | None,
    ) -> list[at.Float[at.Array, "l b s d"] | None] | None:
        control_image, control_text = self.embed_control_prompt(obs)
        if (control_image is None or image_target_span is None) and (control_text is None or text_target_span is None):
            return None

        paligemma_injection = jnp.zeros(
            (self._control_depth, target_tokens.shape[0], target_tokens.shape[1], target_tokens.shape[2]),
            dtype=target_tokens.dtype,
        )
        if control_image is not None and image_target_span is not None:
            paligemma_injection = self._set_control_injection_span(
                paligemma_injection, control_image, image_target_span, target_tokens.dtype
            )
        if control_text is not None and text_target_span is not None:
            control_text = self._project_control_text_to_target_span(
                control_text, target_span_len=text_target_span[1] - text_target_span[0]
            )
            paligemma_injection = self._set_projected_control_injection_span(
                paligemma_injection, control_text, text_target_span, target_tokens.dtype
            )
        return [paligemma_injection, None]

    def _set_control_injection_span(
        self,
        paligemma_injection: at.Float[at.Array, "l b s d"],
        control_tokens: at.Float[at.Array, "l b span d"],
        target_span: tuple[int, int],
        target_dtype: jnp.dtype,
    ) -> at.Float[at.Array, "l b s d"]:
        span_start, span_end = target_span
        target_span_len = span_end - span_start
        if control_tokens.shape[2] != target_span_len:
            raise ValueError(
                "Control token count must match target token count for token-wise injection: "
                f"{control_tokens.shape[2]} != {target_span_len}."
            )

        span_injection = jnp.zeros_like(control_tokens, dtype=target_dtype)
        for layer in self._control_injection_layer_indices:
            layer_injection = self.control_zero_projs[f"layer_{layer}"](control_tokens[layer]).astype(target_dtype)
            span_injection = span_injection.at[layer].set(layer_injection)
        return paligemma_injection.at[:, :, span_start:span_end, :].set(span_injection)

    def _set_projected_control_injection_span(
        self,
        paligemma_injection: at.Float[at.Array, "l b s d"],
        projected_control_tokens: at.Float[at.Array, "l b span d"],
        target_span: tuple[int, int],
        target_dtype: jnp.dtype,
    ) -> at.Float[at.Array, "l b s d"]:
        span_start, span_end = target_span
        target_span_len = span_end - span_start
        if projected_control_tokens.shape[2] != target_span_len:
            raise ValueError(
                "Projected control token count must match target token count for injection: "
                f"{projected_control_tokens.shape[2]} != {target_span_len}."
            )
        span_injection = jnp.zeros_like(projected_control_tokens, dtype=target_dtype)
        for layer in self._control_injection_layer_indices:
            span_injection = span_injection.at[layer].set(projected_control_tokens[layer].astype(target_dtype))
        return paligemma_injection.at[:, :, span_start:span_end, :].set(span_injection)

    def _project_control_text_to_target_span(
        self,
        control_text: at.Float[at.Array, "l b s d"],
        *,
        target_span_len: int,
    ) -> at.Float[at.Array, "l b target_s d"]:
        text_by_feature = einops.rearrange(control_text, "l b s d -> l b d s")
        text_by_feature = self.control_text_seq_mlp_in(text_by_feature)
        text_by_feature = nnx.swish(text_by_feature)
        text_by_feature = self.control_text_seq_mlp_out(text_by_feature)
        projected_text = einops.rearrange(text_by_feature, "l b d s -> l b s d")
        return projected_text[:, :, :target_span_len]

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

        prefix_tokens, prefix_mask, prefix_ar_mask, image_spans, text_span = self._embed_prefix_with_spans(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        layer_injections = self.control_layer_injections(
            observation,
            prefix_tokens,
            image_target_span=image_spans.get(self._control_injection_image_key),
            text_target_span=text_span,
        )
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            layer_injections=layer_injections,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        pred_actions = x_t - time_expanded * v_t

        # Debug norms: control tower token norms vs main tower activation norms
        control_image, control_text = self.embed_control_prompt(observation)

        control_token_norm = jnp.array(0.0)
        if control_image is not None:
            control_token_norm = jnp.maximum(control_token_norm, jnp.linalg.norm(control_image, axis=-1).mean())
        if control_text is not None:
            control_token_norm = jnp.maximum(control_token_norm, jnp.linalg.norm(control_text, axis=-1).mean())

        suffix_out_norm = jnp.linalg.norm(suffix_out, axis=-1).mean()

        injection_norm = jnp.array(0.0)
        if layer_injections is not None and layer_injections[0] is not None:
            injection_norm = jnp.linalg.norm(layer_injections[0], axis=-1).mean()

        info = {
            "control_token_norm": control_token_norm,
            "suffix_out_norm": suffix_out_norm,
            "injection_norm": injection_norm,
            "dropout": jnp.where(jnp.asarray(should_dropout), 1.0, 0.0),
        }
        return loss, info

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask, image_spans, text_span = self._embed_prefix_with_spans(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        layer_injections = self.control_layer_injections(
            observation,
            prefix_tokens,
            image_target_span=image_spans.get(self._control_injection_image_key),
            text_target_span=text_span,
        )
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            layer_injections=layer_injections,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
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
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
