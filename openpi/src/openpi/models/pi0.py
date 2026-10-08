import logging
from collections.abc import Callable
from typing import Literal

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

PrefixAttentionSchedule = Literal["ones", "zeros", "linear", "exp"]


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
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
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        phase2_steps: float | at.Float[at.Array, ""] = 0.0,
        phase1_actions: at.Float[at.Array, "b ah ad"] | at.Float[at.Array, "ah ad"] | None = None,
        frs: bool = False,
        random_noise_ratio: float | at.Float[at.Array, ""] = 0.0,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        phase2_steps = jnp.clip(jnp.asarray(phase2_steps), 0, num_steps)
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        if phase1_actions is not None:
            phase1_actions = jnp.asarray(phase1_actions, dtype=noise.dtype)
            if phase1_actions.ndim == 2:
                phase1_actions = jnp.broadcast_to(phase1_actions[None, ...], noise.shape)
            elif phase1_actions.shape != noise.shape:
                raise ValueError(
                    f"phase1_actions must have shape {(self.action_horizon, self.action_dim)} "
                    f"or {noise.shape}, got {phase1_actions.shape}."
                )

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def predict_velocity(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step(carry):
            x_t, time = carry
            v_t = predict_velocity(x_t, time)

            current_dt = -jnp.minimum(-dt, time) # denoise 1 step when time > 0.1, denoise remaining time at last.
            return x_t + current_dt * v_t, time + current_dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time > 1e-6

        def denoise(initial_x, start_time):
            x_0, _ = jax.lax.while_loop(cond, step, (initial_x, start_time))
            return x_0

        def reverse_ode(initial_x, target_time):
            def reverse_step(carry):
                x_t, time = carry
                v_t = predict_velocity(x_t, time)
                current_dt = jnp.minimum(-dt, target_time - time) # positive
                return x_t + current_dt * v_t, time + current_dt

            def reverse_cond(carry):
                _, time = carry
                return time < target_time - 1e-6

            x_t, _ = jax.lax.while_loop(
                reverse_cond,
                reverse_step,
                (initial_x, jnp.asarray(0.0, dtype=target_time.dtype)),
            )
            return x_t

        if phase1_actions is None:
            return denoise(noise, 1.0)

        phase2_time = phase2_steps.astype(noise.dtype) / jnp.asarray(num_steps, dtype=noise.dtype)
        if frs:
            phase2_x_t = reverse_ode(phase1_actions, phase2_time)
            sigma = jnp.clip(jnp.asarray(random_noise_ratio, dtype=noise.dtype), 0.0, 1.0)
            eps = jax.random.normal(jax.random.fold_in(rng, 2), phase2_x_t.shape)
            safe_phase2_time = jnp.maximum(phase2_time, jnp.asarray(1e-6, dtype=noise.dtype))
            reverse_noise = (phase2_x_t - (1.0 - phase2_time) * phase1_actions) / safe_phase2_time
            mixed_noise = jnp.sqrt(1.0 - sigma**2) * reverse_noise + sigma * eps
            mixed_x_t = (1.0 - phase2_time) * phase1_actions + phase2_time * mixed_noise
            phase2_x_t = jnp.where(phase2_time > 0.0, mixed_x_t, phase2_x_t)
        else:
            phase2_rng = jax.random.fold_in(rng, 1)
            phase2_noise = jax.random.normal(phase2_rng, phase1_actions.shape)
            phase2_x_t = phase2_time * phase2_noise + (1.0 - phase2_time) * phase1_actions
        return denoise(phase2_x_t, phase2_time)

    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_actions: at.Float[at.Array, "b ah ad"] | at.Float[at.Array, "ah ad"],
        inference_delay: int,
        prefix_attention_horizon: int,
        max_guidance_weight: float,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        prefix_attention_schedule: PrefixAttentionSchedule = "exp",
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        elif noise.ndim == 2:
            noise = noise[None, ...]

        prefix_actions = jnp.asarray(prefix_actions, dtype=noise.dtype)
        if prefix_actions.ndim == 2:
            prefix_actions = prefix_actions[None, ...]
        if prefix_actions.shape[0] == 1 and batch_size != 1:
            prefix_actions = jnp.broadcast_to(prefix_actions, (batch_size, *prefix_actions.shape[1:]))
        if prefix_actions.shape[0] != batch_size:
            raise ValueError(f"prefix_actions batch size {prefix_actions.shape[0]} does not match {batch_size}.")
        if prefix_actions.shape[1] != self.action_horizon:
            raise ValueError(
                f"prefix_actions horizon {prefix_actions.shape[1]} does not match {self.action_horizon}."
            )
        if prefix_actions.shape[-1] < self.action_dim:
            pad_width = self.action_dim - prefix_actions.shape[-1]
            prefix_actions = jnp.concatenate(
                [prefix_actions, jnp.zeros((*prefix_actions.shape[:-1], pad_width), dtype=prefix_actions.dtype)],
                axis=-1,
            )
        elif prefix_actions.shape[-1] > self.action_dim:
            prefix_actions = prefix_actions[..., : self.action_dim]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
            start = jnp.minimum(start, end)
            if schedule == "ones":
                weights = jnp.ones(total)
            elif schedule == "zeros":
                weights = (jnp.arange(total) < start).astype(jnp.float32)
            elif schedule == "linear" or schedule == "exp":
                weights = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
                if schedule == "exp":
                    weights = weights * jnp.expm1(weights) / (jnp.e - 1)
            else:
                raise ValueError(f"Invalid schedule: {schedule}")
            return jnp.where(jnp.arange(total) >= end, 0, weights)

        def pinv_corrected_velocity(
            v_t_fn: Callable[[jax.Array, float], jax.Array],
            x_t: jax.Array,
            t: float,
            prefix_actions: jax.Array,
            inference_delay: int,
            prefix_attention_horizon: int,
            max_guidance_weight: float,
            prefix_attention_schedule: PrefixAttentionSchedule,
        ) -> jax.Array:
            @jax.vmap
            def _pinv_corrected_velocity(x_t: jax.Array, prefix: jax.Array) -> jax.Array:
                def denoiser(x_t: jax.Array) -> tuple[jax.Array, jax.Array]:
                    v_t = v_t_fn(x_t, t)
                    return x_t - v_t * t, v_t

                x_0, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                weights = get_prefix_weights(
                    inference_delay,
                    prefix_attention_horizon + inference_delay,
                    prefix.shape[0],
                    prefix_attention_schedule,
                )
                error = (prefix - x_0) * weights[:, None]
                pinv_correction = vjp_fun(error)[0]
                inv_r2 = (t**2 + (1 - t) ** 2) / (t**2)
                c = jnp.nan_to_num(t / (1 - t), posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)

                del pinv_correction
                return v_t - guidance_weight * error

            return _pinv_corrected_velocity(x_t, prefix_actions)

        def v_t_step(x_t: jax.Array, time: jax.Array) -> jax.Array:
            x_t = x_t[None, ...]
            time = time[None, ...]
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
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
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])[0, ...]

        def rtc_step(carry):
            x_t, time = carry
            guided_vt = pinv_corrected_velocity(
                v_t_step,
                x_t,
                time,
                prefix_actions,
                inference_delay,
                prefix_attention_horizon,
                max_guidance_weight,
                prefix_attention_schedule,
            )
            return x_t + dt * guided_vt, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, rtc_step, (noise, 1.0))
        return x_0
