from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
try:
    import robomimic.models.obs_core as rmoc
except ImportError:
    import robomimic.models.base_nets as rmoc
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmoc.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            init_sample=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps

        if init_sample is None:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator)
        else:
            trajectory = init_sample.to(device=condition_data.device, dtype=condition_data.dtype).clone()
            refinement_steps = getattr(self, "refinement_steps", None)
            if refinement_steps is not None:
                steps_to_run = max(0, min(int(round(float(refinement_steps))), int(timesteps.shape[0])))
                frs = bool(getattr(self, "frs", False))
                random_noise_ratio = float(max(0.0, min(1.0, getattr(self, "random_noise_ratio", 0.0))))
                if steps_to_run <= 0 and not (frs and random_noise_ratio > 0.0):
                    trajectory[condition_mask] = condition_data[condition_mask]
                    return trajectory
                full_timesteps = timesteps
                if steps_to_run > 0:
                    timesteps = full_timesteps[-steps_to_run:]
                else:
                    timesteps = full_timesteps[:0]
                if frs:
                    trajectory = self._reverse_diffusion_to_timestep(
                        trajectory=trajectory,
                        timesteps=timesteps,
                        condition_data=condition_data,
                        condition_mask=condition_mask,
                        cond=cond,
                    )
                    if random_noise_ratio > 0.0 and timesteps.numel() > 0:
                        noise = torch.randn(
                            size=condition_data.shape,
                            dtype=condition_data.dtype,
                            device=condition_data.device,
                            generator=generator,
                        )
                        signal_scale = math.sqrt(max(0.0, 1.0 - random_noise_ratio**2))
                        alpha_prod_t = scheduler.alphas_cumprod[timesteps[0]].to(
                            device=trajectory.device,
                            dtype=trajectory.dtype,
                        )
                        while alpha_prod_t.ndim < trajectory.ndim:
                            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
                        beta_prod_t = 1 - alpha_prod_t
                        reverse_noise = (
                            trajectory - alpha_prod_t.sqrt() * init_sample.to(device=trajectory.device, dtype=trajectory.dtype)
                        ) / beta_prod_t.sqrt().clamp_min(1e-12)
                        mixed_noise = signal_scale * reverse_noise + random_noise_ratio * noise
                        trajectory = alpha_prod_t.sqrt() * init_sample.to(
                            device=trajectory.device,
                            dtype=trajectory.dtype,
                        ) + beta_prod_t.sqrt() * mixed_noise
                else:
                    noise = torch.randn(
                        size=condition_data.shape,
                        dtype=condition_data.dtype,
                        device=condition_data.device,
                        generator=generator,
                    )
                    trajectory = scheduler.add_noise(trajectory, noise, timesteps[:1])

        for t in timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def _reverse_diffusion_to_timestep(
            self,
            trajectory,
            timesteps,
            condition_data,
            condition_mask,
            cond=None):
        if timesteps.numel() <= 1:
            return trajectory
        model = self.model
        reverse_timesteps = torch.flip(timesteps, dims=[0])
        for lower_t, upper_t in zip(reverse_timesteps[:-1], reverse_timesteps[1:]):
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, lower_t, cond)
            trajectory = self._ddim_reverse_step(trajectory, model_output, lower_t, upper_t)
        return trajectory

    def _ddim_reverse_step(self, sample, model_output, lower_t, upper_t):
        scheduler = self.noise_scheduler
        pred_original_sample, pred_epsilon = self._predict_x0_and_epsilon(sample, model_output, lower_t)
        alpha_prod_upper = scheduler.alphas_cumprod[upper_t].to(device=sample.device, dtype=sample.dtype)
        while alpha_prod_upper.ndim < sample.ndim:
            alpha_prod_upper = alpha_prod_upper.unsqueeze(-1)
        beta_prod_upper = 1 - alpha_prod_upper
        return alpha_prod_upper.sqrt() * pred_original_sample + beta_prod_upper.sqrt() * pred_epsilon

    def _predict_x0_and_epsilon(self, sample, model_output, timestep):
        scheduler = self.noise_scheduler
        alpha_prod_t = scheduler.alphas_cumprod[timestep].to(device=sample.device, dtype=sample.dtype)
        while alpha_prod_t.ndim < sample.ndim:
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            pred_epsilon = model_output
            pred_original_sample = (sample - beta_prod_t.sqrt() * pred_epsilon) / alpha_prod_t.sqrt()
        elif prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt().clamp_min(1e-12)
        elif prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
        else:
            raise ValueError(f"Unsupported prediction_type {prediction_type!r}")
        return pred_original_sample, pred_epsilon

    @staticmethod
    def _effective_refinement_steps(base_steps, num_steps, sigma):
        if num_steps <= 0:
            return 0
        base_time = max(0.0, min(1.0, float(base_steps) / float(num_steps)))
        effective_time = 1.0 - math.sqrt(max(0.0, 1.0 - float(sigma) ** 2)) * (1.0 - base_time)
        return max(0, min(num_steps, int(round(effective_time * float(num_steps)))))


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        past_action = obs_dict.get('past_action')
        obs_only_dict = {key: value for key, value in obs_dict.items() if key != 'past_action'}
        # normalize input
        nobs = self.normalizer.normalize(obs_only_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        init_sample = None
        if past_action is not None:
            init_sample = self._fit_past_action_sample(past_action, cond_data.shape, Da)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            init_sample=init_sample,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    @staticmethod
    def _fit_past_action_sample(past_action: torch.Tensor, sample_shape: torch.Size, action_dim: int) -> torch.Tensor:
        if past_action.ndim == 2:
            past_action = past_action.unsqueeze(0)
        if past_action.ndim != 3:
            raise ValueError(f"Expected past_action [B,T,D] or [T,D], got {tuple(past_action.shape)}")
        batch_size, horizon = int(sample_shape[0]), int(sample_shape[1])
        if past_action.shape[0] == 1 and batch_size > 1:
            past_action = past_action.expand(batch_size, -1, -1)
        if past_action.shape[0] != batch_size:
            raise ValueError(f"past_action batch {past_action.shape[0]} does not match obs batch {batch_size}")
        out = torch.zeros(sample_shape, device=past_action.device, dtype=past_action.dtype)
        horizon_n = min(horizon, past_action.shape[1])
        dim_n = min(action_dim, past_action.shape[2])
        out[:, :horizon_n, :dim_n] = past_action[:, :horizon_n, :dim_n]
        return out

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
