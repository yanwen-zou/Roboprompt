import dataclasses


@dataclasses.dataclass(frozen=True)
class SteeringConfig:
    """Runtime settings for phase-1 steering and phase-2 policy refinement."""

    checkpoint_dir: str
    mode: str = "evo"
    device: str = "cuda"
    num_inference_timesteps: int = 32
    inference_seed: int | None = None
    enable_by_default: bool = False
    prompt: str | None = None
    image_keys: tuple[str, ...] = ("base_0_rgb", "prompt_0_rgb")
    image_size: int = 448
    max_views: int = 2
    max_state_dim: int = 24
    max_action_dim: int = 24
    action_horizon: int | None = None
    action_dim: int | None = None


Evo1SteererConfig = SteeringConfig
SteeredPolicyConfig = SteeringConfig
