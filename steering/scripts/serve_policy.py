from __future__ import annotations

import argparse
import logging
import pathlib
import socket
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT,
    REPO_ROOT / "diffusion_policy",
    REPO_ROOT / "openpi" / "src",
    REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi import transforms  # noqa: E402
from openpi.policies import policy as _policy  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _openpi_config  # noqa: E402
from steering.config import SteeringConfig  # noqa: E402
from steering.policies.diffusion_policy import DiffusionPolicyRuntimeConfig  # noqa: E402
from steering.policies.diffusion_policy import create_steered_diffusion_policy  # noqa: E402
from steering.policies.fastwam import FastWAMRuntimeConfig  # noqa: E402
from steering.policies.fastwam import create_steered_fastwam_policy  # noqa: E402
from steering.policies.openpi import create_steered_openpi_policy  # noqa: E402


def _optional_int(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "off", "false"}:
        return None
    return int(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a steered phase-2 policy.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--default-prompt", default=None)

    parser.add_argument("--steerer.mode", dest="steerer_mode", default="evo", choices=("evo",))
    parser.add_argument("--steerer.ckpt", dest="steerer_ckpt", required=True)
    parser.add_argument("--steerer.device", dest="steerer_device", default="cuda")
    parser.add_argument("--steerer.steps", dest="steerer_steps", type=int, default=32)
    parser.add_argument("--steerer.seed", dest="steerer_seed", type=_optional_int, default=None)

    parser.add_argument("--policy.type", dest="policy_type", default="openpi", choices=("openpi", "fastwam", "diffusion_policy"))
    parser.add_argument("--policy.config", dest="policy_config", required=True)
    parser.add_argument("--policy.dir", dest="policy_dir", required=True)
    parser.add_argument("--policy.stats", dest="policy_stats", default=None)
    parser.add_argument("--policy.device", dest="policy_device", default=None)
    parser.add_argument("--policy.mixed-precision", dest="policy_mixed_precision", default="bf16")
    parser.add_argument("--policy.steps", dest="policy_steps", type=int, default=20)
    parser.add_argument("--policy.sigma-shift", dest="policy_sigma_shift", type=float, default=None)
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="Use a DDIM scheduler for Diffusion Policy inference. Defaults to the checkpoint scheduler, usually DDPM.",
    )
    parser.add_argument(
        "--frs",
        action="store_true",
        help=(
            "Use FRS-style phase-2 initialization for OpenPI/FastWAM/DP. "
            "DP maps this to past-action refinement."
        ),
    )
    parser.add_argument(
        "--random-noise-ratio-step",
        dest="random_noise_ratio_step",
        type=float,
        default=0.2,
        help="Prompt-memory random-noise ratio increment used by clients.",
    )
    args = parser.parse_args()
    if args.frs and args.policy_type not in {"openpi", "fastwam", "diffusion_policy"}:
        parser.error("--frs is only supported with --policy.type=openpi, fastwam, or diffusion_policy.")
    if args.ddim and args.policy_type != "diffusion_policy":
        parser.error("--ddim is only supported with --policy.type=diffusion_policy.")
    if args.random_noise_ratio_step < 0.0:
        parser.error("--random-noise-ratio-step must be non-negative.")
    return args


def _norm_stats_from_metadata(metadata: dict) -> transforms.NormStats | None:
    stats = metadata.get("action_norm_stats")
    if not isinstance(stats, dict):
        return None
    return transforms.NormStats(
        mean=np.asarray(stats["mean"], dtype=np.float32),
        std=np.asarray(stats["std"], dtype=np.float32),
        q01=None if stats.get("q01") is None else np.asarray(stats["q01"], dtype=np.float32),
        q99=None if stats.get("q99") is None else np.asarray(stats["q99"], dtype=np.float32),
    )


def _create_openpi_policy(args: argparse.Namespace) -> _policy.BasePolicy:
    train_config = _openpi_config.get_config(args.policy_config)
    sample_kwargs = {"num_steps": int(args.policy_steps)}
    if args.frs:
        sample_kwargs["frs"] = True
    base_policy = _policy_config.create_trained_policy(
        train_config,
        args.policy_dir,
        default_prompt=args.default_prompt,
        sample_kwargs=sample_kwargs,
    )
    steerer_config = SteeringConfig(
        checkpoint_dir=args.steerer_ckpt,
        mode=args.steerer_mode,
        device=args.steerer_device,
        num_inference_timesteps=args.steerer_steps,
        inference_seed=args.steerer_seed,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
    )
    return create_steered_openpi_policy(
        base_policy,
        steerer_config,
        action_norm_stats=_norm_stats_from_metadata(base_policy.metadata),
        use_quantile_norm=bool(base_policy.metadata.get("use_quantile_norm", False)),
    )


def _create_fastwam_policy(args: argparse.Namespace) -> _policy.BasePolicy:
    if args.policy_stats is None:
        raise ValueError("--policy.stats is required when --policy.type=fastwam.")
    steerer_config = SteeringConfig(
        checkpoint_dir=args.steerer_ckpt,
        mode=args.steerer_mode,
        device=args.steerer_device,
        num_inference_timesteps=args.steerer_steps,
        inference_seed=args.steerer_seed,
    )
    runtime_config = FastWAMRuntimeConfig(
        checkpoint_path=args.policy_dir,
        dataset_stats_path=args.policy_stats,
        config=args.policy_config,
        device=args.policy_device or args.steerer_device,
        mixed_precision=args.policy_mixed_precision,
        num_inference_steps=args.policy_steps,
        sigma_shift=args.policy_sigma_shift,
        frs=bool(args.frs),
    )
    return create_steered_fastwam_policy(runtime_config, steerer_config)


def _create_diffusion_policy(args: argparse.Namespace) -> _policy.BasePolicy:
    steerer_config = SteeringConfig(
        checkpoint_dir=args.steerer_ckpt,
        mode=args.steerer_mode,
        device=args.steerer_device,
        num_inference_timesteps=args.steerer_steps,
        inference_seed=args.steerer_seed,
    )
    runtime_config = DiffusionPolicyRuntimeConfig(
        checkpoint_path=args.policy_dir,
        config=args.policy_config,
        device=args.policy_device or args.steerer_device,
        num_inference_steps=args.policy_steps,
        frs=bool(args.frs),
        use_ddim=bool(args.ddim),
    )
    return create_steered_diffusion_policy(runtime_config, steerer_config)


def create_policy(args: argparse.Namespace) -> _policy.BasePolicy:
    if args.policy_type == "openpi":
        return _create_openpi_policy(args)
    if args.policy_type == "fastwam":
        return _create_fastwam_policy(args)
    if args.policy_type == "diffusion_policy":
        return _create_diffusion_policy(args)
    raise ValueError(f"Unsupported policy type: {args.policy_type}")


def main() -> None:
    args = _parse_args()
    policy = create_policy(args)
    policy_metadata = dict(policy.metadata)
    steerer_metadata = {
        "enabled": True,
        "mode": args.steerer_mode,
        "checkpoint_dir": args.steerer_ckpt,
        "inference_seed": args.steerer_seed,
    }
    policy_metadata.setdefault("steerer", steerer_metadata)
    policy_metadata.setdefault(f"{args.steerer_mode}_steerer", steerer_metadata)
    if args.steerer_mode == "evo":
        policy_metadata.setdefault("evo1_steerer", steerer_metadata)
    policy_metadata.setdefault("phase2_policy_type", args.policy_type)
    policy_metadata.setdefault("phase2_policy_config", args.policy_config)
    policy_metadata.setdefault("phase2_policy_dir", args.policy_dir)
    policy_metadata["policy_inference_steps"] = int(args.policy_steps)
    policy_metadata["max_phase2_steps"] = float(args.policy_steps)
    policy_metadata["phase2_max_steps"] = float(args.policy_steps)
    if args.policy_type in {"openpi", "fastwam", "diffusion_policy"}:
        policy_metadata["num_steps"] = int(args.policy_steps)
        policy_metadata["frs"] = bool(args.frs)
        policy_metadata["random_noise_ratio_step"] = float(args.random_noise_ratio_step)
    if args.policy_type == "diffusion_policy":
        policy_metadata["ddim"] = bool(args.ddim)
        policy_metadata["scheduler"] = (
            policy_metadata.get("diffusion_policy", {}).get("scheduler")
            if isinstance(policy_metadata.get("diffusion_policy"), dict)
            else None
        )

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating steered policy server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
