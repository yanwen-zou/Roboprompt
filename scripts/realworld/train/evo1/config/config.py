from __future__ import annotations

from datetime import datetime
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Missing Python package: PyYAML. Install it in the active environment.") from exc


DEFAULT_CONFIG_NAME = "bread_action"
RP_DATA_ROOT_ENV = "RP_DATA_ROOT"


def repo_root_from_here() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Evo-1").is_dir():
            return parent
    raise FileNotFoundError("Could not find repository root containing Evo-1.")


REPO_ROOT = repo_root_from_here()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.realworld_paths import normalize_dataset_config_paths  # noqa: E402


def resolve_config_path(config_name: str | None, config_dir: Path) -> Path:
    name = config_name or DEFAULT_CONFIG_NAME
    path = Path(name)
    if path.suffix in {".yaml", ".yml"} or path.parent != Path("."):
        return path.expanduser().resolve()
    return (config_dir / f"{name}.yaml").resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Config must be a YAML mapping: {path}")
    return value


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping.")
    return value


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LauncherConfig:
    raw: dict[str, Any]
    repo_root: Path
    evo1_root: Path
    config_path: Path

    def __post_init__(self) -> None:
        self.run = section(self.raw, "run")
        self.runtime = section(self.raw, "runtime")
        self.model = section(self.raw, "model")
        self.dataset = section(self.raw, "dataset")
        self.shape = section(self.raw, "shape")
        self.optimization = section(self.raw, "optimization")
        self.logging = section(self.raw, "logging")
        self.resume = section(self.raw, "resume")
        self.finetuning = section(self.raw, "finetuning")

        time = os.environ.get("TIME", datetime.now().strftime("%Y%m%d"))
        self.context = {
            "repo_root": str(self.repo_root),
            "evo1_root": str(self.evo1_root),
            "TIME": time,
            "home": str(Path.home()),
        }
        self.run_name = self.expand(self.run.get("name", "evo1_stage1"))
        self.context["run_name"] = self.run_name

    def expand(self, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        text = os.path.expanduser(str(value))
        try:
            text = text.format(**self.context)
        except KeyError as exc:
            raise KeyError(f"Unknown placeholder in config value '{value}': {exc}") from exc
        return os.path.expandvars(text)


def add_arg(args: list[str], cfg: LauncherConfig, name: str, value: Any) -> None:
    value = cfg.expand(value)
    if value is not None and value != "":
        args.extend([name, str(value)])


def add_flag(args: list[str], name: str, value: Any) -> None:
    if truthy(value):
        args.append(name)


def require_rp_data_root_expanded(path_value: str, config_key: str) -> None:
    if f"${RP_DATA_ROOT_ENV}" in path_value or f"${{{RP_DATA_ROOT_ENV}}}" in path_value:
        raise RuntimeError(
            f"{RP_DATA_ROOT_ENV} is not set, but {config_key} uses it. "
            f"Export {RP_DATA_ROOT_ENV} before launching Evo-1 training."
        )


def load_launcher_config(config_name: str | None = None) -> LauncherConfig:
    repo_root = REPO_ROOT
    evo1_root = repo_root / "Evo-1"
    config_dir = Path(__file__).resolve().parent
    config_path = resolve_config_path(config_name or os.environ.get("CONFIG_NAME"), config_dir)

    if not evo1_root.is_dir():
        raise FileNotFoundError(f"Missing Evo-1 directory: {evo1_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    return LauncherConfig(
        raw=load_yaml(config_path),
        repo_root=repo_root,
        evo1_root=evo1_root,
        config_path=config_path,
    )


def build_dataset_config(cfg: LauncherConfig) -> dict[str, Any]:
    """Build the dataset config dict expected by LeRobotDatasetRP.

    If dataset.config_path is set, read from that external YAML file.
    Otherwise, build from the inline fields in the dataset section.
    """
    config_path = cfg.dataset.get("config_path")
    if config_path:
        path = Path(cfg.expand(config_path))
        if not path.is_absolute():
            candidate = cfg.evo1_root / path
            if candidate.exists():
                path = candidate
        return normalize_dataset_config_paths(load_yaml(path))

    # Build inline dataset config from the launcher YAML
    data_groups = cfg.dataset.get("data_groups")
    if not data_groups:
        raise ValueError(
            "Dataset config missing. Either set dataset.config_path to an external YAML, "
            "or define dataset.data_groups inline in the launcher config."
        )
    dataset_config = {
        "max_action_dim": cfg.dataset.get("max_action_dim", 24),
        "max_state_dim": cfg.dataset.get("max_state_dim", 24),
        "max_views": cfg.dataset.get("max_views", 2),
        "prompt_task_dropout_prob": cfg.dataset.get("prompt_task_dropout_prob", 0.0),
        "prompt_motion_noise_std": cfg.dataset.get("prompt_motion_noise_std", 0.005),
        "video_dropout": cfg.dataset.get("video_dropout", cfg.dataset.get("prompt_video_dropout_prob", 0.3)),
        "cmd_type": cfg.dataset.get("cmd_type", "sigma"),
        "data_groups": data_groups,
    }
    if "prompt_task_dropout_prob" in cfg.dataset:
        dataset_config["prompt_task_dropout_prob"] = cfg.dataset["prompt_task_dropout_prob"]
    if "prompt_motion_noise_std" in cfg.dataset:
        dataset_config["prompt_motion_noise_std"] = cfg.dataset["prompt_motion_noise_std"]
    if "video_dropout" in cfg.dataset:
        dataset_config["video_dropout"] = cfg.dataset["video_dropout"]
    elif "prompt_video_dropout_prob" in cfg.dataset:
        dataset_config["video_dropout"] = cfg.dataset["prompt_video_dropout_prob"]
    return normalize_dataset_config_paths(dataset_config)


def _write_dataset_config_to_file(cfg: LauncherConfig, cache_dir: Path) -> Path:
    """Write inline dataset config to a stable per-run YAML under cache_dir."""
    ds_cfg = build_dataset_config(cfg)
    out_path = Path(cache_dir) / f"{cfg.run_name}_dataset_config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(ds_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return out_path


def build_train_argv(cfg: LauncherConfig) -> list[str]:
    """Translate YAML config into the CLI argv expected by Evo-1/scripts/train.py."""
    save_dir = cfg.expand(cfg.run.get("save_dir", f"${RP_DATA_ROOT_ENV}/evo1/{{run_name}}"))
    cache_dir = cfg.expand(cfg.run.get("cache_dir", "{repo_root}/.cache/evo1_training_data"))
    require_rp_data_root_expanded(save_dir, "run.save_dir")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Resolve dataset config path (external or auto-generated)
    dataset_config_path = cfg.dataset.get("config_path")
    if not dataset_config_path:
        dataset_config_path = str(_write_dataset_config_to_file(cfg, Path(cache_dir)))
    else:
        dataset_config_path = cfg.expand(dataset_config_path)

    args = ["scripts/train.py"]

    add_arg(args, cfg, "--device", cfg.model.get("device", "cuda"))
    add_arg(args, cfg, "--run_name", cfg.run_name)
    add_arg(args, cfg, "--vlm_name", cfg.model.get("vlm_name", "OpenGVLab/InternVL3-1B"))
    add_arg(args, cfg, "--action_head", cfg.model.get("action_head", "flowmatching"))
    add_flag(args, "--return_cls_only", cfg.model.get("return_cls_only", False))
    add_flag(args, "--disable_wandb", cfg.logging.get("disable_wandb", True))

    add_arg(args, cfg, "--dataset_type", cfg.dataset.get("type", "lerobot"))
    add_arg(args, cfg, "--data_paths", cfg.dataset.get("data_paths"))
    add_arg(args, cfg, "--dataset_config_path", dataset_config_path)
    add_arg(args, cfg, "--image_size", cfg.dataset.get("image_size", 448))
    add_arg(args, cfg, "--cache_dir", cache_dir)
    add_arg(args, cfg, "--cmd_type", cfg.dataset.get("cmd_type", "sigma"))
    add_flag(args, "--binarize_gripper", cfg.dataset.get("binarize_gripper", False))
    add_flag(args, "--use_augmentation", cfg.dataset.get("use_augmentation", True))

    add_arg(args, cfg, "--lr", cfg.optimization.get("lr", "1e-5"))
    add_arg(args, cfg, "--batch_size", cfg.optimization.get("batch_size", 8))
    add_arg(args, cfg, "--max_steps", cfg.optimization.get("max_steps", 5000))
    add_arg(args, cfg, "--warmup_steps", cfg.optimization.get("warmup_steps", 1000))
    add_arg(args, cfg, "--grad_clip_norm", cfg.optimization.get("grad_clip_norm", 1.0))
    add_arg(args, cfg, "--weight_decay", cfg.optimization.get("weight_decay", "1e-5"))
    loss_cfg = section(cfg.raw, "loss")
    add_arg(args, cfg, "--wrist_loss_weight", loss_cfg.get("wrist_loss_weight", 0.0))
    add_arg(args, cfg, "--primitive_loss_weight", loss_cfg.get("primitive_loss_weight", 0.0))
    add_arg(args, cfg, "--traj_loss_weight", loss_cfg.get("traj_loss_weight", 0.0))
    add_arg(args, cfg, "--point_loss_weight", loss_cfg.get("point_loss_weight", 0.0))
    add_arg(
        args,
        cfg,
        "--action_smoothness_weight",
        loss_cfg.get("action_smoothness_weight", 0.0),
    )
    add_arg(
        args,
        cfg,
        "--action_smoothness_dims",
        loss_cfg.get("action_smoothness_dims", 3),
    )
    add_arg(
        args,
        cfg,
        "--action_smoothness_min_norm",
        loss_cfg.get("action_smoothness_min_norm", 1e-4),
    )

    add_arg(args, cfg, "--log_interval", cfg.logging.get("log_interval", 10))
    add_arg(args, cfg, "--ckpt_interval", cfg.logging.get("ckpt_interval", 2500))
    add_arg(args, cfg, "--save_dir", save_dir)

    resume_path = cfg.resume.get("path")
    if resume_path:
        resume_path = cfg.expand(resume_path)
        require_rp_data_root_expanded(resume_path, "resume.path")
    resume_enabled = truthy(cfg.resume.get("enabled", False)) or bool(resume_path)
    add_flag(args, "--resume", resume_enabled)
    add_arg(args, cfg, "--resume_path", resume_path)
    add_flag(args, "--resume_pretrain", cfg.resume.get("pretrain", False))

    add_flag(args, "--finetune_vlm", cfg.finetuning.get("vlm", False))
    add_flag(args, "--finetune_action_head", cfg.finetuning.get("action_head", True))

    add_arg(args, cfg, "--per_action_dim", cfg.shape.get("per_action_dim", 24))
    add_arg(args, cfg, "--state_dim", cfg.shape.get("state_dim", 24))
    add_arg(args, cfg, "--horizon", cfg.shape.get("horizon", 16))
    add_arg(args, cfg, "--num_layers", cfg.optimization.get("num_layers", 8))
    add_arg(args, cfg, "--num_workers", cfg.dataset.get("num_workers", 4))
    add_arg(args, cfg, "--dropout", cfg.optimization.get("dropout", 0.0))

    return args


def build_command(cfg: LauncherConfig) -> list[str]:
    python_bin = cfg.expand(cfg.runtime.get("python_bin", "auto"))
    if python_bin in (None, "", "auto"):
        venv_python = cfg.repo_root / ".venv" / "bin" / "python"
        python_bin = str(venv_python) if venv_python.exists() else "python3"

    deepspeed_cfg = cfg.runtime.get("deepspeed_config_file", "ds_config.json")
    use_deepspeed = bool(deepspeed_cfg)

    cmd = [
        str(python_bin),
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(cfg.runtime.get("num_processes", 1)),
        "--num_machines",
        str(cfg.runtime.get("num_machines", 1)),
    ]
    if use_deepspeed:
        cmd.append("--use_deepspeed")
        cmd.extend(["--deepspeed_config_file", str(cfg.expand(deepspeed_cfg))])
    cmd.extend(build_train_argv(cfg))
    return cmd
