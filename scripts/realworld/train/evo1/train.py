#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex

# Respect the selected CUDA toolkit; otherwise use normal toolchain discovery.
if os.environ.get("CUDA_HOME"):
    os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + os.pathsep + os.environ.get("PATH", "")

from config.config import DEFAULT_CONFIG_NAME, build_command, load_launcher_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Evo-1 training from a YAML config.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_NAME,
        help=(
            "Config name under scripts/realworld/train/evo1/config without .yaml, or a YAML path. "
            f"Default: {DEFAULT_CONFIG_NAME}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved training command without launching it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_launcher_config(args.config)
    cmd = build_command(cfg)

    if args.dry_run:
        print(shlex.join(cmd))
        return

    os.chdir(cfg.evo1_root)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
