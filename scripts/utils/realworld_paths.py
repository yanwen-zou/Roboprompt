from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REAL_ROBOT_DATA_DIR = "real_robot_data"


def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except PermissionError:
        return False


def get_data_root(data_root: str | Path | None = None, env_var: str = "RP_DATA_ROOT") -> Path:
    value = str(data_root) if data_root is not None else os.environ.get(env_var)
    if not value:
        raise RuntimeError(f"{env_var} is not set. Export {env_var} or pass an explicit data root.")
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else path.resolve()


def resolve_dataset_path(path_value: str | Path, data_root: str | Path | None = None) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))

    if path.is_absolute():
        parts = path.parts
        if REAL_ROBOT_DATA_DIR not in parts:
            return path
        root = get_data_root(data_root)
        root_index = len(parts) - 1 - parts[::-1].index(REAL_ROBOT_DATA_DIR)
        rel_parts = parts[root_index + 1 :]
        if rel_parts:
            return (root / REAL_ROBOT_DATA_DIR / Path(*rel_parts)).resolve()
        return (root / REAL_ROBOT_DATA_DIR).resolve()

    root = get_data_root(data_root)
    if path.parts and path.parts[0] == REAL_ROBOT_DATA_DIR:
        return (root / path).resolve()
    return (root / REAL_ROBOT_DATA_DIR / path).resolve()


def normalize_dataset_config_paths(
    dataset_config: dict[str, Any],
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    data_groups = dataset_config.get("data_groups", {})
    if not isinstance(data_groups, dict):
        return dataset_config

    for arm_config in data_groups.values():
        if not isinstance(arm_config, dict):
            continue
        for single_dataset_config in arm_config.values():
            if not isinstance(single_dataset_config, dict) or "path" not in single_dataset_config:
                continue
            single_dataset_config["path"] = str(resolve_dataset_path(single_dataset_config["path"], data_root))
    return dataset_config


def resolve_repo_local_path(
    path_value: str | Path,
    *,
    repo_root: Path,
    fallback_roots: list[Path] | tuple[Path, ...] = (),
    must_exist: bool,
) -> Path:
    path = Path(path_value).expanduser()
    candidates: list[Path] = []

    if path.is_absolute():
        parts = path.parts
        if repo_root.name in parts:
            repo_index = len(parts) - 1 - parts[::-1].index(repo_root.name)
            rel_parts = parts[repo_index + 1 :]
            if rel_parts:
                candidates.append(repo_root.joinpath(*rel_parts))
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        candidates.append(repo_root / path)
        candidates.extend(root / path for root in fallback_roots)

    if must_exist:
        for candidate in candidates:
            if safe_exists(candidate):
                return candidate.resolve()
        return candidates[0].resolve() if candidates else path.resolve()

    for candidate in candidates:
        if safe_exists(candidate):
            return candidate.resolve()
    return candidates[-1].resolve() if len(candidates) > 1 else candidates[0].resolve()
