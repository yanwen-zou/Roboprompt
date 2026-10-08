"""Hardware constants from an explicitly selected site calibration file."""
import json
import os
from pathlib import Path

import numpy as np


def load_hardware_config(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Hardware configuration must be a JSON object")
    serials = config.get("CAM_SERIAL")
    if not isinstance(serials, list) or len(serials) < 2 or not all(isinstance(s, str) and s for s in serials):
        raise ValueError("CAM_SERIAL must contain environment and wrist camera serial strings, in that order")
    if len(set(serials)) != len(serials):
        raise ValueError("CAM_SERIAL entries must be unique")
    if not isinstance(config.get("ROBOT_SN"), str) or not config["ROBOT_SN"].strip():
        raise ValueError("ROBOT_SN must be a nonempty string")
    for name, shape in {
        "EIH_INTRINSIC": (3, 3), "E2H_INTRINSIC": (3, 3),
        "EIH_CAM_T": (4, 4), "E2H_CAM_T": (4, 4), "Gripper_TCP_T": (4, 4),
        "HOME_JOINT_DEG": (7,),
    }.items():
        value = np.asarray(config.get(name), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite array of shape {shape}")
        config[name] = value
    return config


_config_path = os.environ.get("ROBOPROMPT_HARDWARE_CONFIG")
if not _config_path:
    raise RuntimeError("Set ROBOPROMPT_HARDWARE_CONFIG to your hardware JSON; use hardware/config/flexiv.reference.json as the schema")
_config = load_hardware_config(_config_path)
CAM_SERIAL = _config["CAM_SERIAL"]
ROBOT_SN = _config["ROBOT_SN"]
HOME_JOINT_DEG = _config["HOME_JOINT_DEG"]
EIH_INTRINSIC = _config["EIH_INTRINSIC"]
EIH_CAM_T = _config["EIH_CAM_T"]
E2H_INTRINSIC = _config["E2H_INTRINSIC"]
E2H_CAM_T = _config["E2H_CAM_T"]
Gripper_TCP_T = _config["Gripper_TCP_T"]

HUMAN = 0
ROBOT = 1
INTV = 3
