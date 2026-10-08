from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class Phase1Action:
    """Canonical output from a phase-1 steerer backend."""

    actions: np.ndarray
    mode: str
    model_actions: np.ndarray | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
