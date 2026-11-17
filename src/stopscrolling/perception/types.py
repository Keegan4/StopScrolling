"""Plain data types produced by the perception layer (no MediaPipe imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FaceObservation:
    landmarks: np.ndarray  # (N, 3) normalised: x, y in [0, 1] of the raw image, z relative depth
    rotation: np.ndarray  # (3, 3) head rotation in camera space
    translation: np.ndarray  # (3,) head position in camera space, cm
    blendshapes: dict[str, float] = field(default_factory=dict)


@dataclass(eq=False)
class PhoneDetection:
    score: float  # object-detector confidence for "cell phone"
    box: tuple[int, int, int, int]  # x, y, width, height in pixels of the raw image
    check: Any = None  # PhoneCheck from the verifier (phone vs calculator etc.), if run

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2
