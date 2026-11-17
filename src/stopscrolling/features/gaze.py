"""Eye-in-head gaze angles from face landmarks and blendshapes.

Horizontal: iris position between the eye corners (image space, so no left/right
labelling ambiguity). Vertical: MediaPipe's eyeLookUp / eyeLookDown blendshapes, which
handle the eyelid partly covering the iris when you look down better than the iris does.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# Face-mesh indices (refined mesh, 478 points). Each eye: two corners + iris centre.
EYES = (
    ((33, 133), 468),  # person's right eye (left side of the raw image)
    ((362, 263), 473),  # person's left eye
)
NUM_REFINED_LANDMARKS = 478


def iris_horizontal_offset(landmarks: np.ndarray) -> float | None:
    """Mean iris offset across both eyes, in [-1, 1].

    0 = centred, +1 = iris at the raw-image-right corner (the person looking to their left).
    """
    if landmarks.shape[0] < NUM_REFINED_LANDMARKS:
        return None
    offsets = []
    for (a, b), iris in EYES:
        lo, hi = sorted((landmarks[a, 0], landmarks[b, 0]))
        width = hi - lo
        if width < 1e-6:
            continue
        offsets.append(((landmarks[iris, 0] - lo) / width - 0.5) * 2.0)
    return float(np.mean(offsets)) if offsets else None


def eye_vertical_score(blendshapes: Mapping[str, float]) -> float:
    """eyeLookUp minus eyeLookDown averaged over both eyes; > 0 = looking up."""
    up = (blendshapes.get("eyeLookUpLeft", 0.0) + blendshapes.get("eyeLookUpRight", 0.0)) / 2
    down = (blendshapes.get("eyeLookDownLeft", 0.0) + blendshapes.get("eyeLookDownRight", 0.0)) / 2
    return up - down


def eye_angles(
    landmarks: np.ndarray,
    blendshapes: Mapping[str, float],
    yaw_gain_deg: float,
    pitch_gain_deg: float,
) -> tuple[float, float]:
    """Approximate eye rotation relative to the head as (yaw_deg, pitch_deg).

    Uses the conventions in :mod:`stopscrolling.features.geometry` (yaw > 0 = the
    person's right, pitch > 0 = up).
    """
    offset = iris_horizontal_offset(landmarks) or 0.0
    # Iris towards raw-image right means the person looks to their left -> negative yaw.
    eye_yaw = -offset * yaw_gain_deg
    eye_pitch = eye_vertical_score(blendshapes) * pitch_gain_deg
    return eye_yaw, eye_pitch
