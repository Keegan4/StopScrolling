"""Per-frame feature vector for the attention model.

Both dataset extraction and (later) live inference build features with
:func:`frame_features`, so the model sees the same numbers at training and run time.
Missing values (no face, no phone) are NaN; tree models handle those directly.
"""

from __future__ import annotations

import math

import numpy as np

from stopscrolling.detection.facing import FacingResult
from stopscrolling.features.gaze import eye_vertical_score, iris_horizontal_offset
from stopscrolling.perception.types import FaceObservation, PhoneDetection

# MediaPipe blendshapes kept as features (eyes, brows, jaw: gaze, blinking, talking, squinting).
BLENDSHAPES = [
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "jawOpen",
]

FEATURE_NAMES = [
    # face / head
    "face_present",
    "face_cx",  # face centre in the raw image, 0..1
    "face_cy",
    "face_width",  # fraction of image width (closeness to the camera)
    "head_dist_cm",
    "head_yaw",
    "head_pitch",
    "head_roll",
    # gaze
    "gaze_yaw",
    "gaze_pitch",
    "iris_offset",
    "eye_vertical",
    *(f"bs_{n}" for n in BLENDSHAPES),
    # phone
    "phone_det",  # best detector score for "cell phone" (0 if none)
    "phone_p",  # CLIP probability that the best candidate really is a phone (NaN if unchecked)
    "phone_confirmed",  # 1 if a candidate passed both checks
    "phone_cx",
    "phone_cy",
    "phone_area",  # fraction of the image
    "angle_to_phone",
    "looking_at_phone",  # the rule-based per-frame decision, as a baseline feature
]

NAN = float("nan")


def head_roll(rotation: np.ndarray) -> float:
    """Head tilt about the viewing axis, degrees (0 = upright)."""
    return math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))


def frame_features(
    face: FaceObservation | None,
    phones: list[PhoneDetection],
    result: FacingResult,
    image_size: tuple[int, int],
) -> dict[str, float]:
    """Feature dict (keys = FEATURE_NAMES) for one frame. ``image_size`` = (width, height)."""
    w, h = image_size
    f = dict.fromkeys(FEATURE_NAMES, NAN)

    f["face_present"] = 0.0 if face is None else 1.0
    if face is not None:
        lm = face.landmarks
        x0, x1 = float(lm[:, 0].min()), float(lm[:, 0].max())
        y0, y1 = float(lm[:, 1].min()), float(lm[:, 1].max())
        f.update(face_cx=(x0 + x1) / 2, face_cy=(y0 + y1) / 2, face_width=x1 - x0)
        f["head_dist_cm"] = -float(face.translation[2])
        f["head_roll"] = head_roll(face.rotation)
        offset = iris_horizontal_offset(lm)
        f["iris_offset"] = NAN if offset is None else offset
        f["eye_vertical"] = eye_vertical_score(face.blendshapes)
        for n in BLENDSHAPES:
            f[f"bs_{n}"] = face.blendshapes.get(n, NAN)

    for key, value in (
        ("head_yaw", result.head_yaw),
        ("head_pitch", result.head_pitch),
        ("gaze_yaw", result.gaze_yaw),
        ("gaze_pitch", result.gaze_pitch),
        ("angle_to_phone", result.angle_to_phone),
    ):
        if value is not None:
            f[key] = float(value)

    best = max(phones, key=lambda p: p.score, default=None)
    f["phone_det"] = best.score if best else 0.0
    shown = result.phone or best  # prefer the confirmed phone for position / CLIP score
    if shown is not None:
        x, y, bw, bh = shown.box
        f.update(phone_cx=(x + bw / 2) / w, phone_cy=(y + bh / 2) / h, phone_area=(bw * bh) / (w * h))
        if shown.check is not None:
            f["phone_p"] = shown.check.p_phone
    f["phone_confirmed"] = 1.0 if result.phone is not None else 0.0
    f["looking_at_phone"] = 1.0 if result.looking_at_phone else 0.0
    return f


def to_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features[n] for n in FEATURE_NAMES], dtype=np.float32)
