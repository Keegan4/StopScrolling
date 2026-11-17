"""Per-frame check: is the person looking directly at a (real) phone?

1. Candidates: "cell phone" detections with detector score >= ``phone_conf_threshold``.
2. If verification is on, a candidate only counts when CLIP agrees it is a phone (not a
   calculator, remote, ...): ``check.p_phone >= phone_verify_threshold``.
3. The phone's distance is estimated from its box (a phone is ~15 cm long). Because a
   phone in use is tilted towards the user it looks shorter than it is, so that estimate
   is the far end of a range of plausible distances.
4. Gaze = head rotation (face transformation matrix) + eye rotation (iris + blendshapes).
   The person is looking at the phone when, somewhere in that range, the angle between the
   gaze and the face->phone direction is within ``facing_tolerance_deg`` plus the phone's
   own angular radius.

This is noisy frame to frame; :mod:`stopscrolling.detection.attention` makes it steady,
and :class:`FaceLossLatch` covers frames where tilting your head down hides your face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from stopscrolling.config import Settings
from stopscrolling.features.gaze import eye_angles
from stopscrolling.features.geometry import (
    FORWARD,
    PinholeCamera,
    angle_between,
    angles_from_direction,
    direction_from_angles,
)
from stopscrolling.perception.types import FaceObservation, PhoneDetection


PHONE_RADIUS_FRACTION = 0.35  # target radius as a fraction of phone length
DEPTH_SAMPLES = 12  # distances tried across the plausible phone depth range


class Status(str, Enum):
    NO_FACE = "no face"
    NO_PHONE = "no phone"
    PHONE_IN_VIEW = "phone in view, not looking"
    LOOKING_AT_PHONE = "looking at phone"


def is_confirmed_phone(phone: PhoneDetection, settings: Settings, require_verification: bool) -> bool:
    if phone.score < settings.phone_conf_threshold:
        return False
    if not require_verification:
        return True
    return phone.check is not None and phone.check.p_phone >= settings.phone_verify_threshold


@dataclass
class FacingResult:
    status: Status
    phone: PhoneDetection | None = None  # the confirmed phone used for the check
    phone_conf: float = 0.0
    head_yaw: float | None = None
    head_pitch: float | None = None
    gaze_yaw: float | None = None
    gaze_pitch: float | None = None
    gaze_dir: np.ndarray | None = None
    angle_to_phone: float | None = None  # degrees between gaze and face->phone direction
    tolerance_deg: float | None = None  # effective tolerance for this phone
    phone_pos: np.ndarray | None = None  # estimated 3D phone position, cm
    phone_dist_cm: float | None = None  # face -> phone distance

    @property
    def looking_at_phone(self) -> bool:
        return self.status is Status.LOOKING_AT_PHONE


def phone_depth_range(
    phone: PhoneDetection, camera: PinholeCamera, face_depth_cm: float, settings: Settings
) -> tuple[float, float]:
    """Plausible distance range (cm from the camera) for a detected phone.

    A phone of known length L spanning ``long_px`` pixels is ``f * L / long_px`` away. While
    someone uses a phone its screen faces them, so the camera sees it tilted and shorter
    than it really is (fingers cover parts too), which makes that estimate too far. So it is
    treated as the far end of the range.
    """
    long_px = max(phone.box[2], phone.box[3], 1)
    far = camera.focal_px * settings.phone_length_cm / long_px
    far = float(np.clip(far, settings.min_phone_depth_cm, face_depth_cm + 10.0))
    near = max(settings.min_phone_depth_cm, far * settings.phone_depth_min_ratio)
    return near, far


def _best_alignment(
    phone: PhoneDetection, camera: PinholeCamera, face_pos: np.ndarray, gaze_dir: np.ndarray, settings: Settings
) -> tuple[float, float, np.ndarray, float, float]:
    """Over the plausible depth range, find where the gaze best lines up with the phone.

    Returns (margin, angle, phone_pos, dist, tolerance); margin <= 0 means looking at it.
    """
    near, far = phone_depth_range(phone, camera, max(-float(face_pos[2]), 1.0), settings)
    best = None
    for depth in np.linspace(near, far, DEPTH_SAMPLES):
        pos = camera.back_project(*phone.center, float(depth))
        to_phone = pos - face_pos
        dist = float(np.linalg.norm(to_phone))
        angle = angle_between(gaze_dir, to_phone)
        # Angular radius of the phone itself: ~half its average side (15 x 7 cm -> ~5 cm).
        radius = math.degrees(math.atan2(PHONE_RADIUS_FRACTION * settings.phone_length_cm, max(dist, 1.0)))
        tol = settings.facing_tolerance_deg + radius
        if best is None or angle - tol < best[0]:
            best = (angle - tol, angle, pos, dist, tol)
    return best


def evaluate(
    face: FaceObservation | None,
    phones: list[PhoneDetection],
    camera: PinholeCamera,
    settings: Settings,
    require_verification: bool = False,
) -> FacingResult:
    confirmed = [p for p in phones if is_confirmed_phone(p, settings, require_verification)]

    if face is None:
        best = max(confirmed, key=lambda p: p.score, default=None)
        return FacingResult(Status.NO_FACE, phone=best, phone_conf=best.score if best else 0.0)

    r, t = face.rotation, face.translation
    head_yaw, head_pitch = angles_from_direction(r @ FORWARD)
    eye_yaw, eye_pitch = eye_angles(
        face.landmarks, face.blendshapes, settings.eye_yaw_gain_deg, settings.eye_pitch_gain_deg
    )
    gaze_dir = r @ direction_from_angles(eye_yaw, eye_pitch)
    gaze_yaw, gaze_pitch = angles_from_direction(gaze_dir)

    result = FacingResult(
        Status.NO_PHONE,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        gaze_yaw=gaze_yaw,
        gaze_pitch=gaze_pitch,
        gaze_dir=gaze_dir,
    )
    if not confirmed:
        return result

    # With several confirmed phones, use the one the gaze is closest to.
    candidates = [(_best_alignment(p, camera, t, gaze_dir, settings), p) for p in confirmed]
    (margin, angle, pos, dist, tol), phone = min(candidates, key=lambda c: c[0][0])
    result.phone, result.phone_conf = phone, phone.score
    result.phone_pos, result.phone_dist_cm = pos, dist
    result.angle_to_phone, result.tolerance_deg = angle, tol
    result.status = Status.LOOKING_AT_PHONE if margin <= 0 else Status.PHONE_IN_VIEW
    return result


class FaceLossLatch:
    """Looking down steeply at a phone can make the face tracker lose your face.

    While the face is missing but a confirmed phone is still in view, keep the answer from
    the last frame where the face was visible. Once the phone is gone too, clear it.
    """

    def __init__(self) -> None:
        self._last = False

    def update(self, result: FacingResult) -> bool:
        if result.status is not Status.NO_FACE:
            self._last = result.looking_at_phone
        elif result.phone is None:
            self._last = False
        return self._last
