import math

import numpy as np
import pytest

from stopscrolling.config import Settings
from stopscrolling.detection.facing import FaceLossLatch, Status, evaluate, phone_depth_range
from stopscrolling.features.gaze import NUM_REFINED_LANDMARKS, iris_horizontal_offset
from stopscrolling.features.geometry import PinholeCamera, direction_from_angles
from stopscrolling.perception.types import FaceObservation, PhoneDetection
from stopscrolling.perception.verifier import PhoneCheck

CAM = PinholeCamera(640, 480)
FACE_POS = np.array([0.0, 0.0, -50.0])
S = Settings()

# A phone held below the face: 20 cm down, 10 cm closer to the camera.
PHONE_BELOW = [0.0, -20.0, -40.0]
IS_PHONE = PhoneCheck(0.9, "phone", 0.9)
IS_CALCULATOR = PhoneCheck(0.01, "calculator", 0.98)


def centred_eye_landmarks() -> np.ndarray:
    """Landmarks with both irises exactly centred between their eye corners."""
    lm = np.full((NUM_REFINED_LANDMARKS, 3), 0.5)
    lm[33, 0], lm[133, 0], lm[468, 0] = 0.40, 0.46, 0.43
    lm[362, 0], lm[263, 0], lm[473, 0] = 0.54, 0.60, 0.57
    return lm


def face_looking(yaw_deg: float, pitch_deg: float) -> FaceObservation:
    """Face at FACE_POS with the head pointing at (yaw, pitch), eyes centred."""
    f = direction_from_angles(yaw_deg, pitch_deg)
    right = np.cross([0.0, 1.0, 0.0], f)
    right /= np.linalg.norm(right)
    up = np.cross(f, right)
    return FaceObservation(centred_eye_landmarks(), np.column_stack([right, up, f]), FACE_POS.copy(), {})


def pitched_face(pitch_deg: float) -> FaceObservation:
    return face_looking(0.0, pitch_deg)


def phone_at(point_3d, score=0.9, check=IS_PHONE, foreshorten=1.0) -> PhoneDetection:
    """Detection whose pixel size matches a phone_length_cm phone at that 3D point
    (``foreshorten`` < 1 shrinks it, like a phone tilted towards the user)."""
    p = np.asarray(point_3d, dtype=float)
    u, v = CAM.project(p)
    long_px = CAM.focal_px * S.phone_length_cm / -p[2] * foreshorten
    w = long_px * 0.5
    return PhoneDetection(score, (int(u - w / 2), int(v - long_px / 2), int(w), int(long_px)), check)


def test_iris_offset_centred_and_shifted():
    lm = centred_eye_landmarks()
    assert iris_horizontal_offset(lm) == pytest.approx(0.0, abs=1e-9)
    lm[468, 0] += 0.012
    lm[473, 0] += 0.012
    assert iris_horizontal_offset(lm) == pytest.approx(0.4)


def test_head_pitch_is_reported():
    res = evaluate(pitched_face(-30), [], CAM, S)
    assert res.head_pitch == pytest.approx(-30)
    assert res.head_yaw == pytest.approx(0, abs=1e-9)
    assert res.status is Status.NO_PHONE


def test_depth_range_from_box_size():
    near, far = phone_depth_range(phone_at(PHONE_BELOW), CAM, 50.0, S)
    assert far == pytest.approx(40.0, abs=1.0)
    assert near == pytest.approx(40.0 * S.phone_depth_min_ratio, abs=1.0)


def test_looking_down_at_phone():
    # Direction face->phone is (0, -20, 10): pitch about -63 degrees.
    res = evaluate(pitched_face(-63), [phone_at(PHONE_BELOW)], CAM, S)
    assert res.status is Status.LOOKING_AT_PHONE
    assert res.angle_to_phone < 3


def test_tilted_phone_still_counts():
    # Seen tilted, the phone looks 30% shorter, so its size alone suggests it is farther away.
    res = evaluate(pitched_face(-63), [phone_at(PHONE_BELOW, foreshorten=0.7)], CAM, S)
    assert res.looking_at_phone


def test_looking_to_the_side_is_not_enough():
    res = evaluate(face_looking(40, -40), [phone_at(PHONE_BELOW)], CAM, S)
    assert res.status is Status.PHONE_IN_VIEW


def test_phone_visible_but_looking_at_screen():
    res = evaluate(pitched_face(0), [phone_at(PHONE_BELOW)], CAM, S)
    assert res.status is Status.PHONE_IN_VIEW
    assert not res.looking_at_phone


def test_low_detector_score_is_ignored():
    res = evaluate(pitched_face(-63), [phone_at(PHONE_BELOW, score=0.3)], CAM, S)
    assert res.status is Status.NO_PHONE


def test_calculator_is_ignored_when_verifying():
    calc = phone_at(PHONE_BELOW, check=IS_CALCULATOR)
    assert evaluate(pitched_face(-63), [calc], CAM, S, require_verification=True).status is Status.NO_PHONE
    # Without verification the detector alone would have counted it.
    assert evaluate(pitched_face(-63), [calc], CAM, S).status is Status.LOOKING_AT_PHONE


def test_unchecked_candidate_does_not_count_when_verifying():
    pending = phone_at(PHONE_BELOW, check=None)
    assert evaluate(pitched_face(-63), [pending], CAM, S, require_verification=True).status is Status.NO_PHONE


def test_picks_the_phone_being_looked_at():
    off_to_side = phone_at([25.0, 15.0, -40.0])
    below = phone_at(PHONE_BELOW)
    res = evaluate(pitched_face(-63), [off_to_side, below], CAM, S, require_verification=True)
    assert res.phone is below
    assert res.looking_at_phone


def test_no_face():
    res = evaluate(None, [phone_at(PHONE_BELOW)], CAM, S)
    assert res.status is Status.NO_FACE


def test_face_loss_latch_keeps_answer_while_phone_visible():
    latch = FaceLossLatch()
    phone = phone_at(PHONE_BELOW)
    assert latch.update(evaluate(pitched_face(-63), [phone], CAM, S)) is True
    # Head tilts further down, face tracker loses the face, phone still in view.
    assert latch.update(evaluate(None, [phone], CAM, S)) is True
    # Phone gone as well -> cleared.
    assert latch.update(evaluate(None, [], CAM, S)) is False


def test_face_loss_latch_does_not_invent_phone_use():
    latch = FaceLossLatch()
    phone = phone_at(PHONE_BELOW)
    assert latch.update(evaluate(pitched_face(0), [phone], CAM, S)) is False
    assert latch.update(evaluate(None, [phone], CAM, S)) is False
