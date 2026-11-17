import math

import numpy as np
import pytest

from stopscrolling.features.geometry import (
    PinholeCamera,
    angle_between,
    angles_from_direction,
    direction_from_angles,
    rotation_and_translation,
)


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (25, 0), (-40, 10), (15, -60), (-80, -30)])
def test_angles_round_trip(yaw, pitch):
    got = angles_from_direction(direction_from_angles(yaw, pitch))
    assert got == pytest.approx((yaw, pitch), abs=1e-9)


def test_sign_conventions():
    # Straight at the camera.
    assert direction_from_angles(0, 0) == pytest.approx([0, 0, 1])
    # Turning to the person's right moves towards raw-image left (-x).
    assert direction_from_angles(30, 0)[0] < 0
    # Looking down is -y.
    assert direction_from_angles(0, -30)[1] < 0


def test_angle_between():
    assert angle_between([1, 0, 0], [0, 1, 0]) == pytest.approx(90)
    assert angle_between([1, 1, 0], [2, 2, 0]) == pytest.approx(0, abs=1e-6)


def test_rotation_and_translation_strips_scale():
    a = math.radians(30)
    r = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    m = np.eye(4)
    m[:3, :3] = 1.7 * r
    m[:3, 3] = [1, 2, -50]
    got_r, got_t = rotation_and_translation(m)
    assert got_r == pytest.approx(r)
    assert got_t == pytest.approx([1, 2, -50])


def test_camera_project_back_project_round_trip():
    cam = PinholeCamera(640, 480)
    p = np.array([5.0, -8.0, -45.0])
    u, v = cam.project(p)
    assert cam.back_project(u, v, 45.0) == pytest.approx(p)
    # Points above the optical axis appear in the top half of the image.
    assert cam.project(np.array([0.0, 10.0, -50.0]))[1] < 240
    assert cam.project(np.array([0.0, 0.0, 5.0])) is None  # behind the camera
