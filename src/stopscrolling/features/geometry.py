"""3D geometry helpers.

Coordinate system (MediaPipe face-geometry "metric" space, units in cm):
    +x  right in the raw (un-mirrored) camera image
    +y  up
    +z  towards the viewer; the camera sits at the origin looking down -z

Angles reported by this module use the *person's* point of view:
    yaw   > 0  the person turns to their own right
    pitch > 0  the person looks up
A face looking straight into the camera has direction (0, 0, +1) and yaw = pitch = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FORWARD = np.array([0.0, 0.0, 1.0])


def direction_from_angles(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit vector for a (yaw, pitch) pair, using the conventions above."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([-math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)])


def angles_from_direction(v: np.ndarray) -> tuple[float, float]:
    """Inverse of :func:`direction_from_angles`. Returns (yaw_deg, pitch_deg)."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    yaw = math.degrees(math.atan2(-v[0], v[2]))
    pitch = math.degrees(math.asin(float(np.clip(v[1], -1.0, 1.0))))
    return yaw, pitch


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # atan2 form stays accurate near 0 and 180 degrees, unlike acos(dot).
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b))))


def rotation_and_translation(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 pose matrix into a clean rotation (scale removed) and a translation."""
    m = np.asarray(matrix, dtype=float).reshape(4, 4)
    u, _, vt = np.linalg.svd(m[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:  # guard against a reflection
        u[:, -1] *= -1
        r = u @ vt
    return r, m[:3, 3].copy()


@dataclass(frozen=True)
class PinholeCamera:
    """Ideal pinhole camera matching MediaPipe's virtual face-geometry camera."""

    width: int
    height: int
    vertical_fov_deg: float = 63.0

    @property
    def focal_px(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.vertical_fov_deg) / 2)

    def project(self, p: np.ndarray) -> tuple[float, float] | None:
        """3D point -> pixel (u, v) in the raw image, or None if behind the camera."""
        depth = -float(p[2])
        if depth <= 1e-6:
            return None
        f = self.focal_px
        return self.width / 2 + f * float(p[0]) / depth, self.height / 2 - f * float(p[1]) / depth

    def back_project(self, u: float, v: float, depth: float) -> np.ndarray:
        """Pixel (u, v) at a given distance in front of the camera -> 3D point."""
        f = self.focal_px
        return np.array([(u - self.width / 2) * depth / f, -(v - self.height / 2) * depth / f, -depth])
