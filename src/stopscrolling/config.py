"""Paths and tunable settings."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if getattr(sys, "frozen", False):  # packaged with PyInstaller
    # Bundled read-only files (models) live in the unpacked bundle; user-editable
    # folders (assets/memes, assets/sounds) sit next to the executable.
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    USER_DIR = Path(sys.executable).resolve().parent
else:
    BUNDLE_DIR = USER_DIR = PROJECT_ROOT

MODELS_DIR = BUNDLE_DIR / "models"

FACE_MODEL = MODELS_DIR / "face_landmarker.task"
PHONE_MODELS = {
    "lite0": MODELS_DIR / "efficientdet_lite0.tflite",  # fast, misses more phones
    "lite2": MODELS_DIR / "efficientdet_lite2.tflite",  # slower, more accurate
}
CLIP_DIR = MODELS_DIR / "clip"  # phone-vs-calculator verifier
DATA_DIR = USER_DIR / "data"  # training data (git-ignored)
CLIPS_DIR = DATA_DIR / "clips"  # recorded webcam clips: <id>.mp4 + <id>.json
PROCESSED_DIR = DATA_DIR / "processed"  # per clip: frames/, landmarks.npz, features.csv
MEMES_DIR = USER_DIR / "assets" / "memes"  # your own meme images (optional)
SOUNDS_DIR = USER_DIR / "assets" / "sounds"  # your own alarm sounds (optional)


@dataclass
class Settings:
    # Camera
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # Phone detection (MediaPipe object detector, COCO "cell phone")
    detector: str = "lite0"
    detect_every: int = 1  # run the object detector every N frames
    phone_conf_threshold: float = 0.5  # min detector score to be a phone candidate
    phone_hold_s: float = 0.5  # keep the last detections this long between detector runs

    # Phone verification (CLIP: phone vs calculator / remote / book / ...)
    verify_phones: bool = True
    phone_verify_threshold: float = 0.5  # min CLIP probability that the candidate is a phone
    verify_interval_s: float = 0.4  # re-check each tracked candidate this often

    # "Looking directly at the phone" geometry
    phone_length_cm: float = 15.0  # typical phone long side; used to estimate its distance
    min_phone_depth_cm: float = 15.0
    phone_depth_min_ratio: float = 0.65  # phone may be up to 35% closer than its size suggests
    facing_tolerance_deg: float = 10.0  # allowed gaze error, added to the phone's own angular radius

    # Eye-in-head gaze (approximate, tune with the debug viewer)
    eye_yaw_gain_deg: float = 60.0  # degrees per unit of normalised iris offset
    eye_pitch_gain_deg: float = 35.0  # degrees per unit of (eyeLookUp - eyeLookDown)

    # Steady decision (less sensitive than the raw per-frame result)
    enter_s: float = 1.5  # must look at the phone this long before it counts
    exit_s: float = 1.0  # must look away this long before it clears
    evidence_smoothing_s: float = 0.4

    # Alerts (while the steady decision says you're on your phone)
    alert_sound: bool = True
    alert_popups: bool = True
    max_popups: int = 40
    escalate_s: float = 6.0  # louder sound + faster popups every N seconds
    snooze_s: float = 300.0  # 's' key in the preview pauses alerts this long

    # Must match MediaPipe's face-geometry virtual camera so 3D positions line up with pixels.
    vertical_fov_deg: float = 63.0
