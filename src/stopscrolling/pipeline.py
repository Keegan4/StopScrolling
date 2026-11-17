"""Frame -> "are you on your phone?" pipeline (no camera, no UI)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from stopscrolling.config import FACE_MODEL, PHONE_MODELS, Settings
from stopscrolling.detection.attention import AttentionState, AttentionTracker
from stopscrolling.detection.facing import FaceLossLatch, FacingResult, evaluate
from stopscrolling.features.geometry import PinholeCamera
from stopscrolling.perception.face import FaceTracker
from stopscrolling.perception.phone import PhoneDetector
from stopscrolling.perception.types import FaceObservation, PhoneDetection
from stopscrolling.perception.verifier import PhoneVerifier, VerifiedPhoneTracker


@dataclass
class Observation:
    face: FaceObservation | None
    phones: list[PhoneDetection]
    result: FacingResult  # this frame
    state: AttentionState  # steady decision
    camera: PinholeCamera


class PhoneWatcher:
    def __init__(self, settings: Settings, verifier: PhoneVerifier | None) -> None:
        self.s = settings
        self.faces = FaceTracker(FACE_MODEL)
        self.detector = PhoneDetector(PHONE_MODELS[settings.detector])
        self.tracker = (
            VerifiedPhoneTracker(verifier, settings.phone_conf_threshold, settings.verify_interval_s)
            if verifier is not None
            else None
        )
        self.attention = AttentionTracker(settings.enter_s, settings.exit_s, settings.evidence_smoothing_s)
        self.face_loss = FaceLossLatch()
        self.camera: PinholeCamera | None = None
        self._last_ts = -1
        self._frame_no = 0
        self._phones: list[PhoneDetection] = []
        self._phones_at = 0.0

    @property
    def verifying(self) -> bool:
        return self.tracker is not None

    def process(self, frame_bgr: np.ndarray, now: float) -> Observation:
        s = self.s
        h, w = frame_bgr.shape[:2]
        if self.camera is None or (self.camera.width, self.camera.height) != (w, h):
            self.camera = PinholeCamera(w, h, s.vertical_fov_deg)

        ts = max(int(now * 1000), self._last_ts + 1)  # MediaPipe needs strictly increasing timestamps
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        face = self.faces.process(rgb, ts)
        if self._frame_no % s.detect_every == 0:
            self._phones, self._phones_at = self.detector.process(rgb, ts), now
        elif now - self._phones_at > s.phone_hold_s:
            self._phones = []
        self._frame_no += 1
        phones = self._phones

        if self.tracker is not None:
            self.tracker.min_det_score = s.phone_conf_threshold
            self.tracker.annotate(rgb, phones, now)
        result = evaluate(face, phones, self.camera, s, require_verification=self.tracker is not None)
        state = self.attention.update(self.face_loss.update(result), now)
        return Observation(face, phones, result, state, self.camera)

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()
        self.faces.close()
        self.detector.close()
