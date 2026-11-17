"""MediaPipe Face Landmarker wrapper."""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from stopscrolling.features.geometry import rotation_and_translation
from stopscrolling.perception.types import FaceObservation


class FaceTracker:
    """Tracks one face. Use ``video=True`` for webcam streams, ``False`` for still images."""

    def __init__(self, model_path: Path, video: bool = True) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found - run: python tools/download_models.py")
        self._video = video
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> FaceObservation | None:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video:
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        else:
            result = self._landmarker.detect(image)

        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return None
        landmarks = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=float)
        rotation, translation = rotation_and_translation(result.facial_transformation_matrixes[0])
        blendshapes = {}
        if result.face_blendshapes:
            blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        return FaceObservation(landmarks, rotation, translation, blendshapes)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
