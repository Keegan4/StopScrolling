"""MediaPipe Object Detector wrapper, restricted to the COCO "cell phone" class."""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from stopscrolling.perception.types import PhoneDetection


class PhoneDetector:
    def __init__(
        self,
        model_path: Path,
        video: bool = True,
        min_score: float = 0.1,
        max_results: int = 3,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found - run: python tools/download_models.py --all")
        self._video = video
        options = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            score_threshold=min_score,  # low on purpose: the decision threshold lives in Settings
            max_results=max_results,
            category_allowlist=["cell phone"],
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def process(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> list[PhoneDetection]:
        """Phone detections, highest score first."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video:
            result = self._detector.detect_for_video(image, int(timestamp_ms))
        else:
            result = self._detector.detect(image)

        phones = []
        for det in result.detections:
            if not det.categories:
                continue
            bb = det.bounding_box
            phones.append(
                PhoneDetection(
                    score=float(det.categories[0].score),
                    box=(bb.origin_x, bb.origin_y, bb.width, bb.height),
                )
            )
        phones.sort(key=lambda p: p.score, reverse=True)
        return phones

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> PhoneDetector:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
