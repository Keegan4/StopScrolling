"""Labelled webcam clips on disk: ``<clip_id>.mp4`` plus a ``<clip_id>.json`` sidecar.

The sidecar holds the label and the capture time of every frame, because webcams don't
deliver frames at a perfectly steady rate and the face tracker needs real timestamps.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from stopscrolling import __version__
from stopscrolling.data import labels

VIDEO_EXTS = (".mp4", ".avi")


@dataclass
class ClipMeta:
    clip_id: str
    label: str
    attentive: bool
    created: str  # local time, ISO 8601
    fps: float  # nominal rate the video file was written at
    width: int
    height: int
    video: str  # file name of the video next to this sidecar
    frame_count: int = 0
    duration_s: float = 0.0
    timestamps_s: list[float] = field(default_factory=list)  # capture time of each frame
    camera_index: int = 0
    app_version: str = __version__

    @classmethod
    def load(cls, path: Path) -> ClipMeta:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")


def _new_clip_id(out_dir: Path, label: str) -> str:
    base = f"{time.strftime('%Y%m%d-%H%M%S')}_{label}"
    clip_id, n = base, 1
    while any((out_dir / f"{clip_id}{ext}").exists() for ext in (*VIDEO_EXTS, ".json")):
        n += 1
        clip_id = f"{base}-{n}"
    return clip_id


class ClipWriter:
    """Writes one clip. Frames are stored un-mirrored, exactly as the camera sees them."""

    def __init__(self, out_dir: Path, label: str, fps: float, size: tuple[int, int], camera_index: int = 0):
        lbl = labels.get(label)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.size = size
        clip_id = _new_clip_id(self.out_dir, label)

        # mp4v ships with every OpenCV build; fall back to MJPG/AVI just in case.
        self.video_path = self.out_dir / f"{clip_id}.mp4"
        self._writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._writer.isOpened():
            self.video_path = self.out_dir / f"{clip_id}.avi"
            self._writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open a video writer in {self.out_dir}")

        self.meta = ClipMeta(
            clip_id=clip_id,
            label=lbl.name,
            attentive=lbl.attentive,
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
            fps=float(fps),
            width=size[0],
            height=size[1],
            video=self.video_path.name,
            camera_index=camera_index,
        )

    @property
    def sidecar_path(self) -> Path:
        return self.out_dir / f"{self.meta.clip_id}.json"

    def write(self, frame_bgr: np.ndarray, t: float) -> None:
        """Append a frame captured ``t`` seconds after the clip started."""
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        self._writer.write(frame_bgr)
        self.meta.timestamps_s.append(round(float(t), 4))

    def close(self) -> ClipMeta:
        self._writer.release()
        ts = self.meta.timestamps_s
        self.meta.frame_count = len(ts)
        self.meta.duration_s = round(ts[-1] - ts[0], 3) if len(ts) > 1 else 0.0
        self.meta.save(self.sidecar_path)
        return self.meta

    def discard(self) -> None:
        self._writer.release()
        for p in (self.video_path, self.sidecar_path):
            p.unlink(missing_ok=True)


def list_clips(clips_dir: Path) -> list[tuple[Path, ClipMeta]]:
    """(video path, metadata) for every clip with a sidecar, oldest first."""
    out = []
    for sidecar in sorted(Path(clips_dir).glob("*.json")):
        try:
            meta = ClipMeta.load(sidecar)
        except (ValueError, TypeError):
            continue
        video = sidecar.parent / meta.video
        if video.exists():
            out.append((video, meta))
    return out


def delete_clip(video: Path, meta: ClipMeta) -> None:
    video.unlink(missing_ok=True)
    (video.parent / f"{meta.clip_id}.json").unlink(missing_ok=True)
