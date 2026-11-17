"""Interactive recorder for labelled training clips.

Pick what you're about to do (1-5), press SPACE, get into position during the countdown,
then act it out naturally. SPACE again (or the time limit) stops and saves the clip.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stopscrolling.data import labels
from stopscrolling.data.clips import ClipMeta, ClipWriter, delete_clip, list_clips

IDLE, COUNTDOWN, RECORDING = "idle", "countdown", "recording"
WINDOW = "StopScrolling - record training clips"


@dataclass
class RecorderConfig:
    out_dir: Path
    max_seconds: float = 60.0
    countdown_s: float = 3.0
    camera_index: int = 0


class Recorder:
    """Recording state machine, independent of the camera and window (so it's testable)."""

    def __init__(self, cfg: RecorderConfig, start_label: str = "screen") -> None:
        self.cfg = cfg
        self.label = labels.get(start_label)
        self.state = IDLE
        self.state_since = 0.0
        self.writer: ClipWriter | None = None
        self.rec_start = 0.0
        self.last_saved: tuple[Path, ClipMeta] | None = None
        self.message = ""
        self._frame_times: deque[float] = deque(maxlen=60)
        self.totals = self._load_totals()

    # ---------------------------------------------------------------- bookkeeping

    def _load_totals(self) -> dict[str, float]:
        totals = dict.fromkeys(labels.BY_NAME, 0.0)
        for _, meta in list_clips(self.cfg.out_dir):
            if meta.label in totals:
                totals[meta.label] += meta.duration_s
        return totals

    def measured_fps(self) -> float:
        if len(self._frame_times) < 5:
            return 30.0
        gaps = [b - a for a, b in zip(self._frame_times, list(self._frame_times)[1:]) if b > a]
        fps = 1.0 / statistics.median(gaps) if gaps else 30.0
        return float(min(max(round(fps), 5), 60))

    # ---------------------------------------------------------------- events

    def handle_key(self, key: str, now: float) -> bool:
        """Process one key. Returns False when the user wants to quit."""
        if key in ("q", "esc"):
            if self.state == RECORDING:
                self.stop(now)
            return False
        if key == " ":
            if self.state == IDLE:
                self.state, self.state_since = COUNTDOWN, now
                self.message = ""
            elif self.state == COUNTDOWN:
                self.state = IDLE  # cancel
            else:
                self.stop(now)
        elif key in labels.BY_KEY and self.state == IDLE:
            self.label = labels.BY_KEY[key]
        elif key in ("x", "backspace") and self.state == IDLE and self.last_saved:
            video, meta = self.last_saved
            delete_clip(video, meta)
            self.totals[meta.label] = max(0.0, self.totals[meta.label] - meta.duration_s)
            self.message = f"deleted {meta.clip_id}"
            self.last_saved = None
        return True

    def on_frame(self, frame_bgr: np.ndarray, now: float) -> None:
        """Feed every new camera frame (BGR, un-mirrored)."""
        self._frame_times.append(now)
        if self.state == COUNTDOWN and now - self.state_since >= self.cfg.countdown_s:
            h, w = frame_bgr.shape[:2]
            self.writer = ClipWriter(self.cfg.out_dir, self.label.name, self.measured_fps(), (w, h), self.cfg.camera_index)
            self.state, self.rec_start = RECORDING, now
        if self.state == RECORDING and self.writer is not None:
            self.writer.write(frame_bgr, now - self.rec_start)
            if now - self.rec_start >= self.cfg.max_seconds:
                self.stop(now)

    def stop(self, now: float) -> ClipMeta | None:
        if self.writer is None:
            self.state = IDLE
            return None
        writer, self.writer = self.writer, None
        self.state = IDLE
        if len(writer.meta.timestamps_s) < 2:
            writer.discard()
            self.message = "clip too short - discarded"
            return None
        meta = writer.close()
        self.totals[meta.label] += meta.duration_s
        self.last_saved = (writer.video_path, meta)
        self.message = f"saved {meta.clip_id} ({meta.duration_s:.1f}s, {meta.frame_count} frames)"
        return meta

    # ---------------------------------------------------------------- drawing

    def render(self, frame_bgr: np.ndarray, now: float) -> np.ndarray:
        vis = cv2.flip(frame_bgr, 1)  # mirror for display only
        h, w = vis.shape[:2]
        lbl = self.label

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 64), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 128), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

        _put(vis, f"[{lbl.key}] {lbl.name.upper()}", (10, 28), lbl.color, 0.8, 2)
        _put(vis, lbl.description, (10, 52), (230, 230, 230), 0.42)

        if self.state == IDLE:
            _put(vis, "SPACE to record", (w - 175, 28), (255, 255, 255), 0.6, 2)
        elif self.state == COUNTDOWN:
            left = max(0.0, self.cfg.countdown_s - (now - self.state_since))
            _put(vis, f"get ready... {left:0.1f}", (w - 210, 28), (0, 220, 255), 0.65, 2)
            _put(vis, f"{int(left) + 1}", (w // 2 - 30, h // 2 + 30), (0, 220, 255), 3.0, 6)
        else:
            t = now - self.rec_start
            if int(now * 2) % 2 == 0:
                cv2.circle(vis, (w - 150, 22), 9, (0, 0, 255), -1)
            _put(vis, f"REC {t:4.1f}/{self.cfg.max_seconds:.0f}s", (w - 135, 28), (0, 0, 255), 0.6, 2)
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

        # Seconds recorded so far per label: aim for a balanced dataset.
        y = h - 108
        for i, lb in enumerate(labels.LABELS):
            x = 10 + i * (w - 20) // len(labels.LABELS)
            selected = lb is lbl
            _put(vis, f"{lb.key} {lb.name}", (x, y), lb.color if selected else (200, 200, 200), 0.45, 2 if selected else 1)
            _put(vis, f"{self.totals.get(lb.name, 0.0):5.0f}s", (x, y + 20), (200, 200, 200), 0.45)
        if self.message:
            _put(vis, self.message, (10, h - 44), (0, 220, 255), 0.45)
        _put(vis, "1-5 label | SPACE start/stop | x delete last clip | q quit", (10, h - 14), (220, 220, 220), 0.45)
        return vis


def _put(img, text, org, color, scale=0.5, thickness=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


_KEYMAP = {27: "esc", 32: " ", 8: "backspace"}


def run(cfg: RecorderConfig, start_label: str, width: int, height: int) -> int:
    from stopscrolling.capture.camera import Camera

    rec = Recorder(cfg, start_label)
    print(f"Saving clips to {cfg.out_dir}")
    with Camera(cfg.camera_index, width, height) as cam:
        last_id = -1
        frame = None
        printed = ""
        try:
            while True:
                got = cam.read()
                now = time.monotonic()
                if got is not None and got[0] != last_id:
                    last_id, frame = got
                    rec.on_frame(frame, now)
                if frame is not None:
                    cv2.imshow(WINDOW, rec.render(frame, now))
                k = cv2.waitKey(1) & 0xFF
                if k != 255 and not rec.handle_key(_KEYMAP.get(k, chr(k).lower()), now):
                    break
                if rec.message != printed:
                    printed = rec.message
                    if printed:
                        print(printed)
                if frame is not None and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if rec.state == RECORDING:
                meta = rec.stop(time.monotonic())
                if meta:
                    print(rec.message)
            cv2.destroyAllWindows()
    total = sum(rec.totals.values())
    print("Recorded so far: " + ", ".join(f"{k} {v:.0f}s" for k, v in rec.totals.items()) + f" (total {total:.0f}s)")
    return 0
