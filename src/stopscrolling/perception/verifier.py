"""Second-opinion check on "cell phone" detections using zero-shot CLIP.

The COCO object detector has no "calculator" class, so calculators, remotes and similar
rectangles often come back as "cell phone" (a scientific calculator scores ~0.75). For each
candidate box we crop the image and ask CLIP which of several text descriptions fits best
(phone, calculator, remote, ...). Only candidates where "phone" wins count as phones.

Runs the quantized CLIP ViT-B/32 ONNX export with onnxruntime (no PyTorch needed).
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stopscrolling.perception.types import PhoneDetection

PHONE = "phone"

# Each class is described by a few prompts; their embeddings are averaged (prompt ensembling).
CLASS_PROMPTS: dict[str, list[str]] = {
    PHONE: [
        "a photo of a smartphone.",
        "a photo of a hand holding a smartphone.",
        "a photo of the back of a smartphone.",
        "a photo of a mobile phone in a phone case.",
    ],
    "calculator": [
        "a photo of a calculator.",
        "a photo of a scientific calculator.",
        "a photo of a hand holding a calculator.",
        "a photo of a calculator with many buttons.",
    ],
    "remote": ["a photo of a TV remote control.", "a photo of a remote control with buttons."],
    "book": ["a photo of a book.", "a photo of a notebook."],
    "pencil case": ["a photo of a pencil case."],
    "eraser": ["a photo of an eraser."],
    "wallet": ["a photo of a wallet."],
    "hand": ["a photo of a hand.", "a photo of fingers."],
}

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_INPUT = 224
_LOGIT_SCALE = 100.0  # CLIP's learned temperature


@dataclass
class PhoneCheck:
    p_phone: float  # probability the crop is a phone (vs. the other classes)
    label: str  # best-matching class
    label_prob: float


def square_crop(rgb: np.ndarray, box: tuple[int, int, int, int], margin: float = 1.25) -> np.ndarray:
    """Square crop around a box (with some context), padded at image borders."""
    x, y, w, h = box
    side = max(int(max(w, h) * margin), 8)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    ih, iw = rgb.shape[:2]
    # Crop the part inside the image, then pad only what falls outside it.
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, iw), min(y1, ih)
    if cx1 <= cx0 or cy1 <= cy0:
        return np.zeros((side, side, 3), dtype=rgb.dtype)
    crop = rgb[cy0:cy1, cx0:cx1]
    return cv2.copyMakeBorder(crop, cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1, cv2.BORDER_REPLICATE)


def preprocess(rgb_crop: np.ndarray) -> np.ndarray:
    img = cv2.resize(rgb_crop, (_INPUT, _INPUT), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])  # NCHW


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class PhoneVerifier:
    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        vision = model_dir / "vision_model_quantized.onnx"
        text = model_dir / "text_model_quantized.onnx"
        tokenizer = model_dir / "tokenizer.json"
        missing = [p.name for p in (vision, text, tokenizer) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing {missing} in {model_dir} - run: python tools/download_models.py")

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self._vision = ort.InferenceSession(str(vision), opts, providers=providers)

        # Text embeddings are computed once at start-up; the text model is then discarded.
        text_session = ort.InferenceSession(str(text), opts, providers=providers)
        tok = Tokenizer.from_file(str(tokenizer))
        self.labels = list(CLASS_PROMPTS)
        class_embeds = []
        for label in self.labels:
            embeds = []
            for prompt in CLASS_PROMPTS[label]:
                ids = np.array([tok.encode(prompt).ids], dtype=np.int64)
                (e,) = text_session.run(["text_embeds"], {"input_ids": ids})
                embeds.append(e[0] / np.linalg.norm(e[0]))
            mean = np.mean(embeds, axis=0)
            class_embeds.append(mean / np.linalg.norm(mean))
        self._text = np.stack(class_embeds).astype(np.float32)  # (classes, 512)
        self._phone_idx = self.labels.index(PHONE)

    def check_crop(self, rgb_crop: np.ndarray) -> PhoneCheck:
        (img_embed,) = self._vision.run(["image_embeds"], {"pixel_values": preprocess(rgb_crop)})
        v = img_embed[0] / np.linalg.norm(img_embed[0])
        logits = _LOGIT_SCALE * (self._text @ v)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best = int(np.argmax(probs))
        return PhoneCheck(float(probs[self._phone_idx]), self.labels[best], float(probs[best]))

    def check(self, rgb: np.ndarray, box: tuple[int, int, int, int]) -> PhoneCheck:
        return self.check_crop(square_crop(rgb, box))


@dataclass(eq=False)
class _Track:
    box: tuple[int, int, int, int]
    seen_at: float
    check: PhoneCheck | None = None
    checked_at: float = -math.inf


class VerifiedPhoneTracker:
    """Verifies phone candidates at a limited rate and carries the result across frames.

    Candidates are matched to tracks by box overlap (IoU), so a box only needs checking
    every ``interval_s``; repeated checks are blended so one odd crop can't flip it. With
    ``background=True`` the (~200 ms) CLIP call runs on a worker thread and results appear
    a frame or two later, keeping the video smooth.
    """

    def __init__(
        self,
        verifier: PhoneVerifier,
        min_det_score: float,
        interval_s: float = 0.4,
        smoothing: float = 0.5,
        keep_s: float = 1.0,
        background: bool = True,
    ) -> None:
        self.verifier = verifier
        self.min_det_score = min_det_score
        self.interval_s = interval_s
        self.smoothing = smoothing
        self.keep_s = keep_s
        self._tracks: list[_Track] = []
        self._lock = threading.Lock()
        self._busy = False
        self._queue: queue.Queue | None = None
        if background:
            self._queue = queue.Queue(maxsize=1)
            self._thread = threading.Thread(target=self._worker, name="phone-verifier", daemon=True)
            self._thread.start()

    def _apply(self, track: _Track, fresh: PhoneCheck, now: float) -> None:
        with self._lock:
            if track.check is not None:
                s = self.smoothing
                fresh = PhoneCheck(s * track.check.p_phone + (1 - s) * fresh.p_phone, fresh.label, fresh.label_prob)
            track.check = fresh
            track.checked_at = now

    def _worker(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            if item is None:
                return
            track, crop = item
            try:
                self._apply(track, self.verifier.check_crop(crop), time.monotonic())
            finally:
                with self._lock:
                    self._busy = False

    def annotate(self, rgb: np.ndarray, phones: list[PhoneDetection], now: float) -> None:
        """Fill ``phone.check`` for candidates scoring >= ``min_det_score`` (in place)."""
        candidates = sorted((p for p in phones if p.score >= self.min_det_score), key=lambda p: -p.score)

        # Match candidates to existing tracks.
        unused = [t for t in self._tracks if now - t.seen_at <= self.keep_s]
        matched: list[tuple[PhoneDetection, _Track]] = []
        for phone in candidates:
            best = max(unused, key=lambda t: iou(t.box, phone.box), default=None)
            if best is None or iou(best.box, phone.box) < 0.3:
                best = _Track(phone.box, now)
            else:
                unused.remove(best)
                best.box, best.seen_at = phone.box, now
            matched.append((phone, best))
        self._tracks = [t for _, t in matched] + unused  # unmatched tracks linger for keep_s

        # Schedule checks for stale tracks (highest detector score first).
        for _, track in matched:
            if now - track.checked_at < self.interval_s:
                continue
            crop = square_crop(rgb, track.box)
            if self._queue is None:
                self._apply(track, self.verifier.check_crop(crop), now)
                continue
            with self._lock:
                if self._busy:
                    break
                self._busy = True
            self._queue.put((track, crop))
            break

        with self._lock:
            for phone, track in matched:
                phone.check = track.check

    def close(self) -> None:
        if self._queue is not None:
            self._queue.put(None)
