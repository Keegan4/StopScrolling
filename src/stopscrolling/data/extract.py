"""Split recorded clips into frames, landmarks and features for training.

For each clip, ``data/processed/<clip_id>/`` gets:

    frames/000123.jpg   sampled video frames (file name = frame index in the clip)
    landmarks.npz       per sampled frame: 478 face landmarks, 52 blendshapes, head
                        rotation/translation, phone boxes/scores - NaN where missing
    features.csv        one row per sampled frame: clip_id, frame, t, label, attentive,
                        then stopscrolling.features.extractor.FEATURE_NAMES
    meta.json           the clip's metadata plus the extraction settings

``data/processed/dataset.csv`` concatenates every clip's features.csv.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from stopscrolling import __version__
from stopscrolling.config import FACE_MODEL, PHONE_MODELS, Settings
from stopscrolling.data.clips import ClipMeta
from stopscrolling.detection.facing import evaluate
from stopscrolling.features.extractor import FEATURE_NAMES, frame_features
from stopscrolling.features.geometry import PinholeCamera
from stopscrolling.perception.verifier import PhoneVerifier, VerifiedPhoneTracker

NUM_LANDMARKS = 478
MAX_PHONES = 3
META_COLUMNS = ["clip_id", "frame", "t", "label", "attentive"]


@dataclass
class ExtractOptions:
    fps: float = 10.0  # sample rate; 0 = every frame
    save_frames: bool = True
    jpeg_quality: int = 90
    force: bool = False


@dataclass
class ClipSummary:
    clip_id: str
    label: str
    frames: int
    face_rate: float  # fraction of sampled frames with a face
    phone_rate: float  # fraction with a confirmed phone
    looking_rate: float  # fraction the rule-based check says "looking at phone"
    seconds: float
    skipped: bool = False


def sample_indices(timestamps: list[float], fps: float) -> list[int]:
    """Frame indices that sample the clip at ~``fps`` using real capture times."""
    if fps <= 0:
        return list(range(len(timestamps)))
    step, keep = 1.0 / fps, []
    next_t: float | None = None
    for i, t in enumerate(timestamps):
        if next_t is None or t >= next_t - 1e-6:
            keep.append(i)
            # Stay on the fixed grid, but resync after a capture gap instead of bursting.
            next_t = t + step if next_t is None or next_t + step <= t else next_t + step
    return keep


def _timestamps(meta: ClipMeta, frame_count: int) -> list[float]:
    ts = list(meta.timestamps_s)
    if len(ts) >= frame_count > 0:
        return ts[:frame_count]
    fps = meta.fps or 30.0
    return [i / fps for i in range(frame_count)]


def extract_clip(
    video: Path,
    meta: ClipMeta,
    out_root: Path,
    settings: Settings,
    verifier: PhoneVerifier | None,
    opts: ExtractOptions,
) -> ClipSummary:
    from stopscrolling.perception.face import FaceTracker
    from stopscrolling.perception.phone import PhoneDetector

    out = Path(out_root) / meta.clip_id
    if (out / "features.csv").exists() and not opts.force:
        return _summary_from_existing(out, meta)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or meta.frame_count
    timestamps = _timestamps(meta, n_video)
    wanted = set(sample_indices(timestamps, opts.fps))

    frames_dir = out / "frames"
    if opts.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("*.jpg"):
            old.unlink()
    else:
        out.mkdir(parents=True, exist_ok=True)

    tracker = VerifiedPhoneTracker(verifier, settings.phone_conf_threshold, settings.verify_interval_s, background=False) if verifier else None
    rows: list[dict] = []
    arrays: dict[str, list] = {k: [] for k in ("frame", "t", "landmarks", "blendshapes", "rotation", "translation", "phone_boxes", "phone_scores", "phone_p")}
    bs_names: list[str] | None = None
    camera = None
    last_ts = -1

    with FaceTracker(FACE_MODEL, video=True) as faces, PhoneDetector(PHONE_MODELS[settings.detector], video=True) as det:
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx not in wanted or idx >= len(timestamps):
                continue
            t = timestamps[idx]
            h, w = frame.shape[:2]
            if camera is None:
                camera = PinholeCamera(w, h, settings.vertical_fov_deg)
            ts = max(int(t * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face = faces.process(rgb, ts)
            phones = det.process(rgb, ts)
            if tracker is not None:
                tracker.annotate(rgb, phones, t)
            result = evaluate(face, phones, camera, settings, require_verification=tracker is not None)
            feats = frame_features(face, phones, result, (w, h))
            rows.append({"clip_id": meta.clip_id, "frame": idx, "t": round(t, 4), "label": meta.label,
                         "attentive": int(meta.attentive), **feats})

            if opts.save_frames:
                cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, opts.jpeg_quality])

            arrays["frame"].append(idx)
            arrays["t"].append(t)
            if face is not None:
                if bs_names is None and face.blendshapes:
                    bs_names = list(face.blendshapes)
                lm = np.full((NUM_LANDMARKS, 3), np.nan, np.float32)
                n = min(NUM_LANDMARKS, len(face.landmarks))
                lm[:n] = face.landmarks[:n]
                arrays["landmarks"].append(lm)
                arrays["blendshapes"].append(face.blendshapes)
                arrays["rotation"].append(face.rotation.astype(np.float32))
                arrays["translation"].append(face.translation.astype(np.float32))
            else:
                arrays["landmarks"].append(np.full((NUM_LANDMARKS, 3), np.nan, np.float32))
                arrays["blendshapes"].append(None)
                arrays["rotation"].append(np.full((3, 3), np.nan, np.float32))
                arrays["translation"].append(np.full(3, np.nan, np.float32))
            boxes = np.full((MAX_PHONES, 4), np.nan, np.float32)
            scores = np.full(MAX_PHONES, np.nan, np.float32)
            p_phone = np.full(MAX_PHONES, np.nan, np.float32)
            for j, p in enumerate(sorted(phones, key=lambda p: -p.score)[:MAX_PHONES]):
                boxes[j], scores[j] = p.box, p.score
                if p.check is not None:
                    p_phone[j] = p.check.p_phone
            arrays["phone_boxes"].append(boxes)
            arrays["phone_scores"].append(scores)
            arrays["phone_p"].append(p_phone)
    cap.release()

    _write_features(out / "features.csv", rows)
    _write_landmarks(out / "landmarks.npz", arrays, bs_names or [])
    summary = _summarise(meta, rows, timestamps)
    info = asdict(meta)
    info.pop("timestamps_s", None)
    info["extraction"] = {
        **asdict(opts),
        "frames_sampled": len(rows),
        "frames_in_video": n_video,
        "verified_phones": verifier is not None,
        "detector": settings.detector,
        "app_version": __version__,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_names": FEATURE_NAMES,
    }
    (out / "meta.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return summary


def _write_features(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=META_COLUMNS + FEATURE_NAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_landmarks(path: Path, arrays: dict[str, list], bs_names: list[str]) -> None:
    n = len(arrays["frame"])
    bs = np.full((n, len(bs_names)), np.nan, np.float32)
    for i, d in enumerate(arrays["blendshapes"]):
        if d:
            bs[i] = [d.get(name, np.nan) for name in bs_names]

    def stack(key, shape):
        return np.stack(arrays[key]).astype(np.float32) if n else np.zeros((0, *shape), np.float32)

    np.savez_compressed(
        path,
        frame=np.asarray(arrays["frame"], np.int32),
        t=np.asarray(arrays["t"], np.float32),
        landmarks=stack("landmarks", (NUM_LANDMARKS, 3)),  # normalised x, y in [0, 1], relative z
        blendshapes=bs,
        blendshape_names=np.asarray(bs_names),
        rotation=stack("rotation", (3, 3)),  # head rotation, camera space
        translation=stack("translation", (3,)),  # head position, cm
        phone_boxes=stack("phone_boxes", (MAX_PHONES, 4)),  # x, y, w, h pixels
        phone_scores=stack("phone_scores", (MAX_PHONES,)),
        phone_p=stack("phone_p", (MAX_PHONES,)),  # CLIP probability it's a phone
    )


def _rate(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] == 1.0 for r in rows])) if rows else 0.0


def _summarise(meta: ClipMeta, rows: list[dict], timestamps: list[float]) -> ClipSummary:
    seconds = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    return ClipSummary(meta.clip_id, meta.label, len(rows), _rate(rows, "face_present"),
                       _rate(rows, "phone_confirmed"), _rate(rows, "looking_at_phone"), seconds)


def _summary_from_existing(out: Path, meta: ClipMeta) -> ClipSummary:
    with open(out / "features.csv", newline="", encoding="utf-8") as fh:
        rows = [{k: float(v) if k in FEATURE_NAMES else v for k, v in r.items()} for r in csv.DictReader(fh)]
    s = _summarise(meta, rows, meta.timestamps_s)
    s.skipped = True
    return s


def build_dataset(processed_root: Path) -> tuple[Path, int]:
    """Concatenate every clip's features.csv into processed_root/dataset.csv."""
    processed_root = Path(processed_root)
    out = processed_root / "dataset.csv"
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(META_COLUMNS + FEATURE_NAMES)
        for f in sorted(processed_root.glob("*/features.csv")):
            with open(f, newline="", encoding="utf-8") as src:
                reader = csv.reader(src)
                header = next(reader, None)
                if header != META_COLUMNS + FEATURE_NAMES:
                    print(f"skipping {f.parent.name}: features are from an older version (re-extract with --force)")
                    continue
                for row in reader:
                    writer.writerow(row)
                    n += 1
    return out, n
