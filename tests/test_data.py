import json
import math

import cv2
import numpy as np
import pytest

from stopscrolling.config import FACE_MODEL, PHONE_MODELS, Settings
from stopscrolling.data import labels
from stopscrolling.data.clips import ClipMeta, ClipWriter, list_clips
from stopscrolling.data.extract import ExtractOptions, build_dataset, extract_clip, sample_indices
from stopscrolling.data.recorder import IDLE, RECORDING, Recorder, RecorderConfig
from stopscrolling.detection.facing import evaluate
from stopscrolling.features.extractor import FEATURE_NAMES, frame_features, to_vector
from stopscrolling.features.geometry import PinholeCamera
from stopscrolling.perception.types import FaceObservation, PhoneDetection
from stopscrolling.perception.verifier import PhoneCheck


def synthetic_frames(n, size=(160, 120)):
    w, h = size
    for i in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(img, (10 + i * 5 % (w - 20), h // 2), 8, (0, 200, 255), -1)
        yield img


# --------------------------------------------------------------------------- sampling


def test_sample_indices_steady_rate():
    ts = [i / 30 for i in range(90)]  # 3 s at 30 fps
    idx = sample_indices(ts, 10)
    assert len(idx) == 30
    assert idx[:4] == [0, 3, 6, 9]


def test_sample_indices_handles_gaps_and_all_frames():
    ts = [0.0, 0.1, 0.2, 2.0, 2.1]  # 1.8 s capture gap
    assert sample_indices(ts, 10) == [0, 1, 2, 3, 4]
    assert sample_indices(ts, 0) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- clips


def test_clip_writer_round_trip(tmp_path):
    w = ClipWriter(tmp_path, "paper", fps=15, size=(160, 120))
    for i, frame in enumerate(synthetic_frames(15)):
        w.write(frame, i / 15)
    meta = w.close()

    assert meta.label == "paper" and meta.attentive is True
    assert meta.frame_count == 15
    assert meta.duration_s == pytest.approx(14 / 15, abs=1e-3)
    on_disk = json.loads(w.sidecar_path.read_text())
    assert on_disk["timestamps_s"][:2] == [0.0, round(1 / 15, 4)]

    cap = cv2.VideoCapture(str(w.video_path))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 15
    assert [m.clip_id for _, m in list_clips(tmp_path)] == [meta.clip_id]


def test_clip_writer_rejects_unknown_label(tmp_path):
    with pytest.raises(ValueError):
        ClipWriter(tmp_path, "napping", fps=15, size=(160, 120))


def test_clip_ids_do_not_collide(tmp_path):
    a = ClipWriter(tmp_path, "phone", 15, (160, 120))
    b = ClipWriter(tmp_path, "phone", 15, (160, 120))
    assert a.meta.clip_id != b.meta.clip_id
    a.discard()
    b.discard()
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- recorder


def drive(rec, seconds, start, fps=30):
    frames = synthetic_frames(int(seconds * fps))
    t = start
    for f in frames:
        rec.on_frame(f, t)
        t += 1 / fps
    return t


def test_recorder_countdown_record_autostop_and_delete(tmp_path):
    rec = Recorder(RecorderConfig(tmp_path, max_seconds=1.0, countdown_s=0.5))
    t = drive(rec, 0.5, 0.0)  # warm up: fps measurement
    assert rec.handle_key("3", t) and rec.label.name == "phone"
    rec.handle_key(" ", t)
    t = drive(rec, 0.4, t)
    assert rec.state != RECORDING  # still counting down
    t = drive(rec, 0.3, t)
    assert rec.state == RECORDING
    rec.handle_key("1", t)  # label can't change mid-clip
    assert rec.label.name == "phone"
    t = drive(rec, 1.2, t)  # passes max_seconds -> auto stop
    assert rec.state == IDLE

    clips = list_clips(tmp_path)
    assert len(clips) == 1
    meta = clips[0][1]
    assert meta.label == "phone" and meta.fps == 30
    assert meta.duration_s == pytest.approx(1.0, abs=0.05)
    assert rec.totals["phone"] == pytest.approx(meta.duration_s)

    rec.handle_key("x", t)  # delete last clip
    assert list_clips(tmp_path) == []
    assert rec.totals["phone"] == 0


def test_recorder_cancel_countdown_and_quit_saves(tmp_path):
    rec = Recorder(RecorderConfig(tmp_path, countdown_s=0.2))
    rec.handle_key(" ", 0.0)
    rec.handle_key(" ", 0.1)  # cancel during countdown
    t = drive(rec, 0.5, 0.1)
    assert rec.state == IDLE and list_clips(tmp_path) == []

    rec.handle_key(" ", t)
    t = drive(rec, 1.0, t)
    assert rec.state == RECORDING
    assert rec.handle_key("q", t) is False  # quitting saves the clip in progress
    assert len(list_clips(tmp_path)) == 1


def test_recorder_render_draws_without_error(tmp_path):
    rec = Recorder(RecorderConfig(tmp_path))
    frame = next(synthetic_frames(1, (640, 480)))
    assert rec.render(frame, 0.0).shape == (480, 640, 3)


def test_labels_are_well_formed():
    assert [lb.key for lb in labels.LABELS] == ["1", "2", "3", "4", "5"]
    assert {lb.name for lb in labels.LABELS if lb.attentive} == {"screen", "paper"}


# --------------------------------------------------------------------------- features


def test_frame_features_with_face_and_phone():
    cam = PinholeCamera(640, 480)
    lm = np.full((478, 3), 0.5)
    lm[:, 0] = np.linspace(0.4, 0.6, 478)
    face = FaceObservation(lm, np.eye(3), np.array([0.0, 0.0, -50.0]), {"eyeBlinkLeft": 0.2, "jawOpen": 0.1})
    phone = PhoneDetection(0.8, (300, 350, 40, 80), PhoneCheck(0.9, "phone", 0.9))
    s = Settings()
    result = evaluate(face, [phone], cam, s, require_verification=True)
    f = frame_features(face, [phone], result, (640, 480))

    assert list(f) == FEATURE_NAMES
    assert f["face_present"] == 1 and f["face_width"] == pytest.approx(0.2)
    assert f["head_dist_cm"] == pytest.approx(50) and f["head_roll"] == pytest.approx(0)
    assert f["bs_eyeBlinkLeft"] == pytest.approx(0.2) and math.isnan(f["bs_eyeBlinkRight"])
    assert f["phone_det"] == pytest.approx(0.8) and f["phone_p"] == pytest.approx(0.9)
    assert f["phone_confirmed"] == 1 and f["phone_cx"] == pytest.approx(320 / 640)
    assert to_vector(f).shape == (len(FEATURE_NAMES),)


def test_frame_features_without_face_or_phone():
    result = evaluate(None, [], PinholeCamera(640, 480), Settings())
    f = frame_features(None, [], result, (640, 480))
    assert f["face_present"] == 0 and f["phone_det"] == 0 and f["phone_confirmed"] == 0
    assert math.isnan(f["head_yaw"]) and math.isnan(f["angle_to_phone"]) and math.isnan(f["phone_p"])


# --------------------------------------------------------------------------- extraction


@pytest.mark.skipif(not (FACE_MODEL.exists() and PHONE_MODELS["lite0"].exists()), reason="models not downloaded")
def test_extract_clip_end_to_end(tmp_path):
    clips_dir, out = tmp_path / "clips", tmp_path / "processed"
    w = ClipWriter(clips_dir, "absent", fps=30, size=(160, 120))
    for i, frame in enumerate(synthetic_frames(60)):  # 2 s
        w.write(frame, i / 30)
    meta = w.close()

    s = extract_clip(w.video_path, meta, out, Settings(), None, ExtractOptions(fps=10))
    assert s.frames == 20 and s.face_rate == 0 and not s.skipped
    clip_out = out / meta.clip_id
    assert len(list((clip_out / "frames").glob("*.jpg"))) == 20

    npz = np.load(clip_out / "landmarks.npz")
    assert npz["landmarks"].shape == (20, 478, 3) and np.isnan(npz["landmarks"]).all()
    assert npz["phone_boxes"].shape == (20, 3, 4)
    assert list(npz["frame"][:3]) == [0, 3, 6]
    assert json.loads((clip_out / "meta.json").read_text())["extraction"]["frames_sampled"] == 20

    again = extract_clip(w.video_path, ClipMeta.load(w.sidecar_path), out, Settings(), None, ExtractOptions(fps=10))
    assert again.skipped  # already done

    path, n = build_dataset(out)
    assert n == 20

    from stopscrolling.data.dataset import load_dataset

    ds = load_dataset(path)
    assert ds.X.shape == (20, len(FEATURE_NAMES))
    assert set(ds.labels) == {"absent"} and set(ds.attentive) == {0}
    assert set(ds.groups) == {meta.clip_id}
