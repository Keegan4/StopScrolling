"""StopScrolling app: watch the webcam, alert (sound + meme popups) while you're on your phone.

    python -m stopscrolling              # runs in the background (console only), Ctrl+C to quit
    python -m stopscrolling --preview    # with the debug camera window
    python tools/debug_viewer.py         # same as --preview

Preview keys: q/Esc quit, s snooze alerts (press again to resume), +/- detector threshold,
[ ] facing tolerance, m mirror.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from stopscrolling import __version__
from stopscrolling.alerts import AlertConfig, AlertController
from stopscrolling.config import CLIP_DIR, FACE_MODEL, MEMES_DIR, MODELS_DIR, PHONE_MODELS, SOUNDS_DIR, Settings
from stopscrolling.detection.attention import AttentionState
from stopscrolling.detection.facing import evaluate
from stopscrolling.features.geometry import PinholeCamera
from stopscrolling.overlay import render
from stopscrolling.perception.verifier import PhoneVerifier, VerifiedPhoneTracker

WINDOW = "StopScrolling"


def parse_args(argv: list[str] | None, preview_default: bool) -> tuple[argparse.Namespace, Settings]:
    s = Settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    view = p.add_mutually_exclusive_group()
    view.add_argument("--preview", dest="preview", action="store_true", default=preview_default, help="show the camera window")
    view.add_argument("--no-preview", dest="preview", action="store_false", help="run without the camera window")

    g = p.add_argument_group("camera")
    g.add_argument("--camera", type=int, default=s.camera_index)
    g.add_argument("--width", type=int, default=s.frame_width)
    g.add_argument("--height", type=int, default=s.frame_height)
    g.add_argument("--no-mirror", action="store_true", help="show the raw (un-mirrored) camera image")

    g = p.add_argument_group("detection")
    g.add_argument("--detector", choices=sorted(PHONE_MODELS), default=s.detector)
    g.add_argument("--detect-every", type=int, default=s.detect_every, help="run phone detector every N frames")
    g.add_argument("--conf", type=float, default=s.phone_conf_threshold, help="detector score threshold")
    g.add_argument("--verify-conf", type=float, default=s.phone_verify_threshold, help="CLIP phone threshold")
    g.add_argument("--no-verify", action="store_true", help="skip the phone-vs-calculator check")
    g.add_argument("--tolerance", type=float, default=s.facing_tolerance_deg, help="gaze tolerance, degrees")
    g.add_argument("--enter-s", type=float, default=s.enter_s, help="seconds of looking before it counts")
    g.add_argument("--exit-s", type=float, default=s.exit_s, help="seconds of looking away before it clears")

    g = p.add_argument_group("alerts")
    g.add_argument("--no-sound", action="store_true")
    g.add_argument("--no-popups", action="store_true")
    g.add_argument("--max-popups", type=int, default=s.max_popups)
    g.add_argument("--escalate-s", type=float, default=s.escalate_s, help="get louder every N seconds")

    g = p.add_argument_group("still image")
    g.add_argument("--image", type=Path, help="process a single image instead of the webcam")
    g.add_argument("--out", type=Path, help="with --image: save the annotated image here")

    p.add_argument("--version", action="version", version=f"StopScrolling {__version__}")
    p.add_argument("--self-test", action="store_true", help="check models and libraries load (no camera), then exit")
    args = p.parse_args(argv)

    s.camera_index, s.frame_width, s.frame_height = args.camera, args.width, args.height
    s.detector, s.detect_every = args.detector, max(1, args.detect_every)
    s.phone_conf_threshold, s.phone_verify_threshold = args.conf, args.verify_conf
    s.verify_phones = not args.no_verify
    s.facing_tolerance_deg = args.tolerance
    s.enter_s, s.exit_s = args.enter_s, args.exit_s
    s.alert_sound, s.alert_popups = not args.no_sound, not args.no_popups
    s.max_popups, s.escalate_s = args.max_popups, args.escalate_s
    return args, s


def load_verifier(s: Settings) -> PhoneVerifier | None:
    if not s.verify_phones:
        return None
    try:
        return PhoneVerifier(CLIP_DIR)
    except (FileNotFoundError, ImportError) as e:
        print(f"Phone verification disabled: {e}", file=sys.stderr)
        return None


def run_image(args: argparse.Namespace, s: Settings) -> int:
    from stopscrolling.perception.face import FaceTracker
    from stopscrolling.perception.phone import PhoneDetector

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"Could not read {args.image}", file=sys.stderr)
        return 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera = PinholeCamera(frame.shape[1], frame.shape[0], s.vertical_fov_deg)
    verifier = load_verifier(s)
    with FaceTracker(FACE_MODEL, video=False) as faces, PhoneDetector(PHONE_MODELS[s.detector], video=False) as det:
        face = faces.process(rgb)
        phones = det.process(rgb)
    if verifier is not None:
        VerifiedPhoneTracker(verifier, s.phone_conf_threshold, background=False).annotate(rgb, phones, 0.0)
    result = evaluate(face, phones, camera, s, require_verification=verifier is not None)

    r = lambda v: "--" if v is None else f"{v:.1f}"  # noqa: E731
    print(f"status         {result.status.value}")
    print(f"head yaw/pitch {r(result.head_yaw)} / {r(result.head_pitch)}")
    print(f"gaze yaw/pitch {r(result.gaze_yaw)} / {r(result.gaze_pitch)}")
    for p in phones:
        chk = f"  clip p_phone={p.check.p_phone:.2f} ({p.check.label})" if p.check else ""
        print(f"candidate      det={p.score:.2f}{chk}")
    print(f"angle_to_phone {r(result.angle_to_phone)} (tolerance {r(result.tolerance_deg)})")

    # A still image has no duration, so show the per-frame result as the decision.
    attention = AttentionState(result.looking_at_phone, float(result.looking_at_phone), 0.0)
    vis = render(frame, face, phones, result, attention, s, verifier is not None, mirror=not args.no_mirror)
    if args.out:
        cv2.imwrite(str(args.out), vis)
        print(f"saved {args.out}")
    else:
        cv2.imshow(WINDOW, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _alert_note(alerts: AlertController, now: float) -> str:
    if not alerts.enabled:
        return "alerts off"
    left = alerts.snooze_remaining(now)
    if left > 0:
        return f"alerts snoozed {int(left) // 60}:{int(left) % 60:02d} (s to resume)"
    return "alerts ON (s to snooze)"


def run_live(args: argparse.Namespace, s: Settings) -> int:
    from stopscrolling.capture.camera import Camera
    from stopscrolling.pipeline import PhoneWatcher

    mirror = not args.no_mirror
    alerts = AlertController(
        AlertConfig(
            sound=s.alert_sound,
            popups=s.alert_popups,
            max_popups=s.max_popups,
            escalate_s=s.escalate_s,
            memes_dir=str(MEMES_DIR),
            sounds_dir=str(SOUNDS_DIR),
        )
    )
    watcher = PhoneWatcher(s, load_verifier(s))
    print("StopScrolling is watching. " + ("Press q in the window to quit." if args.preview else "Ctrl+C to quit."))

    try:
        with Camera(s.camera_index, s.frame_width, s.frame_height) as cam:
            last_id = -1
            fps, fps_t0, fps_n = None, time.monotonic(), 0
            was_on_phone, on_since = False, 0.0

            while True:
                got = cam.read()
                if got is None or got[0] == last_id:
                    if args.preview:
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                    else:
                        time.sleep(0.005)
                    continue
                last_id, frame = got
                now = time.monotonic()

                obs = watcher.process(frame, now)
                alerts.update(obs.state.on_phone, now)

                if obs.state.on_phone != was_on_phone:
                    stamp = time.strftime("%H:%M:%S")
                    if obs.state.on_phone:
                        on_since = now
                        print(f"[{stamp}] on your phone - get back to work!")
                    else:
                        print(f"[{stamp}] back to work (on phone for {now - on_since:.1f}s)")
                    was_on_phone = obs.state.on_phone

                if not args.preview:
                    continue

                fps_n += 1
                if now - fps_t0 >= 1.0:
                    fps, fps_t0, fps_n = fps_n / (now - fps_t0), now, 0
                vis = render(
                    frame, obs.face, obs.phones, obs.result, obs.state, s, watcher.verifying, mirror, fps,
                    note=_alert_note(alerts, now),
                )
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s") and alerts.enabled:
                    if alerts.snooze_remaining(now) > 0:
                        alerts.unsnooze()
                        print("alerts resumed")
                    else:
                        alerts.snooze(s.snooze_s, now)
                        print(f"alerts snoozed for {s.snooze_s / 60:.0f} min")
                elif key in (ord("+"), ord("=")):
                    s.phone_conf_threshold = min(0.95, s.phone_conf_threshold + 0.05)
                elif key in (ord("-"), ord("_")):
                    s.phone_conf_threshold = max(0.05, s.phone_conf_threshold - 0.05)
                elif key == ord("]"):
                    s.facing_tolerance_deg = min(60.0, s.facing_tolerance_deg + 2)
                elif key == ord("["):
                    s.facing_tolerance_deg = max(2.0, s.facing_tolerance_deg - 2)
                elif key == ord("m"):
                    mirror = not mirror
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        alerts.close()
        watcher.close()
        cv2.destroyAllWindows()
    return 0


def _child_ping(q) -> None:
    q.put("pong")


def self_test(s: Settings) -> int:
    """Load every model and library on a blank image. Used to check packaged builds in CI."""
    import multiprocessing as mp
    import random

    import numpy as np

    from stopscrolling.alerts import memes, sounds
    from stopscrolling.perception.face import FaceTracker
    from stopscrolling.perception.phone import PhoneDetector

    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    def check_face():
        with FaceTracker(FACE_MODEL, video=False) as f:
            return f"no face in blank image: {f.process(blank) is None}"

    def check_detectors():
        found = []
        for name, path in PHONE_MODELS.items():
            if path.exists():
                with PhoneDetector(path, video=False) as d:
                    found.append(f"{name}: {len(d.process(blank))} detections")
        if not found:
            raise FileNotFoundError("no phone detector models bundled")
        return ", ".join(found)

    def check_verifier():
        c = PhoneVerifier(CLIP_DIR).check_crop(blank[:120, :120])
        return f"blank crop -> {c.label} ({c.label_prob:.2f})"

    def check_alert_assets():
        img = memes.make_meme(random.Random(0), 30)
        n = sum(len(fn()) for fn in sounds.LEVEL_SOUNDS.values())
        import os

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame  # noqa: F401  (import only; CI machines have no audio device)
        import tkinter  # noqa: F401

        return f"meme {img.size}, {n} audio samples"

    def check_subprocess():
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(target=_child_ping, args=(q,))
        proc.start()
        reply = q.get(timeout=60)
        proc.join(timeout=10)
        return f"spawned child replied {reply!r}"

    checks = [
        ("face landmarker", check_face),
        ("phone detectors", check_detectors),
        ("phone verifier (CLIP)", check_verifier),
        ("memes + sounds", check_alert_assets),
        ("alert subprocess", check_subprocess),
    ]
    failed = 0
    print(f"StopScrolling {__version__} self-test (models: {MODELS_DIR})")
    for name, fn in checks:
        try:
            print(f"  OK    {name}: {fn()}")
        except Exception as e:  # report every failure, not just the first
            failed += 1
            cause = f" (caused by {type(e.__cause__).__name__}: {e.__cause__})" if e.__cause__ else ""
            print(f"  FAIL  {name}: {type(e).__name__}: {e}{cause}")
    print("self-test passed" if not failed else f"self-test FAILED ({failed})")
    return 1 if failed else 0


def main(argv: list[str] | None = None, preview_default: bool = False) -> int:
    args, settings = parse_args(argv, preview_default)
    if args.self_test:
        return self_test(settings)
    return run_image(args, settings) if args.image else run_live(args, settings)
