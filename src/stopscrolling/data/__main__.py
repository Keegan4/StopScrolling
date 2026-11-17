"""Collect training data for the attention model.

    python -m stopscrolling.data record      # record labelled webcam clips
    python -m stopscrolling.data extract     # clips -> frames, landmarks, features
    python -m stopscrolling.data summary     # what you've recorded / extracted so far

Labels: 1 screen, 2 paper, 3 phone, 4 looking_away, 5 absent.
Clips and extracted data live in data/ (git-ignored; raw video never leaves your machine).
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

from stopscrolling.config import CLIP_DIR, CLIPS_DIR, PHONE_MODELS, PROCESSED_DIR, Settings
from stopscrolling.data import labels


def cmd_record(args: argparse.Namespace) -> int:
    from stopscrolling.data.recorder import RecorderConfig, run

    cfg = RecorderConfig(args.out, args.max_seconds, args.countdown, args.camera)
    return run(cfg, args.label, args.width, args.height)


def cmd_extract(args: argparse.Namespace) -> int:
    from stopscrolling.data.clips import ClipMeta, list_clips
    from stopscrolling.data.extract import ExtractOptions, build_dataset, extract_clip
    from stopscrolling.perception.verifier import PhoneVerifier

    if args.clips:
        clips = [(Path(c), ClipMeta.load(Path(c).with_suffix(".json"))) for c in args.clips]
    else:
        clips = list_clips(args.clips_dir)
    if not clips:
        print(f"No clips found in {args.clips_dir}. Record some first: python -m stopscrolling.data record")
        return 1

    settings = Settings()
    settings.detector = args.detector
    verifier = None
    if not args.no_verify:
        try:
            verifier = PhoneVerifier(CLIP_DIR)
        except (FileNotFoundError, ImportError) as e:
            print(f"Phone verification disabled: {e}", file=sys.stderr)
    opts = ExtractOptions(fps=args.fps, save_frames=not args.no_frames, jpeg_quality=args.jpeg_quality, force=args.force)

    print(f"Extracting {len(clips)} clip(s) at {args.fps or 'all'} fps into {args.out}")
    print(f"{'clip':40s} {'label':13s} {'frames':>6s} {'face':>5s} {'phone':>6s} {'look@phone':>10s}")
    for video, meta in clips:
        t0 = time.monotonic()
        s = extract_clip(video, meta, args.out, settings, verifier, opts)
        note = " (already done; --force to redo)" if s.skipped else f" {time.monotonic() - t0:4.1f}s"
        print(f"{s.clip_id:40s} {s.label:13s} {s.frames:6d} {s.face_rate:5.0%} {s.phone_rate:6.0%} {s.looking_rate:10.0%}{note}")
    path, n = build_dataset(args.out)
    print(f"\n{n} frames in {path}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from stopscrolling.data.clips import list_clips

    clips = list_clips(args.clips_dir)
    per = {lbl.name: [0, 0.0, 0] for lbl in labels.LABELS}
    for _, meta in clips:
        row = per.setdefault(meta.label, [0, 0.0, 0])
        row[0] += 1
        row[1] += meta.duration_s
        if (args.out / meta.clip_id / "features.csv").exists():
            row[2] += 1
    print(f"Clips in {args.clips_dir}\n")
    print(f"{'label':14s} {'attentive':>9s} {'clips':>6s} {'seconds':>8s} {'extracted':>9s}")
    for lbl in labels.LABELS:
        n, secs, done = per[lbl.name]
        print(f"{lbl.name:14s} {'yes' if lbl.attentive else 'no':>9s} {n:6d} {secs:8.0f} {done:6d}/{n}")
    total = sum(v[1] for v in per.values())
    print(f"\nTotal {total:.0f}s. Aim for at least a few minutes per label, recorded over")
    print("several sessions (different lighting, seating, clothes) for a model that generalises.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m stopscrolling.data", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record labelled webcam clips")
    r.add_argument("--label", default="screen", choices=list(labels.BY_NAME), help="label selected at start")
    r.add_argument("--max-seconds", type=float, default=60.0, help="auto-stop each clip after this long")
    r.add_argument("--countdown", type=float, default=3.0, help="seconds to get into position")
    r.add_argument("--camera", type=int, default=0)
    r.add_argument("--width", type=int, default=640)
    r.add_argument("--height", type=int, default=480)
    r.add_argument("--out", type=Path, default=CLIPS_DIR)
    r.set_defaults(func=cmd_record)

    e = sub.add_parser("extract", help="split clips into frames, landmarks and features")
    e.add_argument("clips", nargs="*", help="specific clip videos (default: all in --clips-dir)")
    e.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    e.add_argument("--out", type=Path, default=PROCESSED_DIR)
    e.add_argument("--fps", type=float, default=10.0, help="frames per second to sample (0 = every frame)")
    e.add_argument("--no-frames", action="store_true", help="skip saving JPEG frames (landmarks/features only)")
    e.add_argument("--jpeg-quality", type=int, default=90)
    e.add_argument("--no-verify", action="store_true", help="skip the CLIP phone check (faster)")
    e.add_argument("--detector", choices=sorted(PHONE_MODELS), default="lite0")
    e.add_argument("--force", action="store_true", help="re-extract clips that were already done")
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("summary", help="show recorded / extracted data per label")
    s.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    s.add_argument("--out", type=Path, default=PROCESSED_DIR)
    s.set_defaults(func=cmd_summary)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
