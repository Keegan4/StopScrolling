"""Try the alerts without a camera.

    python -m stopscrolling.alerts                  # 15 s of escalating alerts, then they clear
    python -m stopscrolling.alerts --seconds 5 --no-sound
    python -m stopscrolling.alerts --save out/memes  # just write sample meme images
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from stopscrolling.alerts import AlertConfig, AlertController
from stopscrolling.config import MEMES_DIR, SOUNDS_DIR


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=15.0, help="how long to stay 'on the phone'")
    p.add_argument("--no-sound", action="store_true")
    p.add_argument("--no-popups", action="store_true")
    p.add_argument("--max-popups", type=int, default=40)
    p.add_argument("--escalate-s", type=float, default=5.0)
    p.add_argument("--save", type=Path, help="write 20 sample memes to this folder and exit")
    args = p.parse_args()

    if args.save:
        from stopscrolling.alerts.memes import list_custom_memes, make_meme

        args.save.mkdir(parents=True, exist_ok=True)
        rng = random.Random()
        custom = list_custom_memes(MEMES_DIR)
        for i in range(20):
            make_meme(rng, rng.uniform(5, 120), custom).save(args.save / f"meme_{i:02d}.png")
        print(f"saved 20 memes to {args.save}")
        return 0

    alerts = AlertController(
        AlertConfig(
            sound=not args.no_sound,
            popups=not args.no_popups,
            max_popups=args.max_popups,
            escalate_s=args.escalate_s,
            memes_dir=str(MEMES_DIR),
            sounds_dir=str(SOUNDS_DIR),
        )
    )
    try:
        print(f"Pretending you're on your phone for {args.seconds:.0f} s ...")
        alerts.update(True, time.monotonic())
        time.sleep(args.seconds)
        print("... and back to work. Popups close.")
        alerts.update(False, time.monotonic())
        time.sleep(2.0)  # let the 'resolved' sound play
    except KeyboardInterrupt:
        pass
    finally:
        alerts.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
