"""Debug viewer: the full app with the camera window shown (same as `python -m stopscrolling --preview`).

    python tools/debug_viewer.py                       # webcam 0, alerts on
    python tools/debug_viewer.py --no-sound --no-popups
    python tools/debug_viewer.py --image photo.jpg --out annotated.png

Run with --help for every option.
"""

import sys

from stopscrolling.app import main

if __name__ == "__main__":
    sys.exit(main(preview_default=True))
