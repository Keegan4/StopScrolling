"""Entry point for the packaged executable.

Double-clicking the executable opens the camera preview window (press q to quit);
pass --no-preview to run with console output only.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    # Required so the alert process (multiprocessing "spawn") works in a frozen app.
    multiprocessing.freeze_support()
    from stopscrolling.app import main

    sys.exit(main(preview_default=True))
