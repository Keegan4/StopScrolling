# StopScrolling

A desktop app that watches your webcam while you study and notices when you pick up your
phone. Everything runs locally. See [PLAN.md](PLAN.md) for the full design.

**Download:** ready-to-run builds for Windows, macOS (Apple Silicon) and Linux are on the
[Releases page](https://github.com/Keegan4/StopScrolling/releases). No Python needed.

## Alerts

While you're on your phone, a new meme popup appears **every 0.1 s**, up to 40 on screen
(`--max-popups` changes the cap). The sound gets louder every 6 seconds:

| Time on phone | Sound |
|---|---|
| 0–6 s | ding-dong chime |
| 6–12 s | siren |
| 12 s+ | air horn |

- **Where they appear:** popups are always on top, land anywhere on your monitors, and
  some shake or bounce around the screen.
- **Closing them:** they have no close button. Clicking one closes it, but more keep
  coming. **All of them close the moment you put the phone down**, and a happy jingle plays.
- **Your own content:** drop meme images into `assets/memes/` and they're mixed in with the
  generated ones. Drop `.wav`, `.ogg` or `.mp3` files into `assets/sounds/` to replace the
  air horn.
- **Snooze:** press `s` in the preview window to pause alerts for 5 minutes, for example
  while you take a call.

To try the alerts without a camera, run this. It pretends you're on your phone for 15 s:

```bash
.venv\Scripts\python -m stopscrolling.alerts
```

## How detection works

The headline reads **ON YOUR PHONE** or **STUDYING**. A real phone has to pass each step:

1. **Detect**: MediaPipe's object detector finds a "cell phone" candidate with a score of at least 0.5.
2. **Verify**: CLIP, a zero-shot image model, checks the cropped candidate against
   phone, calculator, remote, book, pencil case, eraser, wallet and hand. It only counts
   if "phone" wins with at least 50% probability. The detector alone scores a scientific
   calculator as about 0.75 "cell phone", while CLIP puts the chance it is a phone at about 0.00.
3. **Looking directly at it**: your gaze (head pose from the face mesh plus eye direction)
   must point within about 20° of where the phone is in 3D. The phone's distance is
   estimated from its size in the image.
4. **Sustained**: the look has to last 1.5 s before it counts and stop for 1 s before it
   clears, so a quick glance is ignored. If tilting your head down makes the face tracker
   lose your face, the last answer is kept while the phone stays in view.

Phone boxes are colour-coded: grey for a low score or not a phone (with what it was taken
for, such as "calculator"), yellow while CLIP is still checking, orange for a phone you're
not looking at, and red for a phone you're looking at.

## Setup (Windows)

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python tools\download_models.py --all
```

## Run

```bash
.venv\Scripts\python -m stopscrolling
```

This runs in the background with console output only. Press Ctrl+C to quit. Add `--preview`,
or run `tools\debug_viewer.py`, to also see the camera window:

```bash
.venv\Scripts\python tools\debug_viewer.py
```

| Key (preview window) | Action |
|---|---|
| `q` / `Esc` | quit |
| `s` | snooze alerts for 5 min, or resume them |
| `+` / `-` | raise or lower the phone confidence threshold |
| `[` / `]` | narrow or widen the facing tolerance (degrees) |
| `m` | toggle the mirrored view |

Options:
- **Sensitivity:** `--enter-s 2.5` (look longer before it triggers), `--tolerance 8`
  (stricter gaze), `--conf 0.6` / `--verify-conf 0.7` (stricter phone checks).
- **Detection:** `--detector lite2` (finds more phones, slower), `--no-verify` (skip the
  calculator check), `--detect-every 3` (saves CPU).
- **Input:** `--camera 1`, or `--image photo.jpg --out annotated.png` to process one still image.
- **Alerts:** `--no-sound`, `--no-popups`, `--max-popups 15`, `--escalate-s 10`.

State changes are printed to the console, for example `[14:02:11] on your phone - get back to work!`.

Angle conventions: yaw `+` means you turned to your right, and pitch `+` means you looked up.

## Tests

```bash
.venv\Scripts\python -m pytest
```

## Building the executable

```bash
.venv\Scripts\python -m pip install pyinstaller
```

```bash
.venv\Scripts\python packaging\build.py
```

This downloads the models, builds a one-folder app with PyInstaller in `dist/StopScrolling/`,
runs its `--self-test`, and creates an archive for the current OS. PyInstaller can't
cross-compile, so each OS has to be built on that OS.

Pushing a `v*` tag runs [.github/workflows/release.yml](.github/workflows/release.yml),
which builds on Windows, macOS and Linux and publishes the three archives as a GitHub release.
