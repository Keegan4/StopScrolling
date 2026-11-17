# StopScrolling — Project Plan

A desktop app that watches you through your webcam while you study, detects when you
pick up your phone, and plays a loud sound to get you back to work. Everything runs
locally; no video leaves the machine.

---

## 1. How it works (pipeline)

```
 Webcam ──► Frame grabber ──► Perception (MediaPipe) ──► Feature extractor ──► Classifier ──► Smoother / state machine ──► Alert (sound)
 (OpenCV)    ~15 fps           • Face Landmarker            per-frame vector     P(phone)      hysteresis + timers           escalating
                               • Object Detector (phone)    + 2 s window stats                                               volume, cooldown
                               • Hand Landmarker (optional)
```

1. **Capture**: OpenCV reads the webcam on a background thread at about 15 fps and keeps only the latest frame.
2. **Perception**: the MediaPipe Tasks API, run in `VIDEO` mode:
   - **Face Landmarker** (`face_landmarker.task`) gives 478 landmarks, a **facial transformation matrix** (head pitch, yaw and roll) and **52 blendshapes** (`eyeLookDownLeft/Right`, `eyeBlink*` and so on). This is the main signal.
   - **Object Detector** (`efficientdet_lite0.tflite`, trained on COCO, which has a `cell phone` class) checks whether a phone is in view. It is expensive, so it runs every 3rd to 5th frame.
   - **Hand Landmarker** (optional, phase 2) detects hands raised in front of the chest or face, which fits holding a phone.
3. **Features**: one small numeric vector per frame (see §3), plus rolling statistics over a window of about 2 s.
4. **Classifier**: outputs `P(on_phone)` for each window. It starts as a hand-tuned heuristic and is replaced by a trained model once data exists.
5. **State machine**: turns noisy probabilities into a decision. See §5.
6. **Alert**: plays a sound and repeats it at a higher volume until you look back up.

---

## 2. The main problem: looking down ≠ using a phone

A laptop webcam usually **cannot see a phone held in your lap**. "Head pitched down" is
the strongest signal, but you also look down when you **read a textbook or write notes**,
which is exactly what we want you to do. The design deals with this in four ways:

- **Calibration at session start**: 5 s of "look at your screen" sets a personal neutral pitch and yaw. All angles are measured relative to that baseline.
- **Several signals together**: phone in view, gaze direction from the eye blendshapes, hand position, and *motion pattern*. Scrolling tends to mean a still head, few blinks and long dwell times. Writing tends to mean small periodic head movement.
- **Training data must include hard negatives**: recordings of reading paper notes and writing, labelled `studying`. Without them the model learns "down = phone".
- **Configurable "desk mode"**: if the classifier is unsure, a user who never studies from paper can choose to treat any downward look longer than N seconds as distraction.

---

## 3. Feature vector (per frame)

| Group | Features |
|---|---|
| Head pose (relative to calibration) | `pitch`, `yaw`, `roll` from the transformation matrix |
| Gaze | `eyeLookDownL/R`, `eyeLookInL/R`, `eyeLookOutL/R`, `eyeLookUpL/R` blendshapes |
| Eyes | `eyeBlinkL/R`, eye aspect ratio |
| Face presence | `face_detected` (0/1), face bbox size (distance to camera), face center y |
| Phone | `phone_conf` (max score), phone bbox center x/y and area (0 if none) |
| Hands (phase 2) | `num_hands`, lowest wrist y, hand-to-face distance |

**Window features** (over about 2 s, roughly 30 frames): mean, std and min/max of pitch,
gaze-down and phone_conf, blink rate, and the fraction of frames with a face.

"Face not detected for more than X s" is its own state, *away*. It can trigger a separate,
milder alert, or none.

---

## 4. Model strategy

Start simple and add complexity only when it helps:

| Stage | Model | Why |
|---|---|---|
| v0 | **Rule-based heuristic** (`pitch < -20° for 3 s` OR `phone_conf > 0.5`) | Works on day one with no data and sets a baseline to beat |
| v1 | **Logistic regression / gradient boosting** (scikit-learn `HistGradientBoostingClassifier`) on window features | Small and fast, trains in seconds on a few thousand windows, and you can inspect it |
| v2 (optional) | Small **1D-CNN / GRU** over the raw per-frame feature sequence | Only if v1 tops out. Export to ONNX for fast inference |

**Labels**: `studying`, `phone`, `away`. These can be collapsed to binary for the alarm.

**Data collection tool** (`tools/collect_data.py`): opens the webcam and shows a preview.
You press a key to set the current label (`1` studying, `2` phone, `3` away). It records
**feature vectors only, not video** to `data/raw/<session>.parquet`. Aim for about 10
sessions of 5 minutes across different lighting, seating and people, with deliberate
hard negatives (writing, reading paper, drinking water).

**Evaluation**: split by **session** rather than by frame, because adjacent frames are
nearly identical and a frame split overstates accuracy. Report precision and recall for
`phone`. **Precision matters most**: a false alarm while you study is the fastest way to
make people uninstall.

---

## 5. Decision logic and alerts

```
          P>0.7 for 2s                 still P>0.7 after 3s more
FOCUSED ───────────────► SUSPECT ───────────────────────────────► DISTRACTED ──► 🔊 alarm (escalates every 5s)
   ▲                        │ P<0.4                                     │ P<0.3 for 1.5s
   └────────────────────────┴────────────────────────────────────────────┘  (+ 10s cooldown before re-arming)
```

- Smooth `P` with an exponential moving average. The gap between the enter and exit thresholds (hysteresis) stops the state from flickering.
- **Alert escalation**: a soft chime, then a louder alarm, then the loudest sound with an on-screen overlay. All sounds and volumes are configurable.
- Audio uses `pygame.mixer`, which is cross-platform, non-blocking and can mix several sounds. Default sounds ship in `assets/sounds/`, and users can add their own.
- Optional extras: a desktop notification, logging each distraction for session stats, and a Pomodoro-style "break" button that pauses detection.

---

## 6. Desktop app (UI)

**PySide6 (Qt)**:
- **Main window**: Start/Stop session, live camera preview (can be turned off), a current-state indicator, a sensitivity slider and a sound picker.
- **System tray icon**: so the app can run in the background with minimal CPU.
- **Session summary**: time focused, number of distractions and a timeline chart.
- **Settings**: stored in `%APPDATA%/StopScrolling/config.json`.

Threading: the camera and inference run in a `QThread` worker that emits a `state_changed` and
`frame_ready` signal; the UI never blocks on inference.

---

## 7. Repository structure

```
StopScrolling/
├── PLAN.md                     ← this file
├── README.md
├── pyproject.toml              # deps + `stopscrolling` entry point
├── .gitignore                  # data/, models/*.task, *.tflite, .venv, build/
│
├── src/stopscrolling/
│   ├── __init__.py
│   ├── __main__.py             # `python -m stopscrolling` → launches UI
│   ├── config.py               # dataclass settings + load/save JSON
│   ├── pipeline.py             # wires capture → perception → features → model → state
│   │
│   ├── capture/
│   │   └── camera.py           # threaded OpenCV webcam reader (latest-frame queue)
│   ├── perception/
│   │   ├── face.py             # MediaPipe FaceLandmarker wrapper → landmarks, matrix, blendshapes
│   │   ├── phone.py            # MediaPipe ObjectDetector wrapper → "cell phone" detections
│   │   └── hands.py            # MediaPipe HandLandmarker wrapper (phase 2)
│   ├── features/
│   │   ├── head_pose.py        # transformation matrix → pitch/yaw/roll
│   │   ├── extractor.py        # perception results → per-frame feature vector
│   │   └── window.py           # rolling-window aggregation
│   ├── model/
│   │   ├── base.py             # Classifier protocol: predict_proba(window) -> float
│   │   ├── heuristic.py        # v0 rules
│   │   └── learned.py          # loads trained sklearn/ONNX model
│   ├── detection/
│   │   ├── calibration.py      # neutral-pose calibration
│   │   └── state_machine.py    # EMA + hysteresis + timers → FOCUSED/SUSPECT/DISTRACTED/AWAY
│   ├── alerts/
│   │   ├── sound.py            # pygame.mixer playback + escalation
│   │   └── notify.py           # desktop notifications (optional)
│   ├── stats/
│   │   └── session_log.py      # records events for session summary
│   └── ui/
│       ├── main_window.py
│       ├── tray.py
│       ├── worker.py           # QThread running pipeline
│       └── widgets/            # preview, state badge, timeline chart
│
├── tools/
│   ├── download_models.py      # fetches MediaPipe .task/.tflite files into models/
│   ├── collect_data.py         # labelled feature recording
│   ├── train.py                # trains v1 classifier → models/classifier.joblib
│   ├── evaluate.py             # session-split eval, confusion matrix, PR curve
│   └── debug_viewer.py         # live overlay of landmarks, pose angles, P(phone)
│
├── models/                     # gitignored binaries (README explains how to fetch)
├── assets/sounds/              # default alert sounds (CC0)
├── data/                       # gitignored: raw/ and processed/ feature sessions
└── tests/
    ├── test_head_pose.py
    ├── test_state_machine.py   # deterministic: feed probability sequences, assert transitions
    └── test_features.py
```

**Dependencies**: `mediapipe`, `opencv-python`, `numpy`, `pandas` + `pyarrow`, `scikit-learn`,
`joblib`, `pygame`, `PySide6`, plus `pytest` and `ruff` for development. Packaged later with
**PyInstaller**, which must bundle the MediaPipe model files.

---

## 8. Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | **Skeleton + debug viewer** | `tools/debug_viewer.py` shows the webcam with face mesh, pitch/yaw/roll and phone boxes drawn on it |
| 2 | **Heuristic MVP (CLI)** | Running `python -m stopscrolling --no-ui` plays a sound when you look down at your phone for 3 s and stops when you look up |
| 3 | **State machine + calibration + tests** | False alarms are rare during 10 min of normal typing at a laptop |
| 4 | **Data collection + v1 model** | The trained model beats the heuristic on phone **precision** on held-out sessions, including writing and reading |
| 5 | **Desktop UI** | Tray app with start/stop, preview, settings and session summary |
| 6 | **Polish + packaging** | One-click `.exe` built by PyInstaller, custom sounds, break mode |

---

## 9. Risks and open questions

- **False positives from reading or writing**: mitigated by calibration, hard-negative data and favouring precision.
- **Phone below the camera's view**: phone detection is a bonus signal only. Pose and gaze carry the decision.
- **Lighting, glasses, webcam angle**: collect varied data and expose the sensitivity slider.
- **CPU usage on laptops**: cap at about 15 fps, run object detection every Nth frame, and lower the resolution to 640×480.
- **Privacy**: no frames are stored, and `collect_data` saves only numeric features. State this clearly in the README.
- **Open**: should the app also *block* distractions on the computer (websites)? This is out of scope for now.
