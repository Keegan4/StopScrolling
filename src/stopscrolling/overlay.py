"""Draw perception + decision results on a frame (used by the debug viewer)."""

from __future__ import annotations

import cv2
import numpy as np

from stopscrolling.config import Settings
from stopscrolling.detection.attention import AttentionState
from stopscrolling.detection.facing import FacingResult, Status, is_confirmed_phone
from stopscrolling.features.gaze import EYES
from stopscrolling.perception.types import FaceObservation, PhoneDetection

# BGR
RED = (40, 40, 230)
GREEN = (80, 200, 80)
ORANGE = (0, 165, 255)
YELLOW = (0, 220, 255)
GREY = (170, 170, 170)
GAZE_COLOR = (255, 200, 0)

NOSE_TIP = 1
ARROW_SCALE = 1.5  # arrow length per unit of sideways/vertical direction, in face widths


def _fmt(v: float | None) -> str:
    return "  --" if v is None else f"{v:+6.1f}"


def _text(img, text, org, color=(255, 255, 255), scale=0.5, thickness=1, outline=True) -> None:
    if outline:  # dark halo so labels stay readable on top of the video
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _phone_style(
    p: PhoneDetection, result: FacingResult, settings: Settings, verify: bool
) -> tuple[tuple[int, int, int], int, str]:
    """Box colour, thickness and label for one detection."""
    if p.score < settings.phone_conf_threshold:
        return GREY, 1, f"det {p.score:.2f}"
    if is_confirmed_phone(p, settings, verify):
        color = RED if (p is result.phone and result.looking_at_phone) else ORANGE
        extra = f" | phone {p.check.p_phone:.0%}" if p.check is not None else ""
        return color, 2, f"det {p.score:.2f}{extra}"
    if p.check is None:
        return YELLOW, 1, f"det {p.score:.2f} | checking..."
    return GREY, 1, f"NOT a phone: {p.check.label} {p.check.label_prob:.0%}"


def render(
    frame_bgr: np.ndarray,
    face: FaceObservation | None,
    phones: list[PhoneDetection],
    result: FacingResult,
    attention: AttentionState,
    settings: Settings,
    verify: bool,
    mirror: bool = True,
    fps: float | None = None,
    note: str | None = None,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    vis = cv2.flip(frame_bgr, 1) if mirror else frame_bgr.copy()

    def pt(u: float, v: float) -> tuple[int, int]:
        """Raw-image pixel -> display pixel."""
        return (int(round(w - 1 - u)) if mirror else int(round(u)), int(round(v)))

    for p in phones:
        color, thick, label = _phone_style(p, result, settings, verify)
        x, y, bw, bh = p.box
        a, b = pt(x, y), pt(x + bw, y + bh)
        cv2.rectangle(vis, a, b, color, thick)
        _text(vis, label, (min(a[0], b[0]), max(12, min(a[1], b[1]) - 6)), color, 0.45)

    if face is not None:
        lm = face.landmarks
        for (a, b), iris in EYES:
            for idx in (a, b):
                cv2.circle(vis, pt(lm[idx, 0] * w, lm[idx, 1] * h), 2, (230, 230, 230), -1)
            if lm.shape[0] > iris:
                cv2.circle(vis, pt(lm[iris, 0] * w, lm[iris, 1] * h), 3, GAZE_COLOR, -1)

        nose = (lm[NOSE_TIP, 0] * w, lm[NOSE_TIP, 1] * h)
        face_w = (lm[:, 0].max() - lm[:, 0].min()) * w

        def arrow(direction: np.ndarray, color, thickness: int) -> None:
            # Orthographic: drop z so the arrow shows where the face points, not perspective.
            length = face_w * ARROW_SCALE
            end = (nose[0] + direction[0] * length, nose[1] - direction[1] * length)
            cv2.arrowedLine(vis, pt(*nose), pt(*end), color, thickness, cv2.LINE_AA, tipLength=0.2)

        arrow(face.rotation @ np.array([0.0, 0.0, 1.0]), (200, 200, 200), 1)  # head only
        if result.gaze_dir is not None:
            arrow(result.gaze_dir, GAZE_COLOR, 2)  # head + eyes

        if result.phone is not None:
            color = RED if result.looking_at_phone else ORANGE
            cv2.line(vis, pt(*nose), pt(*result.phone.center), color, 2 if result.looking_at_phone else 1, cv2.LINE_AA)

    _hud(vis, result, attention, settings, verify, fps, mirror)
    if note:
        color = GREEN if note == "alerts on" else YELLOW
        tw = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        _text(vis, note, (w - tw - 10, 20), color, 0.5)
    return vis


def _hud(vis, result, attention: AttentionState, settings: Settings, verify: bool, fps, mirror: bool) -> None:
    panel_w, panel_h = 330, 204
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    # Headline: the steady decision.
    if attention.on_phone:
        headline, color = "ON YOUR PHONE", RED
        bar_label = "looking away" if attention.progress > 0 else ""
    else:
        headline, color = "STUDYING", GREEN
        bar_label = "looking at phone" if attention.progress > 0 else ""
    _text(vis, headline, (10, 30), color, 0.8, 2, outline=False)

    # Progress towards flipping the decision + smoothed evidence.
    x0, x1, y = 10, panel_w - 10, 42
    cv2.rectangle(vis, (x0, y), (x1, y + 8), (80, 80, 80), -1)
    flip_color = GREEN if attention.on_phone else RED
    cv2.rectangle(vis, (x0, y), (x0 + int((x1 - x0) * attention.progress), y + 8), flip_color, -1)
    cv2.line(vis, (x0, y + 12), (x0 + int((x1 - x0) * attention.evidence), y + 12), YELLOW, 2)
    if bar_label:
        _text(vis, bar_label, (x0, y + 28), (220, 220, 220), 0.4, outline=False)

    frame_color = {
        Status.NO_FACE: GREY,
        Status.NO_PHONE: GREEN,
        Status.PHONE_IN_VIEW: ORANGE,
        Status.LOOKING_AT_PHONE: RED,
    }[result.status]
    angle = "--" if result.angle_to_phone is None else f"{result.angle_to_phone:4.1f}"
    tol = "--" if result.tolerance_deg is None else f"{result.tolerance_deg:.0f}"
    dist = "" if result.phone_dist_cm is None else f"  ({result.phone_dist_cm:.0f} cm away)"
    clip = ""
    if result.phone is not None and result.phone.check is not None:
        clip = f"  clip {result.phone.check.p_phone:.2f}"
    lines = [
        (f"this frame: {result.status.value}", frame_color),
        (f"head   yaw {_fmt(result.head_yaw)}  pitch {_fmt(result.head_pitch)}", None),
        (f"gaze   yaw {_fmt(result.gaze_yaw)}  pitch {_fmt(result.gaze_pitch)}", None),
        (f"phone  det {result.phone_conf:.2f}{clip}{dist}", None),
        (f"gaze->phone {angle} deg (need <= {tol})", None),
        (
            f"det >= {settings.phone_conf_threshold:.2f}  clip >= "
            f"{settings.phone_verify_threshold:.2f}{'' if verify else ' (off)'}  hold {settings.enter_s:.1f}s",
            (180, 180, 180),
        ),
    ]
    for i, (line, c) in enumerate(lines):
        _text(vis, line, (10, 88 + i * 19), c or (255, 255, 255), 0.45, outline=False)

    h = vis.shape[0]
    fps_txt = f"{fps:4.1f} fps | " if fps else ""
    keys = f"{fps_txt}q quit | s snooze/resume | +/- det | [ ] tolerance | m mirror"
    _text(vis, keys, (10, h - 10), (220, 220, 220), 0.42)
