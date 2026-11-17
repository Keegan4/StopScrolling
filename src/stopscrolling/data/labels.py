"""Attention labels used when recording training clips.

"Attentive" covers both looking at the screen and working on paper: reading notes and
writing look a lot like looking at a phone (head down), so they need their own examples.
That's the main thing the learned model has to get right.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    key: str  # keyboard key in the recorder
    name: str  # stored in clip metadata / features.csv
    description: str  # what to do while recording
    attentive: bool  # collapsed binary target: paying attention to your work?
    color: tuple[int, int, int]  # BGR, for the recorder UI


LABELS: list[Label] = [
    Label("1", "screen", "Working on the computer: reading, typing, watching a lecture", True, (80, 200, 80)),
    Label("2", "paper", "Working on paper: reading notes or a textbook, writing, using a calculator", True, (60, 170, 30)),
    Label("3", "phone", "Using your phone: scrolling, texting, watching videos", False, (40, 40, 230)),
    Label("4", "looking_away", "At the desk but not working: looking around, daydreaming, chatting", False, (0, 165, 255)),
    Label("5", "absent", "Not at the desk, or leaning out of the camera view", False, (160, 160, 160)),
]
BY_NAME = {lbl.name: lbl for lbl in LABELS}
BY_KEY = {lbl.key: lbl for lbl in LABELS}


def get(name: str) -> Label:
    try:
        return BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown label {name!r}; choose from {', '.join(BY_NAME)}") from None
