"""Alarm sounds synthesised with numpy (no audio files needed).

Escalation levels:
    1  "ding-dong" chime, repeating
    2  siren sweep
    3  air-horn blasts
Plus a short happy arpeggio when you put the phone down.

Drop .wav/.ogg/.mp3 files into ``assets/sounds/`` to use them for level 3 instead.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE))


def _saw(phase: np.ndarray) -> np.ndarray:
    return 2.0 * (phase - np.floor(0.5 + phase))


def _normalise(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def _fade(x: np.ndarray, attack: float = 0.005, release: float = 0.02) -> np.ndarray:
    x = x.copy()
    a, r = int(attack * SAMPLE_RATE), int(release * SAMPLE_RATE)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if r:
        x[-r:] *= np.linspace(1, 0, r)
    return x


def bell(freq: float, seconds: float) -> np.ndarray:
    t = _t(seconds)
    tone = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * 2.76 * freq * t) + 0.2 * np.sin(2 * np.pi * 5.4 * freq * t)
    return _fade(tone * np.exp(-t * 5.0), release=0.05)


def chime() -> np.ndarray:
    """Ding-dong, then a pause (the pause makes it repeat politely when looped)."""
    return _normalise(np.concatenate([bell(988, 0.45), bell(784, 0.9), _silence(1.0)]), 0.7)


def siren(cycle_s: float = 1.2, cycles: int = 2) -> np.ndarray:
    """Rising and falling wail; starts and ends at the same pitch so it loops seamlessly."""
    t = _t(cycle_s * cycles)
    freq = 950 + 350 * np.sin(2 * np.pi * t / cycle_s - np.pi / 2)
    phase = np.concatenate([[0.0], np.cumsum(freq[:-1])]) / SAMPLE_RATE
    end = float(np.sum(freq)) / SAMPLE_RATE
    phase *= round(end) / end  # whole number of waves, so the loop point doesn't click
    tone = np.sin(2 * np.pi * phase) + 0.35 * np.sin(2 * np.pi * 3 * phase)
    return _normalise(np.tanh(1.5 * tone), 0.85)


def horn_blast(seconds: float) -> np.ndarray:
    t = _t(seconds)
    vibrato = 1 + 0.006 * np.sin(2 * np.pi * 6 * t)
    chord = sum(_saw(f * vibrato * t) for f in (311.0, 415.0, 466.0, 622.0))
    bend = np.clip(t / 0.06, 0, 1)  # quick swell at the start like a real horn
    return _fade(np.tanh(2.5 * chord * bend), attack=0.01, release=0.04)


def airhorn() -> np.ndarray:
    gap = _silence(0.07)
    parts = [horn_blast(0.18), gap, horn_blast(0.18), gap, horn_blast(0.18), gap, horn_blast(0.8), _silence(0.5)]
    return _normalise(np.concatenate(parts), 0.95)


def resolved() -> np.ndarray:
    """Short happy C-E-G-C arpeggio."""
    notes = [523.25, 659.25, 783.99, 1046.5]
    return _normalise(np.concatenate([bell(f, 0.16 if i < 3 else 0.6) for i, f in enumerate(notes)]), 0.6)


def to_stereo_int16(mono: np.ndarray) -> np.ndarray:
    pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
    return np.ascontiguousarray(np.column_stack([pcm, pcm]))


LEVEL_SOUNDS = {1: chime, 2: siren, 3: airhorn}
LEVEL_VOLUME = {1: 0.6, 2: 0.85, 3: 1.0}
