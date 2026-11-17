import random
import sys
import tkinter as tk

import numpy as np
import pytest

from stopscrolling.alerts import AlertConfig, AlertController, memes, sounds
from stopscrolling.alerts.worker import PopupManager


@pytest.mark.parametrize("seed", range(12))
def test_make_meme_produces_reasonable_images(seed):
    img = memes.make_meme(random.Random(seed), seconds_on_phone=42)
    assert 150 <= img.width <= 500 and 150 <= img.height <= 600
    assert img.mode == "RGB"


def test_custom_memes_are_used(tmp_path):
    from PIL import Image

    Image.new("RGB", (800, 600), (1, 2, 3)).save(tmp_path / "mine.png")
    (tmp_path / "notes.txt").write_text("not an image")
    custom = memes.list_custom_memes(tmp_path)
    assert [p.name for p in custom] == ["mine.png"]
    sizes = {memes.make_meme(random.Random(i), 10, custom).size for i in range(20)}
    assert (440, 330) in sizes  # the custom image, scaled to fit


@pytest.mark.parametrize("fn", [sounds.chime, sounds.siren, sounds.airhorn, sounds.resolved])
def test_sounds_are_audible_and_unclipped(fn):
    pcm = sounds.to_stereo_int16(fn())
    assert pcm.shape[1] == 2 and pcm.dtype == np.int16
    assert 0.5 < len(pcm) / sounds.SAMPLE_RATE < 5
    assert 15000 < np.abs(pcm).max() <= 32767


def test_siren_loops_seamlessly():
    s = sounds.siren()
    # Wrapping from the last sample to the first must look like any other step (no click).
    assert abs(s[0] - s[-1]) <= 1.05 * np.max(np.abs(np.diff(s)))


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    root.withdraw()
    yield root
    root.destroy()


def test_popups_spawn_escalate_and_clear(tk_root):
    mgr = PopupManager(tk_root, AlertConfig(max_popups=5), random.Random(0))
    now = 0.0
    for _ in range(3):  # 0.25 s: a new popup every 0.1 s -> 3 so far
        mgr.update(now, elapsed=now, level=1)
        now += 0.11
    assert len(mgr.popups) == 3
    for _ in range(20):  # keeps going, capped at max_popups
        mgr.update(now, elapsed=now, level=3)
        mgr.animate()
        now += 0.11
    tk_root.update()
    assert len(mgr.popups) == 5
    wins = [p.win for p in mgr.popups]
    assert all(w.winfo_exists() for w in wins)
    if sys.platform != "darwin":  # macOS Tk doesn't report -topmost for borderless windows
        assert all(w.attributes("-topmost") for w in wins)

    mgr.close(mgr.popups[0])  # clicking one closes just that one
    assert len(mgr.popups) == 4

    mgr.clear()  # back to work
    tk_root.update()
    assert mgr.popups == []
    assert not any(w.winfo_exists() for w in wins)


def test_bouncing_popups_stay_on_screen(tk_root):
    mgr = PopupManager(tk_root, AlertConfig(max_popups=10), random.Random(1))
    for i in range(10):
        mgr.spawn(elapsed=1)
    for p in mgr.popups:
        p.mode = "bounce"
    sx, sy, sw, sh = mgr.screen
    for _ in range(300):
        mgr.animate()
    for p in mgr.popups:
        assert sx <= p.x and p.x + p.w <= sx + sw
        assert sy <= p.y and p.y + p.h <= sy + sh
    mgr.clear()


def test_controller_snooze_without_process():
    alerts = AlertController(AlertConfig(sound=False, popups=False))  # no process started
    assert not alerts.enabled
    alerts.update(True, 0.0)
    assert not alerts.active  # nothing to start
    alerts.snooze(300, now=10.0)
    assert alerts.snooze_remaining(100.0) == pytest.approx(210)
    alerts.unsnooze()
    assert alerts.snooze_remaining(100.0) == 0
