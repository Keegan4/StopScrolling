"""The alert process: plays alarm sounds and throws meme popups around the screen.

Runs in its own process (Tk needs its own main thread; the camera loop has OpenCV's).
The main process sends "start" / "stop" / "quit" over a queue.
"""

from __future__ import annotations

import os
import queue
import random
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from stopscrolling.alerts import memes, sounds

TICK_MS = 30
AUDIO_EXTS = {".wav", ".ogg", ".mp3"}


@dataclass
class AlertConfig:
    sound: bool = True
    popups: bool = True
    max_popups: int = 40
    escalate_s: float = 6.0  # move up one level (louder, more popups) every N seconds
    memes_dir: str | None = None
    sounds_dir: str | None = None
    seed: int | None = None


def _enable_dpi_awareness() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def virtual_screen(root: tk.Tk) -> tuple[int, int, int, int]:
    """(x, y, width, height) of the whole desktop (all monitors on Windows)."""
    if sys.platform == "win32":
        try:
            import ctypes

            m = ctypes.windll.user32.GetSystemMetrics
            x, y, w, h = m(76), m(77), m(78), m(79)
            if w > 0 and h > 0:
                return x, y, w, h
        except Exception:
            pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


class Audio:
    def __init__(self, sounds_dir: str | None) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg = pygame
        pygame.mixer.init(sounds.SAMPLE_RATE, -16, 2, 512)
        make = pygame.sndarray.make_sound
        self.level_sounds = {lvl: make(sounds.to_stereo_int16(fn())) for lvl, fn in sounds.LEVEL_SOUNDS.items()}
        self.resolved = make(sounds.to_stereo_int16(sounds.resolved()))
        custom = []
        if sounds_dir and Path(sounds_dir).is_dir():
            for p in sorted(Path(sounds_dir).iterdir()):
                if p.suffix.lower() in AUDIO_EXTS:
                    try:
                        custom.append(pygame.mixer.Sound(str(p)))
                    except pygame.error:
                        pass
        if custom:
            self.level_sounds[3] = random.choice(custom)
        self.channel = pygame.mixer.Channel(0)
        self.level = 0

    def set_level(self, level: int) -> None:
        if level == self.level:
            return
        self.level = level
        self.channel.set_volume(sounds.LEVEL_VOLUME[level])
        self.channel.play(self.level_sounds[level], loops=-1)

    def stop(self, celebrate: bool = True) -> None:
        if self.level:
            self.channel.stop()
            if celebrate:
                self.channel.set_volume(0.8)
                self.channel.play(self.resolved)
        self.level = 0

    def close(self) -> None:
        self.pg.mixer.quit()


@dataclass(eq=False)
class Popup:
    win: tk.Toplevel
    photo: object  # keep a reference or Tk drops the image
    x: float
    y: float
    w: int
    h: int
    mode: str  # "still" | "shake" | "bounce"
    vx: float = 0.0
    vy: float = 0.0


class PopupManager:
    SPAWN_INTERVAL = {1: 0.1, 2: 0.1, 3: 0.1}  # seconds between new popups per level

    def __init__(self, root: tk.Tk, cfg: AlertConfig, rng: random.Random) -> None:
        self.root = root
        self.cfg = cfg
        self.rng = rng
        self.screen = virtual_screen(root)
        self.custom = memes.list_custom_memes(Path(cfg.memes_dir) if cfg.memes_dir else None)
        self.popups: list[Popup] = []
        self.next_spawn = 0.0

    def update(self, now: float, elapsed: float, level: int) -> None:
        if now >= self.next_spawn and len(self.popups) < self.cfg.max_popups:
            self.spawn(elapsed)
            self.next_spawn = now + self.SPAWN_INTERVAL[level]

    def spawn(self, elapsed: float) -> None:
        from PIL import ImageTk

        img = memes.make_meme(self.rng, elapsed, self.custom)
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)  # no title bar: you can't just close them, you have to stop
        win.attributes("-topmost", True)
        photo = ImageTk.PhotoImage(img, master=win)
        label = tk.Label(win, image=photo, bd=0, cursor="hand2")
        label.pack()

        sx, sy, sw, sh = self.screen
        w, h = img.width, img.height
        x = sx + self.rng.randint(0, max(0, sw - w))
        y = sy + self.rng.randint(0, max(0, sh - h))
        mode = self.rng.choices(["still", "shake", "bounce"], weights=[5, 3, 2])[0]
        speed = self.rng.uniform(4, 9)
        popup = Popup(win, photo, x, y, w, h, mode, self.rng.choice([-1, 1]) * speed, self.rng.choice([-1, 1]) * speed)
        win.geometry(f"{w}x{h}+{int(x)}+{int(y)}")
        label.bind("<Button-1>", lambda _e, p=popup: self.close(p))  # click one away (more will come)
        self.popups.append(popup)

    def animate(self) -> None:
        sx, sy, sw, sh = self.screen
        for p in self.popups:
            if p.mode == "bounce":
                p.x += p.vx
                p.y += p.vy
                if p.x < sx or p.x + p.w > sx + sw:
                    p.vx = -p.vx
                    p.x = min(max(p.x, sx), sx + sw - p.w)
                if p.y < sy or p.y + p.h > sy + sh:
                    p.vy = -p.vy
                    p.y = min(max(p.y, sy), sy + sh - p.h)
                p.win.geometry(f"+{int(p.x)}+{int(p.y)}")
            elif p.mode == "shake":
                dx, dy = self.rng.randint(-4, 4), self.rng.randint(-4, 4)
                p.win.geometry(f"+{int(p.x) + dx}+{int(p.y) + dy}")

    def close(self, popup: Popup) -> None:
        if popup in self.popups:
            self.popups.remove(popup)
        try:
            popup.win.destroy()
        except tk.TclError:
            pass

    def clear(self) -> None:
        for p in list(self.popups):
            self.close(p)
        self.next_spawn = 0.0


def run(commands, cfg: AlertConfig) -> None:
    """Process entry point."""
    import multiprocessing

    _enable_dpi_awareness()
    rng = random.Random(cfg.seed)
    root = tk.Tk()
    root.withdraw()

    audio = None
    if cfg.sound:
        try:
            audio = Audio(cfg.sounds_dir)
        except Exception as e:  # no audio device etc. - popups still work
            print(f"[alerts] sound disabled: {e}", file=sys.stderr)
    popups = PopupManager(root, cfg, rng) if cfg.popups else None
    parent = multiprocessing.parent_process()

    state = {"active": False, "since": 0.0}

    def shutdown() -> None:
        if popups:
            popups.clear()
        if audio:
            audio.stop(celebrate=False)
            audio.close()
        root.destroy()

    def tick() -> None:
        now = time.monotonic()
        while True:
            try:
                cmd = commands.get_nowait()
            except queue.Empty:
                break
            if cmd == "start" and not state["active"]:
                state.update(active=True, since=now)
            elif cmd == "stop" and state["active"]:
                state["active"] = False
                if popups:
                    popups.clear()
                if audio:
                    audio.stop()
            elif cmd == "quit":
                shutdown()
                return
        if parent is not None and not parent.is_alive():  # main app died: clean up
            shutdown()
            return

        if state["active"]:
            elapsed = now - state["since"]
            level = 1 + min(2, int(elapsed // cfg.escalate_s))
            if audio:
                audio.set_level(level)
            if popups:
                popups.update(now, elapsed, level)
        if popups:
            popups.animate()
        root.after(TICK_MS, tick)

    root.after(TICK_MS, tick)
    root.mainloop()
