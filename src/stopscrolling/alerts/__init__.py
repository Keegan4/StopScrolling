"""Alerts: escalating alarm sounds + meme popups while you're on your phone."""

from __future__ import annotations

import multiprocessing as mp

from stopscrolling.alerts.worker import AlertConfig


class AlertController:
    """Main-process handle for the alert process. Call :meth:`update` every frame."""

    def __init__(self, cfg: AlertConfig) -> None:
        self.cfg = cfg
        self._active = False
        self._snooze_until = 0.0
        self._proc = None
        if not (cfg.sound or cfg.popups):
            return
        from stopscrolling.alerts import worker

        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        self._proc = ctx.Process(target=worker.run, args=(self._queue, cfg), name="stopscrolling-alerts", daemon=True)
        self._proc.start()

    @property
    def enabled(self) -> bool:
        return self._proc is not None

    @property
    def active(self) -> bool:
        return self._active

    def update(self, on_phone: bool, now: float) -> None:
        want = on_phone and now >= self._snooze_until
        if want != self._active and self._proc is not None:
            self._queue.put("start" if want else "stop")
            self._active = want

    def snooze(self, seconds: float, now: float) -> None:
        self._snooze_until = now + seconds
        self.update(False, now)

    def unsnooze(self) -> None:
        self._snooze_until = 0.0

    def snooze_remaining(self, now: float) -> float:
        return max(0.0, self._snooze_until - now)

    def close(self) -> None:
        if self._proc is None:
            return
        self._queue.put("quit")
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None


__all__ = ["AlertConfig", "AlertController"]
