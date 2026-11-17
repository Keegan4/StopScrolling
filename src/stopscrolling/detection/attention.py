"""Turn noisy per-frame "looking at phone" results into a steady decision.

Per-frame results flicker (a missed detection, a blink, gaze noise). This smooths them
and requires the evidence to stay high for ``enter_s`` before switching to "on phone",
and low for ``exit_s`` before switching back, so glancing down for a moment does not count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AttentionState:
    on_phone: bool  # the steady decision
    evidence: float  # smoothed per-frame evidence, 0..1
    progress: float  # 0..1 progress towards flipping the decision


class AttentionTracker:
    def __init__(
        self,
        enter_s: float = 1.5,
        exit_s: float = 1.0,
        smoothing_s: float = 0.4,
        on_level: float = 0.6,
        off_level: float = 0.4,
    ) -> None:
        self.enter_s = enter_s
        self.exit_s = exit_s
        self.smoothing_s = smoothing_s
        self.on_level = on_level
        self.off_level = off_level
        self.reset()

    def reset(self) -> None:
        self.on_phone = False
        self.evidence = 0.0
        self._timer = 0.0
        self._last: float | None = None

    def update(self, looking: bool, now: float) -> AttentionState:
        dt = 0.0 if self._last is None else min(max(now - self._last, 0.0), 0.5)
        self._last = now

        alpha = 1.0 - math.exp(-dt / self.smoothing_s) if self.smoothing_s > 0 else 1.0
        self.evidence += alpha * (float(looking) - self.evidence)

        if not self.on_phone:
            self._timer = self._timer + dt if self.evidence >= self.on_level else 0.0
            if self._timer >= self.enter_s:
                self.on_phone, self._timer = True, 0.0
            progress = self._timer / self.enter_s if self.enter_s else 0.0
        else:
            self._timer = self._timer + dt if self.evidence <= self.off_level else 0.0
            if self._timer >= self.exit_s:
                self.on_phone, self._timer = False, 0.0
            progress = self._timer / self.exit_s if self.exit_s else 0.0

        return AttentionState(self.on_phone, self.evidence, min(progress, 1.0))
