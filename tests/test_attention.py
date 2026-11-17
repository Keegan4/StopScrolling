from stopscrolling.detection.attention import AttentionTracker

FPS = 15
DT = 1 / FPS


def feed(tracker, pattern, start=0.0):
    """Feed (looking, seconds) segments at 15 fps; return list of states and end time."""
    t, states = start, []
    for looking, seconds in pattern:
        for _ in range(round(seconds * FPS)):
            states.append(tracker.update(looking, t))
            t += DT
    return states, t


def test_brief_glance_does_not_trigger():
    tr = AttentionTracker(enter_s=1.5, exit_s=1.0)
    states, _ = feed(tr, [(False, 1), (True, 1.0), (False, 2)])
    assert not any(s.on_phone for s in states)


def test_sustained_look_triggers_after_hold():
    tr = AttentionTracker(enter_s=1.5, exit_s=1.0)
    states, _ = feed(tr, [(True, 3)])
    first = next(i for i, s in enumerate(states) if s.on_phone)
    # Smoothing lag + the 1.5 s hold: should flip roughly 1.5-2.5 s in.
    assert 1.5 * FPS <= first <= 2.5 * FPS


def test_flicker_while_on_phone_does_not_clear():
    tr = AttentionTracker(enter_s=1.5, exit_s=1.0)
    feed(tr, [(True, 3)])
    # Detector drops out every few frames.
    states, _ = feed(tr, [(True, 0.2), (False, 0.13)] * 10, start=3.0)
    assert all(s.on_phone for s in states)


def test_clears_after_looking_away():
    tr = AttentionTracker(enter_s=1.5, exit_s=1.0)
    _, t = feed(tr, [(True, 3)])
    states, _ = feed(tr, [(False, 3)], start=t)
    assert states[-1].on_phone is False
