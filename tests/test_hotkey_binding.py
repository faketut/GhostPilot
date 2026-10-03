"""Hotkey names: what reaches the `keyboard` library, and what it does with it.

`add_hotkey` raises on a name it does not know — inside `HotkeyManager.start()`,
where the failure was a single log line and the binding simply never existed.
The symptom is a documented hotkey that does nothing, which is what
`SCREENSHOT_FULL_HOTKEY=Caps` (the value in `.env.example`) produced: the
library knows `capslock` and `caps lock`, not bare `caps`.

Nothing here installs an OS hook. The seam that broke is the *call into the
library*, so these tests drive that seam directly (and use the library's own
parser to prove a combo is registrable), which keeps them deterministic.
"""
from __future__ import annotations

import keyboard
import pytest

from src.hotkey_manager import HotkeyManager, normalize_hotkey


def _capture_add_hotkey(monkeypatch, *, fail: bool = False):
    """Record the library call instead of installing a global hook."""
    calls: list[dict] = []

    def _add(hotkey, callback, suppress=False, trigger_on_release=False):
        calls.append({
            "hotkey": hotkey, "suppress": suppress,
            "trigger_on_release": trigger_on_release,
        })
        if fail:
            raise ValueError(f"Key {hotkey!r} is not mapped to any known key.")
        return lambda: None

    monkeypatch.setattr("src.hotkey_manager.keyboard.add_hotkey", _add)
    return calls


# ── names ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "typed,expected",
    [
        ("caps", "caps lock"),
        ("Caps", "caps lock"),
        ("CAPS", "caps lock"),
        ("caps-lock", "caps lock"),
        ("caps_lock", "caps lock"),
        # Left alone: the library's own alias table already maps it. Duplicating
        # that table here would be a second source of truth for key names.
        ("capslock", "capslock"),
        ("num_lock", "num lock"),
        ("alt+p", "alt+p"),
        ("Alt+Shift+P", "alt+shift+p"),
        ("ctrl+alt+a", "ctrl+alt+a"),
        ("", ""),
    ],
)
def test_hotkey_names_are_normalized_to_the_librarys_spelling(typed, expected):
    assert normalize_hotkey(typed) == expected


@pytest.mark.parametrize("typed", ["caps", "caps-lock", "capslock"])
def test_caps_variants_resolve_to_a_registrable_scan_code(typed):
    """The regression: `keyboard.add_hotkey('caps', …)` raised, so the binding
    never existed. The normalized name must resolve to the physical key."""
    assert keyboard.key_to_scan_codes(normalize_hotkey(typed)) == (58,)


def test_an_unknown_name_still_fails_where_the_library_can_report_it():
    with pytest.raises(ValueError):
        keyboard.parse_hotkey("definitely-not-a-key")


# ── suppression ──────────────────────────────────────────────────────────


def test_lone_latching_key_is_suppressed_so_it_does_not_toggle(monkeypatch):
    """Pressing a bound `caps` must not also flip Caps Lock — but suppression
    applies only to that shape: on `alt+p` it would swallow a real keystroke."""
    calls = _capture_add_hotkey(monkeypatch)

    HotkeyManager("caps", lambda: None).start()
    HotkeyManager("num lock", lambda: None).start()
    HotkeyManager("alt+p", lambda: None).start()
    HotkeyManager("alt+shift+p", lambda: None).start()

    assert [(c["hotkey"], c["suppress"]) for c in calls] == [
        ("caps lock", True),
        ("num lock", True),
        ("alt+p", False),
        ("alt+shift+p", False),
    ]


def test_explicit_suppress_still_wins(monkeypatch):
    calls = _capture_add_hotkey(monkeypatch)

    HotkeyManager("caps", lambda: None, suppress=False).start()
    HotkeyManager("alt+p", lambda: None, suppress=True).start()

    assert [c["suppress"] for c in calls] == [False, True]


# ── registration outcome ─────────────────────────────────────────────────


def test_start_reports_a_binding_that_could_not_be_registered(monkeypatch):
    """The caller decides what to do with a dead binding (warn, and keep it out
    of the footer hint); swallowing the failure is what made it invisible."""
    _capture_add_hotkey(monkeypatch, fail=True)
    hk = HotkeyManager("caps", lambda: None)

    assert hk.start() is False
    assert hk._is_running is False


def test_start_reports_success_and_is_idempotent(monkeypatch):
    calls = _capture_add_hotkey(monkeypatch)
    hk = HotkeyManager("caps", lambda: None)

    assert hk.start() is True
    assert hk.start() is True          # second call is a no-op, not a re-register
    assert len(calls) == 1


def test_stop_survives_two_managers_on_one_key(monkeypatch):
    """A hotkey rebuild can leave two managers on the same physical key — via
    aliases (`caps` and `caps lock`) or a settings change. The library keeps
    removers in a global dict keyed by the combo, so the second removal raises;
    that must not abort the rebuild or shutdown that called it."""
    _capture_add_hotkey(monkeypatch)
    first, second = HotkeyManager("caps", lambda: None), HotkeyManager("caps lock", lambda: None)
    first.start()
    second.start()

    def _remove(_handle):
        raise KeyError("caps lock")

    monkeypatch.setattr("src.hotkey_manager.keyboard.remove_hotkey", _remove)
    first.stop()
    second.stop()

    assert first._is_running is False and second._is_running is False


# ── dispatch ─────────────────────────────────────────────────────────────


def test_async_callback_is_scheduled_on_the_app_loop():
    """Every screenshot hotkey is an ``async def`` on the qasync loop, and the
    library calls its handler from a **listener thread** — so the callback must
    be handed to the loop, not awaited from that thread."""
    import asyncio
    import threading

    ran: list[str] = []

    async def _handler():
        ran.append("fired")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        hk = HotkeyManager("caps", _handler, loop)
        # Called from this thread, exactly as the listener thread would.
        hk._on_hotkey_pressed()
        deadline = threading.Event()
        for _ in range(100):
            if ran:
                break
            deadline.wait(0.02)
        assert ran == ["fired"]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
