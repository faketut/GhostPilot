from __future__ import annotations

import asyncio
import inspect
import logging
import re
from typing import Callable

import keyboard

logger = logging.getLogger(__name__)

# The name users type that the `keyboard` library's alias table does not cover.
# It knows 'capslock' — and anything hyphenated/spaced becomes the canonical
# 'caps lock' below — but not bare 'caps', and an unknown name makes
# `add_hotkey` raise, so the binding silently never existed.
_KEY_ALIASES = {"caps": "caps lock"}

# Latching keys: pressing one flips an OS toggle that has nothing to do with the
# action bound to it. A hotkey bound to a lone toggle key is registered
# suppressed, so `caps` takes a screenshot and does not also flip Caps Lock.
_TOGGLE_KEYS = {"caps lock", "num lock", "scroll lock"}

_STEP_SPLIT = re.compile(r"([+,])")


def normalize_hotkey(hotkey: str) -> str:
    """Rewrite key names the `keyboard` library cannot resolve.

    ``add_hotkey('caps')`` raises ``ValueError: Key 'caps' is not mapped to any
    known key`` — inside ``start()``, where the failure was only a log line, so
    the binding simply never existed. The library's canonical spelling is
    ``'caps lock'``.
    """
    out = []
    for part in _STEP_SPLIT.split(hotkey or ""):
        if part in ("+", ","):
            out.append(part)
            continue
        token = " ".join(part.strip().lower().replace("_", " ").replace("-", " ").split())
        # The space-normalized token is what the library itself canonicalizes
        # to, so 'caps-lock' / 'Caps' / 'num_lock' all reach it spelled right.
        out.append(_KEY_ALIASES.get(token, token))
    return "".join(out)


def _is_lone_toggle(hotkey: str) -> bool:
    """True for a single-key hotkey on a latching key (``caps``, ``num lock``)."""
    return len(_STEP_SPLIT.split(hotkey)) == 1 and hotkey.strip().lower() in _TOGGLE_KEYS


# Combos `add_hotkey` refused. Kept so the UI can stop *advertising* a key that
# does nothing — a hotkey hint for a dead binding is worse than no hint, and the
# banner that reports it goes to a log file the windowed app never shows.
_dead_combos: set[str] = set()


def note_unregistered(combos) -> None:
    """Replace the set of combos that failed to register (normalized, lowercased)."""
    _dead_combos.clear()
    _dead_combos.update(normalize_hotkey(c).lower() for c in combos if c)


def is_unregistered(combo: str) -> bool:
    """True when this combo is known to have failed registration."""
    return normalize_hotkey(combo or "").lower() in _dead_combos


class HotkeyManager:
    def __init__(
        self,
        hotkey: str,
        callback: Callable,
        loop=None,
        *,
        suppress: bool | None = None,
        trigger_on_release: bool = False,
    ):
        """``suppress=None`` (the default) suppresses exactly the combos that need
        it: a lone latching key. Any other combo is left alone, because
        suppression there would swallow keys the user is actually typing."""
        self.hotkey = normalize_hotkey(hotkey)
        self.callback = callback
        self.loop = loop
        self.suppress = _is_lone_toggle(self.hotkey) if suppress is None else bool(suppress)
        self.trigger_on_release = trigger_on_release
        self._is_running = False
        self._hotkey_handle = None

    def start(self) -> bool:
        """Starts listening for the hotkey in the background.

        Returns False when the combo is not registrable (a name the `keyboard`
        library does not know, or a key another hook owns) so callers can tell a
        live binding from a dead one instead of showing a hint for a key that
        does nothing.
        """
        if self._is_running:
            return True

        try:
            self._hotkey_handle = keyboard.add_hotkey(
                self.hotkey,
                self._on_hotkey_pressed,
                suppress=self.suppress,
                trigger_on_release=self.trigger_on_release,
            )
            self._is_running = True
            logger.info(
                "Started listening for hotkey: %s (suppress=%s)", self.hotkey, self.suppress
            )
            return True
        except Exception as e:
            logger.error(f"Failed to register hotkey {self.hotkey}: {e}")
            return False

    def _on_hotkey_pressed(self):
        logger.info(f"Hotkey {self.hotkey} pressed.")
        # `inspect`, not `asyncio.iscoroutinefunction`: the latter is deprecated
        # and slated for removal in 3.16, and it is the same predicate here.
        if inspect.iscoroutinefunction(self.callback):
            if self.loop:
                try:
                    asyncio.run_coroutine_threadsafe(self.callback(), self.loop)
                except Exception as e:
                    logger.error(f"Hotkey coroutine schedule failed ({self.hotkey}): {e}")
            else:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.callback())
                except RuntimeError:
                    asyncio.run(self.callback())
        else:
            if self.loop:
                def _safe_call():
                    try:
                        self.callback()
                    except Exception as e:
                        logger.error(f"Hotkey callback failed ({self.hotkey}): {e}")
                self.loop.call_soon_threadsafe(_safe_call)
            else:
                try:
                    self.callback()
                except Exception as e:
                    logger.error(f"Hotkey callback failed ({self.hotkey}): {e}")

    def stop(self):
        """Stops listening for hotkeys.

        Best-effort: the library tracks removers in a **global** dict keyed by
        the combo string, so two live managers on the same combo (possible when
        settings are rebound without a restart) share one entry and the second
        removal raises ``KeyError``. A failed removal must not abort the
        hotkey rebuild or shutdown that called this.
        """
        if self._is_running:
            try:
                if self._hotkey_handle is not None:
                    keyboard.remove_hotkey(self._hotkey_handle)
                else:
                    keyboard.remove_hotkey(self.hotkey)
            except Exception as e:
                logger.warning("Could not unregister hotkey %r: %s", self.hotkey, e)
            finally:
                self._hotkey_handle = None
            self._is_running = False
            logger.info("Stopped listening for hotkeys.")
