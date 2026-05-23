"""
Theme palette
-------------
Single source of truth for overlay/settings colours.  Reads the OS colour
scheme (Qt 6.5+) so the overlay matches Windows light/dark mode at startup,
and re-emits `theme_changed` if the user flips it at runtime.
"""

from __future__ import annotations

import logging
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QGuiApplication

logger = logging.getLogger(__name__)


DARK: dict[str, str] = {
    "bg_glass":       "rgba(15, 15, 20, 210)",
    "bg_header":      "rgba(30, 30, 45, 230)",
    "border":         "rgba(255, 255, 255, 40)",
    "text_primary":   "#e8eaf6",
    "text_dim":       "#90939e",
    "accent_green":   "#4caf50",
    "accent_blue":    "#89b4fa",
    "accent_amber":   "#f9a825",
    "code_bg":        "rgba(0, 0, 0, 0.5)",
    "code_inline_bg": "rgba(0, 0, 0, 0.45)",
    "code_text":      "#e5e7eb",
    "code_lang":      "#6b7280",
    "code_inline":    "#a8d8a8",
    "footer_text":    "#7a7d88",
    "dialog_bg":      "#1e1e2e",
    "dialog_field":   "#313244",
    "dialog_border":  "#45475a",
    "dialog_btn":     "#89b4fa",
    "dialog_btn_hi":  "#b4befe",
    "dialog_btn2":    "#45475a",
    "dialog_btn2_hi": "#585b70",
}

LIGHT: dict[str, str] = {
    "bg_glass":       "rgba(245, 245, 250, 220)",
    "bg_header":      "rgba(220, 220, 230, 235)",
    "border":         "rgba(0, 0, 0, 35)",
    "text_primary":   "#1e1e2e",
    "text_dim":       "#5c5f6e",
    "accent_green":   "#2e7d32",
    "accent_blue":    "#1d4ed8",
    "accent_amber":   "#b45309",
    "code_bg":        "rgba(0, 0, 0, 0.06)",
    "code_inline_bg": "rgba(0, 0, 0, 0.08)",
    "code_text":      "#1f2937",
    "code_lang":      "#6b7280",
    "code_inline":    "#166534",
    "footer_text":    "#6b6f7a",
    "dialog_bg":      "#f4f4f8",
    "dialog_field":   "#ffffff",
    "dialog_border":  "#c4c6cf",
    "dialog_btn":     "#1d4ed8",
    "dialog_btn_hi":  "#3b5bdb",
    "dialog_btn2":    "#d4d6df",
    "dialog_btn2_hi": "#bcbec8",
}


class _ThemeStore(QObject):
    theme_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._palette: dict[str, str] = DARK
        self._wired = False

    def palette(self) -> dict[str, str]:
        return self._palette

    def detect_and_apply(self) -> None:
        """Pick palette based on OS colour scheme (Qt 6.5+; falls back to dark)."""
        try:
            hints = QGuiApplication.styleHints()
            # Qt 6.5+: ColorScheme enum on styleHints
            scheme = getattr(hints, "colorScheme", lambda: None)()
            if scheme is not None and int(scheme) == 1:  # Light = 1
                self._palette = LIGHT
            else:
                self._palette = DARK
        except Exception as e:
            logger.debug(f"Theme detect failed, using DARK: {e}")
            self._palette = DARK

        if not self._wired:
            try:
                QGuiApplication.styleHints().colorSchemeChanged.connect(self._on_os_change)
                self._wired = True
            except Exception:
                pass

    def _on_os_change(self, *_a) -> None:
        prev = self._palette
        self.detect_and_apply()
        if self._palette is not prev:
            self.theme_changed.emit()


_store: _ThemeStore | None = None


def store() -> _ThemeStore:
    global _store
    if _store is None:
        _store = _ThemeStore()
        _store.detect_and_apply()
    return _store


def palette() -> dict[str, str]:
    return store().palette()


def on_changed(callback: Callable[[], None]) -> None:
    store().theme_changed.connect(callback)
