"""Centralized icon factory using qtawesome (Font Awesome / Material Design).

Falls back to emoji text if qtawesome isn't installed so the app still runs.

Usage:
    from src import icons
    btn.setIcon(icons.icon("stop"))
    tabs.addTab(w, icons.icon("microphone"), "Azure")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtGui import QIcon

# Bundled assets (icons/). Frozen builds extract to sys._MEIPASS; source runs
# use the repo root, one level up from src/.
_BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
APP_ICON_PATH = _BASE_DIR / "assets" / "icon.ico"

try:
    import qtawesome as qta  # type: ignore
    _HAS_QTA = True
except Exception:
    qta = None  # type: ignore
    _HAS_QTA = False


# Semantic name → (qtawesome glyph, emoji fallback)
_NAMES = {
    "stop":         ("fa5s.stop",            "⏹"),
    "save":         ("fa5s.save",            "💾"),
    "settings":     ("fa5s.cog",             "⚙️"),
    "search":       ("fa5s.search",          "🔍"),
    "refresh":      ("fa5s.sync-alt",        "🔄"),
    "test":         ("fa5s.flask",           "🧪"),
    "ghost":        ("fa5s.ghost",           "👻"),
    "cursor":       ("fa5s.mouse-pointer",   "🖱️"),
    "clear":        ("fa5s.broom",           "🧹"),
    "copy":         ("fa5s.copy",            "📋"),
    "microphone":   ("fa5s.microphone",      "🎙️"),
    "camera":       ("fa5s.camera",          "📸"),
    "robot":        ("fa5s.robot",           "🤖"),
    "keyboard":     ("fa5s.keyboard",        "⌨️"),
    "edit":         ("fa5s.edit",            "📝"),
    "record":       ("fa5s.circle",          "●"),
    "language":     ("fa5s.globe",           "🌐"),
    "chart":        ("fa5s.chart-bar",       "📊"),
}


def has_qtawesome() -> bool:
    return _HAS_QTA


def icon(name: str, color: Optional[str] = None) -> QIcon:
    """Return a QIcon for `name`. Returns an empty QIcon if qtawesome is missing
    (callers should fall back to setText(emoji_fallback(name))).
    """
    if not _HAS_QTA:
        return QIcon()
    glyph, _ = _NAMES.get(name, (None, None))
    if not glyph:
        return QIcon()
    kw = {}
    if color:
        kw["color"] = color
    try:
        return qta.icon(glyph, **kw)
    except Exception:
        return QIcon()


def emoji_fallback(name: str) -> str:
    """Return the emoji fallback for `name` (for environments without qtawesome)."""
    _, em = _NAMES.get(name, ("", ""))
    return em


def app_icon() -> QIcon:
    """The GhostPilot application icon: tray, taskbar grouping and dialogs.

    Falls back to the qtawesome ghost glyph when the .ico is missing (e.g. a
    dev checkout without assets/), and to an empty QIcon when neither exists —
    callers must handle a null QIcon.
    """
    if APP_ICON_PATH.exists():
        ic = QIcon(str(APP_ICON_PATH))
        if not ic.isNull():
            return ic
    glyph = icon("ghost", color="#00C850")
    if not glyph.isNull():
        return glyph
    return QIcon()
