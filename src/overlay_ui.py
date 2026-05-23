"""
Overlay UI Module
-----------------
A glassmorphism-style, frameless, transparent overlay window that:
  - Is invisible to screen capture (WDA_EXCLUDEFROMCAPTURE)
  - Starts in click-through (stealth) mode, toggled by Alt+A
  - Renders streamed LLM tokens with Markdown-aware formatting into a
    QTextBrowser (delta-append → smooth streaming, no per-token re-render)
  - Shows a "thinking" pulse animation while the LLM is generating
  - Has a thin drag-handle header with state/clear/copy controls
  - Displays question type badge (🎯 behavioral / 💻 algorithm / 📖 technical)
  - Persists per-overlay geometry across sessions, clamped to the visible screen
  - Optional Windows 11 Mica/Acrylic backdrop (silent no-op elsewhere)
"""

import json
import os
import re
import sys
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSystemTrayIcon, QMenu, QTextBrowser, QToolButton, QSizeGrip,
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QSize
from PyQt6.QtGui import (
    QColor, QIcon, QPixmap, QPainter, QBrush, QGuiApplication, QTextCursor,
    QShortcut, QKeySequence,
)

from src.windows_api import enable_window_stealth, set_window_interaction_mode, enable_mica
from src.config import config
from src.settings_ui import SettingsUI
from src import theme
from src import icons

logger = logging.getLogger(__name__)

# Pygments is optional. Without it we fall back to a plain <pre> code block.
try:
    from pygments import highlight as _pyg_highlight
    from pygments.lexers import get_lexer_by_name as _pyg_get_lexer
    from pygments.util import ClassNotFound as _PygClassNotFound
    from pygments.formatters import HtmlFormatter as _PygHtmlFormatter
    _PYG_FORMATTER = _PygHtmlFormatter(noclasses=True, nowrap=False, style="monokai")
    HAS_PYGMENTS = True
except Exception:
    HAS_PYGMENTS = False

# Precompiled Markdown regexes (perf — issue #16)
_RE_FENCE = re.compile(r"```(\w*)\n([\s\S]*?)```")
_RE_INLINE = re.compile(r"`([^`]+)`")
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")

CONFIG_FILE = "config.json"
_GEOMETRY_SAVE_DEBOUNCE_MS = 500
_STICKY_BOTTOM_TOLERANCE_PX = 4


# Badge colours per question type (label + base hex; alpha applied at runtime)
_BADGE = {
    "behavioral": ("#d97706", "Behavioral"),
    "algorithm":  ("#2563eb", "Algorithm"),
    "technical":  ("#7c3aed", "Technical"),
    "vision":     ("#065f46", "Vision"),
}


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 1.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class _PulseDot(QWidget):
    """Three bouncing dots, mirroring OpenGhostPilot's loading animation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 14)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._timer.start(120)
        self.show()

    def stop(self):
        self._timer.stop()
        self.hide()

    def is_running(self) -> bool:
        return self._timer.isActive()

    def _tick(self):
        self._phase = (self._phase + 1) % 6
        self.update()

    def paintEvent(self, _):
        pal = theme.palette()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for i in range(3):
            active = (i == self._phase % 3)
            radius = 5 if active else 3
            colour = QColor(pal["accent_green"]) if active else QColor(100, 100, 100)
            p.setBrush(QBrush(colour))
            p.setPen(Qt.PenStyle.NoPen)
            cx = 6 + i * 14
            cy = 7
            p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)


class _DragHeader(QWidget):
    """Thin drag-handle bar at the top of the overlay, with controls cluster."""

    def __init__(
        self,
        parent_window: QMainWindow,
        *,
        on_clear,
        on_copy,
        on_toggle_interaction,
        on_stop=None,
        on_export=None,
    ):
        super().__init__(parent_window)
        self._win = parent_window
        self._drag_pos: Optional[QPoint] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)

        self._badge = QLabel("● GhostPilot")
        layout.addWidget(self._badge)
        layout.addStretch()

        self._type_badge = QLabel("")
        self._type_badge.hide()
        layout.addWidget(self._type_badge)

        def _tb(icon_name: str, tip: str, slot) -> QToolButton:
            b = QToolButton(self)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedSize(22, 22)
            b.clicked.connect(slot)
            ic = icons.icon(icon_name, color=theme.palette()["text_dim"])
            if not ic.isNull():
                b.setIcon(ic)
                from PyQt6.QtCore import QSize
                b.setIconSize(QSize(14, 14))
            else:
                b.setText(icons.emoji_fallback(icon_name))
            return b

        # Header controls: lock state · stop · clear · copy · export
        self._lock_btn = _tb("ghost", "Click-through (Alt+A to toggle)", on_toggle_interaction)
        self._stop_btn = _tb("stop", "Stop generation (Esc)", on_stop or (lambda: None))
        self._stop_btn.setVisible(False)
        self._clear_btn = _tb("clear", "Clear", on_clear)
        self._copy_btn = _tb("copy", "Copy last answer", on_copy)
        self._export_btn = _tb("save", "Export conversation to ~/Documents/GhostPilot", on_export or (lambda: None))
        layout.addWidget(self._lock_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._clear_btn)
        layout.addWidget(self._copy_btn)
        layout.addWidget(self._export_btn)

        self.setFixedHeight(26)
        self._apply_style()

    def _apply_style(self) -> None:
        pal = theme.palette()
        self._badge.setStyleSheet(
            f"color: {pal['accent_green']}; font-size: 11px; font-weight: 600;"
        )
        btn_css = (
            f"QToolButton {{ color: {pal['text_dim']}; background: transparent; "
            f"border: 0; font-size: 12px; }} "
            f"QToolButton:hover {{ color: {pal['text_primary']}; }}"
        )
        for b in (self._lock_btn, self._stop_btn, self._clear_btn, self._copy_btn, self._export_btn):
            b.setStyleSheet(btn_css)
        self.setStyleSheet(
            f"background: {pal['bg_header']}; border-bottom: 1px solid {pal['border']};"
        )

    def refresh_theme(self) -> None:
        self._apply_style()
        # Re-apply type badge colour with current style scaffold (badge has its own colour)

    def set_lock_state(self, interactive: bool) -> None:
        if interactive:
            ic = icons.icon("cursor", color=theme.palette()["text_dim"])
            if not ic.isNull():
                self._lock_btn.setIcon(ic)
            else:
                self._lock_btn.setText(icons.emoji_fallback("cursor"))
            self._lock_btn.setToolTip("Interactive (Alt+A to lock)")
        else:
            ic = icons.icon("ghost", color=theme.palette()["text_dim"])
            if not ic.isNull():
                self._lock_btn.setIcon(ic)
            else:
                self._lock_btn.setText(icons.emoji_fallback("ghost"))
            self._lock_btn.setToolTip("Click-through (Alt+A to toggle)")

    def set_question_type(self, q_type: str):
        colour, label = _BADGE.get(q_type, ("#555", q_type))
        self._type_badge.setText(label)
        self._type_badge.setStyleSheet(
            f"background: {colour}33; color: {colour}; border: 1px solid {colour}55;"
            "border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: bold;"
        )
        self._type_badge.show()

    def clear_badge(self):
        self._type_badge.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self._win.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos:
            self._win.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, _):
        self._drag_pos = None


def _read_config_json() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read {CONFIG_FILE}: {e}")
        return {}


def _write_config_json(updates: dict) -> None:
    data = _read_config_json()
    data.update(updates)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save geometry to {CONFIG_FILE}: {e}")


def _clamp_geometry(geom: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = geom
    screens = QGuiApplication.screens()
    if not screens:
        return geom
    # Pick the screen the saved top-left falls in, or the primary.
    target = QGuiApplication.primaryScreen()
    for s in screens:
        if s.availableGeometry().contains(QPoint(x, y)):
            target = s
            break
    avail = target.availableGeometry()
    w = max(320, min(int(w), avail.width()))
    h = max(80, min(int(h), avail.height()))
    x = max(avail.left(), min(int(x), avail.right() - w))
    y = max(avail.top(), min(int(y), avail.bottom() - h))
    return x, y, w, h


class OverlayUI(QMainWindow):
    """`max_conversation_blocks`: trim ASR history to the last N blocks (split on `—` separators). None or 0 = no limit."""

    CONVERSATION_BLOCK_SEP = "\n\n—\n\n"

    def __init__(
        self,
        *,
        title: str = "GhostPilot Copilot",
        with_tray: bool = True,
        start_y: int = 20,
        accent: str = "● GhostPilot",
        on_settings_saved=None,
        max_conversation_blocks: int | None = None,
        geometry_key: str = "OVERLAY_ASR_GEOMETRY",
        hotkey_hint: str | None = None,
        on_stop=None,
        on_clear=None,
        on_rebuild_kb=None,
    ):
        super().__init__()
        self.is_interactive = True   # toggled by Alt+A
        self._full_text = ""         # accumulated streamed text (history)
        self._max_conversation_blocks = max_conversation_blocks
        self._title = title
        self._with_tray = with_tray
        self._start_y = start_y
        self._accent = accent
        self._on_settings_saved = on_settings_saved
        self._on_stop = on_stop
        self._on_clear = on_clear
        self._on_rebuild_kb = on_rebuild_kb
        self._geometry_key = geometry_key
        self._hotkey_hint = hotkey_hint
        self._status_flash_timer: Optional[QTimer] = None
        self._geom_save_timer: Optional[QTimer] = None
        self._usage_text: str = ""
        self._info_text: str = ""
        self._initUI()
        theme.on_changed(self._on_theme_changed)

    # ── Init / styling ───────────────────────────────────────────────────

    def _apply_root_style(self) -> None:
        pal = theme.palette()
        self.centralWidget().setStyleSheet(f"""
            QWidget#root {{
                background: {pal['bg_glass']};
                border: 1px solid {pal['border']};
                border-radius: 10px;
            }}
        """)
        self._content.setStyleSheet(f"""
            QTextBrowser {{
                color: {pal['text_primary']};
                font-family: 'Segoe UI', 'PingFang SC', sans-serif;
                font-size: 13px;
                background: transparent;
                border: 0;
                padding: 10px 14px;
            }}
            QScrollBar:vertical {{ width: 8px; background: transparent; }}
            QScrollBar::handle:vertical {{
                background: {pal['border']};
                border-radius: 4px;
                min-height: 24px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)
        self._thinking_label.setStyleSheet(f"color: {pal['text_dim']}; font-size: 11px;")
        self._footer.setStyleSheet(f"color: {pal['footer_text']}; font-size: 10px; padding: 0 12px 4px 12px;")

    def _on_theme_changed(self) -> None:
        self._apply_root_style()
        self._header.refresh_theme()
        # Re-render existing body so colour-bearing inline HTML matches the new theme.
        self._rerender_full()

    def _initUI(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(QSize(360, 80))
        self.resize(640, 260)

        self.setWindowOpacity(_clamp01(getattr(config, "OVERLAY_OPACITY", 1.0)))

        # OS-level stealth & native backdrop (Windows only)
        if sys.platform == "win32":
            self.show()
            hwnd = int(self.winId())
            enable_window_stealth(hwnd)
            enable_mica(hwnd, acrylic=True)
            self.is_interactive = False
            set_window_interaction_mode(hwnd, False)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Header
        self._header = _DragHeader(
            self,
            on_clear=self.clear,
            on_copy=self.copy_last_block,
            on_toggle_interaction=self.toggle_interaction,
            on_stop=self._handle_stop,
            on_export=self.export_conversation,
        )
        self._header._badge.setText(self._accent)
        vbox.addWidget(self._header)

        # Esc → stop generation (works only when overlay is interactive)
        self._esc_shortcut = QShortcut(QKeySequence("Escape"), self)
        self._esc_shortcut.activated.connect(self._handle_stop)

        # Streamed-text widget (issue #16): QTextBrowser + delta append.
        self._content = QTextBrowser()
        self._content.setReadOnly(True)
        self._content.setOpenExternalLinks(False)
        self._content.setFrameShape(QTextBrowser.Shape.NoFrame)
        self._content.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._content.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._content.setStyleSheet("background: transparent;")
        # Placeholder
        pal = theme.palette()
        self._content.setHtml(
            f'<span style="color:{pal["text_dim"]}; font-style:italic;">'
            f'GhostPilot ready · waiting for question…</span>'
        )
        vbox.addWidget(self._content, 1)

        # Thinking animation row
        anim_row = QHBoxLayout()
        anim_row.setContentsMargins(12, 4, 12, 6)
        self._pulse = _PulseDot()
        self._thinking_label = QLabel("Thinking…")
        self._anim_frame = QWidget()
        inner = QHBoxLayout(self._anim_frame)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(6)
        inner.addWidget(self._pulse)
        inner.addWidget(self._thinking_label)
        inner.addStretch()
        self._anim_frame.hide()
        anim_row.addWidget(self._anim_frame)
        vbox.addLayout(anim_row)

        # Persistent hotkey footer + size grip (issues #25, #26)
        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(0)
        self._footer = QLabel(self._build_footer_text())
        footer_row.addWidget(self._footer, 1)
        grip = QSizeGrip(root)
        grip.setFixedSize(14, 14)
        footer_row.addWidget(grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        vbox.addLayout(footer_row)

        self._apply_root_style()

        self._status_flash_timer = QTimer(self)
        self._status_flash_timer.setSingleShot(True)

        # Debounced geometry save
        self._geom_save_timer = QTimer(self)
        self._geom_save_timer.setSingleShot(True)
        self._geom_save_timer.setInterval(_GEOMETRY_SAVE_DEBOUNCE_MS)
        self._geom_save_timer.timeout.connect(self._save_geometry)

        # System tray
        if self._with_tray:
            self._init_tray()

        # Initial geometry: load saved (clamped) or use sensible default top-right.
        self.setWindowTitle(self._title)
        self._apply_initial_geometry()
        self._header.set_lock_state(self.is_interactive)

    def _build_footer_text(self) -> str:
        bits = [
            f"{config.SCREENSHOT_HOTKEY} screenshot",
            f"{getattr(config, 'ASR_INTERACTION_HOTKEY', 'alt+a')} drag",
            f"{getattr(config, 'FORCE_STEALTH_HOTKEY', 'alt+s')} stealth",
        ]
        if self._hotkey_hint:
            bits.insert(0, self._hotkey_hint)
        text = "  ·  ".join(bits)
        suffix_bits = []
        if self._info_text:
            suffix_bits.append(self._info_text)
        if self._usage_text:
            suffix_bits.append(self._usage_text)
        if suffix_bits:
            text = f"{text}    │   " + "  ·  ".join(suffix_bits)
        return text

    def refresh_footer(self) -> None:
        self._footer.setText(self._build_footer_text())

    def set_usage_footer(self, text: str) -> None:
        """Display a token / cost suffix in the footer (Phase 2.3)."""
        self._usage_text = text or ""
        self.refresh_footer()

    def set_info_footer(self, provider: str = "", rag_hits: int | None = None) -> None:
        """Display a small "provider · rag:N" status in the footer."""
        parts = []
        if provider:
            parts.append(provider)
        if rag_hits is not None:
            parts.append(f"rag:{rag_hits}")
        self._info_text = "  ·  ".join(parts)
        self.refresh_footer()

    def _apply_initial_geometry(self) -> None:
        saved = getattr(config, self._geometry_key, None)
        if saved and isinstance(saved, (list, tuple)) and len(saved) == 4:
            try:
                x, y, w, h = _clamp_geometry(tuple(int(v) for v in saved))  # type: ignore[arg-type]
                self.setGeometry(x, y, w, h)
                return
            except Exception as e:
                logger.warning(f"Bad saved geometry {saved!r}: {e}")
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 20, screen.top() + self._start_y)

    # ── Tray / settings ──────────────────────────────────────────────────

    def _init_tray(self):
        px = QPixmap(32, 32)
        px.fill(QColor(0, 200, 80))
        tray = QSystemTrayIcon(QIcon(px), self)
        tray.setToolTip("GhostPilot Copilot")
        menu = QMenu()
        settings_act = menu.addAction("Settings", self._open_settings)
        ic_settings = icons.icon("settings")
        if not ic_settings.isNull():
            settings_act.setIcon(ic_settings)
        menu.addSeparator()
        self._record_action = menu.addAction("Start recording", self._toggle_recording)
        ic_rec = icons.icon("record", color="#d32f2f")
        if not ic_rec.isNull():
            self._record_action.setIcon(ic_rec)
        menu.addSeparator()
        menu.addAction("Quit", QApplication.instance().quit)
        tray.setContextMenu(menu)
        tray.show()
        self._tray = tray

    def _toggle_recording(self):
        from src.session_recorder import recorder
        if recorder.is_recording():
            out = recorder.stop()
            self._record_action.setText("Start recording")
            ic = icons.icon("record", color="#d32f2f")
            if not ic.isNull():
                self._record_action.setIcon(ic)
            self._flash_bottom_status(f"Recording saved: {out.name if out else '?'}", duration_ms=2500)
        else:
            d = recorder.start()
            if d:
                self._record_action.setText("Stop recording")
                ic = icons.icon("stop")
                if not ic.isNull():
                    self._record_action.setIcon(ic)
                self._flash_bottom_status("Recording…", duration_ms=1500)
            else:
                self._flash_bottom_status("Recording failed to start", duration_ms=2000)

    def _open_settings(self):
        dlg = SettingsUI(self, on_saved=self._on_settings_saved, on_rebuild_kb=self._on_rebuild_kb)
        dlg.exec()

    # ── Public API ────────────────────────────────────────────────────────

    def set_streaming(self, streaming: bool) -> None:
        """Show/hide the ⏹ Stop button. Called by main.py around stream tasks."""
        try:
            self._header._stop_btn.setVisible(bool(streaming))
        except Exception:
            pass

    def _handle_stop(self) -> None:
        """Invoke the on_stop callback (if any) and flash a status."""
        if self._on_stop is not None:
            try:
                self._on_stop()
                self._flash_bottom_status("Stopping…", duration_ms=900)
            except Exception as e:
                logger.warning(f"Stop callback failed: {e}")

    def show_thinking(self, q_type: str = ""):
        self._full_text = self._full_text or ""
        if q_type:
            self._header.set_question_type(q_type)
        self._anim_frame.show()
        self._pulse.start()

    def _trim_conversation_blocks(self) -> bool:
        """Return True if any blocks were trimmed (means the body must be re-rendered)."""
        mb = self._max_conversation_blocks
        if mb is None or mb <= 0 or not self._full_text:
            return False
        parts = self._full_text.split(self.CONVERSATION_BLOCK_SEP)
        if len(parts) <= mb:
            return False
        self._full_text = self.CONVERSATION_BLOCK_SEP.join(parts[-mb:])
        return True

    def _retrim_and_render(self) -> None:
        self._trim_conversation_blocks()
        self._rerender_full()

    def _rerender_full(self) -> None:
        html = self._to_html(self._full_text)
        self._content.setHtml(html)
        self._schedule_scroll_to_bottom()

    def append_block(self, text: str):
        """Append a new conversation block (keeps history)."""
        if self._full_text:
            self._full_text += self.CONVERSATION_BLOCK_SEP
        self._full_text += text
        trimmed = self._trim_conversation_blocks()
        if trimmed:
            self._rerender_full()
        else:
            # Append only the new content's HTML.
            self._append_delta_html(self.CONVERSATION_BLOCK_SEP + text if self._full_text != text else text)

    def clear(self):
        """Empty the body but keep state/thinking indicator intact."""
        self._full_text = ""
        self._content.clear()
        self._header.clear_badge()
        if self._on_clear is not None:
            try:
                self._on_clear()
            except Exception:
                pass

    def copy_last_block(self):
        """Copy the most recent Q/A block to the clipboard."""
        if not self._full_text:
            self._flash_bottom_status("Nothing to copy", duration_ms=1200)
            return
        last = self._full_text.split(self.CONVERSATION_BLOCK_SEP)[-1]
        try:
            QGuiApplication.clipboard().setText(last)
            self._flash_bottom_status("Copied ✓", duration_ms=1200)
        except Exception as e:
            logger.warning(f"Clipboard copy failed: {e}")

    def export_conversation(self) -> None:
        """Save the visible Q/A history to a timestamped markdown file."""
        if not self._full_text.strip():
            self._flash_bottom_status("Nothing to export", duration_ms=1500)
            return
        import datetime
        from pathlib import Path
        out_dir = Path.home() / "Documents" / "GhostPilot"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = out_dir / f"session-{stamp}.md"
            header = f"# GhostPilot session — {stamp}\n\nWindow: {self._title}\n\n---\n\n"
            path.write_text(header + self._full_text, encoding="utf-8")
            self._flash_bottom_status(f"Saved ✓ {path.name}", duration_ms=2500)
            logger.info("Exported conversation to %s", path)
        except Exception as e:
            logger.warning("Export failed: %s", e)
            self._flash_bottom_status(f"Export failed: {e}", duration_ms=2500)

    def set_status(self, text: str):
        """Show a lightweight status line without overwriting main content."""
        if text:
            self._thinking_label.setText(text)
            self._anim_frame.show()
            if not self._pulse.is_running():
                self._pulse.start()
        else:
            self._thinking_label.setText("Thinking…")
            self._pulse.stop()
            self._anim_frame.hide()

    def update_text(self, text: str, append: bool = True):
        """Receives streamed tokens from the UI queue (perf — issue #16)."""
        if not append:
            self._full_text = text
            trimmed = False
            self._content.clear()
            if text:
                self._append_delta_html(text)
        else:
            self._full_text += text
            trimmed = self._trim_conversation_blocks()
            if trimmed:
                self._rerender_full()
            elif text:
                self._append_delta_html(text)

        # Stop thinking animation on first real token
        if self._anim_frame.isVisible() and text:
            self._pulse.stop()
            self._anim_frame.hide()

    def _append_delta_html(self, text: str) -> None:
        """Append only the delta as HTML (sticky-bottom — issue #18)."""
        sb = self._content.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - _STICKY_BOTTOM_TOLERANCE_PX
        cursor = self._content.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(self._to_html(text))
        if at_bottom:
            self._schedule_scroll_to_bottom()

    def _schedule_scroll_to_bottom(self) -> None:
        # Read scrollbar maximum AFTER layout (perf — issue #16).
        def _do_scroll():
            sb = self._content.verticalScrollBar()
            sb.setValue(sb.maximum())
        QTimer.singleShot(0, _do_scroll)

    # ── HTML rendering ────────────────────────────────────────────────────

    @staticmethod
    def _to_html(text: str) -> str:
        """
        Lightweight Markdown-to-HTML: handles **bold**, `code`, fenced code
        blocks (with optional Pygments syntax highlighting), and line breaks.
        Uses module-level precompiled regexes.
        """
        pal = theme.palette()

        # Escape HTML special chars first
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def replace_fence(m):
            lang = m.group(1) or "text"
            code = m.group(2).rstrip()
            if HAS_PYGMENTS:
                try:
                    lexer = _pyg_get_lexer(lang.strip().lower() or "text")
                    inner = _pyg_highlight(code, lexer, _PYG_FORMATTER)
                    return (
                        f'<div style="background:{pal["code_bg"]};border-radius:6px;'
                        f'border:1px solid {pal["border"]};margin:6px 0;padding:8px 12px;">'
                        f'<span style="color:{pal["code_lang"]};font-size:10px;">{lang}</span>'
                        f'{inner}</div>'
                    )
                except _PygClassNotFound:
                    pass
                except Exception:
                    pass
            return (
                f'<div style="background:{pal["code_bg"]};border-radius:6px;'
                f'border:1px solid {pal["border"]};margin:6px 0;padding:8px 12px;">'
                f'<span style="color:{pal["code_lang"]};font-size:10px;">{lang}</span><br>'
                f'<pre style="margin:0;color:{pal["code_text"]};font-family:Consolas,monospace;'
                f'font-size:12px;white-space:pre-wrap;">{code}</pre></div>'
            )
        text = _RE_FENCE.sub(replace_fence, text)

        text = _RE_INLINE.sub(
            f'<code style="background:{pal["code_inline_bg"]};border-radius:3px;'
            f'padding:1px 4px;font-family:Consolas,monospace;color:{pal["code_inline"]};">'
            r'\1</code>',
            text,
        )
        text = _RE_BOLD.sub(r"<b>\1</b>", text)
        text = text.replace("\n", "<br>")

        return f'<span style="color:{pal["text_primary"]}">{text}</span>'

    # ── Interaction / stealth ────────────────────────────────────────────

    def _flash_bottom_status(self, message: str, duration_ms: int = 1500) -> None:
        timer = self._status_flash_timer
        if timer is None:
            return
        try:
            timer.timeout.disconnect()
        except TypeError:
            pass
        timer.stop()

        prev_text = self._thinking_label.text()
        prev_anim_visible = self._anim_frame.isVisible()
        prev_pulse = self._pulse.is_running()

        self._thinking_label.setText(message)
        self._anim_frame.show()
        self._pulse.stop()

        def restore():
            self._thinking_label.setText(prev_text)
            if prev_anim_visible:
                self._anim_frame.show()
                if prev_pulse:
                    self._pulse.start()
            else:
                self._anim_frame.hide()
                self._pulse.stop()

        timer.timeout.connect(restore)
        timer.start(duration_ms)

    def toggle_interaction(self):
        self.is_interactive = not self.is_interactive
        if sys.platform == "win32":
            set_window_interaction_mode(int(self.winId()), self.is_interactive)
        mode = "Interactive" if self.is_interactive else "Click-through"
        logger.info(f"Interaction mode → {mode}")
        self._header.set_lock_state(self.is_interactive)
        self._flash_bottom_status(f"Switched to {mode}")

    def set_interaction(self, interactive: bool):
        self.is_interactive = bool(interactive)
        if sys.platform == "win32":
            set_window_interaction_mode(int(self.winId()), self.is_interactive)
        mode = "Interactive" if self.is_interactive else "Click-through"
        logger.info(f"Interaction mode → {mode}")
        self._header.set_lock_state(self.is_interactive)
        self._flash_bottom_status(f"Switched to {mode}")

    # ── Geometry persistence (issue #19) ─────────────────────────────────

    def moveEvent(self, e):
        super().moveEvent(e)
        self._schedule_geometry_save()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._schedule_geometry_save()

    def _schedule_geometry_save(self) -> None:
        if self._geom_save_timer is not None:
            self._geom_save_timer.start()

    def _save_geometry(self) -> None:
        g = self.geometry()
        value = [g.x(), g.y(), g.width(), g.height()]
        try:
            setattr(config, self._geometry_key, value)
            _write_config_json({self._geometry_key: value})
        except Exception as e:
            logger.debug(f"Geometry persist failed: {e}")
