"""
Overlay UI Module
-----------------
A glassmorphism-style, frameless, transparent overlay window that:
  - Is invisible to screen capture (WDA_EXCLUDEFROMCAPTURE)
  - Starts in click-through (stealth) mode, toggled by Alt+A
  - Renders streamed LLM tokens with Markdown-aware formatting
  - Shows a "thinking" pulse animation while the LLM is generating
  - Has a thin drag-handle header for repositioning
  - Displays question type badge (🎯 behavioral / 💻 algorithm / 📖 technical)
"""

import sys
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QScrollArea, QSystemTrayIcon, QMenu, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QSize
from PyQt6.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QBrush, QPen

from src.windows_api import enable_window_stealth, set_window_interaction_mode
from src.config import config
from src.settings_ui import SettingsUI

logger = logging.getLogger(__name__)

# ── Colour palette (matches OpenGhostPilot dark glass aesthetic) ──────────────
_BG_GLASS = "rgba(15, 15, 20, 210)"
_BG_HEADER = "rgba(30, 30, 45, 230)"
_BORDER = "rgba(255, 255, 255, 40)"
_TEXT_PRIMARY = "#e8eaf6"
_TEXT_DIM = "#90939e"
_ACCENT_GREEN = "#4caf50"
_ACCENT_BLUE = "#89b4fa"
_ACCENT_AMBER = "#f9a825"

# Badge colours per question type
_BADGE = {
    "behavioral": ("#d97706", "🎯 行为"),
    "algorithm":  ("#2563eb", "💻 算法"),
    "technical":  ("#7c3aed", "📖 技术"),
    "vision":     ("#065f46", "📸 视觉"),
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

    def _tick(self):
        self._phase = (self._phase + 1) % 6
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for i in range(3):
            active = (i == self._phase % 3)
            radius = 5 if active else 3
            colour = QColor(_ACCENT_GREEN) if active else QColor(100, 100, 100)
            p.setBrush(QBrush(colour))
            p.setPen(Qt.PenStyle.NoPen)
            cx = 6 + i * 14
            cy = 7
            p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)


class _DragHeader(QWidget):
    """Thin drag-handle bar at the top of the overlay."""

    def __init__(self, parent_window: QMainWindow):
        super().__init__(parent_window)
        self._win = parent_window
        self._drag_pos: QPoint | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)

        self._badge = QLabel("● GhostPilot")
        self._badge.setStyleSheet(f"color: {_ACCENT_GREEN}; font-size: 11px; font-weight: 600;")
        layout.addWidget(self._badge)
        layout.addStretch()

        self._type_badge = QLabel("")
        self._type_badge.setStyleSheet(
            "border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: bold;"
        )
        self._type_badge.hide()
        layout.addWidget(self._type_badge)

        self.setFixedHeight(26)
        self.setStyleSheet(
            f"background: {_BG_HEADER}; border-bottom: 1px solid {_BORDER};"
        )

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


class OverlayUI(QMainWindow):
    def __init__(
        self,
        *,
        title: str = "GhostPilot Copilot",
        with_tray: bool = True,
        start_y: int = 20,
        accent: str = "● GhostPilot",
        on_settings_saved=None,
    ):
        super().__init__()
        self.is_interactive = True   # toggled by Alt+A
        self._full_text = ""         # accumulated streamed text
        self._title = title
        self._with_tray = with_tray
        self._start_y = start_y
        self._accent = accent
        self._on_settings_saved = on_settings_saved
        self._initUI()

    def _initUI(self):
        # ── Window flags ─────────────────────────────────────────────────
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(QSize(480, 80))
        self.resize(640, 260)

        # ── Config-driven opacity ─────────────────────────────────────────
        # Note: this affects the entire window (including text). Keep defaults
        # conservative; background glass already uses alpha in CSS.
        self.setWindowOpacity(_clamp01(getattr(config, "OVERLAY_OPACITY", 1.0)))

        # ── OS-level stealth (invisible to screen capture) ────────────────
        if sys.platform == "win32":
            self.show()
            hwnd = int(self.winId())
            enable_window_stealth(hwnd)
            # Default: click-through stealth
            self.is_interactive = False
            set_window_interaction_mode(hwnd, False)

        # ── Root widget with glass background ─────────────────────────────
        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet(f"""
            QWidget#root {{
                background: {_BG_GLASS};
                border: 1px solid {_BORDER};
                border-radius: 10px;
            }}
        """)
        self.setCentralWidget(root)

        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── Drag header ───────────────────────────────────────────────────
        self._header = _DragHeader(self)
        self._header._badge.setText(self._accent)
        vbox.addWidget(self._header)

        # ── Scroll area for streamed text ──────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("background: transparent;")

        self._content = QLabel()
        self._content.setWordWrap(True)
        self._content.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._content.setTextFormat(Qt.TextFormat.RichText)
        self._content.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._content.setStyleSheet(f"""
            color: {_TEXT_PRIMARY};
            font-family: 'Segoe UI', 'PingFang SC', sans-serif;
            font-size: 13px;
            line-height: 1.6;
            padding: 10px 14px;
            background: transparent;
        """)
        self._content.setText(
            f'<span style="color:{_TEXT_DIM}; font-style:italic;">GhostPilot 就绪 · 等待问题…</span>'
        )
        self._scroll.setWidget(self._content)
        vbox.addWidget(self._scroll, 1)

        # ── Thinking animation ────────────────────────────────────────────
        anim_row = QHBoxLayout()
        anim_row.setContentsMargins(12, 4, 12, 6)
        self._pulse = _PulseDot()
        self._thinking_label = QLabel("思考中")
        self._thinking_label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
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

        # ── System tray (optional) ────────────────────────────────────────
        if self._with_tray:
            self._init_tray()

        # Position top-right
        screen = QApplication.primaryScreen().geometry()
        self.setWindowTitle(self._title)
        self.move(screen.width() - self.width() - 20, self._start_y)

    def _init_tray(self):
        px = QPixmap(32, 32)
        px.fill(QColor(0, 200, 80))
        tray = QSystemTrayIcon(QIcon(px), self)
        tray.setToolTip("GhostPilot Copilot")
        menu = QMenu()
        menu.addAction("⚙️ 设置", self._open_settings)
        menu.addSeparator()
        menu.addAction("❌ 退出", QApplication.instance().quit)
        tray.setContextMenu(menu)
        tray.show()
        self._tray = tray

    def _open_settings(self):
        dlg = SettingsUI(self, on_saved=self._on_settings_saved)
        dlg.exec()

    # ── Public API ────────────────────────────────────────────────────────

    def show_thinking(self, q_type: str = ""):
        """Called when a final ASR transcript arrives and LLM has been invoked."""
        # Default behaviour keeps compatibility; caller can preserve history by not clearing via update_text.
        self._full_text = self._full_text or ""
        if q_type:
            self._header.set_question_type(q_type)
        self._anim_frame.show()
        self._pulse.start()

    def append_block(self, text: str):
        """Append a new block to the overlay (keeps history)."""
        if self._full_text:
            self._full_text += "\n\n—\n\n"
        self._full_text += text
        self.update_text("", append=True)  # re-render current full text

    def set_status(self, text: str):
        """Show a lightweight status line without overwriting main content."""
        # Reuse thinking label area to avoid new layout complexity.
        if text:
            self._thinking_label.setText(text)
            self._anim_frame.show()
            if not self._timer_is_running():
                self._pulse.start()
        else:
            self._thinking_label.setText("思考中")
            self._pulse.stop()
            self._anim_frame.hide()

    def _timer_is_running(self) -> bool:
        try:
            return self._pulse._timer.isActive()
        except Exception:
            return False

    def update_text(self, text: str, append: bool = True):
        """Receives streamed tokens from the UI queue."""
        if not append:
            self._full_text = text
        else:
            self._full_text += text

        # Stop thinking animation on first real token
        if self._anim_frame.isVisible() and text:
            self._pulse.stop()
            self._anim_frame.hide()

        # Render as basic HTML (bold, code, newlines)
        html = self._to_html(self._full_text)
        self._content.setText(html)

        # Auto-scroll to bottom
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    @staticmethod
    def _to_html(text: str) -> str:
        """
        Lightweight Markdown-to-HTML: handles **bold**, `code`, code blocks,
        and line breaks — enough for the structured prompts we generate.
        """
        import re
        # Escape HTML special chars first
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Code fences ```lang\n...\n``` → styled block
        def replace_fence(m):
            lang = m.group(1) or "text"
            code = m.group(2).strip()
            return (
                f'<div style="background:rgba(0,0,0,0.5);border-radius:6px;'
                f'border:1px solid rgba(255,255,255,0.12);margin:6px 0;padding:8px 12px;">'
                f'<span style="color:#6b7280;font-size:10px;">{lang}</span><br>'
                f'<pre style="margin:0;color:#e5e7eb;font-family:Consolas,monospace;'
                f'font-size:12px;white-space:pre-wrap;">{code}</pre></div>'
            )
        text = re.sub(r"```(\w*)\n([\s\S]*?)```", replace_fence, text)

        # Inline `code`
        text = re.sub(
            r"`([^`]+)`",
            r'<code style="background:rgba(0,0,0,0.45);border-radius:3px;'
            r'padding:1px 4px;font-family:Consolas,monospace;color:#a8d8a8;">\1</code>',
            text,
        )
        # **bold**
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        # newlines → <br>
        text = text.replace("\n", "<br>")

        return f'<span style="color:{_TEXT_PRIMARY}">{text}</span>'

    def toggle_interaction(self):
        self.is_interactive = not self.is_interactive
        if sys.platform == "win32":
            set_window_interaction_mode(int(self.winId()), self.is_interactive)
        mode = "🖱️ 可拖拽" if self.is_interactive else "👻 隐身穿透"
        logger.info(f"Interaction mode → {mode}")
        # Brief status flash
        self.update_text(
            f"已切换至 {mode} 模式",
            append=False,
        )
        QTimer.singleShot(1500, lambda: self.update_text(self._full_text or "", append=False))

    def set_interaction(self, interactive: bool):
        """Force interaction mode (does not toggle)."""
        self.is_interactive = bool(interactive)
        if sys.platform == "win32":
            set_window_interaction_mode(int(self.winId()), self.is_interactive)
        mode = "🖱️ 可拖拽" if self.is_interactive else "👻 隐身穿透"
        logger.info(f"Interaction mode → {mode}")
        self.update_text(f"已切换至 {mode} 模式", append=False)
        QTimer.singleShot(1500, lambda: self.update_text(self._full_text or "", append=False))
