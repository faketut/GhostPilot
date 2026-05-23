"""
Area Capture Module
-------------------
Provides a semi-transparent, full-screen drag-to-select overlay so the user can
capture an exact rectangular region of the screen.  The captured region is returned
as raw PNG bytes (in memory — never written to disk).

Usage:
    image_bytes = await AreaCapture.capture_async()
    # image_bytes is None if the user pressed Escape or selected a degenerate region
"""

import logging
import io
from typing import Optional

import asyncio
import mss
import mss.tools
from PIL import Image

logger = logging.getLogger(__name__)

# Qt is imported lazily to avoid pulling it in at module-level before QApplication exists
try:
    from PyQt6.QtWidgets import QWidget, QApplication
    from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
    from PyQt6.QtGui import QPainter, QColor, QPen, QScreen, QPixmap, QFont
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logger.warning("PyQt6 not available — AreaCapture will fall back to full-screen capture.")


_MIN_SELECTION_PX = 10


class _SelectionOverlay(QWidget):
    """
    A full-screen, semi-transparent widget that lets the user drag a rectangle.
    After the user releases the mouse the selected region is stored in `self.selection`.

    Keyboard shortcuts:
      - Esc                 → cancel
      - Enter / Return      → confirm current selection
      - Arrow keys          → grow/shrink the selection by 1px (Shift = 10px)
    """

    closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.selection: Optional[QRect] = None
        self._origin: Optional[QPoint] = None
        self._cursor_pos: Optional[QPoint] = None
        self._rubber: QRect = QRect()
        self._cancelled: bool = False

        # Grab the current desktop screenshot as a background image
        screen: QScreen = QApplication.primaryScreen()
        self._background: QPixmap = screen.grabWindow(0)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        # Cover the full primary screen
        geo = screen.geometry()
        self.setGeometry(geo)
        self.showFullScreen()
        self.setFocus()

    def closeEvent(self, event):
        try:
            self.closed.emit()
        finally:
            super().closeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self._background)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if not self._rubber.isNull():
            painter.fillRect(self._rubber, QColor(0, 0, 0, 0))
            pen = QPen(QColor(0, 200, 255), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self._rubber)
            self._paint_size_label(painter)

    def _paint_size_label(self, painter: QPainter) -> None:
        w = self._rubber.width()
        h = self._rubber.height()
        if w <= 0 or h <= 0:
            return
        text = f"{w} × {h}"
        font = QFont("Segoe UI", 10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(text) + 12
        text_h = metrics.height() + 6

        anchor = self._cursor_pos or self._rubber.bottomRight()
        x = anchor.x() + 14
        y = anchor.y() + 14
        sw = self.width()
        sh = self.height()
        if x + text_w > sw:
            x = anchor.x() - text_w - 14
        if y + text_h > sh:
            y = anchor.y() - text_h - 14
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        bg = QRect(x, y, text_w, text_h)
        painter.fillRect(bg, QColor(0, 0, 0, 180))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(bg, Qt.AlignmentFlag.AlignCenter, text)

    def _confirm(self) -> None:
        if self._rubber.isNull():
            self.close()
            return
        rect = self._rubber.normalized()
        if rect.width() > _MIN_SELECTION_PX and rect.height() > _MIN_SELECTION_PX:
            self.selection = rect
        self.close()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._cancelled = True
            self.close()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._confirm()
            return

        if self._rubber.isNull():
            return
        step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
        r = QRect(self._rubber)
        if key == Qt.Key.Key_Left:
            r.setRight(max(r.left() + _MIN_SELECTION_PX, r.right() - step))
        elif key == Qt.Key.Key_Right:
            r.setRight(min(self.width() - 1, r.right() + step))
        elif key == Qt.Key.Key_Up:
            r.setBottom(max(r.top() + _MIN_SELECTION_PX, r.bottom() - step))
        elif key == Qt.Key.Key_Down:
            r.setBottom(min(self.height() - 1, r.bottom() + step))
        else:
            return
        self._rubber = r.normalized()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.pos()
            self._cursor_pos = event.pos()
            self._rubber = QRect(self._origin, self._origin)
            self.update()

    def mouseMoveEvent(self, event):
        if self._origin:
            self._cursor_pos = event.pos()
            self._rubber = QRect(self._origin, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._origin:
            self._cursor_pos = event.pos()
            self._rubber = QRect(self._origin, event.pos()).normalized()
            self._confirm()


class AreaCapture:
    """
    Public interface for triggering an interactive area capture.
    Returns PNG bytes (in-memory) or None if cancelled.
    """

    @staticmethod
    async def capture_async() -> Optional[bytes]:
        """
        Async-friendly area capture (for qasync).
        Creates the overlay in the Qt UI thread and awaits window close without
        entering nested event loops.
        """
        if not HAS_QT:
            return AreaCapture._full_screen_fallback()

        overlay = _SelectionOverlay()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()

        def _on_closed():
            if fut.done():
                return
            fut.set_result((overlay._cancelled, overlay.selection))

        overlay.closed.connect(_on_closed)

        cancelled, rect = await fut
        if cancelled or rect is None:
            logger.info("Area capture cancelled.")
            return None

        return await asyncio.to_thread(AreaCapture._grab_rect_png, rect)

    @staticmethod
    def _grab_rect_png(rect) -> bytes:
        logger.info(f"Area selected: {rect.x()},{rect.y()} {rect.width()}x{rect.height()}")
        with mss.mss() as sct:
            monitor = {
                "top": rect.y(),
                "left": rect.x(),
                "width": rect.width(),
                "height": rect.height(),
            }
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _full_screen_fallback() -> bytes:
        """Falls back to capturing the entire primary monitor when PyQt6 is not available."""
        logger.warning("Using full-screen fallback capture (no area selection).")
        with mss.mss() as sct:
            sct_img = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
