"""
Area Capture Module
-------------------
Provides a semi-transparent, full-screen drag-to-select overlay so the user can
capture an exact rectangular region of the screen.  The captured region is returned
as raw PNG bytes (in memory — never written to disk).

Usage:
    area_bytes = AreaCapture.capture()   # blocks until selection is done / cancelled
    # area_bytes is None if the user pressed Escape or selected a degenerate region
"""

import sys
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
    from PyQt6.QtGui import QPainter, QColor, QPen, QScreen, QPixmap
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logger.warning("PyQt6 not available — AreaCapture will fall back to full-screen capture.")


class _SelectionOverlay(QWidget):
    """
    A full-screen, semi-transparent widget that lets the user drag a rectangle.
    After the user releases the mouse the selected region is stored in `self.selection`.
    """

    closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.selection: Optional[QRect] = None
        self._origin: Optional[QPoint] = None
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

    def closeEvent(self, event):
        try:
            self.closed.emit()
        finally:
            super().closeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw the desktop snapshot as the background
        painter.drawPixmap(self.rect(), self._background)

        # Dark translucent overlay over everything outside the selection
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if not self._rubber.isNull():
            # Cut out (highlight) the selected region
            painter.fillRect(self._rubber, QColor(0, 0, 0, 0))

            # Draw a bright border around the selection
            pen = QPen(QColor(0, 200, 255), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self._rubber)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._cancelled = True
            self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.pos()
            self._rubber = QRect(self._origin, self._origin)
            self.update()

    def mouseMoveEvent(self, event):
        if self._origin:
            self._rubber = QRect(self._origin, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._origin:
            final = QRect(self._origin, event.pos()).normalized()
            if final.width() > 10 and final.height() > 10:
                self.selection = final
            self.close()


class AreaCapture:
    """
    Public interface for triggering an interactive area capture.
    Returns PNG bytes (in-memory) or None if cancelled.
    """

    @staticmethod
    def capture() -> Optional[bytes]:
        """
        Displays a full-screen drag-to-select overlay and returns the selected region
        as compressed JPEG bytes ready to be passed to the vision LLM.
        Returns None if the user cancelled.
        """
        if not HAS_QT:
            return AreaCapture._full_screen_fallback()

        overlay = _SelectionOverlay()
        # Legacy sync capture kept for compatibility.
        # Prefer capture_async() in async apps (qasync) to avoid nested loops.
        app = QApplication.instance()
        if app is None:
            logger.error("QApplication instance not found; cannot run area capture.")
            return None

        # Wait by pumping events (lightweight) with a tiny sleep to avoid CPU spin.
        # This avoids nested Qt event loops, but still blocks the caller thread.
        while overlay.isVisible():
            app.processEvents()
            try:
                import time
                time.sleep(0.005)
            except Exception:
                pass

        if overlay._cancelled or overlay.selection is None:
            logger.info("Area capture cancelled.")
            return None

        return AreaCapture._grab_rect_png(overlay.selection)

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

        # Grab pixels off the UI thread (can be slow on some machines).
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
