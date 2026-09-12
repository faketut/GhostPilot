"""Area-capture coordinate mapping.

Qt reports screen geometry in logical (device-independent) pixels — on a 200%
display the primary screen is 1500x1000 logical — while mss grabs physical
pixels (3000x2000). Passing the logical selection straight to mss used to
capture a top-left, half-size slice of the intended region, so a selected code
block arrived at OCR without its body. These tests pin the conversion.
"""
from __future__ import annotations

import io

from PIL import Image
from PyQt6.QtCore import QRect

from src.area_capture import AreaCapture, _physical_box

# Primary monitor as mss reports it on a 200% display.
_MON_2X = {"left": 0, "top": 0, "width": 3000, "height": 2000}


def test_physical_box_scales_by_device_pixel_ratio():
    """A 400x300 logical selection is a 800x600 physical grab at 2x."""
    box = _physical_box(QRect(100, 50, 400, 300), _MON_2X, 2.0)
    assert box == {"left": 200, "top": 100, "width": 800, "height": 600}


def test_physical_box_is_identity_at_100_percent():
    box = _physical_box(QRect(100, 50, 400, 300), _MON_2X, 1.0)
    assert box == {"left": 100, "top": 50, "width": 400, "height": 300}


def test_physical_box_respects_monitor_origin():
    """A primary monitor with a non-zero virtual-desktop origin is offset."""
    mon = {"left": 1920, "top": -200, "width": 3000, "height": 2000}
    box = _physical_box(QRect(10, 10, 100, 100), mon, 2.0)
    assert box == {"left": 1940, "top": -180, "width": 200, "height": 200}


def test_physical_box_clamps_to_the_monitor():
    """A selection running past the screen edge cannot produce an out-of-bounds box."""
    box = _physical_box(QRect(1400, 900, 400, 300), _MON_2X, 2.0)
    assert box["left"] + box["width"] <= 3000
    assert box["top"] + box["height"] <= 2000
    assert box["width"] >= 1 and box["height"] >= 1


def test_grab_uses_the_physical_box(monkeypatch):
    """The grab itself must receive the scaled box, not the raw logical rect."""
    seen: dict = {}

    class _FakeShot:
        size = (2, 2)
        bgra = bytes(2 * 2 * 4)

    class _FakeSct:
        monitors = [None, _MON_2X]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def grab(self, monitor):
            seen["monitor"] = monitor
            return _FakeShot()

    monkeypatch.setattr("src.area_capture.mss.mss", lambda: _FakeSct())
    out = AreaCapture._grab_rect_png(QRect(100, 50, 400, 300), 2.0)

    assert seen["monitor"] == {"left": 200, "top": 100, "width": 800, "height": 600}
    assert Image.open(io.BytesIO(out)).size == (2, 2)
