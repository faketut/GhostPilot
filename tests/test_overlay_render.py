"""Rendering contract for streamed answers.

The overlay receives an answer one token at a time and renders it incrementally.
Code structure must survive that: indentation, tab-separated columns and the
code-block treatment itself. These tests drive the real widget (`update_text`,
the same entry point `ui_updater` uses) and assert on what a reader actually
sees — the plain text of the rendered document.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.overlay_ui import OverlayUI  # noqa: E402

CODE_ANSWER = (
    "[U] iterative reverse, O(n)\n"
    "\n"
    "```python\n"
    "def reverse(head):\n"
    "    prev = None\n"
    "    while head:\n"
    "        nxt = head.next\n"
    "        head.next = prev\n"
    "        head = nxt\n"
    "    return prev\n"
    "```\n"
)
TABBED_LINE = "[R] two pointers\t[E] O(n) time\t[A] left/right\t[C] sorted input\n"


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def _render(qt_app, text: str, *, chunk: int = 3) -> str:
    """Stream `text` through the real overlay widget, token by token.

    Starts with the `clear` update `ui_updater` sends before a turn, so the
    widget is in the same state it is in when a real answer streams in.
    """
    w = OverlayUI(title="test", with_tray=False, start_y=0, accent="ASR")
    try:
        w.update_text("", append=False)
        for i in range(0, len(text), chunk):
            w.update_text(text[i:i + chunk], append=True)
        return w._content.toPlainText().replace("\u00a0", " ")
    finally:
        w.close()


def test_code_indentation_survives_token_streaming(qt_app):
    out = _render(qt_app, CODE_ANSWER)
    lines = out.splitlines()
    for source_line in ("    prev = None", "        nxt = head.next", "    return prev"):
        assert source_line in lines, f"indentation lost: {source_line!r}\nin: {out!r}"


def test_open_fence_renders_as_code_not_literal_markers(qt_app):
    """Mid-stream there is no closing fence; the block must still be code."""
    out = _render(qt_app, "```python\ndef f():\n    return 1\n")
    assert "```" not in out, f"fence markers leaked into the answer: {out!r}"
    assert "    return 1" in out.splitlines()


def test_tab_separated_columns_stay_visible(qt_app):
    """Prompts emit tab-delimited lines; HTML would collapse a tab to one space."""
    out = _render(qt_app, TABBED_LINE)
    assert "[R] two pointers    [E] O(n) time" in out, repr(out)


def test_streamed_render_matches_single_shot(qt_app):
    """Incremental and whole-text rendering must agree for the same input."""
    streamed = _render(qt_app, CODE_ANSWER)

    w = OverlayUI(title="test", with_tray=False, start_y=0, accent="ASR")
    try:
        w.update_text("", append=False)
        w.update_text(CODE_ANSWER, append=False)
        single = w._content.toPlainText().replace("\u00a0", " ")
    finally:
        w.close()

    # Only the document's trailing newline may differ (setHtml drops it).
    assert streamed.rstrip("\n") == single.rstrip("\n")


def test_code_content_is_not_reinterpreted_as_markdown(qt_app):
    """`**`, backticks and `<`/`&` inside a fence are literal source text."""
    answer = "```python\nif a < b and c & d:\n    x = a ** 2  # `tick`\n```\n"
    out = _render(qt_app, answer)
    assert "if a < b and c & d:" in out, repr(out)
    assert "    x = a ** 2  # `tick`" in out, repr(out)
    assert "&lt;" not in out and "&amp;" not in out, repr(out)


def test_prose_formatting_applies_to_a_complete_segment(qt_app):
    """Inline spans need the whole construct in one render.

    Deltas are rendered as they arrive, so a `**span**` split across two
    provider chunks keeps its markers (pre-existing; a re-render on the next
    fence/trim/clear repairs it). Rendering a segment whole is what the
    full-render path does, and formatting must hold there.
    """
    out = _render(qt_app, "Use **two pointers** and `while` here.\n", chunk=10_000)
    assert out.strip() == "Use two pointers and while here."
