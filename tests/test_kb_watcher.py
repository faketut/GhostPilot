"""Tests for incremental rebuild + mtime staleness detection (v0.8.0)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.rag_manager import RAGManager


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "knowledge"
    d.mkdir()
    (d / "resume.md").write_text("# Resume\n\nPython, async, distributed systems.", encoding="utf-8")
    (d / "notes.md").write_text("# Notes\n\nProject GhostPilot — desktop copilot.", encoding="utf-8")
    return d


def test_rebuild_from_dir_returns_chunk_count(kb_dir: Path):
    rm = RAGManager()
    n = rm.rebuild_from_dir(str(kb_dir))
    assert n > 0


def test_is_stale_false_immediately_after_rebuild(kb_dir: Path):
    rm = RAGManager()
    rm.rebuild_from_dir(str(kb_dir))
    assert rm.is_stale() is False


def test_is_stale_false_when_never_built():
    rm = RAGManager()
    # Without _kb_root set, the watcher must not trigger phantom rebuilds.
    assert rm.is_stale() is False
    assert rm.rebuild_if_stale() is None


def test_is_stale_after_file_edit(kb_dir: Path):
    rm = RAGManager()
    rm.rebuild_from_dir(str(kb_dir))
    # Bump mtime — touch + new content
    target = kb_dir / "resume.md"
    new_mtime = time.time() + 2
    target.write_text(target.read_text() + "\n\nAdditional bullet.", encoding="utf-8")
    os.utime(target, (new_mtime, new_mtime))
    assert rm.is_stale() is True


def test_rebuild_if_stale_returns_count_and_clears_stale(kb_dir: Path):
    rm = RAGManager()
    rm.rebuild_from_dir(str(kb_dir))
    target = kb_dir / "notes.md"
    new_mtime = time.time() + 2
    target.write_text(target.read_text() + "\n\nNew section.", encoding="utf-8")
    os.utime(target, (new_mtime, new_mtime))
    n = rm.rebuild_if_stale()
    assert isinstance(n, int) and n > 0
    # After rebuild, no longer stale.
    assert rm.is_stale() is False
    assert rm.rebuild_if_stale() is None


def test_is_stale_after_new_file_added(kb_dir: Path):
    rm = RAGManager()
    rm.rebuild_from_dir(str(kb_dir))
    (kb_dir / "new_doc.md").write_text("# New\n\nAdded after initial build.", encoding="utf-8")
    assert rm.is_stale() is True


def test_is_stale_after_file_deleted(kb_dir: Path):
    rm = RAGManager()
    rm.rebuild_from_dir(str(kb_dir))
    (kb_dir / "notes.md").unlink()
    assert rm.is_stale() is True
