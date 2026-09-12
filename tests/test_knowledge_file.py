"""Whole-file knowledge loading (`knowledge/algorithm.md` and friends).

Single-file injection is a different job from the chunked RAG index: the caller
wants the whole document, resolved the same way the index resolves its root
(bundle-aware), with a visible marker if a size bound cuts it.
"""
from __future__ import annotations

from src.knowledge_loader import load_knowledge_file, resolve_knowledge_file


def test_resolves_bare_name_and_relative_path(tmp_path):
    (tmp_path / "algorithm.md").write_text("A", encoding="utf-8")
    assert resolve_knowledge_file(str(tmp_path), "algorithm.md") == tmp_path / "algorithm.md"
    assert resolve_knowledge_file(str(tmp_path), "missing.md") is None


def test_resolves_absolute_path(tmp_path):
    f = tmp_path / "dsa.md"
    f.write_text("B", encoding="utf-8")
    assert resolve_knowledge_file("unused-root", str(f)) == f
    assert resolve_knowledge_file("unused-root", str(tmp_path / "nope.md")) is None


def test_finds_the_file_at_any_depth(tmp_path):
    """Users organise notes in folders; a bare filename should still resolve."""
    nested = tmp_path / "dsa" / "patterns"
    nested.mkdir(parents=True)
    (nested / "algorithm.md").write_text("C", encoding="utf-8")
    assert resolve_knowledge_file(str(tmp_path), "algorithm.md") == nested / "algorithm.md"


def test_missing_root_or_empty_name_is_none(tmp_path):
    assert resolve_knowledge_file(str(tmp_path / "absent"), "algorithm.md") is None
    assert resolve_knowledge_file(str(tmp_path), "") is None
    assert resolve_knowledge_file(str(tmp_path), "   ") is None


def test_loads_text_and_reports_path(tmp_path):
    (tmp_path / "algorithm.md").write_text("  body  \n", encoding="utf-8")
    text, path = load_knowledge_file(str(tmp_path), "algorithm.md")
    assert text == "body"
    assert path == tmp_path / "algorithm.md"


def test_missing_file_yields_empty_text(tmp_path):
    assert load_knowledge_file(str(tmp_path), "algorithm.md") == ("", None)


def test_size_bound_cuts_with_a_marker(tmp_path):
    (tmp_path / "algorithm.md").write_text("x" * 50, encoding="utf-8")
    text, _ = load_knowledge_file(str(tmp_path), "algorithm.md", max_chars=10)
    assert text.startswith("x" * 10)
    assert text.endswith("[cheatsheet truncated]")


def test_no_bound_leaves_the_document_intact(tmp_path):
    body = "y" * 50
    (tmp_path / "algorithm.md").write_text(body, encoding="utf-8")
    text, _ = load_knowledge_file(str(tmp_path), "algorithm.md", max_chars=0)
    assert text == body
