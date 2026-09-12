from __future__ import annotations

import threading

import numpy as np
import pytest

from src.rag_manager import RAGManager, kb_relpath_from_chunk


def test_tokenize_ascii():
    assert RAGManager._tokenize("Hello, World! Foo-bar") == ["hello", "world", "foo", "bar"]


def test_tokenize_cjk():
    toks = RAGManager._tokenize("你好世界 hello")
    assert "hello" in toks
    assert any("你" in t for t in toks)


def test_kb_relpath_parse():
    assert kb_relpath_from_chunk("[kb:resume.md#3]\nbody") == "resume.md"
    assert kb_relpath_from_chunk("no header here") is None


def test_rrf_combines():
    rm = RAGManager.__new__(RAGManager)  # bypass __init__
    # 2 is rank-1 in the first list and rank-1 in the second → unambiguous winner.
    fused = rm._rrf([2, 1, 3], [2, 3, 1])
    assert set(fused) == {1, 2, 3}
    assert fused[0] == 2


# ── Dense (sentence-transformers) load lifecycle ─────────────────────────


class _FakeST:
    """Stands in for SentenceTransformer: 2 dims per doc, deterministic."""

    def __init__(self, _name):
        pass

    def encode(self, texts, convert_to_numpy=True):
        return np.array([[float(len(t) % 5 + 1), 1.0] for t in texts])


@pytest.fixture
def _dense(monkeypatch):
    """Enable a fake, working sentence-transformers in the imported module."""
    import src.rag_manager as rm

    monkeypatch.setattr(rm, "SentenceTransformer", _FakeST)
    monkeypatch.setattr(rm, "HAS_ST", True)
    return rm


def test_load_documents_does_not_block_on_the_dense_probe(_dense, monkeypatch):
    """The ST/torch import must happen off the caller's thread.

    A broken torch install costs ~10s to *fail*; paying that on the main
    thread delayed every launch (overlays were created afterwards). The
    background thread is held open here, so a synchronous probe would hang
    the load_documents call instead of returning.
    """
    rm = _dense
    release = threading.Event()
    probed_on: list[threading.Thread] = []
    probed = threading.Event()

    def _held_probe():
        probed_on.append(threading.current_thread())
        probed.set()
        release.wait(10)  # hold the load open until we are done asserting

    monkeypatch.setattr(rm, "_probe_st", _held_probe)
    rm_inst = RAGManager()
    try:
        rm_inst.load_documents(["[kb:a.md#0]\nhello world"])
        # Returned while the loader is still parked inside the probe.
        assert probed.wait(5), "dense probe never ran"
        assert probed_on[0] is not threading.current_thread()
        assert rm_inst.model is None
        # ...and search is already usable via BM25.
        assert rm_inst._bm25 is not None
    finally:
        release.set()


def test_background_dense_load_publishes_embeddings(_dense):
    """Once the model is warm, search upgrades from BM25 to hybrid ranking."""
    rm_inst = RAGManager()
    rm_inst.load_documents(["[kb:a.md#0]\nfirst document", "[kb:b.md#0]\nsecond"])
    rm_inst._load_dense_model()  # what the background thread does
    assert rm_inst.embeddings is not None
    assert rm_inst.embeddings.shape[0] == 2


def test_rebuild_replaces_stale_embeddings_with_the_new_documents(_dense):
    """Vectors must never stay paired with a different chunk list.

    Left stale, `search` would rank new documents by old vectors — and index
    past the end of a shrunken KB.
    """
    rm_inst = RAGManager()
    rm_inst.load_documents(["[kb:a.md#0]\none", "[kb:b.md#0]\ntwo", "[kb:c.md#0]\nthree"])
    rm_inst._load_dense_model()
    assert rm_inst.embeddings.shape[0] == 3

    rm_inst.load_documents(["[kb:a.md#0]\nonly chunk now"])
    assert rm_inst.embeddings.shape[0] == 1
    assert rm_inst.search("only chunk", top_k=3, min_score=0.0)

