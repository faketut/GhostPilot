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
    fused = rm._rrf([1, 2, 3], [3, 2, 1])
    assert set(fused) == {1, 2, 3}
    assert fused[0] == 2  # 2 is mid-rank in both → highest combined
