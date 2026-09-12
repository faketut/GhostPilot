"""What each route injects ahead of the question.

Turns differ in *how* reference material reaches the model:

* behavioral / technical — retrieve snippets from the knowledge base;
* algorithm — the patterns cheatsheet, injected **whole** (a fragment of a
  cheatsheet is worth little, and this route must not depend on retrieval
  quality).

The two are tracked separately: only retrieval may be reported as `rag_hits`.
"""
from __future__ import annotations

import asyncio
import types
from collections import deque

import pytest

from src import prompt_loader
from src.config import config
from src.llm_engine import LLMEngine, TurnContext, _lang_suffix


def _engine(rag=None) -> LLMEngine:
    eng = LLMEngine.__new__(LLMEngine)
    eng._active_tasks = {}
    eng._text_history = deque(maxlen=0)
    eng.last_usage = {}
    eng.rag = rag
    return eng


@pytest.fixture
def cheatsheet(tmp_path, monkeypatch):
    """Point the algorithm route at a temp cheatsheet (keep tests hermetic)."""
    path = tmp_path / "algorithm.md"
    path.write_text("PATTERN: sliding window -> O(n)\n", encoding="utf-8")
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(config, "ALGORITHM_KNOWLEDGE_FILE", "algorithm.md", raising=False)
    monkeypatch.setattr(config, "ALGORITHM_KNOWLEDGE_MAX_CHARS", 8000, raising=False)
    return path


# ── Route selection ──────────────────────────────────────────────────────


def test_algorithm_route_takes_the_cheatsheet_and_does_not_retrieve(cheatsheet):
    """Algorithm turns must not query the knowledge base: a coding question
    scores poorly against a resume, and the cheatsheet is not a retrieval hit."""

    class _ExplodingRAG:
        def search(self, *_a, **_k):
            raise AssertionError("algorithm turn must not query the knowledge base")

    ctx = _engine(_ExplodingRAG())._gather_context("algorithm", "sum two digits")
    assert ctx.snippets == []
    assert ctx.algorithm_ref == "PATTERN: sliding window -> O(n)"


def test_behavioral_and_technical_routes_retrieve_with_their_filters():
    class _RAG:
        def __init__(self):
            self.calls: list[dict] = []

        def search(self, question, **kwargs):
            self.calls.append({"question": question, **kwargs})
            return ["snippet"]

    rag = _RAG()
    eng = _engine(rag)

    assert eng._gather_context("behavioral", "conflict?").snippets == ["snippet"]
    assert eng._gather_context("technical", "index?").snippets == ["snippet"]
    assert len(rag.calls) == 2
    # Behavioral is scoped to resume/JD; technical excludes the JD.
    assert rag.calls[0]["source_filter"]("knowledge/resume.md") is True
    assert rag.calls[0]["source_filter"]("knowledge/notes.md") is False
    assert rag.calls[1]["source_filter"]("knowledge/jd.md") is False
    assert rag.calls[1]["source_filter"]("knowledge/resume.md") is True


def test_missing_cheatsheet_is_not_an_error_and_warns_once(cheatsheet, tmp_path, caplog, monkeypatch):
    """A missing file degrades to no reference — but says so once, not on every
    algorithm question (the per-turn warning used to bury the signal)."""
    monkeypatch.setattr(config, "ALGORITHM_KNOWLEDGE_FILE", "nope.md", raising=False)
    eng = _engine()

    with caplog.at_level("WARNING"):
        first = eng._gather_context("algorithm", "q")
        second = eng._gather_context("algorithm", "q")

    assert first == TurnContext() and second == TurnContext()
    assert sum("cheatsheet" in r.message for r in caplog.records) == 1


def test_oversized_cheatsheet_is_cut_visibly(cheatsheet, monkeypatch):
    """The bound must never look like an absent file."""
    monkeypatch.setattr(config, "ALGORITHM_KNOWLEDGE_MAX_CHARS", 10, raising=False)
    ref = _engine()._gather_context("algorithm", "q").algorithm_ref
    assert ref.startswith("PATTERN: s")
    assert ref.endswith("[cheatsheet truncated]")


# ── Message assembly ─────────────────────────────────────────────────────


def test_cheatsheet_is_injected_as_its_own_reference_block():
    eng = _engine()
    msgs = eng._build_interview_messages(
        "algorithm", "sum two digits", TurnContext(algorithm_ref="PATTERN: two pointers"),
    )
    user = msgs[1]["content"]
    assert "[Algorithm patterns reference]\nPATTERN: two pointers" in user
    # The question still follows the reference.
    assert user.index("PATTERN: two pointers") < user.index("sum two digits")


@pytest.mark.asyncio
async def test_algorithm_turn_injects_once_and_reports_rag_hits_zero(cheatsheet):
    """The cheatsheet reaches the model once, and the footer's `rag:N` stays 0
    — it is an injection, not a retrieval."""
    sent: dict = {}

    def _chat_stream(messages, **_kw):
        sent["messages"] = messages

        async def _gen():
            yield types.SimpleNamespace(text="ok", usage=None, finish_reason="stop")

        return _gen()

    eng = _engine()
    eng.text_provider = types.SimpleNamespace(name="deepseek", chat_stream=_chat_stream)
    eng.register_task = lambda *_a, **_k: None

    ui: asyncio.Queue = asyncio.Queue()
    await eng.generate_answer_stream("sum two digits", ui, q_type="algorithm")

    msgs = sent["messages"]
    assert sum(m["content"].count("PATTERN: sliding window") for m in msgs) == 1
    # The system message still carries the route's instructions (UMPIR spec).
    assert "[I]" in msgs[0]["content"]

    events = []
    while not ui.empty():
        events.append(ui.get_nowait())
    info = next(e for e in events if e["type"] == "info")
    assert info["rag_hits"] == 0
    assert info["algo_chars"] == len("PATTERN: sliding window -> O(n)")


@pytest.mark.asyncio
async def test_prompt_override_still_reaches_the_algorithm_turn(cheatsheet):
    """Session replay A/B-tests prompts through ``override_prompts``; the system
    message remains the single delivery point for the route's instructions."""
    sent: dict = {}

    def _chat_stream(messages, **_kw):
        sent["messages"] = messages

        async def _gen():
            yield types.SimpleNamespace(text="ok", usage=None, finish_reason="stop")

        return _gen()

    eng = _engine()
    eng.text_provider = types.SimpleNamespace(name="deepseek", chat_stream=_chat_stream)
    eng.register_task = lambda *_a, **_k: None

    with prompt_loader.override_prompts({"algorithm": "CUSTOM ALGO PROMPT"}):
        await eng.generate_answer_stream("q", asyncio.Queue(), q_type="algorithm")

    assert sent["messages"][0]["content"] == "CUSTOM ALGO PROMPT" + _lang_suffix()
