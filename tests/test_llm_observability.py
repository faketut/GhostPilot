"""Tests for the v0.8.0 observability + latency telemetry payload shape."""
from __future__ import annotations

import asyncio
import types
from collections import deque

import pytest

from src.config import config
from src.llm import Usage
from src.llm_engine import LLMEngine, TurnContext


def _make_engine(*, text_provider_name: str = "openai", deltas=None):
    """Build an LLMEngine bypassing real client init."""
    eng = LLMEngine.__new__(LLMEngine)
    # Minimal required state.
    eng._active_tasks = {}
    eng._text_history = deque(maxlen=0)
    eng.last_usage = {}

    # Fake provider that yields the supplied deltas.
    async def _chat_stream(*_a, **_k):
        for d in deltas or []:
            yield d

    eng.text_provider = types.SimpleNamespace(name=text_provider_name, chat_stream=_chat_stream)

    # Stub out helpers that touch real state.
    eng._gather_context = lambda *_a, **_k: TurnContext()
    eng._build_interview_messages = lambda *_a, **_k: [
        {"role": "system", "content": ""},
        {"role": "user", "content": "?"},
    ]
    eng.register_task = lambda *_a, **_k: None
    return eng


async def _drain(q: asyncio.Queue) -> list[dict]:
    out: list[dict] = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


@pytest.mark.asyncio
async def test_info_event_emits_provider_and_rag_hits():
    eng = _make_engine(text_provider_name="deepseek")
    ui = asyncio.Queue()
    await eng.generate_answer_stream("hi", ui, q_type="technical")
    msgs = await _drain(ui)
    info = next(m for m in msgs if m["type"] == "info")
    assert info["provider"] == "deepseek"
    assert info["rag_hits"] == 0


@pytest.mark.asyncio
async def test_usage_event_includes_latency_and_provider():
    delta = types.SimpleNamespace(text="hello", usage=None)
    final = types.SimpleNamespace(
        text="",
        usage=Usage(in_tokens=12, out_tokens=34),
    )
    eng = _make_engine(text_provider_name="openai", deltas=[delta, final])
    ui = asyncio.Queue()
    await eng.generate_answer_stream("hi", ui, q_type="behavioral")
    msgs = await _drain(ui)
    usage = next(m for m in msgs if m["type"] == "usage")
    assert usage["in"] == 12
    assert usage["out"] == 34
    assert usage["provider"] == "openai"
    assert isinstance(usage["total_ms"], int) and usage["total_ms"] >= 0
    assert isinstance(usage["ttft_ms"], int) and usage["ttft_ms"] >= 0
    # ttft must arrive no later than total
    assert usage["ttft_ms"] <= usage["total_ms"]


@pytest.mark.asyncio
async def test_ttft_remains_none_when_no_text_emitted():
    """If provider yields only a usage frame (no text), ttft_ms stays None."""
    final = types.SimpleNamespace(text="", usage=Usage(in_tokens=5, out_tokens=0))
    eng = _make_engine(deltas=[final])
    ui = asyncio.Queue()
    await eng.generate_answer_stream("?", ui, q_type="technical")
    msgs = await _drain(ui)
    usage = next(m for m in msgs if m["type"] == "usage")
    assert usage["ttft_ms"] is None


@pytest.mark.asyncio
async def test_truncated_answer_names_the_reasoning_share():
    """A cut answer must say what cut it.

    `deepseek-flash` is a reasoning model: the hidden chain-of-thought is billed
    against the output budget *before* the answer is written, so the visible text
    can stop right after a code fence opens — a function header with no body.
    Reconstructing that after the fact is impossible, so the notice carries the
    reasoning figure that ate the budget.

    Uncapped (the default) the provider's own limit did the cutting, so blaming a
    configuration value we no longer send would be wrong.
    """
    partial = types.SimpleNamespace(
        text="[I]\n```python\ndef solution(n):\n", usage=None, finish_reason=None,
    )
    final = types.SimpleNamespace(
        text="",
        usage=Usage(in_tokens=587, out_tokens=1200, reasoning_tokens=980),
        finish_reason="length",
    )
    eng = _make_engine(deltas=[partial, final])
    ui = asyncio.Queue()
    await eng.generate_answer_stream("?", ui, q_type="algorithm")
    msgs = await _drain(ui)

    usage = next(m for m in msgs if m["type"] == "usage")
    assert usage["reasoning"] == 980

    notice = next(m["text"] for m in msgs if m["type"] == "token" and "truncated" in m["text"])
    assert "the model's own output limit" in notice
    assert "980" in notice and "reasoning" in notice


@pytest.mark.asyncio
async def test_truncation_notice_names_a_configured_cap_when_one_is_set(monkeypatch):
    """If the user *does* set a cap, the notice must point at that setting."""
    monkeypatch.setattr(config, "ANSWER_MAX_TOKENS", 300, raising=False)
    final = types.SimpleNamespace(
        text="", usage=Usage(out_tokens=300, reasoning_tokens=120), finish_reason="length",
    )
    eng = _make_engine(deltas=[final])
    ui = asyncio.Queue()
    await eng.generate_answer_stream("?", ui, q_type="algorithm")
    msgs = await _drain(ui)

    notice = next(m["text"] for m in msgs if m["type"] == "token" and "truncated" in m["text"])
    assert "300-token ANSWER_MAX_TOKENS cap" in notice


def test_zero_means_uncapped(monkeypatch):
    """0 must resolve to "no cap" — passing 0 would create a zero-length budget."""
    from src.llm_engine import _answer_token_cap

    monkeypatch.setattr(config, "ANSWER_MAX_TOKENS", 0, raising=False)
    assert _answer_token_cap() is None

    monkeypatch.setattr(config, "ANSWER_MAX_TOKENS", 2000, raising=False)
    assert _answer_token_cap() == 2000


# ── Task registration ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_second_answer_in_one_task_does_not_cancel_that_task():
    """`asr_router` is a single long-lived task awaiting an answer per question.

    Registering the second answer supersedes the first registration — which is
    the same task object. Cancelling it killed the router, and because the router
    swallows CancelledError the only symptom was an answer that never arrived
    (every second question). Uses a real engine so the registration logic runs.
    """
    import types

    from src.llm.base import Delta

    eng = LLMEngine.__new__(LLMEngine)
    eng._active_tasks = {}
    eng._text_history = deque(maxlen=0)
    eng.last_usage = {}

    async def _chat_stream(*_a, **_k):
        yield Delta(text="answer")

    eng.text_provider = types.SimpleNamespace(name="stub", chat_stream=_chat_stream)
    eng._gather_context = lambda *_a, **_k: TurnContext()
    eng._build_interview_messages = lambda *_a, **_k: [{"role": "user", "content": "?"}]

    ui: asyncio.Queue = asyncio.Queue()

    async def router():
        for i in range(3):
            await eng.generate_answer_stream(f"q{i}", ui, q_type="technical")
        return "survived"

    task = asyncio.get_running_loop().create_task(router())
    assert await asyncio.wait_for(task, timeout=5) == "survived"


@pytest.mark.asyncio
async def test_a_new_stream_still_cancels_a_previous_one():
    """The supersede behaviour must survive the self-cancel guard: a Stop on a
    running answer still has to cancel it."""
    eng = LLMEngine.__new__(LLMEngine)
    eng._active_tasks = {}

    first = asyncio.get_running_loop().create_task(asyncio.sleep(30))
    await asyncio.sleep(0)  # let it start
    eng.register_task("text", first)

    second = asyncio.get_running_loop().create_task(asyncio.sleep(30))
    await asyncio.sleep(0)
    eng.register_task("text", second)

    with pytest.raises(asyncio.CancelledError):
        await first
    second.cancel()
