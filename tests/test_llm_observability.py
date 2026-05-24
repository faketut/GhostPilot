"""Tests for the v0.8.0 observability + latency telemetry payload shape."""
from __future__ import annotations

import asyncio
import types
from collections import deque

import pytest

from src.llm import Usage
from src.llm_engine import LLMEngine


def _make_engine(*, text_provider_name: str = "openai", deltas=None):
    """Build an LLMEngine bypassing real client init."""
    eng = LLMEngine.__new__(LLMEngine)
    # Minimal required state.
    eng._active_tasks = {}
    eng._text_history = deque(maxlen=0)
    eng._vision_history = deque(maxlen=0)
    eng.last_usage = {}

    # Fake provider that yields the supplied deltas.
    async def _chat_stream(*_a, **_k):
        for d in deltas or []:
            yield d

    eng.text_provider = types.SimpleNamespace(name=text_provider_name, chat_stream=_chat_stream)

    # Stub out helpers that touch real state.
    eng._gather_context = lambda *_a, **_k: ([], None)
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
