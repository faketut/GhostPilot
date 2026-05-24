"""Vision pipeline tests: provider plumbing + failover hook wiring."""
from __future__ import annotations

import asyncio
import types
from collections import deque
from unittest.mock import patch

import pytest

from src.llm.base import Delta
from src.llm.failover import FailoverProvider
from src.llm_engine import LLMEngine


def _make_engine(vision_provider) -> LLMEngine:
    eng = LLMEngine.__new__(LLMEngine)
    eng._active_tasks = {}
    eng._text_history = deque(maxlen=0)
    eng._vision_history = deque(maxlen=0)
    eng.last_usage = {}
    eng.vision_provider = vision_provider
    eng._vision_provider = "openai_compatible"
    eng._gather_context = lambda *_a, **_k: ([], None)
    eng._build_interview_messages = lambda *_a, **_k: [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    return eng


async def _drain(q: asyncio.Queue) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


@pytest.mark.asyncio
async def test_vision_step_b_routes_through_provider():
    async def _stream(payload, *, model, system_prompt, max_tokens, temperature):
        # Confirm engine actually called us with the openai-style payload.
        assert isinstance(payload, list) and payload[0]["role"] == "system"
        assert system_prompt == "sys"
        yield Delta(text="hello ")
        yield Delta(text="world")

    fake = types.SimpleNamespace(name="openai-vision", vision_stream=_stream)
    eng = _make_engine(fake)
    ui: asyncio.Queue = asyncio.Queue()
    await eng._vision_step_b(b"\x00\x00", "technical", "what now", ui)
    msgs = await _drain(ui)
    assert [m["text"] for m in msgs if m["type"] == "token"] == ["hello ", "world"]


@pytest.mark.asyncio
async def test_vision_failover_hook_emits_info_event():
    """When primary errors before yielding, failover hook should push an info event."""
    async def _bad(*_a, **_k):
        raise RuntimeError("429 rate limit")
        yield  # pragma: no cover

    async def _good(*_a, **_k):
        yield Delta(text="recovered")

    primary = types.SimpleNamespace(name="openai-vision", vision_stream=_bad)
    fallback = types.SimpleNamespace(name="deepseek-vision", vision_stream=_good)
    fp = FailoverProvider(primary, [fallback])
    eng = _make_engine(fp)

    ui: asyncio.Queue = asyncio.Queue()
    await eng._vision_step_b(b"\x00\x00", "technical", "q", ui)
    msgs = await _drain(ui)

    info = [m for m in msgs if m["type"] == "info"]
    assert len(info) == 1
    assert info[0]["kind"] == "vision"
    assert info[0]["provider"] == "deepseek-vision"
    assert "failover from openai-vision" in info[0]["note"]
    assert "429" in info[0]["error"]
    # And the fallback's tokens did stream through.
    assert any(m.get("text") == "recovered" for m in msgs if m["type"] == "token")


@pytest.mark.asyncio
async def test_vision_payload_shape_for_gemini():
    """Gemini family → payload is a [text, image-part] list, not chat messages."""
    captured = {}

    async def _stream(payload, *, model, system_prompt, max_tokens, temperature):
        captured["payload"] = payload
        captured["system_prompt"] = system_prompt
        yield Delta(text="ok")

    fake = types.SimpleNamespace(name="gemini-flash", vision_stream=_stream)
    eng = _make_engine(fake)
    eng._vision_provider = "gemini"

    # Stub out the genai types.Part.from_bytes import so the test doesn't
    # require the google-genai SDK at runtime.
    fake_part = object()
    fake_types = types.SimpleNamespace(
        Part=types.SimpleNamespace(from_bytes=lambda **_kw: fake_part),
    )
    with patch.dict("sys.modules", {"google": types.SimpleNamespace(genai=types.SimpleNamespace(types=fake_types)),
                                    "google.genai": types.SimpleNamespace(types=fake_types),
                                    "google.genai.types": fake_types}):
        ui: asyncio.Queue = asyncio.Queue()
        await eng._vision_step_b(b"\x00\x00", "technical", "q", ui)

    payload = captured["payload"]
    assert isinstance(payload, list) and len(payload) == 2
    assert payload[0] == "user"  # user_text
    assert payload[1] is fake_part
    assert captured["system_prompt"] == "sys"
