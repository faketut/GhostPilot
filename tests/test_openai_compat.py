"""Unit tests for OpenAICompatProvider.chat_stream (no real network)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.llm import openai_compat as oc


# ── Fake stream plumbing ─────────────────────────────────────────────────

class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c
        return gen()

    async def close(self):
        self.closed = True


class _FakeChatCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(chunks))


def _delta_chunk(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
        usage=None,
    )


def _usage_chunk(in_tok: int, out_tok: int):
    # Final chunk with no choices but populated usage (OpenAI's pattern when
    # include_usage=True).
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
    )


def _empty_choice_chunk():
    # Some providers send a choices entry with delta=None as a keepalive.
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=None)],
        usage=None,
    )


def _collect(provider, **kwargs):
    async def run():
        out = []
        async for d in provider.chat_stream([{"role": "user", "content": "hi"}], **kwargs):
            out.append(d)
        return out
    return asyncio.run(run())


# ── Tests ────────────────────────────────────────────────────────────────

def test_b64_image_url_data_uri():
    url = oc.OpenAICompatProvider.b64_image_url(b"abc", mime="image/png")
    assert url.startswith("data:image/png;base64,")
    # Decodes back to original bytes.
    import base64
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == b"abc"


def test_chat_stream_yields_text_deltas_and_usage(monkeypatch):
    p = oc.OpenAICompatProvider(api_key="k", label="openai")
    chunks = [
        _delta_chunk("Hel"),
        _delta_chunk("lo"),
        _usage_chunk(7, 2),
    ]
    fake = _FakeClient(chunks)
    p._client = fake

    deltas = _collect(p, model="gpt-4o-mini")
    texts = [d.text for d in deltas]
    assert "".join(texts) == "Hello"
    # Last delta carries the usage.
    usages = [d.usage for d in deltas if d.usage is not None]
    assert len(usages) == 1
    assert usages[0].in_tokens == 7
    assert usages[0].out_tokens == 2


def test_chat_stream_drops_empty_keepalive_chunks(monkeypatch):
    p = oc.OpenAICompatProvider(api_key="k", label="openai")
    chunks = [
        _empty_choice_chunk(),         # no text, no usage → skipped
        _delta_chunk("ok"),
        _empty_choice_chunk(),
    ]
    p._client = _FakeClient(chunks)

    deltas = _collect(p, model="gpt-4o-mini")
    assert len(deltas) == 1
    assert deltas[0].text == "ok"


def test_chat_stream_closes_underlying_stream():
    p = oc.OpenAICompatProvider(api_key="k", label="openai")
    chunks = [_delta_chunk("x")]
    fake = _FakeClient(chunks)
    p._client = fake

    _collect(p, model="gpt-4o-mini")
    # The stream produced by create() should have been closed in the finally.
    # We can't capture the returned _FakeStream directly; instead, patch create
    # to expose it.
    captured: dict = {}
    orig_create = fake.chat.completions.create

    async def spy(**kwargs):
        s = await orig_create(**kwargs)
        captured["stream"] = s
        return s

    fake.chat.completions.create = spy
    _collect(p, model="gpt-4o-mini")
    assert captured["stream"].closed is True


def test_ollama_base_url_disables_include_usage():
    """Ollama doesn't support stream_options.include_usage → must be omitted."""
    p = oc.OpenAICompatProvider(api_key="ollama", base_url="http://localhost:11434/v1", label="ollama")
    p._client = _FakeClient([_delta_chunk("x")])
    _collect(p, model="llama3.1")
    kwargs = p._client.chat.completions.last_kwargs
    assert kwargs is not None
    # Either omitted entirely or explicitly None — both are acceptable.
    assert kwargs.get("stream_options") in (None,)


def test_openai_base_url_enables_include_usage():
    p = oc.OpenAICompatProvider(api_key="k", label="openai")  # base_url=None → OpenAI default
    p._client = _FakeClient([_delta_chunk("x")])
    _collect(p, model="gpt-4o-mini")
    kwargs = p._client.chat.completions.last_kwargs
    assert kwargs is not None
    assert kwargs.get("stream_options") == {"include_usage": True}


def test_vision_stream_delegates_to_chat_stream():
    p = oc.OpenAICompatProvider(api_key="k", label="openai")
    p._client = _FakeClient([_delta_chunk("answer")])

    async def run():
        out = []
        async for d in p.vision_stream(
            [{"role": "user", "content": [{"type": "text", "text": "describe"}]}],
            model="gpt-4o", system_prompt="be brief",
        ):
            out.append(d)
        return out

    out = asyncio.run(run())
    assert "".join(d.text for d in out) == "answer"
