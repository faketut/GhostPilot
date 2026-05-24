"""Tests for the FailoverProvider wrapper."""
from __future__ import annotations

import pytest

from src.llm.base import Delta, Usage
from src.llm.failover import FailoverProvider


class _FakeProvider:
    """Minimal LLMProvider duck — matches the chat_stream / chat_complete surface."""

    def __init__(self, name: str, *, raises: Exception | None = None, deltas=None, raise_after: int | None = None):
        self.name = name
        self._raises = raises
        self._deltas = deltas or []
        self._raise_after = raise_after  # raise mid-stream after yielding N deltas
        self.calls = 0

    async def chat_complete(self, *_a, **_k) -> str:
        self.calls += 1
        if self._raises:
            raise self._raises
        return f"{self.name}-ok"

    async def chat_stream(self, *_a, **_k):
        self.calls += 1
        if self._raises and self._raise_after is None:
            raise self._raises
        for i, d in enumerate(self._deltas):
            if self._raise_after is not None and i == self._raise_after:
                raise self._raises  # type: ignore[misc]
            yield d

    async def vision_stream(self, *_a, **_k):
        async for d in self.chat_stream():
            yield d


@pytest.mark.asyncio
async def test_failover_uses_primary_when_healthy():
    primary = _FakeProvider("primary", deltas=[Delta(text="hi", usage=Usage(1, 1))])
    fallback = _FakeProvider("fallback", deltas=[Delta(text="never")])
    fp = FailoverProvider(primary, [fallback])

    chunks = [d async for d in fp.chat_stream([], model="m")]
    assert [c.text for c in chunks] == ["hi"]
    assert fp.last_used is primary
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_failover_switches_when_primary_errors_before_emit():
    primary = _FakeProvider("primary", raises=RuntimeError("429"))
    fallback = _FakeProvider("fallback", deltas=[Delta(text="ok", usage=Usage(2, 3))])
    # retries_per_provider=0 keeps this test focused on the *switch*, not the retry.
    fp = FailoverProvider(primary, [fallback], retries_per_provider=0)

    chunks = [d async for d in fp.chat_stream([], model="m")]
    assert [c.text for c in chunks] == ["ok"]
    assert fp.last_used is fallback
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_failover_propagates_when_primary_errors_mid_stream():
    primary = _FakeProvider(
        "primary",
        raises=RuntimeError("boom"),
        deltas=[Delta(text="partial"), Delta(text="never")],
        raise_after=1,
    )
    fallback = _FakeProvider("fallback", deltas=[Delta(text="should-not-run")])
    fp = FailoverProvider(primary, [fallback])

    out: list[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for d in fp.chat_stream([], model="m"):
            out.append(d.text)
    assert out == ["partial"]
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_failover_chat_complete_retries_chain():
    p = _FakeProvider("p", raises=RuntimeError("x"))
    f1 = _FakeProvider("f1", raises=RuntimeError("y"))
    f2 = _FakeProvider("f2")
    fp = FailoverProvider(p, [f1, f2])

    result = await fp.chat_complete([], model="m")
    assert result == "f2-ok"
    assert fp.last_used is f2


@pytest.mark.asyncio
async def test_failover_raises_last_error_when_all_fail():
    p = _FakeProvider("p", raises=RuntimeError("p-err"))
    f1 = _FakeProvider("f1", raises=RuntimeError("f1-err"))
    fp = FailoverProvider(p, [f1])

    with pytest.raises(RuntimeError, match="f1-err"):
        await fp.chat_complete([], model="m")


@pytest.mark.asyncio
async def test_failover_callback_fires_with_prev_next_error():
    primary = _FakeProvider("primary", raises=RuntimeError("429"))
    fallback = _FakeProvider("fallback", deltas=[Delta(text="ok")])
    events: list[tuple[str, str, str]] = []
    fp = FailoverProvider(
        primary,
        [fallback],
        on_failover=lambda prev, nxt, err: events.append((prev.name, nxt.name, str(err))),
    )

    _ = [d async for d in fp.chat_stream([], model="m")]
    assert events == [("primary", "fallback", "429")]


@pytest.mark.asyncio
async def test_failover_callback_silent_when_no_switch():
    primary = _FakeProvider("primary", deltas=[Delta(text="hi")])
    fallback = _FakeProvider("fallback")
    fired: list = []
    fp = FailoverProvider(primary, [fallback], on_failover=lambda *a: fired.append(a))

    _ = [d async for d in fp.chat_stream([], model="m")]
    assert fired == []


# ── Error classification + backoff ─────────────────────────────────


class _StatusError(Exception):
    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.status_code = status_code


@pytest.mark.asyncio
async def test_classify_fatal_4xx_does_not_failover():
    """401/403/400/404 → raise immediately; never touch fallback."""
    primary = _FakeProvider("primary", raises=_StatusError("Unauthorized", 401))
    fallback = _FakeProvider("fallback", deltas=[Delta(text="should-not-run")])
    fp = FailoverProvider(primary, [fallback])

    with pytest.raises(_StatusError):
        _ = [d async for d in fp.chat_stream([], model="m")]
    assert primary.calls == 1  # no in-place retry on fatal either
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_classify_retryable_429_triggers_in_place_retry(monkeypatch):
    """429 → retry once on primary, then fail over to fallback."""
    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)

    import asyncio as _aio
    monkeypatch.setattr(_aio, "sleep", fake_sleep)

    primary = _FakeProvider("primary", raises=_StatusError("rate limit", 429))
    fallback = _FakeProvider("fallback", deltas=[Delta(text="ok")])
    fp = FailoverProvider(primary, [fallback], retries_per_provider=1, backoff_base=0.1)

    chunks = [d async for d in fp.chat_stream([], model="m")]
    assert [c.text for c in chunks] == ["ok"]
    assert primary.calls == 2  # first attempt + 1 in-place retry
    assert fallback.calls == 1
    assert sleeps == [0.1]  # one backoff before retry


@pytest.mark.asyncio
async def test_classify_5xx_is_retryable():
    primary = _FakeProvider("primary", raises=_StatusError("server boom", 503))
    fallback = _FakeProvider("fallback", deltas=[Delta(text="ok")])
    fp = FailoverProvider(primary, [fallback], retries_per_provider=0, backoff_base=0)
    chunks = [d async for d in fp.chat_stream([], model="m")]
    assert [c.text for c in chunks] == ["ok"]


@pytest.mark.asyncio
async def test_classify_fatal_in_chat_complete_propagates():
    primary = _FakeProvider("primary", raises=_StatusError("bad request", 400))
    fallback = _FakeProvider("fallback")
    fp = FailoverProvider(primary, [fallback])
    with pytest.raises(_StatusError):
        await fp.chat_complete([], model="m")
    assert fallback.calls == 0


def test_classify_error_helper():
    from src.llm.failover import classify_error

    assert classify_error(_StatusError("x", 401)) == "fatal"
    assert classify_error(_StatusError("x", 429)) == "retryable"
    assert classify_error(_StatusError("x", 502)) == "retryable"
    assert classify_error(TimeoutError("slow")) == "retryable"
    assert classify_error(ConnectionError("net")) == "retryable"
    assert classify_error(RuntimeError("Invalid API key")) == "fatal"
    assert classify_error(RuntimeError("nothing to classify")) == "unknown"
