"""Failover wrapper that retries against secondary providers on primary error.

The wrapper only "fails over" when the primary raises *before* emitting any
streamed output. Once tokens have been delivered to the UI we propagate the
error rather than restart from scratch (the user has already seen partial
text).

Errors are classified before deciding whether to retry/failover:

- **retryable** (network glitches, timeouts, 429, 5xx) → retry on same
  provider up to ``retries_per_provider`` times with exponential backoff,
  then fail over.
- **fatal** (auth 401/403, bad-request 400, model-not-found 404) → skip
  failover entirely and raise immediately. Switching providers won't fix
  a malformed request or a missing API key.
- **unknown** → treat as retryable (failover but no in-place retry).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator, Callable, Optional

from src.llm.base import Delta, LLMProvider, Usage

logger = logging.getLogger(__name__)


_FATAL_STATUS = {400, 401, 403, 404}
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_EXC_NAMES = {
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
    "PoolTimeout", "TimeoutException", "RemoteProtocolError",
    "APIConnectionError", "APITimeoutError",
}


def classify_error(err: BaseException) -> str:
    """Return ``"fatal"``, ``"retryable"``, or ``"unknown"`` for ``err``.

    Inspects ``status_code``/``status``/``response.status_code`` attrs first,
    then exception class name, then falls back to a regex on ``str(err)`` for
    HTTP-ish substrings like ``"429"`` or ``"401 Unauthorized"``.
    """
    status = (
        getattr(err, "status_code", None)
        or getattr(err, "status", None)
        or getattr(getattr(err, "response", None), "status_code", None)
    )
    if isinstance(status, int):
        if status in _FATAL_STATUS:
            return "fatal"
        if status in _RETRYABLE_STATUS or 500 <= status < 600:
            return "retryable"

    name = type(err).__name__
    if name in _RETRYABLE_EXC_NAMES:
        return "retryable"
    if isinstance(err, (asyncio.TimeoutError, ConnectionError, TimeoutError)):
        return "retryable"

    msg = str(err)
    m = re.search(r"\b(\d{3})\b", msg)
    if m:
        code = int(m.group(1))
        if code in _FATAL_STATUS:
            return "fatal"
        if code in _RETRYABLE_STATUS or 500 <= code < 600:
            return "retryable"
    if re.search(r"\b(rate.?limit|timeout|timed out|temporarily unavailable)\b", msg, re.I):
        return "retryable"
    if re.search(r"\b(unauthorized|forbidden|invalid api key|bad request)\b", msg, re.I):
        return "fatal"

    return "unknown"


class FailoverProvider(LLMProvider):
    """Wraps a primary `LLMProvider` plus an ordered list of fallbacks."""

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider],
        on_failover: Optional[Callable[[LLMProvider, LLMProvider, Exception], None]] = None,
        retries_per_provider: int = 1,
        backoff_base: float = 0.5,
    ):
        self.primary = primary
        self.fallbacks = list(fallbacks)
        self.name = primary.name
        # Tracks which provider served the most recent request.
        self.last_used: LLMProvider = primary
        # Optional hook fired when a switch happens (prev, next, error).
        self.on_failover = on_failover
        self.retries_per_provider = max(0, int(retries_per_provider))
        self.backoff_base = float(backoff_base)

    def _chain(self) -> list[LLMProvider]:
        return [self.primary, *self.fallbacks]

    async def _backoff(self, attempt: int) -> None:
        delay = self.backoff_base * (2 ** attempt)
        await asyncio.sleep(delay)

    async def chat_complete(self, messages, *, model, max_tokens=256, temperature=0.1) -> str:
        last_err: Exception | None = None
        prev: LLMProvider | None = None
        for prov in self._chain():
            if prev is not None and last_err is not None:
                self._notify(prev, prov, last_err)
            for attempt in range(self.retries_per_provider + 1):
                try:
                    self.last_used = prov
                    return await prov.chat_complete(
                        messages, model=model, max_tokens=max_tokens, temperature=temperature,
                    )
                except Exception as e:  # noqa: BLE001
                    kind = classify_error(e)
                    last_err = e
                    if kind == "fatal":
                        logger.warning(
                            "Provider %s raised fatal error (%s): %s — not failing over",
                            prov.name, type(e).__name__, e,
                        )
                        raise
                    if kind == "retryable" and attempt < self.retries_per_provider:
                        logger.info(
                            "Provider %s retryable error (%s); attempt %d/%d after backoff",
                            prov.name, type(e).__name__, attempt + 1, self.retries_per_provider,
                        )
                        await self._backoff(attempt)
                        continue
                    prev = prov
                    logger.warning(
                        "Provider %s failed (chat_complete): %s — trying next", prov.name, e,
                    )
                    break
        assert last_err is not None
        raise last_err

    def _notify(self, prev: LLMProvider, nxt: LLMProvider, err: Exception) -> None:
        if self.on_failover is None:
            return
        try:
            self.on_failover(prev, nxt, err)
        except Exception:  # noqa: BLE001
            logger.exception("on_failover callback raised; ignoring")

    async def _stream_with_failover(
        self, method: str, *args, **kwargs,
    ) -> AsyncIterator[Delta]:
        last_err: Exception | None = None
        prev: LLMProvider | None = None
        for prov in self._chain():
            if prev is not None and last_err is not None:
                self._notify(prev, prov, last_err)
            for attempt in range(self.retries_per_provider + 1):
                self.last_used = prov
                gen = getattr(prov, method)(*args, **kwargs)
                emitted = False
                try:
                    async for delta in gen:
                        emitted = True
                        yield delta
                    return  # finished cleanly
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if emitted:
                        # Already streamed text — don't restart on a different provider.
                        logger.warning("Provider %s failed mid-stream: %s — propagating", prov.name, e)
                        raise
                    kind = classify_error(e)
                    if kind == "fatal":
                        logger.warning(
                            "Provider %s raised fatal error (%s): %s — not failing over",
                            prov.name, type(e).__name__, e,
                        )
                        raise
                    if kind == "retryable" and attempt < self.retries_per_provider:
                        logger.info(
                            "Provider %s retryable error (%s); attempt %d/%d after backoff",
                            prov.name, type(e).__name__, attempt + 1, self.retries_per_provider,
                        )
                        await self._backoff(attempt)
                        continue
                    prev = prov
                    logger.warning("Provider %s failed (%s): %s — trying next", prov.name, method, e)
                    break
        assert last_err is not None
        raise last_err

    def chat_stream(self, messages, *, model, max_tokens=512, temperature=0.25):
        return self._stream_with_failover(
            "chat_stream", messages, model=model, max_tokens=max_tokens, temperature=temperature,
        )

    def vision_stream(self, messages_or_parts, *, model, system_prompt, max_tokens=550, temperature=0.25):
        return self._stream_with_failover(
            "vision_stream",
            messages_or_parts,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


__all__ = ["FailoverProvider", "Usage", "classify_error"]
