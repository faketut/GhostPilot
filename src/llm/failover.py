"""Failover wrapper that retries against secondary providers on primary error.

The wrapper only "fails over" when the primary raises *before* emitting any
streamed output. Once tokens have been delivered to the UI we propagate the
error rather than restart from scratch (the user has already seen partial
text).
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from src.llm.base import Delta, LLMProvider, Usage

logger = logging.getLogger(__name__)


class FailoverProvider(LLMProvider):
    """Wraps a primary `LLMProvider` plus an ordered list of fallbacks."""

    def __init__(self, primary: LLMProvider, fallbacks: list[LLMProvider]):
        self.primary = primary
        self.fallbacks = list(fallbacks)
        self.name = primary.name
        # Tracks which provider served the most recent request.
        self.last_used: LLMProvider = primary

    def _chain(self) -> list[LLMProvider]:
        return [self.primary, *self.fallbacks]

    async def chat_complete(self, messages, *, model, max_tokens=256, temperature=0.1) -> str:
        last_err: Exception | None = None
        for prov in self._chain():
            try:
                self.last_used = prov
                return await prov.chat_complete(
                    messages, model=model, max_tokens=max_tokens, temperature=temperature,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("Provider %s failed (chat_complete): %s — trying next", prov.name, e)
        assert last_err is not None
        raise last_err

    async def _stream_with_failover(
        self, method: str, *args, **kwargs,
    ) -> AsyncIterator[Delta]:
        last_err: Exception | None = None
        for prov in self._chain():
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
                logger.warning("Provider %s failed (%s): %s — trying next", prov.name, method, e)
                continue
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


__all__ = ["FailoverProvider", "Usage"]
