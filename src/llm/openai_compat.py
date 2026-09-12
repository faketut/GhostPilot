"""OpenAI-compatible provider (OpenAI, DeepSeek, Ollama, etc.)."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

from src.llm.base import Delta, LLMProvider, Usage

logger = logging.getLogger(__name__)


def _limit_kwarg(key: str, value: int | None) -> dict:
    """Build the output-cap kwarg, omitting it entirely when unset.

    Omitting the key is what makes a call uncapped; passing
    ``max_tokens=None`` would instead serialise ``"max_tokens": null``. That
    happens to be tolerated by DeepSeek but is not part of the OpenAI schema, so
    the field is dropped rather than nulled.
    """
    return {} if value is None else {key: value}


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, *, api_key: str, base_url: str | None = None, label: str | None = None):
        kwargs: dict = {"api_key": api_key or "sk-none"}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self.name = label or ("openai" if not base_url else base_url)
        self._base_url = base_url

    async def chat_complete(self, messages, *, model: str, max_tokens: int | None = 256, temperature: float = 0.1) -> str:
        resp = await self._client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, stream=False,
            **_limit_kwarg("max_tokens", max_tokens),
        )
        return resp.choices[0].message.content or ""

    async def chat_stream(self, messages, *, model: str, max_tokens: int | None = None, temperature: float = 0.25) -> AsyncIterator[Delta]:
        stream = await self._client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            **_limit_kwarg("max_tokens", max_tokens),
            stream=True, stream_options={"include_usage": True} if not self._base_url or "openai" in (self._base_url or "") or "deepseek" in (self._base_url or "") else None,
        )
        try:
            async for chunk in stream:
                delta_text = ""
                finish_reason = None
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta_text = (choice.delta.content or "") if choice.delta else ""
                    # Arrives on its own chunk (empty delta), so it must count
                    # towards "something to yield" or it would be dropped.
                    finish_reason = getattr(choice, "finish_reason", None)
                usage = None
                if getattr(chunk, "usage", None):
                    details = getattr(chunk.usage, "completion_tokens_details", None)
                    usage = Usage(
                        in_tokens=getattr(chunk.usage, "prompt_tokens", 0) or 0,
                        out_tokens=getattr(chunk.usage, "completion_tokens", 0) or 0,
                        # Absent on non-reasoning servers; 0 is the right default.
                        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
                    )
                if delta_text or usage or finish_reason:
                    yield Delta(text=delta_text, usage=usage, finish_reason=finish_reason)
        finally:
            try:
                await stream.close()
            except Exception:
                pass
