"""OpenAI-compatible provider (OpenAI, DeepSeek, Ollama, etc.)."""

from __future__ import annotations

import base64
import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

from src.llm.base import Delta, LLMProvider, Usage

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, *, api_key: str, base_url: str | None = None, label: str | None = None):
        kwargs: dict = {"api_key": api_key or "sk-none"}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self.name = label or ("openai" if not base_url else base_url)
        self._base_url = base_url

    async def chat_complete(self, messages, *, model: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        resp = await self._client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=False,
        )
        return resp.choices[0].message.content or ""

    async def chat_stream(self, messages, *, model: str, max_tokens: int = 512, temperature: float = 0.25) -> AsyncIterator[Delta]:
        stream = await self._client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            stream=True, stream_options={"include_usage": True} if not self._base_url or "openai" in (self._base_url or "") or "deepseek" in (self._base_url or "") else None,
        )
        try:
            async for chunk in stream:
                delta_text = ""
                if chunk.choices:
                    delta_text = (chunk.choices[0].delta.content or "") if chunk.choices[0].delta else ""
                usage = None
                if getattr(chunk, "usage", None):
                    usage = Usage(
                        in_tokens=getattr(chunk.usage, "prompt_tokens", 0) or 0,
                        out_tokens=getattr(chunk.usage, "completion_tokens", 0) or 0,
                    )
                if delta_text or usage:
                    yield Delta(text=delta_text, usage=usage)
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    async def vision_stream(self, messages, *, model: str, system_prompt: str, max_tokens: int = 550, temperature: float = 0.25) -> AsyncIterator[Delta]:
        # `messages` here is already an OpenAI-style content list. The caller
        # builds it (text + image_url blocks).
        async for d in self.chat_stream(messages, model=model, max_tokens=max_tokens, temperature=temperature):
            yield d

    @staticmethod
    def b64_image_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64}"
