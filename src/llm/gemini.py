"""Google Gemini provider (google-genai SDK)."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from src.llm.base import Delta, LLMProvider, Usage

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, *, api_key: str):
        self._api_key = api_key
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _usage_from(resp) -> Usage | None:
        um = getattr(resp, "usage_metadata", None)
        if not um:
            return None
        return Usage(
            in_tokens=getattr(um, "prompt_token_count", 0) or 0,
            out_tokens=getattr(um, "candidates_token_count", 0) or 0,
        )

    async def chat_complete(self, messages, *, model: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        from google.genai import types
        client = self._ensure_client()
        # `messages` is OpenAI-style; flatten to one text input for the classifier.
        text = "\n\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        resp = await client.aio.models.generate_content(
            model=model, contents=text,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=temperature),
        )
        return getattr(resp, "text", "") or ""

    async def chat_stream(self, messages, *, model: str, max_tokens: int = 512, temperature: float = 0.25) -> AsyncIterator[Delta]:
        # Text-only — used rarely (most Gemini use is vision).
        from google.genai import types
        client = self._ensure_client()
        text = "\n\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        stream = await client.aio.models.generate_content_stream(
            model=model, contents=text,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=temperature),
        )
        try:
            async for chunk in stream:
                t = getattr(chunk, "text", "") or ""
                usage = self._usage_from(chunk)
                if t or usage:
                    yield Delta(text=t, usage=usage)
        finally:
            close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
            if close:
                try:
                    res = close()
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    pass

    async def vision_stream(self, contents, *, model: str, system_prompt: str, max_tokens: int = 550, temperature: float = 0.25) -> AsyncIterator[Delta]:
        from google.genai import types
        client = self._ensure_client()
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        stream = await client.aio.models.generate_content_stream(
            model=model, contents=contents, config=cfg,
        )
        try:
            async for chunk in stream:
                t = getattr(chunk, "text", "") or ""
                usage = self._usage_from(chunk)
                if t or usage:
                    yield Delta(text=t, usage=usage)
        finally:
            close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
            if close:
                try:
                    res = close()
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    pass
