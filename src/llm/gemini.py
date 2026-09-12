"""Google Gemini provider (google-genai SDK)."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from src.llm.base import Delta, LLMProvider, Usage

logger = logging.getLogger(__name__)


def _gemini_limit(value: int | None) -> dict:
    """Output-cap field for a GenerateContentConfig, or nothing when uncapped.

    google-genai requires an explicit value in the config dict, so an absent
    field (rather than a `None`) is what leaves the model on its own limit.
    """
    return {} if value is None else {"max_output_tokens": value}


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

    async def chat_complete(self, messages, *, model: str, max_tokens: int | None = 256, temperature: float = 0.1) -> str:
        from google.genai import types
        client = self._ensure_client()
        # `messages` is OpenAI-style; flatten to one text input for the classifier.
        text = "\n\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        resp = await client.aio.models.generate_content(
            model=model, contents=text,
            config=types.GenerateContentConfig(
                temperature=temperature, **_gemini_limit(max_tokens),
            ),
        )
        return getattr(resp, "text", "") or ""

    async def chat_stream(self, messages, *, model: str, max_tokens: int | None = None, temperature: float = 0.25) -> AsyncIterator[Delta]:
        from google.genai import types
        client = self._ensure_client()
        text = "\n\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        stream = await client.aio.models.generate_content_stream(
            model=model, contents=text,
            config=types.GenerateContentConfig(
                temperature=temperature, **_gemini_limit(max_tokens),
            ),
        )
        try:
            async for chunk in stream:
                t = getattr(chunk, "text", "") or ""
                usage = self._usage_from(chunk)
                finish_reason = None
                candidates = getattr(chunk, "candidates", None) or []
                if candidates:
                    fr = getattr(candidates[0], "finish_reason", None)
                    if fr is not None:
                        # google-genai reports an enum; its str() form is the
                        # readable name (e.g. "FinishReason.MAX_TOKENS").
                        finish_reason = getattr(fr, "name", None) or str(fr).rsplit(".", 1)[-1]
                if t or usage or finish_reason:
                    yield Delta(text=t, usage=usage, finish_reason=finish_reason)
        finally:
            close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
            if close:
                try:
                    res = close()
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    pass
