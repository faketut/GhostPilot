"""Provider ABC + small data classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class Usage:
    in_tokens: int = 0
    out_tokens: int = 0
    # Reasoning models bill hidden chain-of-thought tokens inside `out_tokens`
    # (DeepSeek reports them as `completion_tokens_details.reasoning_tokens`),
    # so `max_tokens` is a budget for reasoning + answer *combined*. When the
    # trace eats it, the visible answer is cut with no other symptom than a
    # `"length"` stop — hence this figure is carried for the truncation notice.
    reasoning_tokens: int = 0
    # Provider-reported cost when available; else None and the caller computes
    # it from pricing.py.
    cost_usd: float | None = None


@dataclass
class Delta:
    """One chunk from a streaming response.

    `text` may be empty for the final chunk that only carries `usage` or
    `finish_reason`.

    `finish_reason` is the provider's stop reason (`"stop"`, `"length"`,
    `"MAX_TOKENS"`, …) or None while the stream is still open. It is the only
    signal that distinguishes "the model finished" from "the output cap cut it
    off", so callers must not drop it: a `"length"` stop mid-code-block is a
    truncated answer, not a complete one.
    """
    text: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Common surface for text LLMs."""

    name: str = "abstract"

    @abstractmethod
    async def chat_complete(self, messages: list[dict], *, model: str, max_tokens: int | None = 256, temperature: float = 0.1) -> str:
        """Non-streaming completion. Used for the classifier."""

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int | None = None,
        temperature: float = 0.25,
    ) -> AsyncIterator[Delta]:
        """Streaming text completion. Yields deltas; the final delta may carry `usage`.

        `max_tokens=None` means "no cap": the provider's own output limit
        applies. A caller-imposed cap is a trap for reasoning models — the
        hidden chain-of-thought is billed against the same budget *before* the
        answer is written, so a cap can be consumed entirely by reasoning and
        leave the visible answer empty (measured: `deepseek-flash` at
        `max_tokens=1200` streamed 1200 reasoning tokens and zero characters of
        answer). Only cap a call whose output size is genuinely bounded, such as
        the classifier's one-word JSON reply.
        """
