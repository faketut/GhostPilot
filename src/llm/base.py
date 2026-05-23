"""Provider ABC + small data classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass(slots=True)
class Usage:
    in_tokens: int = 0
    out_tokens: int = 0
    # Provider-reported cost when available; else None and the caller computes
    # it from pricing.py.
    cost_usd: float | None = None


@dataclass(slots=True)
class Delta:
    """One chunk from a streaming response.

    `text` may be empty for the final chunk that only carries `usage`.
    """
    text: str = ""
    usage: Usage | None = None


class LLMProvider(ABC):
    """Common surface for text / vision LLMs."""

    name: str = "abstract"

    @abstractmethod
    async def chat_complete(self, messages: list[dict], *, model: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        """Non-streaming completion. Used for the classifier."""

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.25,
    ) -> AsyncIterator[Delta]:
        """Streaming text completion. Yields deltas; the final delta may carry `usage`."""

    @abstractmethod
    def vision_stream(
        self,
        messages_or_parts,
        *,
        model: str,
        system_prompt: str,
        max_tokens: int = 550,
        temperature: float = 0.25,
    ) -> AsyncIterator[Delta]:
        """Streaming vision completion. Image bytes are embedded in the input."""
