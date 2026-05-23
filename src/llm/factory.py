"""Build providers from current `config`."""

from __future__ import annotations

import logging

from src.config import config
from src.llm.base import LLMProvider
from src.llm.gemini import GeminiProvider
from src.llm.openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)


def _is_deepseek(model: str) -> bool:
    return "deepseek" in (model or "").lower()


def _is_gemini(model: str) -> bool:
    return "gemini" in (model or "").lower()


def _is_ollama_model(model: str) -> bool:
    # Heuristic: anything starting with a recognised local-model family OR
    # explicitly prefixed "ollama/".
    m = (model or "").lower()
    return m.startswith("ollama/") or m.startswith("llama") or m.startswith("qwen") or m.startswith("mistral") or m.startswith("phi")


def _ollama_base() -> str:
    return getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")


def make_text_provider() -> LLMProvider:
    provider = (getattr(config, "TEXT_PROVIDER", "") or "").strip().lower()
    model = config.TEXT_MODEL or ""

    # Explicit provider always wins; model-name inference only kicks in when
    # provider is left blank (auto).
    if provider == "openai":
        return OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai")
    if provider == "ollama" or (not provider and _is_ollama_model(model)):
        return OpenAICompatProvider(api_key="ollama", base_url=_ollama_base(), label="ollama")
    if provider == "deepseek" or (not provider and _is_deepseek(model)):
        return OpenAICompatProvider(
            api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", label="deepseek",
        )
    if provider == "gemini" or (not provider and _is_gemini(model)):
        return GeminiProvider(api_key=config.GEMINI_API_KEY)
    # Default → OpenAI
    return OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai")


def make_vision_provider() -> LLMProvider:
    provider = (getattr(config, "VISION_PROVIDER", "") or "").strip().lower()
    model = config.VISION_MODEL or ""

    if provider == "openai":
        return OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai-vision")
    if provider == "gemini" or (not provider and _is_gemini(model)):
        return GeminiProvider(api_key=config.GEMINI_API_KEY)
    if not provider and _is_deepseek(model):
        # DeepSeek vision: not supported; raise at first use via engine.
        # Still return a placeholder so construction succeeds.
        return OpenAICompatProvider(api_key=config.DEEPSEEK_API_KEY, label="deepseek-vision-unsupported")
    return OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai-vision")
